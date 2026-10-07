#!/usr/bin/env python3
"""Download ann-benchmarks datasets (HDF5) and convert them to the raw dataset format
of mhr_vector_bench (crates/mhr_vector_oracle/src/dataset.rs).

    python3 scripts/fetch_ann.py glove-100-angular nytimes-256-angular --out datasets
    python3 scripts/fetch_ann.py glove-25-angular --out datasets --max-base 100000 --max-queries 1000

Needs h5py and numpy (scripts/requirements.txt). The HDF5 file is deleted after the
conversion unless --keep-hdf5. A sliced base (--max-base) drops the source's neighbour
lists (they index the full base); the harness computes exact ground truth itself in any
case (`mhr_vector_bench gt`) and uses the source's lists only as a cross-check.

Angular (cosine) sets can hold zero vectors (nytimes-*-angular does), which a cosine
store refuses by contract: they are dropped from the base and the queries, the counts
are recorded in dataset.json, and the source's neighbour lists are dropped too when a
base row was removed (their indexes would shift).
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request

USER_AGENT = "mhr-vector-dataset-fetch/0.1"
BASE_URL = "http://ann-benchmarks.com/{name}.hdf5"
METRICS = {"angular": "cosine", "euclidean": "l2", "dot": "dot"}
CHUNK_ROWS = 50_000


def download(url, path):
    if os.path.exists(path):
        print(f"using {path}", flush=True)
        return
    tmp = path + ".part"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = response.read(1 << 22)
            if not block:
                break
            out.write(block)
            done += len(block)
            if total:
                print(f"\r{os.path.basename(path)}: {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print(flush=True)
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def convert(name, hdf5_path, out_root, max_base, max_queries, out_name):
    import h5py
    import numpy as np

    with h5py.File(hdf5_path, "r") as f:
        distance = f.attrs.get("distance")
        if isinstance(distance, bytes):
            distance = distance.decode()
        metric = METRICS.get(distance)
        if metric is None:
            sys.exit(f"{name}: unsupported distance {distance!r}")
        train, test = f["train"], f["test"]
        n = train.shape[0] if not max_base else min(max_base, train.shape[0])
        q = test.shape[0] if not max_queries else min(max_queries, test.shape[0])
        dim = train.shape[1]
        out = os.path.join(out_root, out_name)
        os.makedirs(out, exist_ok=True)

        def keep(rows):
            if metric != "cosine":
                return np.ones(len(rows), dtype=bool)
            return np.sum(np.asarray(rows, dtype=np.float64) ** 2, axis=1) >= 1e-30

        kept_base = 0
        with open(os.path.join(out, "base.f32"), "wb") as w:
            for start in range(0, n, CHUNK_ROWS):
                rows = np.asarray(train[start:min(n, start + CHUNK_ROWS)], dtype="<f4")
                rows = rows[keep(rows)]
                kept_base += len(rows)
                rows.tofile(w)
        queries = np.asarray(test[:q], dtype="<f4")
        query_mask = keep(queries)
        queries = queries[query_mask]
        with open(os.path.join(out, "queries.f32"), "wb") as w:
            queries.tofile(w)
        neighbors, neighbors_k = None, 0
        if n == train.shape[0] and kept_base == n and "neighbors" in f:
            neighbors_k = f["neighbors"].shape[1]
            with open(os.path.join(out, "neighbors.i32"), "wb") as w:
                np.asarray(f["neighbors"][:q], dtype="<i4")[query_mask].tofile(w)
            neighbors = "neighbors.i32"
        dropped_base, dropped_queries = n - kept_base, q - len(queries)
        n, q = kept_base, len(queries)
        info = {
            "format": "mhr-vector-dataset/1",
            "name": out_name,
            "metric": metric,
            "dim": int(dim),
            "base_count": int(n),
            "query_count": int(q),
            "base": "base.f32",
            "queries": "queries.f32",
            "neighbors": neighbors,
            "neighbors_k": int(neighbors_k),
            "source": {
                "url": BASE_URL.format(name=name),
                "sha256": sha256(hdf5_path),
                "distance": distance,
                "full_base": int(train.shape[0]),
                "full_queries": int(test.shape[0]),
                "dropped_zero_base": int(dropped_base),
                "dropped_zero_queries": int(dropped_queries),
                "converted_by": "scripts/fetch_ann.py",
            },
        }
        with open(os.path.join(out, "dataset.json"), "w") as w:
            json.dump(info, w, indent=2)
            w.write("\n")
    print(f"{out}: {n} x {dim} base, {q} queries, {metric}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="+", help="ann-benchmarks dataset names, e.g. glove-100-angular")
    p.add_argument("--out", required=True, help="datasets root")
    p.add_argument("--max-base", type=int, default=0, help="keep the first N base vectors")
    p.add_argument("--max-queries", type=int, default=0, help="keep the first N queries")
    p.add_argument("--name", help="output directory name (one dataset only)")
    p.add_argument("--hdf5-dir", help="where downloads go (default: the datasets root)")
    p.add_argument("--keep-hdf5", action="store_true")
    a = p.parse_args()
    if a.name and len(a.names) != 1:
        sys.exit("--name needs exactly one dataset")
    hdf5_dir = a.hdf5_dir or a.out
    os.makedirs(hdf5_dir, exist_ok=True)
    for name in a.names:
        path = os.path.join(hdf5_dir, f"{name}.hdf5")
        download(BASE_URL.format(name=name), path)
        out_name = a.name or (f"{name}-{a.max_base // 1000}k" if a.max_base else name)
        convert(name, path, a.out, a.max_base, a.max_queries, out_name)
        if not a.keep_hdf5:
            os.remove(path)


if __name__ == "__main__":
    main()
