#!/usr/bin/env python3
"""Download 1M OpenAI text-embedding-3-large vectors (DBpedia entities, 1536 dimensions,
Hugging Face: Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M, MIT) and
write raw datasets at several dimensions:

    python3 scripts/fetch_openai3.py --out datasets --dims 1536,1024,256

writes datasets/dbp3l-1536, dbp3l-1024 and dbp3l-256: the last 10,000 rows are the
queries, the first 990,000 the base. text-embedding-3 models are Matryoshka-trained:
the first d components of a vector are its d-dimensional embedding up to scale, and the
store's cosine metric normalizes, so a prefix is a faithful lower-dimension vector (the
same thing voyage-code-4's 256/512/1024/2048 options are). These are the closest public
stand-in for voyage-code-4 at scale until the code set exists (build_code_set.py).

About 9.6 GB of parquet is downloaded (deleted afterwards unless --keep-shards); the
outputs take 4 bytes x dims x 1M each (6.1 GB at 1536). Needs pyarrow and numpy.
"""

import argparse
import json
import os
import urllib.request

USER_AGENT = "mhr-vector-dataset-fetch/0.1"
REPO = "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M"
SHARDS = 26
COLUMN = "text-embedding-3-large-1536-embedding"
TOTAL = 1_000_000


def shard_url(i):
    return f"https://huggingface.co/datasets/{REPO}/resolve/main/data/train-{i:05d}-of-{SHARDS:05d}.parquet"


def download(url, path):
    if os.path.exists(path):
        return
    tmp = path + ".part"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response, open(tmp, "wb") as out:
        while True:
            block = response.read(1 << 22)
            if not block:
                break
            out.write(block)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--dims", default="1536,1024,256")
    p.add_argument("--queries", type=int, default=10_000)
    p.add_argument("--max-base", type=int, default=0, help="keep the first N base rows")
    p.add_argument("--shards-dir", help="download directory (default: <out>/.dbp3l-shards)")
    p.add_argument("--keep-shards", action="store_true")
    a = p.parse_args()

    import numpy as np
    import pyarrow.parquet as pq

    dims = [int(d) for d in a.dims.split(",")]
    base_n = TOTAL - a.queries
    if a.max_base:
        base_n = min(base_n, a.max_base)
    shards_dir = a.shards_dir or os.path.join(a.out, ".dbp3l-shards")
    os.makedirs(shards_dir, exist_ok=True)
    outs = {}
    for d in dims:
        out = os.path.join(a.out, f"dbp3l-{d}")
        os.makedirs(out, exist_ok=True)
        outs[d] = (open(os.path.join(out, "base.f32"), "wb"), open(os.path.join(out, "queries.f32"), "wb"))
    row = 0
    written_q = 0
    for i in range(SHARDS):
        path = os.path.join(shards_dir, f"train-{i:05d}.parquet")
        print(f"shard {i + 1}/{SHARDS}", flush=True)
        download(shard_url(i), path)
        table = pq.read_table(path, columns=[COLUMN])
        column = table.column(COLUMN)
        parts = [
            np.asarray(chunk.flatten().to_numpy(zero_copy_only=False), dtype=np.float64).reshape(-1, 1536)
            for chunk in column.chunks
        ]
        vectors = np.concatenate(parts).astype("<f4")
        if len(vectors) != table.num_rows:
            raise SystemExit(f"shard {i}: {len(vectors)} vectors for {table.num_rows} rows (null or short lists)")
        lo, hi = row, row + len(vectors)
        base_rows = (lo, min(hi, base_n))
        query_rows = (max(lo, TOTAL - a.queries), hi)
        for d in dims:
            if base_rows[0] < base_rows[1]:
                vectors[base_rows[0] - lo:base_rows[1] - lo, :d].tofile(outs[d][0])
            if query_rows[0] < query_rows[1]:
                vectors[query_rows[0] - lo:query_rows[1] - lo, :d].tofile(outs[d][1])
        if query_rows[0] < query_rows[1]:
            written_q += query_rows[1] - query_rows[0]
        row += len(vectors)
        if not a.keep_shards:
            os.remove(path)
    for d in dims:
        outs[d][0].close()
        outs[d][1].close()
        info = {
            "format": "mhr-vector-dataset/1",
            "name": f"dbp3l-{d}",
            "metric": "cosine",
            "dim": d,
            "base_count": base_n,
            "query_count": written_q,
            "base": "base.f32",
            "queries": "queries.f32",
            "source": {
                "huggingface": REPO,
                "model": "text-embedding-3-large",
                "truncated_from": 1536,
                "split": f"base = rows 0..{base_n}, queries = the last {a.queries} rows",
                "license": "MIT (dataset card)",
                "converted_by": "scripts/fetch_openai3.py",
            },
        }
        with open(os.path.join(a.out, f"dbp3l-{d}", "dataset.json"), "w") as w:
            json.dump(info, w, indent=2)
            w.write("\n")
        print(f"dbp3l-{d}: {base_n} base, {written_q} queries", flush=True)


if __name__ == "__main__":
    main()
