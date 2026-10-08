# Benchmarks

Search speed, recall and build times on two sets of 1024-dimensional embeddings:

- **code**: 199,999 chunks of source code from 12 open-source repositories in 8 languages
  (Go, Java, TypeScript, C, C++, Python, JavaScript and Rust), embedded with
  voyage-code-4. The queries are the first 1,000 of 2,000 doc-comment sentences.
- **docs**: 34,601 chunks of the Cloudflare documentation and MDN, embedded with
  voyage-4. The 2,000 queries are 16 questions about Cloudflare and 1,984 page titles and
  headings.

Both use cosine similarity, and recall is measured against an exact search.

## Search

One thread unless noted, k = 10. `flat-int8` scans the int8 codes of every vector and
rescores the best 40 with the full-precision vectors, and `flat-binary` scans binary codes
and rescores the best 160. `exact` scans f32 codes, the vectors themselves. `vamana-int8`
is the default configuration, a Vamana graph over int8 codes, at its default beam of 64.

| Configuration | Set | Recall@10 | QPS | p50 | p99 | QPS, 6 threads |
|---|---|---:|---:|---:|---:|---:|
| `flat-int8` | code | 1.0000 | 75.1 | 13.4 ms | 14.9 ms | 361 |
| `flat-binary` | code | 0.9957 | 110 | 9.03 ms | 10.1 ms | 555 |
| `exact` | code | 1.0000 | 38.3 | 26.2 ms | 28.1 ms | 121 |
| `vamana-int8` | code | 0.9655 | 1,018 | 0.95 ms | 2.01 ms | 4,835 |
| `flat-int8` | docs | 1.0000 | 437 | 2.27 ms | 2.81 ms | 2,022 |
| `flat-binary` | docs | 0.9998 | 590 | 1.69 ms | 1.85 ms | 2,929 |
| `exact` | docs | 1.0000 | 212 | 4.71 ms | 5.23 ms | 737 |
| `vamana-int8` | docs | 0.9987 | 2,223 | 0.41 ms | 1.05 ms | 11,167 |

Flat search reads every code on every query, so its time grows with the index: about
65 ns per vector with int8 codes, 45 ns with binary codes and 130 ns with f32 codes here.
The scans use AVX2 kernels when the CPU has them, which score four int8 or f32 codes at a
time, or eight binary codes in the lanes of one register. They return the same bits as
the portable kernels, so results are the same on every machine.

### Vamana beams

The beam trades recall for speed. A search always looks at as many candidates as it
rescores (40 for k = 10), so beams of 40 and below behave alike.

| Beam | Code recall@10 | Code QPS | Code p50 | Docs recall@10 | Docs QPS | Docs p50 |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 0.9251 | 1,502 | 0.63 ms | 0.9964 | 3,213 | 0.29 ms |
| 48 | 0.9416 | 1,311 | 0.73 ms | 0.9975 | 2,749 | 0.34 ms |
| 56 | 0.9554 | 1,159 | 0.83 ms | 0.9981 | 2,474 | 0.37 ms |
| 64 | 0.9655 | 1,036 | 0.93 ms | 0.9987 | 2,249 | 0.41 ms |
| 80 | 0.9748 | 837 | 1.14 ms | 0.9991 | 1,854 | 0.50 ms |
| 96 | 0.9815 | 732 | 1.29 ms | 0.9994 | 1,685 | 0.55 ms |
| 128 | 0.9900 | 544 | 1.78 ms | 0.9996 | 1,374 | 0.67 ms |

Beam 64 ran twice, as the default and on this ladder, at 1,018 and 1,036 QPS for code and
2,223 and 2,249 for docs, which gives an idea of the noise.

## Builds

Six threads and a 2 GiB build budget: loading in batches of 10,000 with a commit every
80,000 vectors, then `compact` into one segment, which builds the graph for
`vamana-int8`. Peak RSS is the build process's. Resident is what the open index keeps in
memory (codes, graph, keys and paths), and the index size counts the files of the last
generation.

| Configuration | Set | Build | Peak RSS | Resident | Index size |
|---|---|---:|---:|---:|---:|
| `flat-int8` | code | 14.2 s | 590 MiB | 213 MiB | 995 MiB |
| `exact` | code | 6.0 s | 1,716 MiB | 799 MiB | 799 MiB |
| `vamana-int8` | code | 246.1 s | 642 MiB | 262 MiB | 1,044 MiB |
| `flat-int8` | docs | 1.6 s | 234 MiB | 38 MiB | 173 MiB |
| `exact` | docs | 0.9 s | 330 MiB | 139 MiB | 139 MiB |
| `vamana-int8` | docs | 16.9 s | 234 MiB | 46 MiB | 182 MiB |

Graph builds run in parallel and write the same bytes at any thread count. On a 32-core
AMD EPYC 9354P, loading and compacting these sets took:

| Graph | Vectors | 1 thread | 16 threads |
|---|---:|---:|---:|
| code, int8 codes | 50,000 | 236.6 s | 18.9 s |
| code, binary codes | 50,000 | 42.4 s | 5.3 s |
| docs, int8 codes | 34,601 | 86.2 s | 8.3 s |

## Method

- **Machine:** 6 vCPUs of a VMware virtual machine on an Intel Core i9-13980HX (AVX2 and
  FMA), 22 GiB of memory, Ubuntu 26.04 on Linux 7.0, Rust 1.97.1. The release profile
  has no `target-cpu` flags, and the SIMD kernels are picked at run time.
- **Queries:** on a warm page cache, each configuration gets one untimed warm-up pass and
  three timed passes over all queries on one thread. The median pass gives the QPS, and
  its per-query latencies give p50 and p99. For the 6-thread QPS, each thread runs the
  whole query list once.
- **Recall:** recall@10 against an exact brute-force search, with results tied at the
  10th score counted as interchangeable.

## Running the benchmarks

The harness, `caprock_bench`, builds indexes from a dataset, measures them and writes
`results.json` and `results.md`. `workloads/` holds the suites behind the tables above
(`bench-flat.json` and `bench-vamana.json`, which expect the two sets as
`code-voyagecode4-200k` and `docs-voyage4-200k`) and a small one to try first
(`smoke.json`):

```sh
cargo build --release -p caprock_bench
B=target/release/caprock_bench
$B synth --out datasets/synth-small --dim 32 --n 20000 --queries 200 --clusters 50
$B run --suite workloads/smoke.json --out results/smoke
```

### Your own data

A dataset is a folder under `datasets/` (or `$MHRV_DATASETS`, or `--datasets DIR`):

```text
dataset.json   the description below
base.f32       the vectors, row-major little-endian f32, base_count × dim
queries.f32    the queries, query_count × dim
meta.jsonl     optional: one {"key", "path", "language"} object per vector
```

```json
{
  "format": "mhr-vector-dataset/1",
  "name": "my-set",
  "metric": "cosine",
  "dim": 768,
  "base_count": 100000,
  "query_count": 1000,
  "meta": "meta.jsonl"
}
```

`metric` is `cosine`, `dot` or `l2`. Without `meta.jsonl`, records get generated keys,
paths and languages. A suite lists the parts to measure:

```json
{
  "name": "my-suite",
  "parts": [
    {"dataset": "my-set", "queries": 1000, "config": "flat-int8", "k": 10},
    {"dataset": "my-set", "queries": 1000, "config": "vamana-int8", "k": 10,
     "filters": ["path:src", "lang:rust"], "open": true},
    {"dataset": "my-set", "queries": 500, "config": "vamana-int8", "k": 10, "updates": true}
  ],
  "build": {"threads": 8, "budget": 2147483648, "batch": 10000, "commit_every": 100000},
  "timed_repeats": 3,
  "qps_threads": [1, 8],
  "beams": [32, 64, 128]
}
```

- **Configurations:** `exact`, `flat-int8`, `flat-binary`, `vamana-int8`,
  `vamana-binary` and `vamana-f32`, or an object with a `name` and the fields of
  `ConfigSpec` in `crates/caprock_bench/src/configs.rs` (codes, graph parameters,
  rescoring).
- **Filters:** `path:<prefix>` and `lang:<language>` measure filtered search.
- **`open`** times opening the index in a fresh process.
- **`updates`** builds over 90% of the vectors, then inserts the rest, deletes 10% and
  reports insert and delete throughput, the latency of `apply` for batches of 100, and
  recall after the updates, after `maintain` and after `compact`.

The first run computes exact ground truth for every filter and caches it in the dataset's
folder. `caprock_bench --help` lists the commands that run each step on its own.
