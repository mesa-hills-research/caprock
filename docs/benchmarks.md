# Benchmarks

Search speed and recall on 1024-dimensional Voyage embeddings of code and documentation,
and build times. [datasets.md](datasets.md) describes the sets:

- **code**: `code-voyagecode4-200k`, 199,999 code chunks in 8 languages (voyage-code-4),
  searched with the first 1,000 of its doc-comment queries.
- **docs**: `docs-voyage4-200k`, 34,601 chunks of the Cloudflare docs and MDN (voyage-4),
  searched with its 2,000 queries.

## Search

One thread unless noted, k = 10, on this machine (see [Method](#method)). `flat-int8`
scans the int8 codes of every vector and rescores the best 40 with the full-precision
vectors, and `flat-binary` scans binary codes and rescores the best 160. `exact` is
`IndexConfig::exact`: f32 codes, results equal to the oracle's bit for bit. `vamana-int8`
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
The scans use AVX2 kernels, which score four int8 or f32 codes at a time while
prefetching the codes ahead, or eight binary codes in the lanes of one register. Their
scores have the bits of the portable kernels, so every machine returns the same results.
Graph search scores one code at a time with the same int8 kernel. Before these kernels
(e660b65), flat-int8 took 75.3 ms per query on the code set, flat-binary 40.5 ms, exact
104.5 ms and `vamana-int8` 2.04 ms (467 QPS).

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

Beam 64 appears twice in the runs (as the default and on the ladder), at 1,018 and 1,036
QPS for code and 2,223 and 2,249 for docs, which gives an idea of the noise.

## Builds

On this machine with 6 threads and a 2 GiB build budget: loading in batches of 10,000
with a commit every 80,000 vectors, then `compact` into one segment, which builds the
graph for `vamana-int8`. Peak RSS is the build process's. Resident is what the open index
keeps in memory (codes, graph, keys and paths), and the index size counts the files of
the last generation.

| Configuration | Set | Build | Peak RSS | Resident | Index size |
|---|---|---:|---:|---:|---:|
| `flat-int8` | code | 14.2 s | 590 MiB | 213 MiB | 995 MiB |
| `exact` | code | 6.0 s | 1,716 MiB | 799 MiB | 799 MiB |
| `vamana-int8` | code | 246.1 s | 642 MiB | 262 MiB | 1,044 MiB |
| `flat-int8` | docs | 1.6 s | 234 MiB | 38 MiB | 173 MiB |
| `exact` | docs | 0.9 s | 330 MiB | 139 MiB | 139 MiB |
| `vamana-int8` | docs | 16.9 s | 234 MiB | 46 MiB | 182 MiB |

### Graph builds on the build machine

Graph construction runs in parallel since f7ad141 and writes the same bytes as the serial
build it replaced (839e575) at any thread count. A check on the build machine built each
index with both commits and compared a SHA-256 over every file of the index directory:

| Graph | Vectors | 839e575, 1 thread | f7ad141, 1 thread | f7ad141, 16 threads | Same bytes |
|---|---:|---:|---:|---:|---|
| code, int8 | 50,000 | 823.2 s | 236.6 s | 18.9 s | yes |
| code, binary | 50,000 | 50.1 s | 42.4 s | 5.3 s | yes |
| docs, int8 | 34,601 | 307.8 s | 86.2 s | 8.3 s | yes |
| sift-128-euclidean, int8 | 50,000 | 105.3 s | 52.8 s | 4.3 s | yes |

The code int8 graph also took 105.7 s on 2 threads, 87.4 s on 4 and 31.8 s on 8, with the
same bytes. The times are whole `mhr_vector_bench build` runs (load and compact).

## Search on the build machine

The speed score of the optimization phase (score-v2) at 839e575, before f7ad141's
kernels: single-thread QPS of `vamana-int8` at the smallest beam reaching recall@10 0.95,
over all 2,000 queries of each set.

| Set | Beam | Recall@10 | QPS | p50 | p99 | Recall at beam 64 |
|---|---:|---:|---:|---:|---:|---:|
| code | 56 | 0.9510 | 398 | 2.40 ms | 5.37 ms | 0.9620 |
| docs | 10 (40 in effect) | 0.9964 | 923 | 1.00 ms | 2.41 ms | 0.9987 |

The recall guards of the same run measured `flat-int8` on docs at recall@10 1.0000.

## Method

- **This machine:** a 13th Gen Intel Core i9-13980HX, 6 vCPUs of a VMware virtual
  machine (one thread per core, AVX2 and FMA), 22 GiB of memory, Ubuntu 26.04 on Linux
  7.0. Rust 1.97.1, the release profile without `target-cpu` flags (pulp picks AVX2 at run
  time). The search tables are the library code of 8622ce5 and the build table that of
  464a273 (the commits between them changed search only), all measured on 2026-10-08.
- **Build machine:** an AMD EPYC 9354P (32 cores, 64 threads), 755 GB of memory, Ubuntu
  24.04, Rust 1.99. The score ran pinned to four cores reserved for it, with their second
  threads idle. The graph builds were pinned with `taskset`: one core for 1 thread, and 12
  cores with 4 of their second threads for 16.
- **Queries:** the first 1,000 queries of the code set (as in score-v3) and all 2,000 of
  the docs set, on a warm page cache. Each configuration gets one untimed warm-up pass and
  three timed passes over all queries on one thread. The median pass gives the QPS, and its
  per-query latencies give p50 and p99. The 6-thread QPS has each thread run the whole
  query list once.
- **Recall:** an untimed pass with 6 threads, tie-aware recall@10
  (`mhr_vector_oracle::check`: results tied at the 10th score are interchangeable) against
  ground truth from the oracle's exact arithmetic. Every result is also checked against
  the contract's rules, and no run broke one.
- **Reproducing:** with the sets in `$MHRV_DATASETS`,

  ```sh
  cargo build --release -p mhr_vector_bench
  target/release/mhr_vector_bench run --suite workloads/bench-flat.json --out results/flat
  target/release/mhr_vector_bench run --suite workloads/bench-vamana.json --out results/vamana
  ```

## The harness

`mhr_vector_bench` builds indexes, measures them and writes JSON, and `run` turns a suite
(a JSON file of parts in `workloads/`) into `results.json` and `results.md`. It measures:

- build time and peak memory, in a child process
- insert and delete throughput, and the latency of `apply` for batches of 100 records
  (read-your-writes)
- recall after updates, after `maintain` and after `compact`
- query latency (p50, p99) and QPS at 1 and N threads, at the default beam and at chosen
  beams, filtered and unfiltered
- recall@k against the oracle, with every result rule checked
- disk size
- warm and cold open time (cold needs root to drop the page cache)

```sh
cargo build --release -p mhr_vector_bench
B=target/release/mhr_vector_bench
export MHRV_DATASETS=datasets            # converted sets, see datasets.md

$B synth --out datasets/synth-small --dim 32 --n 20000 --queries 200 --clusters 50
$B gt --dataset synth-small --filters p10,p01,go
$B run --suite workloads/smoke.json --out results/smoke
$B run --suite workloads/smoke.json --out results/smoke-oracle --engine oracle
```

`--engine oracle` runs the measurements against the oracle itself, which checks the
harness. The workloads in `workloads/`:

| File | What |
|---|---|
| `score.json` | the speed score of the optimization phase (score-v3): single-thread QPS of `flat-int8` on the two Voyage sets, recall@10 at least 0.95, combined by geometric mean |
| `build-score.json` | the build score: vectors per second to load and compact the two sets with 16 threads |
| `guards.json` | recall floors with filters, k = 50, binary codes and updates |
| `full.json` | the milestone suite: every set at full size, filters, updates, opens, scale |
| `smoke.json`, `smoke-score.json` | small local checks |

`prepare` then `score` measure a score workload, and `build-score` the build workload.
`mhr_vector_bench --help` lists every command.
