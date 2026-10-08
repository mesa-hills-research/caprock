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
vectors. `exact` is `IndexConfig::exact`: f32 codes, results equal to the oracle's bit for
bit. `vamana-int8` is the default configuration, a Vamana graph over int8 codes, at its
default beam of 64.

| Configuration | Set | Recall@10 | QPS | p50 | p99 | QPS, 6 threads |
|---|---|---:|---:|---:|---:|---:|
| `flat-int8` | code | 1.0000 | 13.1 | 75.8 ms | 86.3 ms | 60.5 |
| `exact` | code | 1.0000 | 9.5 | 104.2 ms | 112.6 ms | 42.6 |
| `vamana-int8` | code | 0.9655 | 440 | 2.17 ms | 4.73 ms | 2,255 |
| `flat-int8` | docs | 1.0000 | 73.2 | 13.8 ms | 15.2 ms | 353 |
| `exact` | docs | 1.0000 | 52.3 | 19.0 ms | 21.3 ms | 264 |
| `vamana-int8` | docs | 0.9987 | 794 | 1.15 ms | 3.09 ms | 4,165 |

Flat search reads every code on every query, so its time grows with the index: about
0.4 µs per vector with int8 codes and 0.5 µs with f32 here. The flat scan scores codes
with the portable query kernel. The AVX2 kernels added in f7ad141 cover code-to-code
scores (graph construction), and extending them to query scores is the next step for flat
search.

### Vamana beams

The beam trades recall for speed. A search always looks at as many candidates as it
rescores (40 for k = 10), so beams of 40 and below behave alike.

| Beam | Code recall@10 | Code QPS | Code p50 | Docs recall@10 | Docs QPS | Docs p50 |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 0.9251 | 682 | 1.39 ms | 0.9964 | 1,202 | 0.77 ms |
| 48 | 0.9416 | 587 | 1.64 ms | 0.9975 | 1,035 | 0.89 ms |
| 56 | 0.9554 | 525 | 1.82 ms | 0.9981 | 932 | 0.98 ms |
| 64 | 0.9655 | 464 | 2.07 ms | 0.9987 | 845 | 1.08 ms |
| 80 | 0.9748 | 374 | 2.55 ms | 0.9991 | 701 | 1.31 ms |
| 96 | 0.9815 | 319 | 3.04 ms | 0.9994 | 624 | 1.48 ms |
| 128 | 0.9900 | 240 | 4.03 ms | 0.9996 | 480 | 1.89 ms |

Beam 64 appears twice in the runs (as the default and on the ladder), at 440 and 464 QPS
for code and 794 and 845 for docs, which gives an idea of the noise.

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
  time). The library code of 464a273, measured on 2026-10-08 with no other heavy job
  running.
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
