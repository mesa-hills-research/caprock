# mhr_vector (groundwork)

A greenfield, pure-Rust, safe-Rust vector store for arko_index's dense search, to be
built and optimized by an `agent_swarm` run. This repository holds the groundwork: the
contract, an exact oracle, a benchmark harness, the crate skeleton and the unit plan. The
store itself is not implemented yet: every function body in `crates/mhr_vector` is a
`todo!("Unn ...")` stub.

| Document | What it is |
|---|---|
| [REQUIREMENTS.md](REQUIREMENTS.md) | the behaviour contract, the design choices, the targets |
| [UNITS.md](UNITS.md) / `units.json` | 19 units with scopes, dependencies and acceptance tests |
| [SWARM-PACK.md](SWARM-PACK.md) | how the swarm runs it: gates, buckets, the score, the profile, stopping rules |
| [HOST_OWNED.md](HOST_OWNED.md) | paths agents may not change |

## Layout

```text
crates/mhr_vector          the store: public API + internal units (skeleton), tests/uNN_*.rs
crates/mhr_vector_oracle   the exact oracle: canonical scores, filters, checks, generators,
                           scenarios, the dataset format, batch ground truth
crates/mhr_vector_bench    mhr_vector_bench (harness, score) and mhr_vector_conform (the
                           conformance runner), property tests
contract/signatures.txt    the frozen interface (scripts/signatures.py --check)
workloads/                 score, build-score, guards, full, smoke suites (JSON)
scripts/                   dataset download/conversion, the Voyage code set, box setup
```

## Commands

```sh
cargo test -p mhr_vector_oracle                                    # the oracle's own tests (pass)
cargo test -p mhr_vector --features internals --test u01_distance  # a unit's acceptance test
cargo build --release -p mhr_vector_bench
B=target/release/mhr_vector_bench; C=target/release/mhr_vector_conform

$C list                                  # 56 conformance checks
$C run --jobs 8                          # JSON lines; signatures like "todo: U17" for now

$B synth --out datasets/synth-small --dim 32 --n 20000 --queries 200 --clusters 50
$B gt --dataset synth-small --filters p10,p01,go
$B run --suite workloads/smoke.json --out results/smoke --engine oracle   # the harness on the oracle
$B prepare --workload workloads/smoke-score.json --index-root /tmp/idx --engine oracle
$B score   --workload workloads/smoke-score.json --index-root /tmp/idx --engine oracle
```

`--engine oracle` runs every measurement against the oracle itself: recall 1.0 and no
violations by construction. That checks the harness. `--engine mhr` (the default) measures
the store.

**What the harness measures**:

- build time and peak memory (in a child process);
- insert and delete throughput, and `apply` latency for 100-record batches
  (read-your-writes);
- recall after updates, after `maintain` and after `compact`;
- query latency (p50, p99) and QPS at 1 and N threads, at the default beam and at chosen
  beams, filtered and unfiltered;
- recall@k against the oracle, with every result rule checked;
- disk size;
- warm and cold open time (cold needs root to drop the page cache).

`run` writes `results.json` and `results.md`.

## Datasets

`datasets/` (ignored by git) holds converted sets in the raw format of
`crates/mhr_vector_oracle/src/dataset.rs`: `dataset.json`, `base.f32`, `queries.f32`,
optionally `meta.jsonl`, and cached ground truth in `gt/`.

| Script | Makes |
|---|---|
| `scripts/fetch_ann.py NAME... --out datasets [--max-base N]` | ann-benchmarks sets (h5py, numpy) |
| `scripts/fetch_openai3.py --out datasets` | dbp3l-1536/1024/256: 1M text-embedding-3-large vectors (pyarrow) |
| `scripts/build_code_set.py ... [--dry-run]` | code-voyage4-1024: ~11.5k arko_index chunks of 11 pinned repositories embedded with voyage-code-4, 1,000 doc-comment queries; ~2.8M tokens estimated, ≤ $0.34 |
| `scripts/box_setup.sh` | everything above except the code set, plus synthetic sets and ground truth, on the box |
| `mhr_vector_bench synth` | synthetic sets of any size |

Locally only two small sets are kept: `glove-25-angular-100k` and `synth-small`.

## Validation

Every test here was run against a throwaway reference implementation of all 19 units,
kept **outside** this repository (`~/Desktop/mhr-work/vector-validation/`; not for the
swarm). It passes:

- every acceptance test;
- all 56 conformance checks;
- the property tests;
- the smoke suite through `--engine mhr`, with no violations.

Two test bugs found that way were fixed. The oracle's ground truth matches
ann-benchmarks' own neighbour lists (random-xs-20-angular: agreement 1.0 at k = 10).

## Building

Rust 1.89 or later; no C compiler; no `unsafe` (`#![forbid(unsafe_code)]`). Run `cargo fmt --all` and keep
`cargo clippy --workspace --all-targets --all-features -- -D warnings` clean.
