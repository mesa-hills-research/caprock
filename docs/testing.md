# Testing

Every result is checked against `mhr_vector_oracle`, a deliberately simple brute-force
store with the same contract: the canonical arithmetic, the filter semantics and the
result order of [requirements.md](requirements.md). Exact search must equal the oracle bit
for bit (same keys, same scores, same order), and approximate search is judged by
recall@k against it.

## Tests

```sh
cargo test --workspace --all-features --locked
cargo test -p mhr_vector --features internals --test u07_quant   # one unit's tests
```

- `crates/mhr_vector/tests/uNN_*.rs`: the acceptance tests of the 19 units the crate was
  built from ([development/units.md](development/units.md)). Tests of internal units need
  the `internals` feature, which exposes the internal modules as
  `mhr_vector::__internals` for these tests only.
- `crates/mhr_vector_bench/tests/properties.rs`: property tests. Random operation
  sequences on random data, under random index configurations, are compared with the
  oracle, and proptest shrinks a failure to a small case.
- The oracle's and the harness's own tests.

The test profile builds with `opt-level = 2` because the oracle-backed tests scan many
vectors.

## Conformance checks

`mhr_vector_conform` runs every check of the contract through the public API, each in its
own process, so a crash or a hang is pinned on its check:

```sh
cargo build --release -p mhr_vector_bench
target/release/mhr_vector_conform list
target/release/mhr_vector_conform run --jobs 8
target/release/mhr_vector_conform one determinism/threads
```

The checks cover the canonical kernels, filter semantics, exact search for every metric
and code kind, recall of the approximate configurations, durability (a reopened index
holds exactly what was committed), byte-identical builds across thread counts and
repeats, garbage collection,
concurrent readers, reader refresh, refusals of bad input, model mismatches and the
embedding cache. `run` prints one JSON line per check and a summary, and exits with 1 when
any check fails.

## Benchmarks as a check

`mhr_vector_bench --engine oracle` runs a suite against the oracle itself, where recall is
1.0 and no rule can be broken, which checks the harness. The oracle's ground truth
matches ann-benchmarks' own neighbour lists (random-xs-20-angular: agreement 1.0 at
k = 10). [benchmarks.md](benchmarks.md) describes the harness.

## Formatting and lints

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
python3 scripts/signatures.py --check docs/development/signatures.txt
```

The last command checks that every public item of the skeleton the crate was built from
is still there.
