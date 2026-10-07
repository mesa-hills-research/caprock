# Host-owned paths

Agents never change these; the gate refuses a branch that does (before any check runs).
The host (the owner, or the coordinating session) changes them, and every change resets
any unit or bucket whose contract it touches.

| Path | Why |
|---|---|
| `REQUIREMENTS.md`, `UNITS.md`, `SWARM-PACK.md`, `HOST_OWNED.md`, `README.md`, `units.json` | the contract and the plan |
| `contract/` | the frozen interface (`signatures.txt`) |
| `Cargo.toml`, `Cargo.lock`, `crates/*/Cargo.toml` | dependencies are an owner decision |
| `crates/mhr_vector_oracle/` | ground truth |
| `crates/mhr_vector_bench/` | the harness, the score, the conformance checks, the property tests |
| `crates/mhr_vector/tests/` | per-unit acceptance tests (and `codec_checks.inc`, `common/`) |
| `crates/mhr_vector/src/lib.rs` | module layout and re-exports (the skeleton `allow` line is removed by the host at the end of the build phase) |
| `crates/mhr_vector/src/config.rs`, `types.rs`, `error.rs` | the public data types and validation |
| `workloads/`, `scripts/` | what is measured and how datasets are made |
| `.gitignore`, `rustfmt.toml` (if added), CI files | repository settings |

Everything else under `crates/mhr_vector/src/` belongs to exactly one unit (UNITS.md;
`src/store/mod.rs` to U17, which U18 may also extend). In the tidy and optimization
phases a bucket's scope names the files it may change.
