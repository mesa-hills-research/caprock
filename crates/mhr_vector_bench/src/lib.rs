//! Host tooling for `mhr_vector`: the benchmark harness and the conformance runner.
//! See docs/benchmarks.md (the harness) and docs/development/swarm-pack.md (how the
//! swarm's gate used them).

#![forbid(unsafe_code)]

pub mod adapter;
pub mod args;
pub mod checks;
pub mod commands;
pub mod configs;
pub mod data;
pub mod engine;
pub mod measure;
pub mod report;
