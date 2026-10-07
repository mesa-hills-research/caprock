//! Host tooling for `mhr_vector`: the benchmark harness and the conformance runner.
//! See README.md (harness) and SWARM-PACK.md (how the swarm's gate uses them).

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
