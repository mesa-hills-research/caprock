//! The benchmark harness and the conformance checks. docs/benchmarks.md describes the
//! harness.

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
