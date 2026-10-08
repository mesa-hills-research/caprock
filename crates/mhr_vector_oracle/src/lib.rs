//! The exact oracle for `mhr_vector`.
//!
//! This crate is deliberately simple and independent of `mhr_vector`: it shares the
//! *specification* (docs/requirements.md, "Scores" and "Filters"), not code. Everything here
//! favours obvious correctness over speed:
//!
//! - [`canonical`]: the canonical score arithmetic (the 16-lane dot product, squared L2,
//!   normalization, validation). Scores an index returns must equal these bit for bit
//!   whenever full-precision rescoring is on.
//! - [`filter`]: the path-prefix and language filter semantics.
//! - [`Oracle`]: a `BTreeMap` of live records, searched by scoring every record and
//!   sorting by (score descending, key ascending).
//! - [`check`]: compares an index's hits with the oracle: ordering, duplicates, filters,
//!   counts, score exactness, exact equality and recall.
//! - [`generate`]: seeded generators for vectors, paths, filters and operation sequences, with
//!   a PRNG defined here so datasets never change with a dependency upgrade.
//! - [`dataset`]: the raw on-disk dataset format the benchmark harness and scripts share.
//! - [`scenario`]: operation sequences run against any index through
//!   [`scenario::IndexUnderTest`], with the oracle replaying them.
//! - [`gt`]: batch ground truth over a dataset (the same arithmetic as [`Oracle::search`],
//!   spread over threads).

#![forbid(unsafe_code)]

pub mod canonical;
pub mod check;
pub mod dataset;
pub mod filter;
pub mod generate;
pub mod gt;
mod oracle;
pub mod rng;
pub mod scenario;

pub use canonical::Metric;
pub use filter::Filter;
pub use oracle::{Entry, Hit, Oracle, OracleError, hit_order, validate_key, validate_meta};

/// Longest key, in bytes (keys are 1..=255 bytes of UTF-8).
pub const MAX_KEY_BYTES: usize = 255;
/// Longest path, in bytes.
pub const MAX_PATH_BYTES: usize = 4096;
/// Longest language name, in bytes.
pub const MAX_LANGUAGE_BYTES: usize = 64;
/// Largest dimension accepted.
pub const MAX_DIM: usize = 4096;
