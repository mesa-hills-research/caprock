//! `mhr_vector`: a pure-Rust vector store for code search.
//!
//! - **Exact and approximate search.** Every index can answer exactly
//!   ([`SearchParams::exact`], or an index made with [`IndexConfig::exact`]); large
//!   segments also carry a Vamana graph (the DiskANN family) for approximate search.
//! - **Quantization with rescoring.** Each segment keeps compact codes in memory
//!   ([`CodeKind::Int8`] or [`CodeKind::Binary`]) to find candidates and rescores them with
//!   the full-precision vectors kept on disk, so returned scores are exact.
//! - **Deterministic.** Scores follow a canonical arithmetic ([`distance`]), ties break by
//!   key, and builds give the same bytes whatever the thread count.
//! - **Incremental and crash-safe.** One [`Writer`] applies upserts and deletes, visible
//!   to new [`Snapshot`]s at once; [`Writer::commit`] makes them durable by switching to a
//!   new generation atomically. Many readers search snapshots concurrently.
//! - **Bounded memory.** Builds stay within [`StoreOptions::build_memory_budget`].
//! - **Embedding-aware.** An index records its [`EmbeddingModel`]; opening it for another
//!   model fails with [`Error::ModelMismatch`]. [`EmbeddingCache`] keeps vectors by
//!   (model, content hash) so unchanged chunks are never embedded twice.
//!
//! The contract is REQUIREMENTS.md at the repository root; the oracle crate
//! (`mhr_vector_oracle`) defines ground truth.
//!
//! ```no_run
//! use mhr_vector::{EmbeddingModel, Filter, IndexConfig, Op, Record, SearchParams, StoreOptions, Writer};
//!
//! # fn main() -> mhr_vector::Result<()> {
//! let model = EmbeddingModel::voyage_code_4(1024);
//! let config = IndexConfig::for_model(model);
//! let mut writer = Writer::create("/tmp/index".as_ref(), config, StoreOptions::default())?;
//! let vector = vec![0.1f32; 1024];
//! writer.apply(&[Op::Upsert(Record {
//!     key: "3f2a...-0",
//!     vector: &vector,
//!     path: "src/lib.rs",
//!     language: "rust",
//! })])?;
//! let filter = Filter::path("src");
//! let hits = writer.snapshot().search(&vector, 10, Some(&filter), &SearchParams::default())?;
//! writer.commit(None)?;
//! # let _ = hits;
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]
// Skeleton phase: stub bodies leave parameters and helpers unused. The host removes this
// line when the build phase ends (UNITS.md, "End of the build phase").
#![allow(
    unused_variables,
    dead_code,
    unreachable_code,
    unused_imports,
    unused_macros
)]

pub mod config;
pub mod distance;
pub mod embedding;
pub mod error;
pub mod filter;
pub mod store;
pub mod types;

/// Declares the internal unit modules: private normally, public (and hidden) with the
/// `internals` feature, which the host's per-unit acceptance tests (tests/uNN_*.rs) use.
/// Nothing in them is a public API.
macro_rules! internal_modules {
    ($($name:ident),* $(,)?) => {
        $(
            #[cfg(feature = "internals")]
            #[doc(hidden)]
            pub mod $name;
            #[cfg(not(feature = "internals"))]
            mod $name;
        )*
    };
}

internal_modules!(
    delta, flat, format, manifest, merge, quant, segment, tombstones, topk, util, vamana,
);

pub use config::{
    AnnConfig, AnnKind, CodeKind, IndexConfig, MergePolicy, Metric, SearchParams, StoreOptions,
    VamanaParams, VerifyMode,
};
pub use embedding::{CacheStats, ContentHash, EmbeddingCache, EmbeddingModel, InputType};
pub use error::{Error, Result, VectorError};
pub use filter::Filter;
pub use store::{BuiltSegment, MaintenanceJob, Reader, Snapshot, Writer};
pub use types::{
    ApplyReport, CommitInfo, Hit, MaintainReport, Op, Record, SegmentStats, Stats, StoredRecord,
};

/// Longest key, in bytes (keys are 1..=255 bytes of UTF-8).
pub const MAX_KEY_BYTES: usize = 255;
/// Longest path, in bytes.
pub const MAX_PATH_BYTES: usize = 4096;
/// Longest language name, in bytes.
pub const MAX_LANGUAGE_BYTES: usize = 64;
/// Largest dimension accepted.
pub const MAX_DIM: u32 = 4096;
/// Largest `k` a search accepts.
pub const MAX_K: usize = 1000;
