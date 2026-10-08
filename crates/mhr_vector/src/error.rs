//! Errors. Every refusal of bad input is an `Err`, never a panic.

use std::path::PathBuf;

use crate::embedding::EmbeddingModel;

/// The result type of this crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Why a vector was refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum VectorError {
    #[error("expected {expected} dimensions, found {found}")]
    Dimension { expected: usize, found: usize },
    #[error("component {index} is not finite")]
    NotFinite { index: usize },
    #[error("component {index} exceeds 1e15 in magnitude")]
    TooLarge { index: usize },
    #[error("zero vector (cosine needs a direction)")]
    Zero,
    #[error("vector cannot be normalized")]
    NotNormalizable,
}

/// Everything that can go wrong.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An operating-system error on a file of the index.
    #[error("I/O error on {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// A file failed a format or checksum check.
    #[error("corrupt index file {path}: {detail}")]
    Corrupt { path: PathBuf, detail: String },
    /// A file was written by another format version.
    #[error("unsupported index format version {found} (this build reads {supported})")]
    UnsupportedFormat { found: u32, supported: u32 },
    /// A vector (record or query) was refused.
    #[error("invalid vector: {0}")]
    InvalidVector(#[from] VectorError),
    /// A key was empty or longer than [`crate::MAX_KEY_BYTES`].
    #[error("invalid key: {0}")]
    InvalidKey(String),
    /// A path or language was too long.
    #[error("invalid record: {0}")]
    InvalidRecord(String),
    /// A call's argument was out of range (`k` of 0 or above [`crate::MAX_K`], ...).
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    /// An [`crate::IndexConfig`] or [`crate::StoreOptions`] value was out of range.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    /// The index was built for another embedding model: its vectors are not comparable
    /// with the requested model's. Re-embed into a new index.
    #[error("the index holds {stored} embeddings, not {requested}: re-embed into a new index")]
    ModelMismatch {
        stored: Box<EmbeddingModel>,
        requested: Box<EmbeddingModel>,
    },
    /// Another process (or another `Writer`) holds the index's writer lock.
    #[error("{path} is locked by another writer")]
    Locked { path: PathBuf },
    /// `create` found an index already there.
    #[error("an index already exists at {0}")]
    AlreadyExists(PathBuf),
    /// `open` found no index.
    #[error("no index at {0}")]
    NotFound(PathBuf),
    /// The build memory budget cannot hold even a minimal segment.
    #[error("memory budget of {budget} bytes is too small: {detail}")]
    Budget { budget: u64, detail: String },
    /// A maintenance job was installed into a writer other than the one that planned it,
    /// or after its inputs were replaced.
    #[error("stale maintenance job: {0}")]
    StaleJob(String),
}

impl Error {
    /// An [`Error::Io`] for `path`.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Error {
        Error::Io {
            path: path.into(),
            source,
        }
    }

    /// An [`Error::Corrupt`] for `path`.
    pub fn corrupt(path: impl Into<PathBuf>, detail: impl Into<String>) -> Error {
        Error::Corrupt {
            path: path.into(),
            detail: detail.into(),
        }
    }
}
