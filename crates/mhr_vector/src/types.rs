//! Records, operations, results and statistics.

use crate::config::CodeKind;

/// A record to store: a stable key, its vector, and the metadata filters use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Record<'a> {
    /// 1..=255 bytes; unique in the index. Results break score ties by key (bytes).
    pub key: &'a str,
    /// `dim` finite components, each at most 1e15 in magnitude (non-zero for cosine).
    pub vector: &'a [f32],
    /// A `/`-separated path, such as a file path (at most 4096 bytes). Filters select
    /// records by path prefix.
    pub path: &'a str,
    /// A short label, such as a programming or natural language (at most 64 bytes).
    /// Filters select records by language.
    pub language: &'a str,
}

/// One change. A batch of them is applied in order; later operations on the same key win.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Op<'a> {
    /// Insert the record, replacing any live record with the same key.
    Upsert(Record<'a>),
    /// Remove the record with this key, if live.
    Delete(&'a str),
}

/// One search result.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub key: String,
    /// Higher is better. Exact: every search rescores its candidates at full precision
    /// with the canonical arithmetic ([`crate::distance`]).
    pub score: f32,
    pub path: String,
    pub language: String,
}

/// A live record as stored.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredRecord {
    pub key: String,
    /// The stored form of the vector (normalized for cosine).
    pub vector: Vec<f32>,
    pub path: String,
    pub language: String,
}

/// What [`crate::Writer::apply`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ApplyReport {
    pub upserted: usize,
    /// Deletes of live keys.
    pub deleted: usize,
    /// Deletes of keys that were not live (no-ops).
    pub missing: usize,
    /// Delta seals into new segments this call caused.
    pub sealed: usize,
}

/// What [`crate::Writer::commit`] published.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitInfo {
    pub generation: u64,
    pub live: u64,
    pub segments: usize,
    /// The number of operations applied since the index was created.
    pub version: u64,
}

/// What maintenance did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MaintainReport {
    pub merges: usize,
    pub graphs_built: usize,
    /// Live vectors written into new segments.
    pub vectors_written: u64,
}

/// One segment's statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentStats {
    pub id: u64,
    pub vectors: u64,
    pub live: u64,
    pub has_graph: bool,
    pub codes: CodeKind,
    /// Memory held while open (codes, graph, keys, paths, tombstones).
    pub resident_bytes: u64,
    pub disk_bytes: u64,
}

/// An index's statistics.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Stats {
    pub live: u64,
    /// Records in the in-memory delta.
    pub delta: u64,
    pub segments: Vec<SegmentStats>,
    pub resident_bytes: u64,
    pub disk_bytes: u64,
    /// The last committed generation.
    pub generation: u64,
    /// Operations applied since creation.
    pub version: u64,
}
