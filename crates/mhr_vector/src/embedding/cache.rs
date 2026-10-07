//! The embedding cache (unit U19).
//!
//! One directory per model under a root: `<root>/<document fingerprint>/`, holding a
//! `model.json` (the [`EmbeddingModel`]) and an append-only log of records:
//!
//! ```text
//! log file "cache-<20 digits>.log": magic b"MHRVEMB1", dimension u32, then records:
//!   content hash [u8; 32], dimension x f32 (little-endian), CRC-32 of the hash and
//!   vector bytes (u32)
//! ```
//!
//! Opening replays the newest log into an in-memory map from hash to file offset
//! (memory: O(entries), about 48 bytes each), truncating a torn or corrupt tail (the
//! records before it stay). A later record for the same hash wins. Only one process may
//! hold a model's cache open for writing (a lock file); [`EmbeddingCache::retain`]
//! rewrites the live records into the next log and switches atomically.

use std::path::Path;

use crate::embedding::{ContentHash, EmbeddingModel};
use crate::error::Result;

/// Cache counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CacheStats {
    pub entries: u64,
    pub log_bytes: u64,
    /// Records dropped at open (a torn tail).
    pub truncated_records: u64,
}

/// A model's embedding cache.
#[derive(Debug)]
pub struct EmbeddingCache {
    model: EmbeddingModel,
}

impl EmbeddingCache {
    /// Opens (or creates) `root/<model.document_fingerprint()>/`. `ModelMismatch` if the
    /// directory's `model.json` describes a model that `requires_reembed` relative to
    /// `model`; `Locked` if another handle has it open.
    pub fn open(root: &Path, model: &EmbeddingModel) -> Result<EmbeddingCache> {
        todo!("U19 embedding: EmbeddingCache::open")
    }

    pub fn model(&self) -> &EmbeddingModel {
        todo!("U19 embedding: EmbeddingCache::model")
    }

    pub fn len(&self) -> usize {
        todo!("U19 embedding: EmbeddingCache::len")
    }

    pub fn is_empty(&self) -> bool {
        todo!("U19 embedding: EmbeddingCache::is_empty")
    }

    pub fn contains(&self, hash: &ContentHash) -> bool {
        todo!("U19 embedding: EmbeddingCache::contains")
    }

    /// The cached vector, if any (read from the log; checksum verified).
    pub fn get(&self, hash: &ContentHash) -> Result<Option<Vec<f32>>> {
        todo!("U19 embedding: EmbeddingCache::get")
    }

    /// The positions in `hashes` with no cached vector, ascending (what to embed).
    pub fn missing(&self, hashes: &[ContentHash]) -> Vec<usize> {
        todo!("U19 embedding: EmbeddingCache::missing")
    }

    /// Adds vectors (`InvalidVector` on a wrong dimension or a non-finite value; nothing
    /// is added then). Buffered: durable after [`flush`](Self::flush).
    pub fn put(&mut self, items: &[(ContentHash, &[f32])]) -> Result<()> {
        todo!("U19 embedding: EmbeddingCache::put")
    }

    /// Writes buffered records and fsyncs the log.
    pub fn flush(&mut self) -> Result<()> {
        todo!("U19 embedding: EmbeddingCache::flush")
    }

    /// Keeps only the entries `keep` accepts, rewriting the log; returns how many were
    /// dropped.
    pub fn retain(&mut self, keep: &dyn Fn(&ContentHash) -> bool) -> Result<usize> {
        todo!("U19 embedding: EmbeddingCache::retain")
    }

    pub fn stats(&self) -> CacheStats {
        todo!("U19 embedding: EmbeddingCache::stats")
    }
}
