//! Sealed segments (unit U11; REQUIREMENTS.md, "Segments" and "On-disk format").
//!
//! A segment is one immutable section file (`seg-<id>.mhrv`) holding records sorted by
//! (path, key): full-precision vectors (read on demand with positional reads; for f32
//! codes the codes section is the full-precision copy and is resident), resident codes, the codec's parameters, keys and a key index, the
//! path and language tables, and optionally a Vamana graph over its ordinals.
//!
//! Resident memory of an open segment: codes, graph, keys, key index, path table and
//! language ids (REQUIREMENTS.md, "Memory"); full-precision vectors stay on disk.

use std::path::Path;

use crate::config::{IndexConfig, VerifyMode};
use crate::error::Result;
use crate::filter::PathTable;
use crate::format::SectionFile;
use crate::quant::AnyCodec;
use crate::topk::KeyLookup;
use crate::vamana::Graph;

/// Records to write into a segment, by index `0..len()`. Keys are distinct.
pub trait RecordSource: Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn key(&self, i: usize) -> &str;
    fn path(&self, i: usize) -> &str;
    fn language(&self, i: usize) -> &str;
    /// Copies record `i`'s stored-form vector into `out` (`dim` long).
    fn read_vector(&self, i: usize, out: &mut [f32]) -> Result<()>;
}

/// Records held in memory (tests, and small callers).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MemoryRecords {
    pub dim: usize,
    pub keys: Vec<String>,
    pub paths: Vec<String>,
    pub languages: Vec<String>,
    /// Stored-form vectors, flattened.
    pub vectors: Vec<f32>,
}

impl RecordSource for MemoryRecords {
    fn len(&self) -> usize {
        todo!("U11 segment: MemoryRecords::len")
    }

    fn key(&self, i: usize) -> &str {
        todo!("U11 segment: MemoryRecords::key")
    }

    fn path(&self, i: usize) -> &str {
        todo!("U11 segment: MemoryRecords::path")
    }

    fn language(&self, i: usize) -> &str {
        todo!("U11 segment: MemoryRecords::language")
    }

    fn read_vector(&self, i: usize, out: &mut [f32]) -> Result<()> {
        todo!("U11 segment: MemoryRecords::read_vector")
    }
}

/// How to build a segment.
#[derive(Debug, Clone, Copy)]
pub struct SegmentBuild<'a> {
    pub id: u64,
    pub config: &'a IndexConfig,
    /// Build a Vamana graph (the configuration's parameters).
    pub graph: bool,
    pub threads: usize,
    /// Peak extra memory the build may use.
    pub memory_budget: u64,
}

/// What [`write_segment`] wrote.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentInfo {
    pub id: u64,
    /// File name within the index directory (`seg-<id>.mhrv`).
    pub file: String,
    pub vectors: u64,
    pub has_graph: bool,
    pub bytes: u64,
}

/// Writes a segment of every record in `source`, sorted by (path, key), into `dir`.
/// The codec is trained on [`crate::quant::training_sample`] rows (taken in sorted
/// order) with the configuration's seed mixed with the segment id. The bytes written
/// depend only on the records, the configuration and the id, never on `threads`.
/// Errors: `Budget` when the records cannot be built within `memory_budget`.
pub fn write_segment(
    dir: &Path,
    source: &dyn RecordSource,
    build: &SegmentBuild<'_>,
) -> Result<SegmentInfo> {
    todo!("U11 segment: write_segment")
}

/// An open sealed segment. `Sync`: searched by many threads at once.
#[derive(Debug)]
pub struct Segment {
    id: u64,
    file_name: String,
    file: SectionFile,
    len: u32,
    dim: usize,
    codec: AnyCodec,
    codes: Vec<u8>,
    graph: Option<Graph>,
    key_bytes: Vec<u8>,
    key_offsets: Vec<u32>,
    key_index: Vec<u8>,
    path_bytes: Vec<u8>,
    path_offsets: Vec<u32>,
    path_of_ordinal: Vec<u32>,
    languages: Vec<String>,
    language_of_ordinal: Vec<u16>,
}

impl Segment {
    /// Opens `dir/file`, checking it against the index configuration (dimension,
    /// metric, codes). `Corrupt` on any mismatch.
    pub fn open(dir: &Path, file: &str, config: &IndexConfig, verify: VerifyMode) -> Result<Segment> {
        todo!("U11 segment: Segment::open")
    }

    pub fn id(&self) -> u64 {
        todo!("U11 segment: Segment::id")
    }

    pub fn file_name(&self) -> &str {
        todo!("U11 segment: Segment::file_name")
    }

    /// Number of ordinals (live or deleted).
    pub fn len(&self) -> u32 {
        todo!("U11 segment: Segment::len")
    }

    pub fn is_empty(&self) -> bool {
        todo!("U11 segment: Segment::is_empty")
    }

    pub fn dim(&self) -> usize {
        todo!("U11 segment: Segment::dim")
    }

    pub fn codec(&self) -> &AnyCodec {
        todo!("U11 segment: Segment::codec")
    }

    /// All codes, `len * code_len` bytes, ordinal order.
    pub fn codes(&self) -> &[u8] {
        todo!("U11 segment: Segment::codes")
    }

    pub fn code(&self, ordinal: u32) -> &[u8] {
        todo!("U11 segment: Segment::code")
    }

    pub fn graph(&self) -> Option<&Graph> {
        todo!("U11 segment: Segment::graph")
    }

    pub fn key(&self, ordinal: u32) -> &str {
        todo!("U11 segment: Segment::key")
    }

    /// The ordinal holding `key` (live or deleted), if any.
    pub fn find(&self, key: &str) -> Option<u32> {
        todo!("U11 segment: Segment::find")
    }

    pub fn path(&self, ordinal: u32) -> &str {
        todo!("U11 segment: Segment::path")
    }

    pub fn language(&self, ordinal: u32) -> &str {
        todo!("U11 segment: Segment::language")
    }

    /// Reads one stored-form vector.
    pub fn read_vector(&self, ordinal: u32, out: &mut [f32]) -> Result<()> {
        todo!("U11 segment: Segment::read_vector")
    }

    /// Reads several vectors into `out` (`ordinals.len() * dim`), in the given order.
    pub fn read_vectors(&self, ordinals: &[u32], out: &mut [f32]) -> Result<()> {
        todo!("U11 segment: Segment::read_vectors")
    }

    /// Memory held while open.
    pub fn resident_bytes(&self) -> u64 {
        todo!("U11 segment: Segment::resident_bytes")
    }

    /// The file's size.
    pub fn disk_bytes(&self) -> u64 {
        todo!("U11 segment: Segment::disk_bytes")
    }
}

impl PathTable for Segment {
    fn ordinals(&self) -> u32 {
        todo!("U11 segment: PathTable::ordinals")
    }

    fn prefix_ranges(&self, prefix: &str) -> Vec<std::ops::Range<u32>> {
        todo!("U11 segment: PathTable::prefix_ranges")
    }

    fn language_id(&self, language: &str) -> Option<u16> {
        todo!("U11 segment: PathTable::language_id")
    }

    fn language_of(&self, ordinal: u32) -> u16 {
        todo!("U11 segment: PathTable::language_of")
    }
}

impl KeyLookup for Segment {
    fn key(&self, id: u32) -> &str {
        todo!("U11 segment: KeyLookup::key")
    }
}
