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
use crate::error::{Error, Result};
use crate::filter::PathTable;
use crate::format::SectionFile;
use crate::quant::AnyCodec;
use crate::topk::KeyLookup;
use crate::vamana::Graph;

mod reading;
mod writing;

// Info deliberately excludes build resources (threads and budget), and timestamps.
#[derive(serde::Serialize, serde::Deserialize)]
struct Info {
    id: u64,
    len: u32,
    dim: u32,
    metric: crate::config::Metric,
    codes: crate::config::CodeKind,
    has_graph: bool,
}

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
        self.keys.len()
    }

    fn key(&self, i: usize) -> &str {
        &self.keys[i]
    }

    fn path(&self, i: usize) -> &str {
        &self.paths[i]
    }

    fn language(&self, i: usize) -> &str {
        &self.languages[i]
    }

    fn read_vector(&self, i: usize, out: &mut [f32]) -> Result<()> {
        if out.len() != self.dim || i >= self.len() {
            return Err(Error::InvalidArgument(
                "record index or vector length out of range".into(),
            ));
        }
        let start = i
            .checked_mul(self.dim)
            .ok_or_else(|| Error::InvalidArgument("vector offset overflow".into()))?;
        let row = start
            .checked_add(self.dim)
            .and_then(|end| self.vectors.get(start..end))
            .ok_or_else(|| Error::InvalidArgument("incomplete record vector".into()))?;
        out.copy_from_slice(row);
        Ok(())
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
    writing::write(dir, source, build)
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
    #[rustfmt::skip]
    pub fn open(dir: &Path, file: &str, config: &IndexConfig, verify: VerifyMode) -> Result<Segment> {
        reading::open(dir, file, config, verify)
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn file_name(&self) -> &str {
        &self.file_name
    }

    /// Number of ordinals (live or deleted).
    pub fn len(&self) -> u32 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn codec(&self) -> &AnyCodec {
        &self.codec
    }

    /// All codes, `len * code_len` bytes, ordinal order.
    pub fn codes(&self) -> &[u8] {
        &self.codes
    }

    pub fn code(&self, ordinal: u32) -> &[u8] {
        let width = self.codec.code_len();
        let start = ordinal as usize * width;
        &self.codes[start..start + width]
    }

    pub fn graph(&self) -> Option<&Graph> {
        self.graph.as_ref()
    }

    pub fn key(&self, ordinal: u32) -> &str {
        string_at(&self.key_bytes, &self.key_offsets, ordinal as usize)
    }

    /// The ordinal holding `key` (live or deleted), if any.
    pub fn find(&self, key: &str) -> Option<u32> {
        let entries = self.key_index.as_chunks::<4>().0;
        entries
            .binary_search_by(|entry| self.key(u32::from_le_bytes(*entry)).cmp(key))
            .ok()
            .map(|i| u32::from_le_bytes(entries[i]))
    }

    pub fn path(&self, ordinal: u32) -> &str {
        string_at(
            &self.path_bytes,
            &self.path_offsets,
            self.path_of_ordinal[ordinal as usize] as usize,
        )
    }

    pub fn language(&self, ordinal: u32) -> &str {
        &self.languages[self.language_of_ordinal[ordinal as usize] as usize]
    }

    /// Reads one stored-form vector.
    pub fn read_vector(&self, ordinal: u32, out: &mut [f32]) -> Result<()> {
        self.read_vectors(&[ordinal], out)
    }

    /// Reads several vectors into `out` (`ordinals.len() * dim`), in the given order.
    pub fn read_vectors(&self, ordinals: &[u32], out: &mut [f32]) -> Result<()> {
        if ordinals.len().checked_mul(self.dim) != Some(out.len())
            || ordinals.iter().any(|&ordinal| ordinal >= self.len)
        {
            return Err(Error::InvalidArgument(
                "ordinal or vector buffer length out of range".into(),
            ));
        }
        if self.codec.kind() == crate::config::CodeKind::F32 {
            for (&ordinal, row) in ordinals.iter().zip(out.chunks_exact_mut(self.dim)) {
                decode_vector(self.code(ordinal), row);
            }
            return Ok(());
        }
        // Coalesce consecutive ordinals but bound scratch independently of the
        // segment size. Repeated or shuffled ordinals retain the caller's order.
        let row_bytes = self.dim * 4;
        let batch = (64 * 1024 / row_bytes).max(1).min(ordinals.len());
        let mut bytes = vec![0; batch * row_bytes];
        let mut first = 0;
        while first < ordinals.len() {
            let mut end = first + 1;
            while end < ordinals.len()
                && end - first < batch
                && ordinals[end] == ordinals[end - 1] + 1
            {
                end += 1;
            }
            let bytes = &mut bytes[..(end - first) * row_bytes];
            self.file.read_at(
                crate::format::SectionKind::Vectors,
                u64::from(ordinals[first]) * row_bytes as u64,
                bytes,
            )?;
            decode_vector(bytes, &mut out[first * self.dim..end * self.dim]);
            first = end;
        }
        Ok(())
    }

    /// Memory held while open.
    pub fn resident_bytes(&self) -> u64 {
        let codec_bytes = match &self.codec {
            AnyCodec::F32(_) => 0,
            AnyCodec::Int8(c) => c.params.capacity() * 4,
            // Binary also caches dim + 1 Hamming correlations.
            AnyCodec::Binary(c) => (c.params.capacity() + self.dim + 1) * 4,
        };
        (std::mem::size_of::<Self>()
            + self.file_name.capacity()
            + self.file.path().as_os_str().len()
            + std::mem::size_of_val(self.file.entries())
            + codec_bytes
            + self.codes.capacity()
            + self
                .graph
                .as_ref()
                .map_or(0, |g| g.adjacency.capacity() * 4)
            + self.key_bytes.capacity()
            + self.key_offsets.capacity() * 4
            + self.key_index.capacity()
            + self.path_bytes.capacity()
            + self.path_offsets.capacity() * 4
            + self.path_of_ordinal.capacity() * 4
            + self.language_of_ordinal.capacity() * 2
            + self.languages.capacity() * std::mem::size_of::<String>()
            + self.languages.iter().map(String::capacity).sum::<usize>()) as u64
    }

    /// The file's size.
    pub fn disk_bytes(&self) -> u64 {
        self.file.file_len()
    }
}

impl PathTable for Segment {
    fn ordinals(&self) -> u32 {
        self.len
    }

    fn prefix_ranges(&self, prefix: &str) -> Vec<std::ops::Range<u32>> {
        let prefix = prefix.strip_suffix('/').unwrap_or(prefix);
        if prefix.is_empty() {
            return std::iter::once(0..self.len)
                .filter(|r| !r.is_empty())
                .collect();
        }
        let path_at = |id: &u32| string_at(&self.path_bytes, &self.path_offsets, *id as usize);
        let start = self
            .path_of_ordinal
            .partition_point(|id| path_at(id) < prefix) as u32;
        let end = self
            .path_of_ordinal
            .partition_point(|id| path_at(id) <= prefix) as u32;
        let mut ranges = Vec::with_capacity(2);
        if start < end {
            ranges.push(start..end);
        }
        let lower = format!("{prefix}/");
        // '/' + 1 is '0': these byte bounds enclose exactly the descendants,
        // not siblings such as prefix.rs, prefix-x, or prefix0.
        let upper = format!("{prefix}0");
        let start = self
            .path_of_ordinal
            .partition_point(|id| path_at(id) < lower.as_str()) as u32;
        let end = self
            .path_of_ordinal
            .partition_point(|id| path_at(id) < upper.as_str()) as u32;
        if start < end {
            ranges.push(start..end);
        }
        ranges
    }

    fn language_id(&self, language: &str) -> Option<u16> {
        self.languages
            .binary_search_by(|name| name.as_str().cmp(language))
            .ok()
            .map(|id| id as u16)
    }

    fn language_of(&self, ordinal: u32) -> u16 {
        self.language_of_ordinal[ordinal as usize]
    }
}

impl KeyLookup for Segment {
    fn key(&self, id: u32) -> &str {
        Segment::key(self, id)
    }
}

// All offsets and UTF-8 boundaries are checked before constructing a Segment.
fn string_at<'a>(bytes: &'a [u8], offsets: &[u32], index: usize) -> &'a str {
    std::str::from_utf8(&bytes[offsets[index] as usize..offsets[index + 1] as usize])
        .expect("validated UTF-8 string table")
}

fn decode_vector(bytes: &[u8], out: &mut [f32]) {
    for (value, &word) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
        *value = f32::from_le_bytes(word);
    }
}
