//! Manifests, the `CURRENT` switch and the writer lock (unit U06; REQUIREMENTS.md,
//! "Persistence").
//!
//! An index directory holds:
//!
//! ```text
//! LOCK                         the writer lock (an exclusive std file lock)
//! CURRENT                      the name of the current manifest, one line
//! gen-<20 digits>.json         one manifest per committed generation
//! seg-<20 digits>.mhrv         segment files (section files, immutable)
//! del-<seg 20>-<gen 20>.bits   tombstones of a segment as of a generation (immutable)
//! *.tmp                        unfinished writes (removed by the next writer)
//! ```
//!
//! A commit writes new segment and tombstone files, then the new manifest (atomically),
//! then replaces `CURRENT` (atomically). A crash at any point leaves the last published
//! generation intact; the files of an unfinished commit are garbage the next writer
//! removes. Files referenced by the previous generation are kept one more commit, so a
//! reader that just read the old `CURRENT` can still open them.

use std::fs::File;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::IndexConfig;
use crate::error::Result;

/// The `CURRENT` file's name.
pub const CURRENT: &str = "CURRENT";
/// The lock file's name.
pub const LOCK: &str = "LOCK";

/// One segment in a manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentEntry {
    pub id: u64,
    /// File name within the index directory.
    pub file: String,
    pub vectors: u64,
    pub live: u64,
    /// The tombstone file in force, if any record is deleted.
    #[serde(default)]
    pub tombstones: Option<String>,
    pub has_graph: bool,
    pub bytes: u64,
}

/// A committed generation's description (`gen-<n>.json`). No timestamps: the same
/// operations give the same manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub format_version: u32,
    pub generation: u64,
    pub config: IndexConfig,
    /// Oldest first.
    pub segments: Vec<SegmentEntry>,
    pub next_segment_id: u64,
    /// The caller's checkpoint token, hex-encoded.
    #[serde(default)]
    pub checkpoint: Option<String>,
    /// Operations applied since creation, as of this generation.
    pub version: u64,
}

/// `gen-<20 digits>.json`.
pub fn manifest_file_name(generation: u64) -> String {
    todo!("U06 manifest: manifest_file_name")
}

/// `seg-<20 digits>.mhrv`.
pub fn segment_file_name(id: u64) -> String {
    todo!("U06 manifest: segment_file_name")
}

/// `del-<segment 20 digits>-<generation 20 digits>.bits`.
pub fn tombstone_file_name(segment: u64, generation: u64) -> String {
    todo!("U06 manifest: tombstone_file_name")
}

/// Lower-case hex, and its inverse (`None` on bad input).
pub fn to_hex(bytes: &[u8]) -> String {
    todo!("U06 manifest: to_hex")
}

pub fn from_hex(text: &str) -> Option<Vec<u8>> {
    todo!("U06 manifest: from_hex")
}

/// Reads `CURRENT` and the manifest it names. `Ok(None)` when there is no `CURRENT`.
/// `Corrupt` when `CURRENT` names a missing or unreadable manifest;
/// `UnsupportedFormat` for a newer format version.
pub fn read_current(dir: &Path) -> Result<Option<Manifest>> {
    todo!("U06 manifest: read_current")
}

/// Publishes `manifest`: writes `gen-<n>.json` atomically, then `CURRENT` atomically.
pub fn publish(dir: &Path, manifest: &Manifest) -> Result<()> {
    todo!("U06 manifest: publish")
}

/// The files of `dir` that neither `current` nor `previous` references (segments,
/// tombstones, manifests, `*.tmp`), sorted. Never `LOCK` or `CURRENT`, never anything
/// whose name this module does not recognise.
pub fn garbage(dir: &Path, current: &Manifest, previous: Option<&Manifest>) -> Result<Vec<PathBuf>> {
    todo!("U06 manifest: garbage")
}

/// The exclusive writer lock on `dir/LOCK`, held until dropped.
#[derive(Debug)]
pub struct IndexLock {
    file: File,
}

impl IndexLock {
    /// Takes the lock without waiting (`File::try_lock`). `Locked` if another holder has
    /// it, in this process or another.
    pub fn acquire(dir: &Path) -> Result<IndexLock> {
        todo!("U06 manifest: IndexLock::acquire")
    }
}
