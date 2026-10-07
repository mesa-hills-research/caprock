//! The store: one [`Writer`], many [`Reader`]s and [`Snapshot`]s (units U18 writer and
//! snapshot, U19 search and reader; REQUIREMENTS.md, "Concurrency", "Updates and
//! visibility", "Persistence").
//!
//! - [`Writer`] owns an index directory (exclusive lock). [`Writer::apply`] changes the
//!   in-memory state and publishes a new [`Snapshot`] before returning (read-your-writes
//!   within the process). [`Writer::commit`] makes the state durable and visible to
//!   other processes by publishing a new generation.
//! - [`Snapshot`] is an immutable, cheaply cloned view: `Send + Sync`, searched by any
//!   number of threads, unaffected by later writes.
//! - [`Reader`] hands out the latest snapshot: of a writer in the same process, or of the
//!   last committed generation of a directory ([`Reader::open`], [`Reader::refresh`]).
//! - Maintenance is split so long builds never block writes: [`Writer::plan_maintenance`]
//!   (fast), [`MaintenanceJob::run`] (long; needs no access to the writer; may run on
//!   another thread), [`Writer::install`] (fast; carries over deletes made meanwhile).
//!   [`Writer::maintain`] and [`Writer::compact`] do all three in place.

mod reader;
mod search;
mod snapshot;
mod writer;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::config::{IndexConfig, StoreOptions};
use crate::delta::{Delta, DeltaView};
use crate::manifest::{IndexLock, Manifest};
use crate::merge::MergePlan;
use crate::segment::{Segment, SegmentInfo};
use crate::tombstones::Tombstones;

/// The single writer of an index directory.
#[derive(Debug)]
pub struct Writer {
    dir: PathBuf,
    config: Arc<IndexConfig>,
    options: StoreOptions,
    lock: IndexLock,
    /// The last committed generation, and the one before it.
    committed: Manifest,
    previous: Option<Manifest>,
    delta: Delta,
    /// Current segments, oldest first (committed and not yet committed).
    segments: Vec<SegmentHandle>,
    next_segment_id: u64,
    version: u64,
    /// Bumped whenever the segment list is replaced (stale-job detection).
    epoch: u64,
    /// The slot in-process readers follow.
    published: Arc<Mutex<Snapshot>>,
}

/// A segment in the writer, with its current deletions.
#[derive(Debug, Clone)]
struct SegmentHandle {
    segment: Arc<Segment>,
    tombstones: Tombstones,
    /// The tombstone file of the last commit, if any.
    committed_tombstones: Option<String>,
    /// Whether `tombstones` changed since the last commit.
    dirty: bool,
}

/// An immutable view of an index. Cheap to clone; `Send + Sync`.
#[derive(Debug, Clone)]
pub struct Snapshot {
    inner: Arc<SnapshotInner>,
}

#[derive(Debug)]
struct SnapshotInner {
    config: Arc<IndexConfig>,
    segments: Vec<(Arc<Segment>, Tombstones)>,
    delta: DeltaView,
    version: u64,
    generation: u64,
}

/// Hands out the latest snapshot of a writer (same process) or of a directory's last
/// committed generation. Cheap to clone; `Send + Sync`.
#[derive(Debug, Clone)]
pub struct Reader {
    source: ReaderSource,
}

#[derive(Debug, Clone)]
enum ReaderSource {
    Writer(Arc<Mutex<Snapshot>>),
    Directory(Arc<DirectoryReader>),
}

#[derive(Debug)]
struct DirectoryReader {
    dir: PathBuf,
    options: StoreOptions,
    current: Mutex<Snapshot>,
}

/// A planned merge, independent of the writer: run it anywhere, then
/// [`Writer::install`] the result.
#[derive(Debug)]
pub struct MaintenanceJob {
    plan: MergePlan,
    inputs: Vec<(Arc<Segment>, Tombstones)>,
    id: u64,
    dir: PathBuf,
    config: Arc<IndexConfig>,
    options: StoreOptions,
    epoch: u64,
}

/// The result of [`MaintenanceJob::run`], to install.
#[derive(Debug)]
pub struct BuiltSegment {
    plan: MergePlan,
    inputs: Vec<(Arc<Segment>, Tombstones)>,
    info: SegmentInfo,
    epoch: u64,
}
