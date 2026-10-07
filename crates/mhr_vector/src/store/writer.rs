//! The writer (unit U17).

use std::path::Path;

use super::{BuiltSegment, MaintenanceJob, Reader, Snapshot, Writer};
use crate::config::{IndexConfig, StoreOptions};
use crate::embedding::EmbeddingModel;
use crate::error::Result;
use crate::types::{ApplyReport, CommitInfo, MaintainReport, Op, Record, Stats};

impl Writer {
    /// Creates an index in `dir` (created if missing; must hold no index) and publishes
    /// generation 1 with no records. Errors: `AlreadyExists`, `InvalidConfig`, `Locked`,
    /// `Io`.
    pub fn create(dir: &Path, config: IndexConfig, options: StoreOptions) -> Result<Writer> {
        todo!("U17 writer: Writer::create")
    }

    /// Opens the index in `dir` at its last committed generation, removes garbage left by
    /// unfinished commits, and takes the writer lock. Errors: `NotFound`, `Locked`,
    /// `Corrupt`, `UnsupportedFormat`, `InvalidConfig`, `Io`.
    pub fn open(dir: &Path, options: StoreOptions) -> Result<Writer> {
        todo!("U17 writer: Writer::open")
    }

    /// [`Writer::open`], refusing an index whose embedding model differs from `model` in
    /// any field that changes document vectors ([`EmbeddingModel::requires_reembed`]):
    /// `ModelMismatch`. An index without a model is a mismatch too (its `stored` is then
    /// `model` with empty `provider` and `model` names).
    pub fn open_for_model(
        dir: &Path,
        model: &EmbeddingModel,
        options: StoreOptions,
    ) -> Result<Writer> {
        todo!("U17 writer: Writer::open_for_model")
    }

    pub fn dir(&self) -> &Path {
        todo!("U17 writer: Writer::dir")
    }

    pub fn config(&self) -> &IndexConfig {
        todo!("U17 writer: Writer::config")
    }

    pub fn options(&self) -> &StoreOptions {
        todo!("U17 writer: Writer::options")
    }

    /// Applies a batch in order (later operations on a key win) and publishes a snapshot
    /// that shows it. The batch is validated first: an invalid operation (key, vector,
    /// path, language) fails the whole call with nothing applied. Whenever the delta
    /// reaches `delta_max_vectors`, also in the middle of a batch, it is sealed into a new
    /// (uncommitted) segment, so the delta never holds more than that.
    pub fn apply(&mut self, ops: &[Op<'_>]) -> Result<ApplyReport> {
        todo!("U17 writer: Writer::apply")
    }

    /// `apply(&[Op::Upsert(record)])`.
    pub fn upsert(&mut self, record: Record<'_>) -> Result<()> {
        todo!("U17 writer: Writer::upsert")
    }

    /// `apply(&[Op::Delete(key)])`; returns whether the key was live.
    pub fn delete(&mut self, key: &str) -> Result<bool> {
        todo!("U17 writer: Writer::delete")
    }

    /// The latest snapshot (everything applied so far).
    pub fn snapshot(&self) -> Snapshot {
        todo!("U17 writer: Writer::snapshot")
    }

    /// A reader that follows this writer's published snapshots.
    pub fn reader(&self) -> Reader {
        todo!("U17 writer: Writer::reader")
    }

    /// Makes everything applied so far durable: seals the delta, writes changed
    /// tombstones, publishes the next generation with `checkpoint` (an opaque caller
    /// token, at most 4 KiB, returned by [`Writer::checkpoint`] after any reopen), and
    /// removes files no longer referenced by this or the previous generation.
    pub fn commit(&mut self, checkpoint: Option<&[u8]>) -> Result<CommitInfo> {
        todo!("U17 writer: Writer::commit")
    }

    /// The checkpoint token of the last commit.
    pub fn checkpoint(&self) -> Option<Vec<u8>> {
        todo!("U17 writer: Writer::checkpoint")
    }

    /// The next merge the policy wants, if any (fast; no I/O).
    pub fn plan_maintenance(&self) -> Option<MaintenanceJob> {
        todo!("U17 writer: Writer::plan_maintenance")
    }

    /// The first merge of a compaction ([`crate::merge::plan_compaction`]), if any.
    pub fn plan_compaction(&self) -> Option<MaintenanceJob> {
        todo!("U17 writer: Writer::plan_compaction")
    }

    /// Replaces a job's inputs by its output, deleting from the output every record
    /// deleted or replaced in the inputs since the job was planned. Visible to new
    /// snapshots at once; durable at the next commit. `StaleJob` if the job came from
    /// another writer or its inputs are gone.
    pub fn install(&mut self, built: BuiltSegment) -> Result<()> {
        todo!("U17 writer: Writer::install")
    }

    /// Plans, runs and installs merges until the policy wants none.
    pub fn maintain(&mut self) -> Result<MaintainReport> {
        todo!("U17 writer: Writer::maintain")
    }

    /// Seals the delta, then merges until [`crate::merge::plan_compaction`] wants
    /// nothing: the fewest segments the memory budget allows, no deletions left, graphs
    /// on every large-enough segment.
    pub fn compact(&mut self) -> Result<MaintainReport> {
        todo!("U17 writer: Writer::compact")
    }

    pub fn stats(&self) -> Stats {
        todo!("U17 writer: Writer::stats")
    }
}

impl MaintenanceJob {
    /// Builds the merged segment (long-running; within the build memory budget).
    pub fn run(self) -> Result<BuiltSegment> {
        todo!("U17 writer: MaintenanceJob::run")
    }

    /// A one-line description for logs.
    pub fn describe(&self) -> String {
        todo!("U17 writer: MaintenanceJob::describe")
    }
}
