//! The writer.

use std::collections::BTreeSet;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::{
    BuiltSegment, JobReservation, MaintenanceJob, Reader, ReaderSource, SegmentHandle, Snapshot,
    SnapshotInner, Writer, WriterState,
};
use crate::config::{IndexConfig, StoreOptions};
use crate::delta::Delta;
use crate::distance;
use crate::embedding::EmbeddingModel;
use crate::error::{Error, Result};
use crate::format::FORMAT_VERSION;
use crate::manifest::{self, IndexLock, Manifest, SegmentEntry};
use crate::merge::{self, LiveRecords, MergePlan, SegmentSummary};
use crate::segment::{RecordSource, Segment, SegmentBuild, write_segment};
use crate::tombstones::Tombstones;
use crate::types::{ApplyReport, CommitInfo, MaintainReport, Op, Record, SegmentStats, Stats};

impl Writer {
    /// Creates an index in `dir` (created if missing; must hold no index) and publishes
    /// generation 1 with no records. Errors: `AlreadyExists`, `InvalidConfig`, `Locked`,
    /// `Io`.
    pub fn create(dir: &Path, config: IndexConfig, options: StoreOptions) -> Result<Writer> {
        config.validate()?;
        options.validate()?;
        let max_segment_vectors = merge::max_segment_vectors(&config, options.build_memory_budget)?;
        if dir
            .join(manifest::CURRENT)
            .try_exists()
            .map_err(|e| Error::io(dir, e))?
        {
            return Err(Error::AlreadyExists(dir.to_owned()));
        }
        fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
        let lock = IndexLock::acquire(dir)?;
        // Another creator could have published between the existence check and lock.
        if manifest::read_current(dir)?.is_some() {
            return Err(Error::AlreadyExists(dir.to_owned()));
        }
        let committed = Manifest {
            format_version: FORMAT_VERSION,
            generation: 1,
            config,
            segments: Vec::new(),
            next_segment_id: 1,
            checkpoint: None,
            version: 0,
        };
        manifest::publish(dir, &committed)?;
        let writer =
            Self::from_committed(dir, options, lock, committed, None, max_segment_vectors)?;
        writer.collect_garbage()?;
        Ok(writer)
    }

    /// Opens the index in `dir` at its last committed generation, removes garbage left by
    /// unfinished commits, and takes the writer lock. Errors: `NotFound`, `Locked`,
    /// `Corrupt`, `UnsupportedFormat`, `InvalidConfig`, `Io`.
    pub fn open(dir: &Path, options: StoreOptions) -> Result<Writer> {
        options.validate()?;
        if !dir
            .join(manifest::CURRENT)
            .try_exists()
            .map_err(|e| Error::io(dir, e))?
        {
            return Err(Error::NotFound(dir.to_owned()));
        }
        let lock = IndexLock::acquire(dir)?;
        let committed =
            manifest::read_current(dir)?.ok_or_else(|| Error::NotFound(dir.to_owned()))?;
        committed.config.validate()?;
        let max_segment_vectors =
            merge::max_segment_vectors(&committed.config, options.build_memory_budget)?;
        let previous = read_previous(dir, committed.generation)?;
        let writer =
            Self::from_committed(dir, options, lock, committed, previous, max_segment_vectors)?;
        writer.collect_garbage()?;
        Ok(writer)
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
        let writer = Self::open(dir, options)?;
        let matches = writer
            .config
            .embedding
            .as_ref()
            .is_some_and(|stored| !stored.requires_reembed(model));
        if !matches {
            let stored = writer.config.embedding.clone().unwrap_or_else(|| {
                let mut unknown = model.clone();
                unknown.provider.clear();
                unknown.model.clear();
                unknown
            });
            return Err(Error::ModelMismatch {
                stored: Box::new(stored),
                requested: Box::new(model.clone()),
            });
        }
        Ok(writer)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn config(&self) -> &IndexConfig {
        &self.config
    }

    pub fn options(&self) -> &StoreOptions {
        &self.options
    }

    /// Applies a batch in order (later operations on a key win) and publishes a snapshot
    /// that shows it. The batch is validated first: an invalid operation (key, vector,
    /// path, language) fails the whole call with nothing applied. Whenever the delta
    /// reaches `delta_max_vectors`, also in the middle of a batch, it is sealed into a new
    /// (uncommitted) segment, so the delta never holds more than that.
    pub fn apply(&mut self, ops: &[Op<'_>]) -> Result<ApplyReport> {
        // Validate without retaining a second full batch of vectors. Preparing again
        // below is cheap compared with keeping an unbounded caller batch resident.
        for op in ops {
            match op {
                Op::Upsert(record) => {
                    validate_key(record.key)?;
                    if record.path.len() > crate::MAX_PATH_BYTES {
                        return Err(Error::InvalidRecord("path exceeds 4096 bytes".into()));
                    }
                    if record.language.len() > crate::MAX_LANGUAGE_BYTES {
                        return Err(Error::InvalidRecord("language exceeds 64 bytes".into()));
                    }
                    distance::prepare(self.config.metric, record.vector, self.config.dim as usize)?;
                }
                Op::Delete(key) => validate_key(key)?,
            }
        }
        let version = self
            .version
            .checked_add(ops.len() as u64)
            .ok_or_else(|| Error::InvalidArgument("operation version exhausted".into()))?;
        let mut report = ApplyReport::default();
        for op in ops {
            match op {
                Op::Upsert(record) => {
                    let vector = distance::prepare(
                        self.config.metric,
                        record.vector,
                        self.config.dim as usize,
                    )?;
                    self.remove_key(record.key);
                    self.delta
                        .upsert(record.key, &vector, record.path, record.language);
                    self.delta_appended += 1;
                    report.upserted += 1;
                    if self.delta_appended >= self.delta_limit() {
                        report.sealed += usize::from(self.seal()?);
                    }
                }
                Op::Delete(key) => {
                    if self.remove_key(key) {
                        report.deleted += 1;
                    } else {
                        report.missing += 1;
                    }
                }
            }
        }
        self.version = version;
        self.publish_snapshot();
        Ok(report)
    }

    /// `apply(&[Op::Upsert(record)])`.
    pub fn upsert(&mut self, record: Record<'_>) -> Result<()> {
        self.apply(&[Op::Upsert(record)])?;
        Ok(())
    }

    /// `apply(&[Op::Delete(key)])`; returns whether the key was live.
    pub fn delete(&mut self, key: &str) -> Result<bool> {
        Ok(self.apply(&[Op::Delete(key)])?.deleted != 0)
    }

    /// The latest snapshot (everything applied so far).
    pub fn snapshot(&self) -> Snapshot {
        self.published
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// A reader that follows this writer's published snapshots.
    pub fn reader(&self) -> Reader {
        Reader {
            source: ReaderSource::Writer(Arc::clone(&self.published)),
        }
    }

    /// Makes everything applied so far durable: seals the delta, writes changed
    /// tombstones, publishes the next generation with `checkpoint` (an opaque caller
    /// token, at most 4 KiB, returned by [`Writer::checkpoint`] after any reopen), and
    /// removes files no longer referenced by this or the previous generation.
    pub fn commit(&mut self, checkpoint: Option<&[u8]>) -> Result<CommitInfo> {
        if checkpoint.is_some_and(|token| token.len() > 4096) {
            return Err(Error::InvalidArgument(
                "checkpoint exceeds 4096 bytes".into(),
            ));
        }
        let generation = self
            .committed
            .generation
            .checked_add(1)
            .ok_or_else(|| Error::InvalidArgument("generation exhausted".into()))?;
        self.seal()?;
        let mut entries = Vec::with_capacity(self.segments.len());
        for handle in &self.segments {
            let segment = &handle.segment;
            let tombstones = if handle.tombstones.deleted() == 0 {
                None
            } else if handle.dirty || handle.committed_tombstones.is_none() {
                let name = manifest::tombstone_file_name(segment.id(), generation);
                handle.tombstones.write(&self.dir, &name)?;
                Some(name)
            } else {
                handle.committed_tombstones.clone()
            };
            entries.push(SegmentEntry {
                id: segment.id(),
                file: segment.file_name().to_owned(),
                vectors: u64::from(segment.len()),
                live: u64::from(handle.tombstones.live()),
                tombstones,
                has_graph: segment.graph().is_some(),
                bytes: segment.disk_bytes(),
            });
        }
        let committed = Manifest {
            format_version: FORMAT_VERSION,
            generation,
            config: (*self.config).clone(),
            segments: entries,
            next_segment_id: self.state.next_segment_id.load(Ordering::Relaxed),
            checkpoint: checkpoint.map(manifest::to_hex),
            version: self.version,
        };
        manifest::publish(&self.dir, &committed)?;
        // No writer state is marked durable until CURRENT has been published.
        for (handle, entry) in self.segments.iter_mut().zip(&committed.segments) {
            handle.committed_tombstones = entry.tombstones.clone();
            handle.dirty = false;
        }
        self.previous = Some(std::mem::replace(&mut self.committed, committed));
        self.publish_snapshot();
        self.collect_garbage()?;
        Ok(CommitInfo {
            generation,
            live: self.snapshot().len(),
            segments: self.segments.len(),
            version: self.version,
        })
    }

    /// The checkpoint token of the last commit.
    pub fn checkpoint(&self) -> Option<Vec<u8>> {
        self.committed
            .checkpoint
            .as_deref()
            .and_then(manifest::from_hex)
    }

    /// The next merge the policy wants, if any (fast; no I/O).
    pub fn plan_maintenance(&self) -> Option<MaintenanceJob> {
        let plan = merge::plan(
            &self.summaries(),
            &self.options.merge,
            &self.config.ann,
            self.max_segment_vectors,
        )
        .into_iter()
        .next()?;
        self.make_job(plan)
    }

    /// The first merge of a compaction (see [`Writer::compact`]), if any.
    pub fn plan_compaction(&self) -> Option<MaintenanceJob> {
        let plan = merge::plan_compaction(
            &self.summaries(),
            &self.config.ann,
            self.max_segment_vectors,
        )
        .into_iter()
        .next()?;
        self.make_job(plan)
    }

    /// Replaces a job's inputs by its output, deleting from the output every record
    /// deleted or replaced in the inputs since the job was planned. Visible to new
    /// snapshots at once; durable at the next commit. `StaleJob` if the job came from
    /// another writer or its inputs are gone.
    pub fn install(&mut self, built: BuiltSegment) -> Result<()> {
        if !Arc::ptr_eq(&self.state, &built.reservation.owner) || built.epoch != self.epoch {
            return Err(Error::StaleJob("writer or segment epoch changed".into()));
        }
        // Arc identity, not just numeric IDs: an input must be the very same
        // immutable segment. Appending delta seals does not invalidate a job.
        let mut positions = Vec::with_capacity(built.inputs.len());
        for (input, _) in &built.inputs {
            let Some(position) = self
                .segments
                .iter()
                .position(|h| Arc::ptr_eq(&h.segment, input))
            else {
                return Err(Error::StaleJob("an input was replaced".into()));
            };
            positions.push(position);
        }
        let mut output = built.output.map(|segment| SegmentHandle {
            tombstones: Tombstones::none(segment.len()),
            segment,
            committed_tombstones: None,
            dirty: false,
        });
        if let Some(output) = &mut output {
            // Tombstones only grow. A transition since the plan represents either
            // a delete or a replacement, even if the key was later inserted again.
            // Key lookup in the output handles the writer's (path,key) reordering.
            let mut removed = Vec::new();
            for ((input, at_plan), &position) in built.inputs.iter().zip(&positions) {
                let now = &self.segments[position].tombstones;
                if now.deleted() == at_plan.deleted() {
                    continue;
                }
                for i in 0..input.len() {
                    if !at_plan.is_deleted(i)
                        && now.is_deleted(i)
                        && let Some(ordinal) = output.segment.find(input.key(i))
                    {
                        removed.push(ordinal);
                    }
                }
            }
            output.tombstones = output.tombstones.with_deleted(&removed);
            output.dirty = !removed.is_empty();
        }
        let inputs: BTreeSet<_> = built.plan.inputs.into_iter().collect();
        self.segments.retain(|h| !inputs.contains(&h.segment.id()));
        if let Some(output) = output {
            // Newest immutable segment. Older copies of every live key were
            // tombstoned before an upsert, so physical order cannot revive them.
            self.segments.push(output);
        }
        self.epoch += 1;
        self.publish_snapshot();
        Ok(())
    }

    /// Plans, runs and installs merges until the policy wants none.
    pub fn maintain(&mut self) -> Result<MaintainReport> {
        self.run_maintenance(false)
    }

    /// Seals the delta, then merges until the index is compact: the fewest segments the
    /// memory budget allows, no deletions left, graphs on every large-enough segment.
    pub fn compact(&mut self) -> Result<MaintainReport> {
        self.seal()?;
        self.publish_snapshot();
        self.run_maintenance(true)
    }

    pub fn stats(&self) -> Stats {
        let segments: Vec<_> = self
            .segments
            .iter()
            .map(|h| SegmentStats {
                id: h.segment.id(),
                vectors: u64::from(h.segment.len()),
                live: u64::from(h.tombstones.live()),
                has_graph: h.segment.graph().is_some(),
                codes: self.config.codes,
                resident_bytes: h.segment.resident_bytes() + h.tombstones.resident_bytes(),
                disk_bytes: h.segment.disk_bytes(),
            })
            .collect();
        let delta = self.delta.len() as u64;
        Stats {
            live: delta + segments.iter().map(|s| s.live).sum::<u64>(),
            delta,
            resident_bytes: self.delta.resident_bytes()
                + segments.iter().map(|s| s.resident_bytes).sum::<u64>(),
            disk_bytes: segments.iter().map(|s| s.disk_bytes).sum(),
            segments,
            generation: self.committed.generation,
            version: self.version,
        }
    }
}

impl MaintenanceJob {
    /// Builds the merged segment (long-running; within the build memory budget).
    pub fn run(self) -> Result<BuiltSegment> {
        let source = LiveRecords::new(self.inputs.iter().map(|(s, t)| (s.as_ref(), t)).collect());
        let output = if source.is_empty() {
            None // A fully dead input is dropped, never written as an empty segment.
        } else {
            let info = write_segment(
                &self.dir,
                &source,
                &SegmentBuild {
                    id: self.id,
                    config: &self.config,
                    graph: self.plan.graph,
                    threads: self.options.threads,
                    // Account for LiveRecords' compact ordinal map, which remains
                    // allocated throughout the streaming build.
                    memory_budget: self
                        .options
                        .build_memory_budget
                        .saturating_sub(source.len() as u64 * 8),
                },
            )?;
            Some(Arc::new(Segment::open(
                &self.dir,
                &info.file,
                &self.config,
                self.options.verify,
            )?))
        };
        Ok(BuiltSegment {
            plan: self.plan,
            inputs: self.inputs,
            output,
            epoch: self.epoch,
            reservation: self.reservation,
        })
    }

    /// A one-line description for logs.
    pub fn describe(&self) -> String {
        format!(
            "{:?}: {:?} -> {} ({} live, graph={})",
            self.plan.reason,
            self.plan.inputs,
            self.id,
            self.inputs
                .iter()
                .map(|(_, t)| u64::from(t.live()))
                .sum::<u64>(),
            self.plan.graph
        )
    }
}

impl Writer {
    fn from_committed(
        dir: &Path,
        options: StoreOptions,
        lock: IndexLock,
        committed: Manifest,
        previous: Option<Manifest>,
        max_segment_vectors: u64,
    ) -> Result<Self> {
        let snapshot = Snapshot::from_manifest(dir, &committed, options.verify)?;
        let segments = snapshot
            .inner
            .segments
            .iter()
            .zip(&committed.segments)
            .map(|((segment, tombstones), entry)| SegmentHandle {
                segment: Arc::clone(segment),
                tombstones: tombstones.clone(),
                committed_tombstones: entry.tombstones.clone(),
                dirty: false,
            })
            .collect();
        Ok(Writer {
            dir: dir.to_owned(),
            config: Arc::clone(&snapshot.inner.config),
            options,
            state: Arc::new(WriterState {
                _lock: lock,
                next_segment_id: AtomicU64::new(committed.next_segment_id),
                reservations: Mutex::new(BTreeSet::new()),
            }),
            delta: Delta::new(committed.config.dim as usize, committed.config.metric),
            version: committed.version,
            committed,
            previous,
            segments,
            max_segment_vectors,
            delta_appended: 0,
            epoch: 0,
            published: Arc::new(Mutex::new(snapshot)),
        })
    }

    fn publish_snapshot(&self) {
        // Assemble everything before locking: in-process readers wait only for an
        // Arc swap, not for a delta copy or a walk over the segments.
        let snapshot = Snapshot {
            inner: Arc::new(SnapshotInner {
                config: Arc::clone(&self.config),
                segments: self
                    .segments
                    .iter()
                    .map(|h| (Arc::clone(&h.segment), h.tombstones.clone()))
                    .collect(),
                delta: self.delta.view(),
                version: self.version,
                generation: self.committed.generation,
            }),
        };
        let old = std::mem::replace(
            &mut *self.published.lock().unwrap_or_else(|e| e.into_inner()),
            snapshot,
        );
        // Destruction of a last old view may be expensive; do it outside the lock.
        drop(old);
    }

    fn delta_limit(&self) -> u64 {
        u64::from(self.options.delta_max_vectors).min(self.max_segment_vectors)
    }

    fn remove_key(&mut self, key: &str) -> bool {
        if self.delta.delete(key) {
            return true;
        }
        for h in self.segments.iter_mut().rev() {
            if let Some(i) = h.segment.find(key)
                && !h.tombstones.is_deleted(i)
            {
                h.tombstones = h.tombstones.with_deleted(&[i]);
                h.dirty = true;
                return true;
            }
        }
        false
    }

    fn allocate_id(&self) -> Option<u64> {
        let counter = &self.state.next_segment_id;
        let mut id = counter.load(Ordering::Relaxed);
        loop {
            match counter.compare_exchange_weak(
                id,
                id.checked_add(1)?,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(id),
                Err(current) => id = current,
            }
        }
    }

    fn seal(&mut self) -> Result<bool> {
        if self.delta.is_empty() {
            self.delta.take();
            self.delta_appended = 0;
            return Ok(false);
        }
        let id = self
            .allocate_id()
            .ok_or_else(|| Error::InvalidArgument("segment IDs exhausted".into()))?;
        // Seals stay flat for apply latency. The maintenance planner builds graphs.
        // Do not consume the delta until the file can be opened successfully.
        let info = write_segment(
            &self.dir,
            &self.delta.view(),
            &SegmentBuild {
                id,
                config: &self.config,
                graph: false,
                threads: self.options.threads,
                memory_budget: self.options.build_memory_budget,
            },
        )?;
        let segment = Arc::new(Segment::open(
            &self.dir,
            &info.file,
            &self.config,
            self.options.verify,
        )?);
        self.segments.push(SegmentHandle {
            tombstones: Tombstones::none(segment.len()),
            segment,
            committed_tombstones: None,
            dirty: false,
        });
        self.delta.take();
        self.delta_appended = 0;
        Ok(true)
    }

    fn summaries(&self) -> Vec<SegmentSummary> {
        self.segments
            .iter()
            .map(|h| SegmentSummary {
                id: h.segment.id(),
                vectors: u64::from(h.segment.len()),
                live: u64::from(h.tombstones.live()),
                has_graph: h.segment.graph().is_some(),
            })
            .collect()
    }

    fn make_job(&self, plan: MergePlan) -> Option<MaintenanceJob> {
        let id = self.allocate_id()?;
        self.state
            .reservations
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id);
        let inputs = plan
            .inputs
            .iter()
            .map(|id| {
                let h = self
                    .segments
                    .iter()
                    .find(|h| h.segment.id() == *id)
                    .expect("planner returned an existing segment");
                (Arc::clone(&h.segment), h.tombstones.clone())
            })
            .collect();
        Some(MaintenanceJob {
            plan,
            inputs,
            id,
            dir: self.dir.clone(),
            config: Arc::clone(&self.config),
            options: self.options.clone(),
            epoch: self.epoch,
            reservation: JobReservation {
                id,
                owner: Arc::clone(&self.state),
            },
        })
    }

    fn run_maintenance(&mut self, compact: bool) -> Result<MaintainReport> {
        let mut report = MaintainReport::default();
        loop {
            let job = if compact {
                self.plan_compaction()
            } else {
                self.plan_maintenance()
            };
            let Some(job) = job else { break };
            let built = job.run()?;
            let written = built.output.as_ref().map_or(0, |s| u64::from(s.len()));
            let graph = built.output.as_ref().is_some_and(|s| s.graph().is_some());
            self.install(built)?;
            report.merges += 1;
            report.graphs_built += usize::from(graph);
            report.vectors_written += written;
        }
        Ok(report)
    }

    fn collect_garbage(&self) -> Result<()> {
        let reservations = self
            .state
            .reservations
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let protected: BTreeSet<_> = reservations
            .iter()
            .flat_map(|&id| {
                let name = manifest::segment_file_name(id);
                [format!("{name}.tmp"), name]
            })
            .collect();
        for path in manifest::garbage(&self.dir, &self.committed, self.previous.as_ref())? {
            if path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| protected.contains(n))
            {
                continue;
            }
            // On Windows an old snapshot can still have the file open. Retrying
            // on a later commit/open is safe; garbage removal is best-effort.
            let _ = fs::remove_file(path);
        }
        Ok(())
    }
}

fn validate_key(key: &str) -> Result<()> {
    if key.is_empty() || key.len() > crate::MAX_KEY_BYTES {
        return Err(Error::InvalidKey("key must contain 1..=255 bytes".into()));
    }
    Ok(())
}

fn read_previous(dir: &Path, generation: u64) -> Result<Option<Manifest>> {
    if generation <= 1 {
        return Ok(None);
    }
    let path = dir.join(manifest::manifest_file_name(generation - 1));
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(Error::io(&path, e)),
    };
    let previous: Manifest = serde_json::from_slice(&bytes)
        .map_err(|e| Error::corrupt(&path, format!("invalid previous manifest: {e}")))?;
    if previous.generation != generation - 1 || previous.format_version != FORMAT_VERSION {
        return Err(Error::corrupt(path, "invalid previous generation"));
    }
    Ok(Some(previous))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CodeKind, Metric};
    use std::path::PathBuf;

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            loop {
                let path = std::env::temp_dir().join(format!(
                    "caprock-writer-test-{}-{}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed),
                ));
                match fs::create_dir(&path) {
                    Ok(()) => return Self(path),
                    Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                    Err(e) => panic!("{e}"),
                }
            }
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn options() -> StoreOptions {
        StoreOptions {
            delta_max_vectors: 2,
            threads: 1,
            ..StoreOptions::default()
        }
    }

    fn record<'a>(key: &'a str, vector: &'a [f32], path: &'a str) -> Record<'a> {
        Record {
            key,
            vector,
            path,
            language: "rust",
        }
    }

    fn populate(dir: &TestDir) -> Writer {
        let mut w = Writer::create(&dir.0, IndexConfig::exact(2, Metric::Dot), options()).unwrap();
        for key in ["a", "b", "c", "d"] {
            w.upsert(record(key, &[1.0, 2.0], key)).unwrap();
        }
        w
    }

    #[test]
    fn operations_are_ordered_even_across_seals_and_invalid_deletes_are_atomic() {
        let dir = TestDir::new();
        let mut w = populate(&dir);
        let old = w.snapshot();
        let report = w
            .apply(&[
                Op::Delete("a"),
                Op::Delete("a"),
                Op::Upsert(record("a", &[3.0, 4.0], "new")),
                Op::Upsert(record("a", &[5.0, 6.0], "newer")),
                Op::Delete("a"),
                Op::Upsert(record("a", &[7.0, 8.0], "newest")),
            ])
            .unwrap();
        assert_eq!(
            report,
            ApplyReport {
                upserted: 3,
                deleted: 2,
                missing: 1,
                sealed: 1
            }
        );
        assert_eq!(w.snapshot().get("a").unwrap().unwrap().vector, [7.0, 8.0]);
        assert_eq!(old.get("a").unwrap().unwrap().vector, [1.0, 2.0]);
        let before = w.snapshot();
        assert!(matches!(
            w.apply(&[Op::Delete("b"), Op::Delete("")]),
            Err(Error::InvalidKey(_))
        ));
        assert_eq!(w.snapshot().keys(), before.keys());
        assert_eq!(w.snapshot().version(), before.version());
        w.commit(Some(&[0, 255, 128])).unwrap();
        drop(w);
        let w = Writer::open(&dir.0, options()).unwrap();
        assert_eq!(w.checkpoint(), Some(vec![0, 255, 128]));
        assert_eq!(w.snapshot().version(), 10);
        assert_eq!(w.snapshot().keys(), old.keys());
    }

    #[test]
    fn jobs_reserve_distinct_ids_and_keep_outputs_through_commits() {
        let dir = TestDir::new();
        let mut w = populate(&dir);
        let job = w.plan_compaction().unwrap();
        let other = w.plan_compaction().unwrap();
        assert_ne!(job.id, other.id);
        let built = job.run().unwrap();
        let output_path = dir.0.join(built.output.as_ref().unwrap().file_name());
        // Several GC cycles while a built result waits for installation.
        w.commit(None).unwrap();
        w.commit(None).unwrap();
        assert!(output_path.exists());
        w.apply(&[
            Op::Delete("a"),
            Op::Upsert(record("b", &[8.0, 9.0], "changed")),
            Op::Upsert(record("e", &[2.0, 3.0], "extra")),
        ])
        .unwrap();
        w.commit(None).unwrap();
        w.install(built).unwrap();
        assert!(!w.snapshot().contains("a"));
        assert_eq!(w.snapshot().get("b").unwrap().unwrap().vector, [8.0, 9.0]);
        let stale = other.run().unwrap();
        let stale_path = dir.0.join(stale.output.as_ref().unwrap().file_name());
        assert!(matches!(w.install(stale), Err(Error::StaleJob(_))));
        w.commit(None).unwrap();
        assert!(!stale_path.exists());
        let keys = w.snapshot().keys();
        drop(w);
        let w = Writer::open(&dir.0, options()).unwrap();
        assert_eq!(w.snapshot().keys(), keys);
        assert_eq!(w.snapshot().len(), 4);
    }

    #[test]
    fn compaction_of_fully_dead_inputs_writes_nothing() {
        let dir = TestDir::new();
        let mut w = populate(&dir);
        for key in ["a", "b", "c", "d"] {
            w.delete(key).unwrap();
        }
        let job = w.plan_compaction().unwrap();
        let path = dir.0.join(manifest::segment_file_name(job.id));
        let built = job.run().unwrap();
        assert!(built.output.is_none());
        assert!(!path.exists());
        w.install(built).unwrap();
        assert!(w.stats().segments.is_empty());
        w.commit(None).unwrap();
        drop(w);
        assert!(
            Writer::open(&dir.0, options())
                .unwrap()
                .snapshot()
                .is_empty()
        );
    }

    #[test]
    fn old_snapshots_hold_vector_files_after_two_generation_gc() {
        let dir = TestDir::new();
        let mut config = IndexConfig::exact(2, Metric::Dot).with_codes(CodeKind::Int8);
        config.ann.kind = crate::config::AnnKind::Flat;
        let mut w = Writer::create(&dir.0, config, options()).unwrap();
        w.upsert(record("a", &[1.0, 2.0], "a")).unwrap();
        w.upsert(record("b", &[3.0, 4.0], "b")).unwrap();
        w.commit(None).unwrap();
        let old = w.snapshot();
        let old_file = dir.0.join(old.inner.segments[0].0.file_name());
        w.delete("a").unwrap();
        w.compact().unwrap();
        w.commit(None).unwrap();
        assert!(old_file.exists(), "previous generation is retained");
        w.commit(None).unwrap();
        assert_eq!(old.get("a").unwrap().unwrap().vector, [1.0, 2.0]);
        assert_eq!(old.get("b").unwrap().unwrap().vector, [3.0, 4.0]);
        assert_eq!(old.keys(), ["a", "b"]);
    }

    #[test]
    fn overwritten_delta_allocations_are_bounded_and_tombstone_files_are_reused() {
        let dir = TestDir::new();
        let mut w = populate(&dir);
        for _ in 0..20 {
            w.upsert(record("x", &[1.0, 0.0], "x")).unwrap();
            assert!(w.delta_appended < w.delta_limit());
        }
        w.delete("a").unwrap();
        w.commit(None).unwrap();
        let deleted = w.committed.segments[0].tombstones.clone().unwrap();
        w.commit(None).unwrap();
        assert_eq!(w.committed.segments[0].tombstones.as_ref(), Some(&deleted));
        let before = w.snapshot();
        let changed = w.apply(&[
            Op::Delete("c"),
            Op::Upsert(record("x", &[f32::NAN, 0.0], "x")),
        ]);
        assert!(matches!(changed, Err(Error::InvalidVector(_))));
        assert_eq!(w.snapshot().keys(), before.keys());
        assert_eq!(w.snapshot().version(), before.version());
    }
}
