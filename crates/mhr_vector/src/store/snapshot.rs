//! Snapshot accessors (unit U17; searching is U18's `search.rs`).

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use super::{Snapshot, SnapshotInner};
use crate::config::{IndexConfig, VerifyMode};
use crate::delta::DeltaView;
use crate::error::{Error, Result};
use crate::manifest::{Manifest, manifest_file_name};
use crate::segment::{RecordSource, Segment};
use crate::tombstones::Tombstones;
use crate::types::StoredRecord;

impl Snapshot {
    pub fn config(&self) -> &IndexConfig {
        &self.inner.config
    }

    /// Live records.
    pub fn len(&self) -> u64 {
        self.inner.delta.live() as u64
            + self
                .inner
                .segments
                .iter()
                .map(|(_, t)| u64::from(t.live()))
                .sum::<u64>()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains(&self, key: &str) -> bool {
        self.inner.delta.contains(key)
            || self
                .inner
                .segments
                .iter()
                .rev()
                .any(|(segment, t)| segment.find(key).is_some_and(|i| !t.is_deleted(i)))
    }

    /// The live record with `key`, if any.
    pub fn get(&self, key: &str) -> Result<Option<StoredRecord>> {
        if let Some(record) = self.inner.delta.get(key) {
            return Ok(Some(record));
        }
        for (segment, t) in self.inner.segments.iter().rev() {
            if let Some(i) = segment.find(key)
                && !t.is_deleted(i)
            {
                let mut vector = vec![0.0; self.config().dim as usize];
                segment.read_vector(i, &mut vector)?;
                return Ok(Some(StoredRecord {
                    key: key.to_owned(),
                    vector,
                    path: segment.path(i).to_owned(),
                    language: segment.language(i).to_owned(),
                }));
            }
        }
        Ok(None)
    }

    /// Operations applied since creation, as of this snapshot: every operation of every
    /// successful `apply`, deletes of absent keys included.
    pub fn version(&self) -> u64 {
        self.inner.version
    }

    /// The last committed generation as of this snapshot.
    pub fn generation(&self) -> u64 {
        self.inner.generation
    }

    /// Every live key, sorted (for checks and tools; O(live)).
    pub fn keys(&self) -> Vec<String> {
        let mut keys = Vec::with_capacity(self.len() as usize);
        for i in 0..self.inner.delta.live() {
            keys.push(self.inner.delta.key(i).to_owned());
        }
        for (segment, t) in &self.inner.segments {
            for i in 0..segment.len() {
                if !t.is_deleted(i) {
                    keys.push(segment.key(i).to_owned());
                }
            }
        }
        keys.sort_unstable();
        keys
    }

    /// Shared generation loader for the writer and directory readers. In particular,
    /// leave missing-file I/O errors intact so a reader can retry a concurrent GC.
    pub(super) fn from_manifest(
        dir: &Path,
        manifest: &Manifest,
        verify: VerifyMode,
    ) -> Result<Self> {
        manifest.config.validate()?;
        let path = dir.join(manifest_file_name(manifest.generation));
        if manifest.generation == 0
            || manifest.next_segment_id == 0
            || manifest.checkpoint.as_ref().is_some_and(|s| s.len() > 8192)
        {
            return Err(Error::corrupt(
                &path,
                "invalid generation, next ID or checkpoint",
            ));
        }
        let mut ids = HashSet::new();
        let mut segments = Vec::with_capacity(manifest.segments.len());
        for entry in &manifest.segments {
            if !ids.insert(entry.id) || entry.id >= manifest.next_segment_id {
                return Err(Error::corrupt(&path, "duplicate or unallocated segment ID"));
            }
            let segment = Arc::new(Segment::open(dir, &entry.file, &manifest.config, verify)?);
            let tombstones = match &entry.tombstones {
                Some(name) => Tombstones::read(&dir.join(name), segment.len())?,
                None => Tombstones::none(segment.len()),
            };
            if segment.id() != entry.id
                || u64::from(segment.len()) != entry.vectors
                || u64::from(tombstones.live()) != entry.live
                || segment.graph().is_some() != entry.has_graph
                || segment.disk_bytes() != entry.bytes
            {
                return Err(Error::corrupt(&path, "segment disagrees with manifest"));
            }
            segments.push((segment, tombstones));
        }
        Ok(Snapshot {
            inner: Arc::new(SnapshotInner {
                config: Arc::new(manifest.config.clone()),
                segments,
                delta: DeltaView::empty(manifest.config.dim as usize, manifest.config.metric),
                version: manifest.version,
                generation: manifest.generation,
            }),
        })
    }
}
