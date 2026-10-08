//! The in-memory delta (unit U12; docs/requirements.md, "Updates and visibility").
//!
//! Recent upserts live here until the writer seals them into a segment. Snapshots hold a
//! [`DeltaView`]: an immutable view that later changes never affect. Taking a view must
//! be cheap (no copy of the whole delta per `apply`): for example append-only chunks
//! shared through `Arc`, with copy-on-write deletion marks. Searches over a view are
//! exact (canonical scores).

use std::collections::HashMap;
use std::mem::size_of;
use std::sync::{Arc, OnceLock};

use crate::config::Metric;
use crate::distance;
use crate::error::{Error, Result, VectorError};
use crate::filter::Filter;
use crate::segment::RecordSource;
use crate::topk::{KeyLookup, TopK};
use crate::types::{Hit, StoredRecord};

const CHUNK_RECORDS: usize = 256;
const ARC_HEADER_BYTES: usize = 2 * size_of::<usize>();

// Records are immutable even in the open tail. Copying a shared tail or changing a
// deletion bit copies at most CHUNK_RECORDS handles, never vectors or metadata.
#[derive(Debug)]
struct Chunk {
    records: Vec<Arc<StoredRecord>>,
    deleted: [u64; CHUNK_RECORDS / 64],
}

impl Chunk {
    fn new() -> Self {
        Self {
            records: Vec::with_capacity(CHUNK_RECORDS),
            deleted: [0; CHUNK_RECORDS / 64],
        }
    }

    fn is_deleted(&self, offset: usize) -> bool {
        self.deleted[offset / 64] & (1 << (offset % 64)) != 0
    }
}

impl Clone for Chunk {
    fn clone(&self) -> Self {
        // Preserve the fixed capacity in a partially filled tail after COW.
        let mut records = Vec::with_capacity(CHUNK_RECORDS);
        records.extend(self.records.iter().cloned());
        Self {
            records,
            deleted: self.deleted,
        }
    }
}

#[derive(Debug, Default)]
struct Contents {
    chunks: Vec<Arc<Chunk>>,
    // Includes obsolete records: IDs are append-only until the delta is taken.
    appended: usize,
    live: usize,
    record_bytes: u64,
    // Not needed by snapshot creation or search. Shared by all clones of a view,
    // initialized only for key lookup or the key-ordered RecordSource interface.
    by_key: OnceLock<Vec<u32>>,
}

impl Clone for Contents {
    fn clone(&self) -> Self {
        Self {
            chunks: self.chunks.clone(),
            appended: self.appended,
            live: self.live,
            record_bytes: self.record_bytes,
            // COW happens only on mutation, so the old index must not be copied.
            by_key: OnceLock::new(),
        }
    }
}

impl Contents {
    fn record(&self, id: u32) -> &StoredRecord {
        let id = id as usize;
        &self.chunks[id / CHUNK_RECORDS].records[id % CHUNK_RECORDS]
    }

    fn mark_deleted(&mut self, id: u32) {
        let id = id as usize;
        let chunk = Arc::make_mut(&mut self.chunks[id / CHUNK_RECORDS]);
        let offset = id % CHUNK_RECORDS;
        debug_assert!(!chunk.is_deleted(offset));
        chunk.deleted[offset / 64] |= 1 << (offset % 64);
        self.live -= 1;
    }

    fn push(&mut self, record: StoredRecord) {
        if self.appended.is_multiple_of(CHUNK_RECORDS) {
            self.chunks.push(Arc::new(Chunk::new()));
        }
        self.record_bytes += (size_of::<StoredRecord>()
            + ARC_HEADER_BYTES
            + record.key.capacity()
            + record.path.capacity()
            + record.language.capacity()
            + record.vector.capacity() * size_of::<f32>()) as u64;
        Arc::make_mut(self.chunks.last_mut().expect("nonempty chunk directory"))
            .records
            .push(Arc::new(record));
        self.appended += 1;
        self.live += 1;
    }

    fn live_records(&self) -> impl Iterator<Item = (u32, &StoredRecord)> {
        self.chunks
            .iter()
            .enumerate()
            .flat_map(|(chunk_id, chunk)| {
                chunk
                    .records
                    .iter()
                    .enumerate()
                    .filter(|(offset, _)| !chunk.is_deleted(*offset))
                    .map(move |(offset, record)| {
                        ((chunk_id * CHUNK_RECORDS + offset) as u32, record.as_ref())
                    })
            })
    }

    fn sorted_ids(&self) -> &[u32] {
        self.by_key.get_or_init(|| {
            let mut ids = Vec::with_capacity(self.live);
            ids.extend(self.live_records().map(|(id, _)| id));
            ids.sort_unstable_by(|&a, &b| self.record(a).key.cmp(&self.record(b).key));
            ids
        })
    }

    fn find(&self, key: &str) -> Option<&StoredRecord> {
        let ids = self.sorted_ids();
        let i = ids
            .binary_search_by(|&id| self.record(id).key.as_str().cmp(key))
            .ok()?;
        Some(self.record(ids[i]))
    }

    fn resident_bytes(&self) -> u64 {
        // Count reachable allocations, including tombstoned records and unused
        // capacity. Allocator-specific bookkeeping is not included.
        self.record_bytes
            + (size_of::<Self>()
                + ARC_HEADER_BYTES
                + self.chunks.capacity() * size_of::<Arc<Chunk>>()
                + self.chunks.len()
                    * (size_of::<Chunk>()
                        + ARC_HEADER_BYTES
                        + CHUNK_RECORDS * size_of::<Arc<StoredRecord>>())
                + self
                    .by_key
                    .get()
                    .map_or(0, |ids| ids.capacity() * size_of::<u32>())) as u64
    }
}

impl KeyLookup for Contents {
    fn key(&self, id: u32) -> &str {
        &self.record(id).key
    }
}

/// The mutable delta, owned by the writer.
#[derive(Debug)]
pub struct Delta {
    dim: usize,
    metric: Metric,
    contents: Arc<Contents>,
    // Writer-only: taking a view never copies the mutable key map.
    keys: HashMap<String, u32>,
    key_bytes: u64,
}

impl Delta {
    pub fn new(dim: usize, metric: Metric) -> Delta {
        Delta {
            dim,
            metric,
            contents: Arc::new(Contents::default()),
            keys: HashMap::new(),
            key_bytes: 0,
        }
    }

    /// Live records.
    pub fn len(&self) -> usize {
        self.contents.live
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Inserts a record (vector in stored form), replacing a live one with the same key.
    pub fn upsert(&mut self, key: &str, vector: &[f32], path: &str, language: &str) {
        // The writer validates whole batches before calling this stored-form API.
        assert_eq!(vector.len(), self.dim, "delta vector dimension mismatch");
        let id = u32::try_from(self.contents.appended).expect("too many delta records");
        let record = StoredRecord {
            key: key.to_owned(),
            vector: vector.to_vec(),
            path: path.to_owned(),
            language: language.to_owned(),
        };
        let contents = Arc::make_mut(&mut self.contents);
        contents.by_key.take();
        if let Some(old) = self.keys.insert(key.to_owned(), id) {
            contents.mark_deleted(old);
        } else {
            self.key_bytes += key.len() as u64;
        }
        contents.push(record);
    }

    /// Removes a live record; returns whether it was live.
    pub fn delete(&mut self, key: &str) -> bool {
        let Some(id) = self.keys.remove(key) else {
            return false;
        };
        self.key_bytes -= key.len() as u64;
        let contents = Arc::make_mut(&mut self.contents);
        contents.by_key.take();
        contents.mark_deleted(id);
        true
    }

    #[cfg(any(test, feature = "internals"))]
    pub fn contains(&self, key: &str) -> bool {
        self.keys.contains_key(key)
    }

    /// An immutable view of the current contents.
    pub fn view(&self) -> DeltaView {
        DeltaView {
            dim: self.dim,
            metric: self.metric,
            contents: Arc::clone(&self.contents),
        }
    }

    /// The current contents as a view, leaving the delta empty.
    pub fn take(&mut self) -> DeltaView {
        let contents = std::mem::take(&mut self.contents);
        // Drop the writer-only map's allocations as well as its live entries.
        self.keys = HashMap::new();
        self.key_bytes = 0;
        DeltaView {
            dim: self.dim,
            metric: self.metric,
            contents,
        }
    }

    /// Memory held.
    pub fn resident_bytes(&self) -> u64 {
        self.contents.resident_bytes()
            + self.key_bytes
            + (size_of::<Self>()
                + self.keys.capacity() * (size_of::<(String, u32)>() + 1)) as u64
    }
}

/// An immutable view of the delta. Cloning is cheap. As a [`RecordSource`] it yields
/// its live records in key order.
#[derive(Debug, Clone)]
pub struct DeltaView {
    dim: usize,
    metric: Metric,
    contents: Arc<Contents>,
}

impl DeltaView {
    /// An empty view.
    pub fn empty(dim: usize, metric: Metric) -> DeltaView {
        DeltaView {
            dim,
            metric,
            contents: Arc::new(Contents::default()),
        }
    }

    /// Live records.
    pub fn live(&self) -> usize {
        self.contents.live
    }

    pub fn contains(&self, key: &str) -> bool {
        self.contents.find(key).is_some()
    }

    pub fn get(&self, key: &str) -> Option<StoredRecord> {
        self.contents.find(key).cloned()
    }

    /// The exact top `k` live records matching `filter` for a stored-form `query`, with
    /// canonical scores, in result order.
    pub fn search(&self, query: &[f32], k: usize, filter: Option<&Filter>) -> Vec<Hit> {
        if k == 0 || self.live() == 0 {
            return Vec::new();
        }
        assert_eq!(query.len(), self.dim, "delta query dimension mismatch");
        let mut top = TopK::new(k.min(self.live()));
        for (id, record) in self.contents.live_records() {
            if filter.is_some_and(|f| !f.matches(&record.path, &record.language)) {
                continue;
            }
            let score = distance::score(self.metric, query, &record.vector);
            top.offer(score, id, self.contents.as_ref());
        }
        top.into_sorted(self.contents.as_ref())
            .into_iter()
            .map(|(score, id)| {
                let record = self.contents.record(id);
                Hit {
                    key: record.key.clone(),
                    score,
                    path: record.path.clone(),
                    language: record.language.clone(),
                }
            })
            .collect()
    }

    /// Memory held by this view (shared memory counted once per view).
    #[cfg(any(test, feature = "internals"))]
    pub fn resident_bytes(&self) -> u64 {
        size_of::<Self>() as u64 + self.contents.resident_bytes()
    }
}

impl RecordSource for DeltaView {
    fn len(&self) -> usize {
        self.live()
    }

    fn key(&self, i: usize) -> &str {
        &self.contents.record(self.contents.sorted_ids()[i]).key
    }

    fn path(&self, i: usize) -> &str {
        &self.contents.record(self.contents.sorted_ids()[i]).path
    }

    fn language(&self, i: usize) -> &str {
        &self.contents.record(self.contents.sorted_ids()[i]).language
    }

    fn read_vector(&self, i: usize, out: &mut [f32]) -> Result<()> {
        if out.len() != self.dim {
            return Err(VectorError::Dimension {
                expected: self.dim,
                found: out.len(),
            }
            .into());
        }
        let id = self.contents.sorted_ids().get(i).ok_or_else(|| {
            Error::InvalidArgument(format!("delta record index {i} is out of bounds"))
        })?;
        out.copy_from_slice(&self.contents.record(*id).vector);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_boundaries_and_cached_key_order_survive_changes() {
        let mut delta = Delta::new(2, Metric::Dot);
        let n = CHUNK_RECORDS * 2 + 1;
        // Insertion order differs from the RecordSource order, over several chunks.
        for i in (0..n).rev() {
            delta.upsert(&format!("k{i:04}"), &[i as f32, 1.0], "old", "rust");
        }
        let before = delta.view();
        for i in 0..n {
            assert_eq!(before.key(i), format!("k{i:04}"));
        }
        let shared = before.clone();
        assert!(Arc::ptr_eq(&before.contents, &shared.contents));
        // Touch bit positions on either side of word/chunk boundaries, including
        // the tail. The old view already has its lazy sorted index initialized.
        for i in [0, 1, 63, 64, 255, 256, 511, 512] {
            let key = format!("k{i:04}");
            assert!(delta.delete(&key));
            assert!(!delta.delete(&key));
            assert!(!delta.contains(&key));
            delta.upsert(&key, &[-1.0, 2.0], "new", "go");
        }
        let after = delta.view();
        assert_eq!(after.live(), n);
        for i in 0..n {
            assert_eq!(after.key(i), before.key(i));
            assert_eq!(before.path(i), "old");
            let mut vector = [0.0; 2];
            shared.read_vector(i, &mut vector).unwrap();
            assert_eq!(vector, [i as f32, 1.0]);
        }
        assert_eq!(after.get("k0256").unwrap().path, "new");
        assert_eq!(after.get("k0256").unwrap().vector, [-1.0, 2.0]);
        assert_eq!(after.get("k0257").unwrap().path, "old");
        let taken = delta.take();
        assert_eq!(taken.live(), n);
        assert!(!delta.contains("k0000"));
        delta.upsert("fresh", &[3.0, 4.0], "fresh", "rust");
        assert_eq!(delta.len(), 1);
        assert!(!taken.contains("fresh"));
        assert!(taken.resident_bytes() >= (n * 2 * size_of::<f32>()) as u64);
    }

    #[test]
    fn unique_cached_contents_are_invalidated_and_get_returns_an_owned_record() {
        let mut delta = Delta::new(1, Metric::Dot);
        delta.upsert("z", &[1.0], "p", "rust");
        // Drop each view after populating the cache, leaving the writer's root unique.
        assert!(delta.view().contains("z"));
        delta.upsert("a", &[2.0], "q", "go");
        assert_eq!(delta.view().key(0), "a");
        assert!(delta.delete("a"));
        assert_eq!(delta.view().key(0), "z");
        let mut owned = delta.view().get("z").unwrap();
        owned.vector[0] = 99.0;
        owned.path.clear();
        assert_eq!(delta.view().get("z").unwrap().vector, [1.0]);
        assert_eq!(delta.view().get("z").unwrap().path, "p");
        assert!(delta.delete("z"));
        assert!(delta.is_empty());
        let empty = delta.take();
        assert_eq!(RecordSource::len(&empty), 0);
        assert!(!empty.contains("z"));
        assert!(empty.get("z").is_none());
        assert!(empty.search(&[1.0], 10, None).is_empty());
    }

    #[test]
    fn score_ties_filters_and_search_limits() {
        let mut delta = Delta::new(1, Metric::Dot);
        for key in ["é", "z", "b", "a"] {
            delta.upsert(key, &[0.0], "x/a.rs", "rust");
        }
        delta.upsert("0", &[100.0], "xy/a.rs", "rust");
        delta.upsert("1", &[100.0], "x/b.rs", "go");
        // An obsolete high-scoring record must never appear in results.
        delta.upsert("gone", &[1000.0], "x/c.rs", "rust");
        assert!(delta.delete("gone"));
        let view = delta.view();
        assert!(view.search(&[1.0], 0, None).is_empty());
        let filter = Filter {
            path_prefixes: vec!["x/".into()],
            languages: vec!["rust".into()],
        };
        for (k, want) in [(1, vec!["a"]), (usize::MAX, vec!["a", "b", "z", "é"])] {
            let hits = view.search(&[-0.0], k, Some(&filter));
            assert_eq!(hits.iter().map(|h| h.key.as_str()).collect::<Vec<_>>(), want);
            assert!(hits.iter().all(|h| h.score == 0.0));
        }
        assert!(view.search(&[1.0], 10, Some(&Filter::path("absent"))).is_empty());
        assert_eq!(view.search(&[1.0], 1, None)[0].key, "0");
    }

    #[test]
    fn record_source_errors_and_parallel_lazy_index() {
        let mut delta = Delta::new(2, Metric::Dot);
        delta.upsert("b", &[2.0, 3.0], "b", "rust");
        delta.upsert("a", &[4.0, 5.0], "a", "go");
        let view = delta.take();
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let view = &view;
                scope.spawn(move || {
                    assert_eq!(view.key(0), "a");
                    assert_eq!(view.get("b").unwrap().vector, [2.0, 3.0]);
                });
            }
        });
        assert!(matches!(
            view.read_vector(0, &mut [0.0]),
            Err(Error::InvalidVector(VectorError::Dimension { expected: 2, found: 1 }))
        ));
        let mut out = [99.0; 2];
        assert!(matches!(view.read_vector(2, &mut out), Err(Error::InvalidArgument(_))));
        assert_eq!(out, [99.0; 2]);
    }
}
