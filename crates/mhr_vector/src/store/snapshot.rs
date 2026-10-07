//! Snapshot accessors (unit U17; searching is U18's `search.rs`).

use super::Snapshot;
use crate::config::IndexConfig;
use crate::error::Result;
use crate::types::StoredRecord;

impl Snapshot {
    pub fn config(&self) -> &IndexConfig {
        todo!("U17 snapshot: Snapshot::config")
    }

    /// Live records.
    pub fn len(&self) -> u64 {
        todo!("U17 snapshot: Snapshot::len")
    }

    pub fn is_empty(&self) -> bool {
        todo!("U17 snapshot: Snapshot::is_empty")
    }

    pub fn contains(&self, key: &str) -> bool {
        todo!("U17 snapshot: Snapshot::contains")
    }

    /// The live record with `key`, if any.
    pub fn get(&self, key: &str) -> Result<Option<StoredRecord>> {
        todo!("U17 snapshot: Snapshot::get")
    }

    /// Operations applied since creation, as of this snapshot: every operation of every
    /// successful `apply`, deletes of absent keys included.
    pub fn version(&self) -> u64 {
        todo!("U17 snapshot: Snapshot::version")
    }

    /// The last committed generation as of this snapshot.
    pub fn generation(&self) -> u64 {
        todo!("U17 snapshot: Snapshot::generation")
    }

    /// Every live key, sorted (for checks and tools; O(live)).
    pub fn keys(&self) -> Vec<String> {
        todo!("U17 snapshot: Snapshot::keys")
    }
}
