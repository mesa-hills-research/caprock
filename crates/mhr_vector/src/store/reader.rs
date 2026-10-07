//! Readers (unit U19).

use std::path::Path;

use super::{Reader, Snapshot};
use crate::config::StoreOptions;
use crate::error::Result;

impl Reader {
    /// A reader of the last committed generation of `dir` (no lock; another process may
    /// be writing). Errors: `NotFound`, `Corrupt`, `UnsupportedFormat`, `Io`.
    pub fn open(dir: &Path, options: StoreOptions) -> Result<Reader> {
        todo!("U19 reader: Reader::open")
    }

    /// Picks up a newer committed generation, if any; returns whether it changed. A
    /// writer's reader always has the latest snapshot (returns `false`). Retries when a
    /// file named by the manifest it read was removed meanwhile.
    pub fn refresh(&self) -> Result<bool> {
        todo!("U19 reader: Reader::refresh")
    }

    /// The current snapshot.
    pub fn snapshot(&self) -> Snapshot {
        todo!("U19 reader: Reader::snapshot")
    }
}
