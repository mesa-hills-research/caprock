//! Deleted ordinals of one sealed segment (unit U10).
//!
//! Tombstones are immutable values shared by snapshots: deleting creates a new value
//! (copy-on-write), so a snapshot never sees a later delete. On disk a tombstone file is
//! `b"MHRVDEL1"`, the segment's ordinal count (u64), the deleted count (u64), the bitset
//! ([`crate::util::Bitset::to_bytes`]) and a CRC-32 of everything before it (u32), all
//! little-endian; it is written atomically ([`crate::format::atomic_write`]).

use std::path::Path;
use std::sync::Arc;

use crate::error::Result;
use crate::util::Bitset;

/// The magic of tombstone files.
pub const TOMBSTONE_MAGIC: [u8; 8] = *b"MHRVDEL1";

/// A segment's deleted ordinals. Cloning is cheap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tombstones {
    bits: Arc<Bitset>,
    deleted: u32,
}

impl Tombstones {
    /// No deletions in a segment of `len` ordinals.
    pub fn none(len: u32) -> Tombstones {
        todo!("U10 tombstones: Tombstones::none")
    }

    /// The segment's ordinal count.
    pub fn ordinals(&self) -> u32 {
        todo!("U10 tombstones: Tombstones::ordinals")
    }

    pub fn is_deleted(&self, ordinal: u32) -> bool {
        todo!("U10 tombstones: Tombstones::is_deleted")
    }

    /// Number of deleted ordinals.
    pub fn deleted(&self) -> u32 {
        todo!("U10 tombstones: Tombstones::deleted")
    }

    /// Live ordinals: `len - deleted`.
    pub fn live(&self) -> u32 {
        todo!("U10 tombstones: Tombstones::live")
    }

    /// A new value with `ordinals` deleted as well (already deleted ones are ignored).
    /// `self` is unchanged.
    pub fn with_deleted(&self, ordinals: &[u32]) -> Tombstones {
        todo!("U10 tombstones: Tombstones::with_deleted")
    }

    /// Writes `dir/name` atomically in the format above.
    pub fn write(&self, dir: &Path, name: &str) -> Result<()> {
        todo!("U10 tombstones: Tombstones::write")
    }

    /// Reads a tombstone file of a segment with `len` ordinals. `Corrupt` on a bad magic,
    /// length, count or checksum.
    pub fn read(path: &Path, len: u32) -> Result<Tombstones> {
        todo!("U10 tombstones: Tombstones::read")
    }

    /// Memory held, in bytes.
    pub fn resident_bytes(&self) -> u64 {
        todo!("U10 tombstones: Tombstones::resident_bytes")
    }
}
