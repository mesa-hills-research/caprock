//! Deleted ordinals of one sealed segment.
//!
//! Tombstones are immutable values shared by snapshots: deleting creates a new value
//! (copy-on-write), so a snapshot never sees a later delete. On disk a tombstone file is
//! `b"MHRVDEL1"`, the segment's ordinal count (u64), the deleted count (u64), the bitset
//! ([`crate::util::Bitset::to_bytes`]) and a CRC-32 of everything before it (u32), all
//! little-endian; it is written atomically ([`crate::format::atomic_write`]).

use std::fs::File;
use std::io::ErrorKind;
use std::path::Path;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::format::{atomic_write, read_exact_at};
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
        Tombstones {
            bits: Arc::new(Bitset::new(len as usize)),
            deleted: 0,
        }
    }

    /// The segment's ordinal count.
    pub fn ordinals(&self) -> u32 {
        self.bits.len() as u32
    }

    /// Whether an ordinal is deleted. Panics if `ordinal >= self.ordinals()`.
    pub fn is_deleted(&self, ordinal: u32) -> bool {
        self.bits.get(ordinal as usize)
    }

    /// Number of deleted ordinals.
    pub fn deleted(&self) -> u32 {
        self.deleted
    }

    /// Live ordinals: `len - deleted`.
    pub fn live(&self) -> u32 {
        self.ordinals() - self.deleted
    }

    /// A new value with `ordinals` deleted as well (already deleted ones are ignored).
    /// `self` is unchanged. Panics if any ordinal is outside the segment.
    pub fn with_deleted(&self, ordinals: &[u32]) -> Tombstones {
        let mut updated = self.clone();
        for &ordinal in ordinals {
            // Keep sharing the allocation when the batch makes no changes.
            if !updated.is_deleted(ordinal) {
                Arc::make_mut(&mut updated.bits).set(ordinal as usize);
                updated.deleted += 1;
            }
        }
        updated
    }

    /// Writes `dir/name` atomically in the format above.
    pub fn write(&self, dir: &Path, name: &str) -> Result<()> {
        let mut bytes = Vec::with_capacity(24 + self.resident_bytes() as usize + 4);
        bytes.extend_from_slice(&TOMBSTONE_MAGIC);
        bytes.extend_from_slice(&u64::from(self.ordinals()).to_le_bytes());
        bytes.extend_from_slice(&u64::from(self.deleted).to_le_bytes());
        bytes.extend_from_slice(&self.bits.to_bytes());
        let crc = crc32fast::hash(&bytes);
        bytes.extend_from_slice(&crc.to_le_bytes());
        atomic_write(dir, name, &bytes)
    }

    /// Reads a tombstone file of a segment with `len` ordinals. `Corrupt` on a bad magic,
    /// length, count or checksum.
    pub fn read(path: &Path, len: u32) -> Result<Tombstones> {
        // Size the read from the caller's segment length, never from untrusted counts.
        let expected_bytes = 24 + (len as usize).div_ceil(64) * 8 + 4;
        let file = File::open(path).map_err(|e| Error::io(path, e))?;
        let file_len = file.metadata().map_err(|e| Error::io(path, e))?.len();
        if file_len != expected_bytes as u64 {
            return Err(Error::corrupt(path, "incorrect tombstone file length"));
        }
        let mut bytes = vec![0; expected_bytes];
        read_exact_at(&file, &mut bytes, 0).map_err(|e| {
            if e.kind() == ErrorKind::UnexpectedEof {
                Error::corrupt(path, "truncated tombstone file")
            } else {
                Error::io(path, e)
            }
        })?;

        // All slices below have fixed lengths within the checked file size (>= 28).
        if bytes[..8] != TOMBSTONE_MAGIC {
            return Err(Error::corrupt(path, "incorrect tombstone magic"));
        }
        let crc_offset = expected_bytes - 4;
        let crc = u32::from_le_bytes(bytes[crc_offset..].try_into().unwrap());
        if crc32fast::hash(&bytes[..crc_offset]) != crc {
            return Err(Error::corrupt(path, "tombstone checksum mismatch"));
        }
        let ordinals = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        if ordinals != u64::from(len) {
            return Err(Error::corrupt(path, "tombstone ordinal count mismatch"));
        }
        let deleted = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        if deleted > u64::from(len) {
            return Err(Error::corrupt(
                path,
                "tombstone deleted count exceeds ordinal count",
            ));
        }
        let bits = Bitset::from_bytes(&bytes[24..crc_offset], len as usize)
            .ok_or_else(|| Error::corrupt(path, "invalid tombstone bitset"))?;
        if bits.count_ones() as u64 != deleted {
            return Err(Error::corrupt(
                path,
                "tombstone deleted count disagrees with bitset",
            ));
        }
        Ok(Tombstones {
            bits: Arc::new(bits),
            deleted: deleted as u32,
        })
    }

    /// Memory held, in bytes.
    pub fn resident_bytes(&self) -> u64 {
        u64::from(self.ordinals()).div_ceil(64) * 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "mhrv-tombstone-unit-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn unchanged_updates_keep_sharing_and_changed_updates_are_independent() {
        let initial = Tombstones::none(130);
        let clone = initial.clone();
        assert!(Arc::ptr_eq(&initial.bits, &clone.bits));
        assert!(Arc::ptr_eq(&initial.bits, &initial.with_deleted(&[]).bits));
        let changed = initial.with_deleted(&[0, 63, 64, 129, 64]);
        assert!(!Arc::ptr_eq(&initial.bits, &changed.bits));
        assert_eq!(initial.deleted(), 0);
        assert_eq!(clone.deleted(), 0);
        assert_eq!(changed.deleted(), 4);
        assert_eq!(changed.live(), 126);
        let unchanged = changed.with_deleted(&[64, 0, 64]);
        assert!(Arc::ptr_eq(&changed.bits, &unchanged.bits));
        assert_eq!(changed, unchanged);
        let all = changed.with_deleted(&(0..130).collect::<Vec<_>>());
        assert_eq!(all.deleted(), 130);
        assert_eq!(all.live(), 0);
        assert_eq!(changed.deleted(), 4);
    }

    #[test]
    fn round_trip_at_word_boundaries_including_empty_and_fully_deleted() {
        let dir = TestDir::new();
        let path = dir.0.join("deletions.bits");
        for len in [0, 1, 63, 64, 65, 127, 128, 129] {
            let none = Tombstones::none(len);
            let all = none.with_deleted(&(0..len).collect::<Vec<_>>());
            for tombstones in [none, all] {
                tombstones.write(&dir.0, "deletions.bits").unwrap();
                let bytes = fs::read(&path).unwrap();
                let mut expected = TOMBSTONE_MAGIC.to_vec();
                expected.extend_from_slice(&u64::from(len).to_le_bytes());
                expected.extend_from_slice(&u64::from(tombstones.deleted()).to_le_bytes());
                expected.extend_from_slice(&tombstones.bits.to_bytes());
                let crc = crc32fast::hash(&expected);
                expected.extend_from_slice(&crc.to_le_bytes());
                assert_eq!(bytes, expected);
                assert_eq!(Tombstones::read(&path, len).unwrap(), tombstones);
                assert_eq!(tombstones.resident_bytes(), u64::from(len).div_ceil(64) * 8);
                assert!(!dir.0.join("deletions.bits.tmp").exists());
            }
        }
    }

    fn assert_corrupt(path: &Path, bytes: &[u8]) {
        fs::write(path, bytes).unwrap();
        match Tombstones::read(path, 65) {
            Err(Error::Corrupt { path: found, .. }) => assert_eq!(found, path),
            other => panic!("expected Corrupt with the file path, got {other:?}"),
        }
    }

    #[test]
    fn rejects_truncation_extra_bytes_damage_and_checksummed_invalid_fields() {
        let dir = TestDir::new();
        let path = dir.0.join("deletions.bits");
        Tombstones::none(65)
            .with_deleted(&[0, 64])
            .write(&dir.0, "deletions.bits")
            .unwrap();
        let original = fs::read(&path).unwrap();
        for end in 0..original.len() {
            assert_corrupt(&path, &original[..end]);
        }
        let mut extended = original.clone();
        extended.push(0);
        assert_corrupt(&path, &extended);
        for offset in 0..original.len() {
            let mut damaged = original.clone();
            damaged[offset] ^= 1;
            assert_corrupt(&path, &damaged);
        }
        for (range, replacement) in [
            (0..8, *b"BADMAGIC"),
            (8..16, 64u64.to_le_bytes()),
            (8..16, u64::MAX.to_le_bytes()),
            (16..24, 1u64.to_le_bytes()),
            (16..24, u64::MAX.to_le_bytes()),
            // Same population count, but a padding bit set instead of ordinal 64.
            (32..40, 2u64.to_le_bytes()),
        ] {
            let mut malformed = original.clone();
            malformed[range].copy_from_slice(&replacement);
            let crc_offset = malformed.len() - 4;
            let crc = crc32fast::hash(&malformed[..crc_offset]);
            malformed[crc_offset..].copy_from_slice(&crc.to_le_bytes());
            assert_corrupt(&path, &malformed);
        }
    }

    #[test]
    fn missing_files_are_io_errors_with_the_path() {
        let dir = TestDir::new();
        let path = dir.0.join("missing.bits");
        match Tombstones::read(&path, 0) {
            Err(Error::Io {
                path: found,
                source,
            }) => {
                assert_eq!(found, path);
                assert_eq!(source.kind(), ErrorKind::NotFound);
            }
            other => panic!("expected Io with the file path, got {other:?}"),
        }
    }
}
