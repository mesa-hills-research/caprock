//! Acceptance tests of unit U10 (tombstones).
#![cfg(feature = "internals")]

mod common;

use common::TempDir;
use mhr_vector::Error;
use mhr_vector::tombstones::{TOMBSTONE_MAGIC, Tombstones};

#[test]
fn copy_on_write() {
    let none = Tombstones::none(100);
    assert_eq!(none.ordinals(), 100);
    assert_eq!(none.deleted(), 0);
    assert_eq!(none.live(), 100);
    assert!((0..100).all(|o| !none.is_deleted(o)));
    let some = none.with_deleted(&[3, 7, 7, 99]);
    assert_eq!(none.deleted(), 0, "the original is unchanged");
    assert_eq!(some.deleted(), 3);
    assert_eq!(some.live(), 97);
    assert!(some.is_deleted(3) && some.is_deleted(7) && some.is_deleted(99));
    assert!(!some.is_deleted(4));
    let more = some.with_deleted(&[3, 4]);
    assert_eq!(more.deleted(), 4);
    assert_eq!(some.deleted(), 3);
    assert!(more.resident_bytes() >= 100 / 8);
    assert_eq!(Tombstones::none(0).live(), 0);
}

#[test]
fn files_round_trip_and_damage_is_detected() {
    let dir = TempDir::new("tombstones");
    let t = Tombstones::none(1000).with_deleted(&[0, 1, 500, 999]);
    t.write(dir.path(), "del-1.bits").unwrap();
    let path = dir.path().join("del-1.bits");
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..8], &TOMBSTONE_MAGIC);
    assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 1000);
    assert_eq!(u64::from_le_bytes(bytes[16..24].try_into().unwrap()), 4);
    assert_eq!(bytes.len(), 24 + 1000usize.div_ceil(64) * 8 + 4);
    assert_eq!(Tombstones::read(&path, 1000).unwrap(), t);
    assert!(matches!(
        Tombstones::read(&path, 999),
        Err(Error::Corrupt { .. })
    ));
    let mut bad = bytes.clone();
    bad[30] ^= 0x10;
    std::fs::write(&path, &bad).unwrap();
    assert!(matches!(
        Tombstones::read(&path, 1000),
        Err(Error::Corrupt { .. })
    ));
    std::fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
    assert!(matches!(
        Tombstones::read(&path, 1000),
        Err(Error::Corrupt { .. })
    ));
    // A count that disagrees with the bits, with a valid checksum.
    let mut lying = bytes.clone();
    lying[16..24].copy_from_slice(&5u64.to_le_bytes());
    let end = lying.len() - 4;
    let mut h = crc32fast::Hasher::new();
    h.update(&lying[..end]);
    let crc = h.finalize();
    lying[end..].copy_from_slice(&crc.to_le_bytes());
    std::fs::write(&path, &lying).unwrap();
    assert!(matches!(
        Tombstones::read(&path, 1000),
        Err(Error::Corrupt { .. })
    ));
}
