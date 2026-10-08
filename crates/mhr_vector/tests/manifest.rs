//! Manifests: file names, CURRENT, garbage collection and the writer lock.
#![cfg(feature = "internals")]

mod common;

use common::TempDir;
use mhr_vector::format::FORMAT_VERSION;
use mhr_vector::manifest::{
    self, CURRENT, IndexLock, Manifest, SegmentEntry, garbage, manifest_file_name, publish,
    read_current, segment_file_name, tombstone_file_name,
};
use mhr_vector::{Error, IndexConfig, Metric};

fn manifest(generation: u64, segments: &[u64]) -> Manifest {
    Manifest {
        format_version: FORMAT_VERSION,
        generation,
        config: IndexConfig::new(8, Metric::Cosine),
        segments: segments
            .iter()
            .map(|&id| SegmentEntry {
                id,
                file: segment_file_name(id),
                vectors: 10,
                live: 9,
                tombstones: Some(tombstone_file_name(id, generation)),
                has_graph: false,
                bytes: 100,
            })
            .collect(),
        next_segment_id: segments.iter().max().map_or(1, |m| m + 1),
        checkpoint: Some(manifest::to_hex(b"\x00\x01seq")),
        version: generation * 10,
    }
}

#[test]
fn names() {
    assert_eq!(manifest_file_name(1), "gen-00000000000000000001.json");
    assert_eq!(segment_file_name(42), "seg-00000000000000000042.mhrv");
    assert_eq!(
        tombstone_file_name(3, 7),
        "del-00000000000000000003-00000000000000000007.bits"
    );
    assert_eq!(
        manifest_file_name(u64::MAX),
        format!("gen-{}.json", u64::MAX)
    );
}

#[test]
fn hex() {
    for bytes in [&b""[..], b"\x00", b"\xff\x10abc"] {
        let h = manifest::to_hex(bytes);
        assert!(
            h.chars()
                .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
        );
        assert_eq!(manifest::from_hex(&h).unwrap(), bytes);
    }
    assert_eq!(manifest::to_hex(b"\x01\xab"), "01ab");
    assert_eq!(manifest::from_hex("abc"), None);
    assert_eq!(manifest::from_hex("zz"), None);
}

#[test]
fn publish_and_read_current() {
    let dir = TempDir::new("manifest-publish");
    assert_eq!(read_current(dir.path()).unwrap(), None);
    let m1 = manifest(1, &[1]);
    publish(dir.path(), &m1).unwrap();
    assert_eq!(read_current(dir.path()).unwrap(), Some(m1.clone()));
    let m2 = manifest(2, &[1, 2]);
    publish(dir.path(), &m2).unwrap();
    assert_eq!(read_current(dir.path()).unwrap(), Some(m2.clone()));
    let current = std::fs::read_to_string(dir.path().join(CURRENT)).unwrap();
    assert_eq!(current.trim(), manifest_file_name(2));
    let on_disk: Manifest = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join(manifest_file_name(2))).unwrap(),
    )
    .unwrap();
    assert_eq!(on_disk, m2);
    assert!(
        dir.path().join(manifest_file_name(1)).exists(),
        "publish removes nothing"
    );
    let leftovers: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|n| n.ends_with(".tmp"))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
}

#[test]
fn bad_current_is_reported() {
    let dir = TempDir::new("manifest-bad");
    std::fs::write(dir.path().join(CURRENT), manifest_file_name(9)).unwrap();
    assert!(matches!(
        read_current(dir.path()),
        Err(Error::Corrupt { .. })
    ));
    let mut newer = manifest(9, &[]);
    newer.format_version = FORMAT_VERSION + 1;
    std::fs::write(
        dir.path().join(manifest_file_name(9)),
        serde_json::to_string(&newer).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        read_current(dir.path()),
        Err(Error::UnsupportedFormat { .. })
    ));
    std::fs::write(dir.path().join(manifest_file_name(9)), "{ not json").unwrap();
    assert!(matches!(
        read_current(dir.path()),
        Err(Error::Corrupt { .. })
    ));
}

#[test]
fn garbage_lists_exactly_the_unreferenced_files() {
    let dir = TempDir::new("manifest-garbage");
    let previous = manifest(1, &[1, 2]);
    let current = manifest(2, &[2, 3]);
    publish(dir.path(), &previous).unwrap();
    publish(dir.path(), &current).unwrap();
    let mut files = vec![
        segment_file_name(1),
        segment_file_name(2),
        segment_file_name(3),
        segment_file_name(4),
        tombstone_file_name(1, 1),
        tombstone_file_name(2, 1),
        tombstone_file_name(2, 2),
        tombstone_file_name(3, 2),
        tombstone_file_name(3, 1),
        format!("{}.tmp", segment_file_name(5)),
        "notes.txt".to_string(),
        manifest_file_name(0),
    ];
    files.sort();
    for f in &files {
        std::fs::write(dir.path().join(f), b"x").unwrap();
    }
    std::fs::write(dir.path().join(manifest::LOCK), b"").unwrap();
    let got = garbage(dir.path(), &current, Some(&previous)).unwrap();
    let mut want: Vec<std::path::PathBuf> = [
        segment_file_name(4),
        tombstone_file_name(3, 1),
        format!("{}.tmp", segment_file_name(5)),
        manifest_file_name(0),
    ]
    .iter()
    .map(|f| dir.path().join(f))
    .collect();
    want.sort();
    assert_eq!(got, want);
    let without_previous = garbage(dir.path(), &current, None).unwrap();
    for f in [
        segment_file_name(1),
        tombstone_file_name(1, 1),
        tombstone_file_name(2, 1),
        manifest_file_name(1),
    ] {
        assert!(without_previous.contains(&dir.path().join(&f)), "{f}");
    }
    assert!(!without_previous.contains(&dir.path().join("notes.txt")));
    assert!(!without_previous.contains(&dir.path().join(CURRENT)));
}

#[test]
fn the_lock_is_exclusive() {
    let dir = TempDir::new("manifest-lock");
    let first = IndexLock::acquire(dir.path()).unwrap();
    assert!(matches!(
        IndexLock::acquire(dir.path()),
        Err(Error::Locked { .. })
    ));
    drop(first);
    let _again = IndexLock::acquire(dir.path()).unwrap();
}
