//! Section files: layout, checksums, positional reads and atomic writes.
#![cfg(feature = "internals")]

mod common;

use caprock::format::{
    self, FORMAT_VERSION, HEADER_LEN, SEGMENT_MAGIC, SectionFile, SectionKind, SectionWriter,
};
use caprock::{Error, VerifyMode};
use common::TempDir;

const ALL: [SectionKind; 13] = [
    SectionKind::Info,
    SectionKind::Vectors,
    SectionKind::Codes,
    SectionKind::CodecParams,
    SectionKind::KeyBytes,
    SectionKind::KeyOffsets,
    SectionKind::KeyIndex,
    SectionKind::PathBytes,
    SectionKind::PathOffsets,
    SectionKind::PathOfOrdinal,
    SectionKind::Languages,
    SectionKind::LanguageOfOrdinal,
    SectionKind::Graph,
];

fn crc(bytes: &[u8]) -> u32 {
    let mut h = crc32fast::Hasher::new();
    h.update(bytes);
    h.finalize()
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

/// Writes a file with three sections; returns (path, info, vectors bytes, codes).
fn sample(dir: &std::path::Path) -> (std::path::PathBuf, Vec<u8>, Vec<u8>, Vec<u8>) {
    let path = dir.join("seg-1.mhrv");
    let info = br#"{"hello":"world"}"#.to_vec();
    let vectors: Vec<u8> = (0..100_003u32)
        .flat_map(|i| (i as f32).to_le_bytes())
        .collect();
    let codes: Vec<u8> = (0..5000u32).map(|i| (i * 7) as u8).collect();
    let mut w = SectionWriter::create(&path, SEGMENT_MAGIC).unwrap();
    w.write_section(SectionKind::Info, &info).unwrap();
    w.begin_section(SectionKind::Vectors).unwrap();
    for chunk in vectors.chunks(4096 * 3 + 5) {
        w.write_chunk(chunk).unwrap();
    }
    w.end_section().unwrap();
    w.write_section(SectionKind::Codes, &codes).unwrap();
    let len = w.finish().unwrap();
    assert_eq!(len, std::fs::metadata(&path).unwrap().len());
    (path, info, vectors, codes)
}

#[test]
fn kinds_and_alignment() {
    for kind in ALL {
        assert_eq!(SectionKind::from_id(kind.id()), Some(kind));
        let want = match kind {
            SectionKind::Vectors | SectionKind::Codes | SectionKind::Graph => 4096,
            _ => 64,
        };
        assert_eq!(kind.alignment(), want, "{kind:?}");
    }
    assert_eq!(SectionKind::from_id(0), None);
    assert_eq!(SectionKind::from_id(14), None);
}

#[test]
fn written_files_follow_the_layout() {
    let dir = TempDir::new("format-layout");
    let (path, info, vectors, codes) = sample(dir.path());
    assert!(!dir.path().join("seg-1.mhrv.tmp").exists());
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[0..8], &SEGMENT_MAGIC);
    assert_eq!(u32_at(&bytes, 8), FORMAT_VERSION);
    assert_eq!(u32_at(&bytes, 12), 3);
    let table_offset = u64_at(&bytes, 16) as usize;
    let table_len = u64_at(&bytes, 24) as usize;
    assert_eq!(table_len, 3 * 32);
    assert_eq!(
        u32_at(&bytes, 32),
        crc(&bytes[table_offset..table_offset + table_len])
    );
    assert!(bytes[36..60].iter().all(|b| *b == 0));
    assert_eq!(u32_at(&bytes, 60), crc(&bytes[0..60]));
    let mut seen = Vec::new();
    for e in 0..3 {
        let at = table_offset + e * 32;
        let kind = SectionKind::from_id(u32_at(&bytes, at)).unwrap();
        assert_eq!(u32_at(&bytes, at + 4), 0);
        let offset = u64_at(&bytes, at + 8) as usize;
        let len = u64_at(&bytes, at + 16) as usize;
        assert_eq!(offset as u64 % kind.alignment(), 0, "{kind:?} aligned");
        assert!(offset as u64 >= HEADER_LEN);
        assert_eq!(u32_at(&bytes, at + 24), crc(&bytes[offset..offset + len]));
        let data = &bytes[offset..offset + len];
        let want = match kind {
            SectionKind::Info => &info,
            SectionKind::Vectors => &vectors,
            SectionKind::Codes => &codes,
            _ => unreachable!(),
        };
        assert_eq!(data, &want[..], "{kind:?}");
        seen.push(kind);
    }
    seen.sort();
    assert_eq!(
        seen,
        vec![SectionKind::Info, SectionKind::Vectors, SectionKind::Codes]
    );
}

#[test]
fn files_read_back() {
    let dir = TempDir::new("format-read");
    let (path, info, vectors, codes) = sample(dir.path());
    for verify in [VerifyMode::Headers, VerifyMode::Full] {
        let f = SectionFile::open(&path, SEGMENT_MAGIC, verify).unwrap();
        assert_eq!(f.path(), path.as_path());
        assert_eq!(f.file_len(), std::fs::metadata(&path).unwrap().len());
        assert_eq!(f.entries().len(), 3);
        assert_eq!(f.read_section(SectionKind::Info).unwrap(), info);
        assert_eq!(f.read_section(SectionKind::Vectors).unwrap(), vectors);
        assert_eq!(f.read_section(SectionKind::Codes).unwrap(), codes);
        assert!(f.entry(SectionKind::Graph).is_none());
        assert!(matches!(
            f.read_section(SectionKind::Graph),
            Err(Error::Corrupt { .. })
        ));
        let mut buf = [0u8; 8];
        f.read_at(SectionKind::Vectors, 4 * 1000, &mut buf).unwrap();
        assert_eq!(
            buf,
            [1000f32.to_le_bytes(), 1001f32.to_le_bytes()].concat()[..]
        );
        let end = vectors.len() as u64;
        assert!(
            f.read_at(SectionKind::Vectors, end - 4, &mut buf).is_err(),
            "past the section"
        );
        f.verify_all().unwrap();
    }
}

#[test]
fn damage_is_detected() {
    let dir = TempDir::new("format-damage");
    let (path, ..) = sample(dir.path());
    let good = std::fs::read(&path).unwrap();
    let f = SectionFile::open(&path, SEGMENT_MAGIC, VerifyMode::Headers).unwrap();
    let codes_at = f.entry(SectionKind::Codes).unwrap().offset as usize;
    drop(f);

    // A flipped byte inside a section: Headers opens, the section read fails, Full fails.
    let mut bad = good.clone();
    bad[codes_at + 10] ^= 0xFF;
    std::fs::write(&path, &bad).unwrap();
    let f = SectionFile::open(&path, SEGMENT_MAGIC, VerifyMode::Headers).unwrap();
    assert!(matches!(
        f.read_section(SectionKind::Codes),
        Err(Error::Corrupt { .. })
    ));
    assert!(f.verify_all().is_err());
    assert!(matches!(
        SectionFile::open(&path, SEGMENT_MAGIC, VerifyMode::Full),
        Err(Error::Corrupt { .. })
    ));

    // A damaged header, a wrong magic, a truncated file.
    let mut bad = good.clone();
    bad[13] ^= 1;
    std::fs::write(&path, &bad).unwrap();
    assert!(matches!(
        SectionFile::open(&path, SEGMENT_MAGIC, VerifyMode::Headers),
        Err(Error::Corrupt { .. })
    ));
    std::fs::write(&path, &good).unwrap();
    assert!(matches!(
        SectionFile::open(&path, *b"MHRVXXXX", VerifyMode::Headers),
        Err(Error::Corrupt { .. })
    ));
    std::fs::write(&path, &good[..good.len() - 7]).unwrap();
    assert!(matches!(
        SectionFile::open(&path, SEGMENT_MAGIC, VerifyMode::Headers),
        Err(Error::Corrupt { .. })
    ));
    std::fs::write(&path, &good[..40]).unwrap();
    assert!(matches!(
        SectionFile::open(&path, SEGMENT_MAGIC, VerifyMode::Headers),
        Err(Error::Corrupt { .. })
    ));

    // A newer version with a valid header checksum.
    let mut newer = good.clone();
    newer[8..12].copy_from_slice(&(FORMAT_VERSION + 1).to_le_bytes());
    let c = crc(&newer[0..60]);
    newer[60..64].copy_from_slice(&c.to_le_bytes());
    std::fs::write(&path, &newer).unwrap();
    assert!(matches!(
        SectionFile::open(&path, SEGMENT_MAGIC, VerifyMode::Headers),
        Err(Error::UnsupportedFormat { .. })
    ));
}

#[test]
fn writers_refuse_misuse() {
    let dir = TempDir::new("format-misuse");
    let (path, ..) = sample(dir.path());
    assert!(
        SectionWriter::create(&path, SEGMENT_MAGIC).is_err(),
        "final file exists"
    );
    let other = dir.path().join("seg-2.mhrv");
    std::fs::write(dir.path().join("seg-2.mhrv.tmp"), b"stale").unwrap();
    let mut w = SectionWriter::create(&other, SEGMENT_MAGIC).unwrap();
    w.write_section(SectionKind::Info, b"{}").unwrap();
    assert!(
        w.write_section(SectionKind::Info, b"{}").is_err(),
        "a kind twice"
    );
    w.finish().unwrap();
    let f = SectionFile::open(&other, SEGMENT_MAGIC, VerifyMode::Full).unwrap();
    assert_eq!(f.read_section(SectionKind::Info).unwrap(), b"{}");
}

#[test]
fn helpers() {
    let dir = TempDir::new("format-helpers");
    format::atomic_write(dir.path(), "x.json", b"one").unwrap();
    format::atomic_write(dir.path(), "x.json", b"two").unwrap();
    assert_eq!(std::fs::read(dir.path().join("x.json")).unwrap(), b"two");
    let names: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(names.len(), 1, "no temporary file left: {names:?}");
    format::sync_dir(dir.path()).unwrap();

    let file = std::fs::File::open(dir.path().join("x.json")).unwrap();
    let mut buf = [0u8; 2];
    format::read_exact_at(&file, &mut buf, 1).unwrap();
    assert_eq!(&buf, b"wo");
    assert!(format::read_exact_at(&file, &mut buf, 2).is_err());

    let floats = [1.5f32, -0.0, f32::MAX, 1e-40];
    let bytes = format::f32s_to_le(&floats);
    assert_eq!(bytes.len(), 16);
    let back = format::f32s_from_le(&bytes).unwrap();
    assert_eq!(
        back.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        floats.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(format::f32s_from_le(&bytes[..5]), None);
    assert_eq!(
        format::u32s_from_le(&format::u32s_to_le(&[1, u32::MAX])).unwrap(),
        vec![1, u32::MAX]
    );
    assert_eq!(format::u32s_from_le(&[0u8; 3]), None);
    assert_eq!(
        format::u16s_from_le(&format::u16s_to_le(&[7, u16::MAX])).unwrap(),
        vec![7, u16::MAX]
    );
    assert_eq!(format::u16s_from_le(&[0u8; 3]), None);
}
