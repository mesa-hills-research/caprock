//! Sealed segments: writing, opening, lookups and deterministic bytes.
#![cfg(feature = "internals")]

mod common;

use common::{TempDir, stored_vectors};
use mhr_vector::filter::PathTable;
use mhr_vector::format::{SEGMENT_MAGIC, SectionFile, SectionKind};
use mhr_vector::segment::{MemoryRecords, RecordSource, Segment, SegmentBuild, write_segment};
use mhr_vector::{CodeKind, Error, IndexConfig, Metric, VerifyMode};
use mhr_vector_oracle::filter::path_has_prefix;
use mhr_vector_oracle::generate::{key, path_for};
use mhr_vector_oracle::rng::Rng;

/// `n` records, about four per file, in a shuffled order.
fn records(metric: Metric, dim: usize, n: usize, seed: u64) -> MemoryRecords {
    let vectors = stored_vectors(metric, dim, n, seed);
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = Rng::new(seed);
    for i in (1..n).rev() {
        order.swap(i, rng.below(i as u64 + 1) as usize);
    }
    let mut r = MemoryRecords {
        dim,
        ..MemoryRecords::default()
    };
    for &i in &order {
        let (path, language) = path_for(i as u64 / 4, seed);
        r.keys.push(key(i as u64));
        r.paths.push(path);
        r.languages.push(language);
        r.vectors.extend_from_slice(&vectors[i]);
    }
    r
}

fn build<'a>(config: &'a IndexConfig, id: u64, threads: usize) -> SegmentBuild<'a> {
    SegmentBuild {
        id,
        config,
        graph: false,
        threads,
        memory_budget: 256 << 20,
    }
}

#[test]
fn memory_records_are_a_record_source() {
    let r = records(Metric::Dot, 8, 10, 1);
    assert_eq!(r.len(), 10);
    let mut v = vec![0f32; 8];
    r.read_vector(3, &mut v).unwrap();
    assert_eq!(&v[..], &r.vectors[24..32]);
    assert_eq!(r.key(3), r.keys[3]);
    assert_eq!(r.path(3), r.paths[3]);
    assert_eq!(r.language(3), r.languages[3]);
}

#[test]
fn segments_hold_their_records_sorted_by_path_and_key() {
    for (metric, codes) in [
        (Metric::Cosine, CodeKind::F32),
        (Metric::Cosine, CodeKind::Int8),
        (Metric::Dot, CodeKind::Binary),
        (Metric::L2, CodeKind::Int8),
    ] {
        let dim = 24;
        let config = IndexConfig::new(dim as u32, metric).with_codes(codes);
        let source = records(metric, dim, 1500, 2);
        let dir = TempDir::new("segment-basic");
        let info = write_segment(dir.path(), &source, &build(&config, 7, 2)).unwrap();
        assert_eq!(info.id, 7);
        assert_eq!(info.file, "seg-00000000000000000007.mhrv");
        assert_eq!(info.vectors, 1500);
        assert!(!info.has_graph);
        assert_eq!(
            info.bytes,
            std::fs::metadata(dir.path().join(&info.file))
                .unwrap()
                .len()
        );
        let seg = Segment::open(dir.path(), &info.file, &config, VerifyMode::Full).unwrap();
        assert_eq!((seg.id(), seg.len(), seg.dim()), (7, 1500, dim));
        assert_eq!(seg.file_name(), info.file);
        assert_eq!(seg.codec().kind(), codes);
        assert_eq!(seg.codes().len(), 1500 * seg.codec().code_len());
        assert!(seg.graph().is_none());
        assert_eq!(seg.disk_bytes(), info.bytes);
        assert!(seg.resident_bytes() >= seg.codes().len() as u64);
        let mut previous: Option<(String, String)> = None;
        let mut v = vec![0f32; dim];
        let mut code = vec![0u8; seg.codec().code_len()];
        for o in 0..seg.len() {
            let pair = (seg.path(o).to_string(), seg.key(o).to_string());
            if let Some(p) = &previous {
                assert!(p < &pair, "sorted by (path, key)");
            }
            previous = Some(pair);
            let i = source.keys.iter().position(|k| k == seg.key(o)).unwrap();
            assert_eq!(seg.find(seg.key(o)), Some(o));
            assert_eq!(seg.path(o), source.paths[i]);
            assert_eq!(seg.language(o), source.languages[i]);
            seg.read_vector(o, &mut v).unwrap();
            assert_eq!(&v[..], &source.vectors[i * dim..(i + 1) * dim]);
            seg.codec().encode(&v, &mut code);
            assert_eq!(seg.code(o), &code[..], "codes are the codec's encoding");
            assert_eq!(
                seg.language_of(o),
                seg.language_id(seg.language(o)).unwrap()
            );
        }
        assert_eq!(seg.find("no such key"), None);
        let ords = [5u32, 0, 1499, 5];
        let mut many = vec![0f32; ords.len() * dim];
        seg.read_vectors(&ords, &mut many).unwrap();
        for (j, &o) in ords.iter().enumerate() {
            seg.read_vector(o, &mut v).unwrap();
            assert_eq!(&many[j * dim..(j + 1) * dim], &v[..]);
        }
        assert_eq!(seg.language_id("nosuch"), None);
    }
}

#[test]
fn prefix_ranges_are_exact() {
    let dim = 8;
    let config = IndexConfig::new(dim as u32, Metric::Cosine);
    let mut source = records(Metric::Cosine, dim, 800, 3);
    for (i, p) in ["p10/d1", "p10/d1-x.rs", "p10/d1.rs", "p10/d1/z.go"]
        .iter()
        .enumerate()
    {
        source.keys.push(format!("special{i}"));
        source.paths.push(p.to_string());
        source.languages.push("go".into());
        let first = source.vectors[..dim].to_vec();
        source.vectors.extend_from_slice(&first);
    }
    let dir = TempDir::new("segment-prefix");
    let info = write_segment(dir.path(), &source, &build(&config, 1, 1)).unwrap();
    let seg = Segment::open(dir.path(), &info.file, &config, VerifyMode::Headers).unwrap();
    assert_eq!(seg.ordinals(), seg.len());
    for prefix in [
        "",
        "/",
        "p10",
        "p10/",
        "p10/d1",
        "p1",
        "p50/d7",
        "rest",
        "zzz",
        "p001",
        "p10/d1.rs",
    ] {
        let want: Vec<u32> = (0..seg.len())
            .filter(|&o| path_has_prefix(seg.path(o), prefix))
            .collect();
        let ranges = seg.prefix_ranges(prefix);
        assert!(ranges.len() <= 2 && ranges.iter().all(|r| r.start < r.end));
        assert!(ranges.windows(2).all(|w| w[0].end <= w[1].start));
        let got: Vec<u32> = ranges.into_iter().flatten().collect();
        assert_eq!(got, want, "prefix {prefix:?}");
    }
}

#[test]
fn bytes_depend_only_on_the_records() {
    let dim = 16;
    for codes in [CodeKind::F32, CodeKind::Int8, CodeKind::Binary] {
        let config = IndexConfig::new(dim as u32, Metric::Cosine).with_codes(codes);
        let a = records(Metric::Cosine, dim, 700, 4);
        // The same records in another order.
        let mut b = MemoryRecords {
            dim,
            ..MemoryRecords::default()
        };
        for i in (0..a.len()).rev() {
            b.keys.push(a.keys[i].clone());
            b.paths.push(a.paths[i].clone());
            b.languages.push(a.languages[i].clone());
            b.vectors
                .extend_from_slice(&a.vectors[i * dim..(i + 1) * dim]);
        }
        let d1 = TempDir::new("segment-det1");
        let d2 = TempDir::new("segment-det2");
        let i1 = write_segment(d1.path(), &a, &build(&config, 3, 1)).unwrap();
        let i2 = write_segment(d2.path(), &b, &build(&config, 3, 7)).unwrap();
        let f1 = std::fs::read(d1.path().join(&i1.file)).unwrap();
        let f2 = std::fs::read(d2.path().join(&i2.file)).unwrap();
        assert!(
            f1 == f2,
            "{codes:?}: identical bytes for any order and thread count"
        );
    }
}

#[test]
fn refusals() {
    let dim = 8;
    let config = IndexConfig::new(dim as u32, Metric::Cosine);
    let dir = TempDir::new("segment-refuse");
    let empty = MemoryRecords {
        dim,
        ..MemoryRecords::default()
    };
    assert!(matches!(
        write_segment(dir.path(), &empty, &build(&config, 1, 1)),
        Err(Error::InvalidArgument(_))
    ));
    let source = records(Metric::Cosine, dim, 5000, 5);
    let mut tiny = build(&config, 2, 1);
    tiny.memory_budget = 1024;
    assert!(matches!(
        write_segment(dir.path(), &source, &tiny),
        Err(Error::Budget { .. })
    ));
    let info = write_segment(dir.path(), &source, &build(&config, 3, 1)).unwrap();
    let other_dim = IndexConfig::new(9, Metric::Cosine);
    assert!(matches!(
        Segment::open(dir.path(), &info.file, &other_dim, VerifyMode::Headers),
        Err(Error::Corrupt { .. })
    ));
    let other_metric = IndexConfig::new(dim as u32, Metric::L2);
    assert!(Segment::open(dir.path(), &info.file, &other_metric, VerifyMode::Headers).is_err());
    let path = dir.path().join(&info.file);
    let at = {
        let f = SectionFile::open(&path, SEGMENT_MAGIC, VerifyMode::Headers).unwrap();
        let e = f.entry(SectionKind::Vectors).unwrap();
        (e.offset + e.len / 2) as usize
    };
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[at] ^= 0x55;
    std::fs::write(&path, &bytes).unwrap();
    assert!(Segment::open(dir.path(), &info.file, &config, VerifyMode::Full).is_err());
}
