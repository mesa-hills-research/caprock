use super::*;
use crate::config::{AnnKind, CodeKind, Metric, VamanaParams};
use crate::format::{SEGMENT_MAGIC, SectionKind, SectionWriter};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

struct Dir(PathBuf);
impl Dir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "mhrv-segment-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("{e}"),
            }
        }
    }
}
impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn records(dim: usize) -> MemoryRecords {
    let paths = [
        "", "", "/", "//x", "/a", "x", "x", "x-", "x.", "x/", "x//", "x//child", "x/a", "x/a/z",
        "x0", "é", "é/子", "éclair",
    ];
    let n = paths.len();
    MemoryRecords {
        dim,
        keys: (0..n).map(|i| format!("key{:03}é", n - i)).collect(),
        paths: paths.iter().map(|p| p.to_string()).collect(),
        languages: (0..n)
            .map(|i| ["", "rust", "\"go\"", "日本語"][i % 4].to_owned())
            .collect(),
        vectors: (0..n)
            .flat_map(|i| {
                (0..dim).map(move |j| {
                    if j % 7 == 0 {
                        -0.0
                    } else {
                        (i as f32 - j as f32) * 0.1
                    }
                })
            })
            .collect(),
    }
}

fn build(config: &IndexConfig) -> SegmentBuild<'_> {
    SegmentBuild {
        config,
        id: 9,
        graph: false,
        threads: 1,
        memory_budget: 64 << 20,
    }
}

#[test]
fn prefixes_and_positional_reads_cover_edge_cases() {
    for kind in [CodeKind::F32, CodeKind::Int8, CodeKind::Binary] {
        let dir = Dir::new();
        let config = IndexConfig::new(17, Metric::Dot).with_codes(kind);
        let source = records(17);
        let info = write_segment(&dir.0, &source, &build(&config)).unwrap();
        let seg = Segment::open(&dir.0, &info.file, &config, VerifyMode::Full).unwrap();
        for prefix in [
            "", "/", "//", "x", "x/", "x//", "x///", "x/a/", "x/a/z", "é", "é/", "éclair",
            "absent", "\0",
        ] {
            let want: Vec<_> = (0..seg.len())
                .filter(|&o| crate::filter::path_has_prefix(seg.path(o), prefix))
                .collect();
            let ranges = seg.prefix_ranges(prefix);
            assert!(ranges.len() <= 2);
            assert!(ranges.iter().all(|r| !r.is_empty()));
            assert!(ranges.windows(2).all(|w| w[0].end <= w[1].start));
            assert_eq!(
                ranges.into_iter().flatten().collect::<Vec<_>>(),
                want,
                "{prefix:?}"
            );
        }
        let verify = || {
            let ordinals: Vec<_> = (0..seg.len())
                .chain((0..seg.len()).rev())
                .chain([0, 0, 3])
                .collect();
            let mut rows = vec![0.0; ordinals.len() * 17];
            seg.read_vectors(&ordinals, &mut rows).unwrap();
            for (&o, row) in ordinals.iter().zip(rows.as_chunks::<17>().0.iter()) {
                let i = source.keys.iter().position(|k| k == seg.key(o)).unwrap();
                assert_eq!(
                    row.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    source.vectors[i * 17..(i + 1) * 17]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
            }
        };
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(verify);
            }
        });
        seg.read_vectors(&[], &mut []).unwrap();
        assert!(matches!(
            seg.read_vector(seg.len(), &mut [0.0; 17]),
            Err(Error::InvalidArgument(_))
        ));
        assert!(seg.read_vector(u32::MAX, &mut [0.0; 17]).is_err());
        assert!(seg.read_vector(0, &mut [0.0; 16]).is_err());
        assert!(seg.read_vectors(&[0, 1], &mut [0.0; 17]).is_err());
        assert!(seg.read_vectors(&[], &mut [0.0; 17]).is_err());
        assert_eq!(KeyLookup::key(&seg, 0), seg.key(0));
        #[cfg(unix)]
        {
            std::fs::remove_file(dir.0.join(&info.file)).unwrap();
            verify(); // Existing open handles survive garbage collection.
        }
    }
}

#[test]
fn headers_verify_every_resident_section_but_not_vectors() {
    for kind in [CodeKind::F32, CodeKind::Int8, CodeKind::Binary] {
        let dir = Dir::new();
        let config = IndexConfig::new(17, Metric::Dot).with_codes(kind);
        let info = write_segment(&dir.0, &records(17), &build(&config)).unwrap();
        let path = dir.0.join(&info.file);
        let f = SectionFile::open(&path, SEGMENT_MAGIC, VerifyMode::Full).unwrap();
        assert_eq!(
            f.entry(SectionKind::Vectors).is_none(),
            kind == CodeKind::F32
        );
        let good = std::fs::read(&path).unwrap();
        for entry in f.entries().iter().filter(|e| e.len > 0) {
            let mut bad = good.clone();
            bad[(entry.offset + entry.len / 2) as usize] ^= 0x01;
            std::fs::write(&path, bad).unwrap();
            let headers = Segment::open(&dir.0, &info.file, &config, VerifyMode::Headers);
            if entry.kind == SectionKind::Vectors {
                assert!(headers.is_ok());
            } else {
                assert!(matches!(headers, Err(Error::Corrupt { path: p, .. }) if p == path));
            }
            assert!(
                matches!(Segment::open(&dir.0, &info.file, &config, VerifyMode::Full), Err(Error::Corrupt { path: p, .. }) if p == path)
            );
        }
    }
}

fn rewrite(
    file: &SectionFile,
    path: &Path,
    mut change: impl FnMut(SectionKind, &mut Vec<u8>) -> bool,
) {
    let mut writer = SectionWriter::create(path, SEGMENT_MAGIC).unwrap();
    for entry in file.entries() {
        let mut bytes = file.read_section(entry.kind).unwrap();
        if change(entry.kind, &mut bytes) {
            writer.write_section(entry.kind, &bytes).unwrap();
        }
    }
    writer.finish().unwrap();
}

#[test]
fn checksummed_invalid_tables_are_corrupt_not_panics() {
    let dir = Dir::new();
    let config = IndexConfig::new(17, Metric::Dot);
    let info = write_segment(&dir.0, &records(17), &build(&config)).unwrap();
    let file = SectionFile::open(&dir.0.join(&info.file), SEGMENT_MAGIC, VerifyMode::Full).unwrap();
    let mutations = [
        (SectionKind::Info, 0, vec![b'?']),
        (SectionKind::CodecParams, 0, vec![0]),
        (SectionKind::KeyBytes, 0, vec![255]),
        (SectionKind::KeyOffsets, 0, 1u32.to_le_bytes().to_vec()),
        (SectionKind::KeyOffsets, 4, u32::MAX.to_le_bytes().to_vec()),
        (SectionKind::KeyOffsets, 8, 1u32.to_le_bytes().to_vec()),
        (SectionKind::KeyIndex, 0, u32::MAX.to_le_bytes().to_vec()),
        (SectionKind::KeyIndex, 0, 0u32.to_le_bytes().to_vec()),
        (SectionKind::PathBytes, 0, vec![255]),
        (SectionKind::PathOffsets, 0, 1u32.to_le_bytes().to_vec()),
        (SectionKind::PathOffsets, 4, u32::MAX.to_le_bytes().to_vec()),
        (SectionKind::PathOfOrdinal, 0, 1u32.to_le_bytes().to_vec()),
        (
            SectionKind::PathOfOrdinal,
            8,
            u32::MAX.to_le_bytes().to_vec(),
        ),
        (
            SectionKind::LanguageOfOrdinal,
            0,
            u16::MAX.to_le_bytes().to_vec(),
        ),
        (SectionKind::Languages, 0, vec![b'?']),
    ];
    let assert_bad = |name: &str| {
        for verify in [VerifyMode::Headers, VerifyMode::Full] {
            assert!(
                matches!(Segment::open(&dir.0, name, &config, verify), Err(Error::Corrupt { path, .. }) if path == dir.0.join(name)),
                "{name}"
            );
        }
    };
    for (i, (kind, at, patch)) in mutations.iter().enumerate() {
        let name = format!("mutation-{i}.mhrv");
        rewrite(&file, &dir.0.join(&name), |k, bytes| {
            if k == *kind {
                bytes[*at..*at + patch.len()].copy_from_slice(patch);
            }
            true
        });
        assert_bad(&name);
    }
    for entry in file.entries() {
        let name = format!("missing-{}.mhrv", entry.kind.id());
        rewrite(&file, &dir.0.join(&name), |k, _| k != entry.kind);
        assert_bad(&name);
        let name = format!("short-{}.mhrv", entry.kind.id());
        rewrite(&file, &dir.0.join(&name), |k, bytes| {
            if k == entry.kind {
                bytes.pop();
            }
            true
        });
        assert_bad(&name);
    }
    for (dim, metric, codes) in [
        (18, Metric::Dot, CodeKind::Int8),
        (17, Metric::L2, CodeKind::Int8),
        (17, Metric::Dot, CodeKind::F32),
    ] {
        let mismatch = IndexConfig::new(dim, metric).with_codes(codes);
        assert!(matches!(
            Segment::open(&dir.0, &info.file, &mismatch, VerifyMode::Headers),
            Err(Error::Corrupt { .. })
        ));
    }
}

#[test]
fn graphs_round_trip_and_ignore_input_order_and_threads() {
    for kind in [CodeKind::F32, CodeKind::Int8, CodeKind::Binary] {
        let dir = Dir::new();
        let other_dir = Dir::new();
        let mut config = IndexConfig::new(17, Metric::Dot).with_codes(kind);
        config.ann.kind = AnnKind::Vamana(VamanaParams {
            max_degree: 4,
            build_beam: 8,
            ..VamanaParams::default()
        });
        let a = records(17);
        let mut b = a.clone();
        b.keys.reverse();
        b.paths.reverse();
        b.languages.reverse();
        b.vectors = a
            .vectors
            .as_chunks::<17>()
            .0
            .iter()
            .rev()
            .flatten()
            .copied()
            .collect();
        let mut one = build(&config);
        one.graph = true;
        let mut seven = one;
        seven.threads = 7;
        let info = write_segment(&dir.0, &a, &one).unwrap();
        let other = write_segment(&other_dir.0, &b, &seven).unwrap();
        assert!(info.has_graph);
        assert_eq!(
            std::fs::read(dir.0.join(&info.file)).unwrap(),
            std::fs::read(other_dir.0.join(other.file)).unwrap()
        );
        let seg = Segment::open(&dir.0, &info.file, &config, VerifyMode::Full).unwrap();
        let graph = seg.graph().unwrap();
        assert_eq!(graph.len(), a.len());
        let f =
            SectionFile::open(&dir.0.join(&info.file), SEGMENT_MAGIC, VerifyMode::Full).unwrap();
        assert_eq!(
            f.read_section(SectionKind::Graph).unwrap(),
            graph.to_bytes()
        );
        rewrite(&f, &dir.0.join("bad-graph"), |k, bytes| {
            if k == SectionKind::Graph {
                bytes[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
            }
            true
        });
        assert!(
            matches!(Segment::open(&dir.0, "bad-graph", &config, VerifyMode::Headers), Err(Error::Corrupt { path, .. }) if path == dir.0.join("bad-graph"))
        );
    }
}

#[test]
fn training_is_sampled_from_sorted_ordinals() {
    use crate::quant::{MAX_TRAINING_VECTORS, training_sample};
    use crate::util::mix64;
    let n = MAX_TRAINING_VECTORS + 17;
    let source = MemoryRecords {
        dim: 1,
        keys: (0..n).rev().map(|i| format!("k{i:06}")).collect(),
        paths: vec![String::new(); n],
        languages: vec![String::new(); n],
        vectors: (0..n).rev().map(|i| ((i * 17) % 1291) as f32).collect(),
    };
    for kind in [CodeKind::Int8, CodeKind::Binary] {
        let dir = Dir::new();
        let mut config = IndexConfig::new(1, Metric::Dot).with_codes(kind);
        config.seed = u64::MAX;
        let spec = build(&config);
        let info = write_segment(&dir.0, &source, &spec).unwrap();
        let seg = Segment::open(&dir.0, &info.file, &config, VerifyMode::Full).unwrap();
        let seed = mix64(config.seed ^ mix64(spec.id));
        let sample: Vec<_> = training_sample(n, seed)
            .iter()
            .map(|&i| ((i * 17) % 1291) as f32)
            .collect();
        let codec = AnyCodec::train(kind, Metric::Dot, 1, &sample, seed).unwrap();
        assert_eq!(seg.codec(), &codec);
    }
}

#[test]
fn invalid_inputs_fail_without_publishing_a_segment() {
    let dir = Dir::new();
    let config = IndexConfig::new(17, Metric::Dot);
    // The workspace allowance is checked before reading vectors or creating files.
    let mut source = records(17);
    let mut spec = build(&config);
    spec.memory_budget = 1;
    assert!(matches!(
        write_segment(&dir.0, &source, &spec),
        Err(Error::Budget { .. })
    ));
    assert!(std::fs::read_dir(&dir.0).unwrap().next().is_none());
    source.keys[0] = source.keys[1].clone();
    assert!(matches!(
        write_segment(&dir.0, &source, &build(&config)),
        Err(Error::InvalidArgument(_))
    ));
    for key in [String::new(), "k".repeat(crate::MAX_KEY_BYTES + 1)] {
        source.keys[0] = key;
        assert!(matches!(
            write_segment(&dir.0, &source, &build(&config)),
            Err(Error::InvalidKey(_))
        ));
    }
    let mut source = records(17);
    source.paths[0] = "p".repeat(crate::MAX_PATH_BYTES + 1);
    assert!(matches!(
        write_segment(&dir.0, &source, &build(&config)),
        Err(Error::InvalidRecord(_))
    ));
    source.paths[0].clear();
    source.languages[0] = "l".repeat(crate::MAX_LANGUAGE_BYTES + 1);
    assert!(matches!(
        write_segment(&dir.0, &source, &build(&config)),
        Err(Error::InvalidRecord(_))
    ));
    let mut source = records(17);
    source.vectors[0] = f32::NAN;
    assert!(matches!(
        write_segment(&dir.0, &source, &build(&config)),
        Err(Error::InvalidVector(_))
    ));
    assert!(!dir.0.join("seg-00000000000000000009.mhrv").exists());
    assert!(source.read_vector(usize::MAX, &mut [0.0; 17]).is_err());
    assert!(source.read_vector(0, &mut []).is_err());
    source.vectors.clear();
    assert!(source.read_vector(0, &mut [0.0; 17]).is_err());
}
