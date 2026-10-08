//! The writer and snapshot accessors, through the public API.

mod common;

use std::collections::BTreeMap;

use caprock::{
    EmbeddingModel, Error, IndexConfig, Metric, Op, Reader, Record, StoreOptions, Writer,
};
use caprock_oracle::canonical;
use caprock_oracle::generate::{Shape, VectorSpec, key, path_for};
use caprock_oracle::rng::Rng;
use common::TempDir;

fn options(delta: u32, threads: usize) -> StoreOptions {
    StoreOptions {
        delta_max_vectors: delta,
        threads,
        build_memory_budget: 256 << 20,
        ..StoreOptions::default()
    }
}

fn config(dim: u32) -> IndexConfig {
    let mut c = IndexConfig::new(dim, Metric::Cosine);
    c.ann.graph_min_vectors = 500;
    c
}

/// The records as raw vectors with their paths, by key.
struct Model {
    spec: VectorSpec,
    live: BTreeMap<String, (Vec<f32>, String, String)>,
    draws: u64,
}

impl Model {
    fn new(dim: usize, seed: u64) -> Model {
        Model {
            spec: VectorSpec {
                dim,
                shape: Shape::Clustered {
                    clusters: 10,
                    spread: 0.4,
                },
                seed,
            },
            live: BTreeMap::new(),
            draws: 0,
        }
    }

    /// Upserts of keys `keys` (fresh vectors) and deletes of `deletes`, applied to `w`.
    fn apply(&mut self, w: &mut Writer, keys: impl Iterator<Item = u64>, deletes: &[u64]) {
        let mut owned = Vec::new();
        for j in keys {
            let v = self.spec.vector(self.draws);
            self.draws += 1;
            let (p, l) = path_for(j, 1);
            owned.push((key(j), v, p, l));
        }
        let del: Vec<String> = deletes.iter().map(|&j| key(j)).collect();
        let mut ops: Vec<Op> = owned
            .iter()
            .map(|(k, v, p, l)| {
                Op::Upsert(Record {
                    key: k,
                    vector: v,
                    path: p,
                    language: l,
                })
            })
            .collect();
        ops.extend(del.iter().map(|k| Op::Delete(k)));
        w.apply(&ops).unwrap();
        for (k, v, p, l) in owned {
            self.live.insert(k, (v, p, l));
        }
        for k in del {
            self.live.remove(&k);
        }
    }

    fn check(&self, w: &Writer) {
        let s = w.snapshot();
        assert_eq!(s.len(), self.live.len() as u64);
        assert_eq!(s.keys(), self.live.keys().cloned().collect::<Vec<_>>());
        for (k, (v, p, l)) in &self.live {
            let r = s.get(k).unwrap().unwrap_or_else(|| panic!("{k} missing"));
            let want = canonical::prepare(canonical::Metric::Cosine, v, v.len()).unwrap();
            assert_eq!(r.vector, want, "{k}: the stored form is the canonical one");
            assert_eq!((&r.path, &r.language), (p, l));
            assert!(s.contains(k));
        }
    }
}

#[test]
fn create_commit_reopen() {
    let dir = TempDir::new("w-basic");
    let mut w = Writer::create(dir.path(), config(16), options(300, 2)).unwrap();
    assert_eq!(w.config(), &config(16));
    assert_eq!(w.snapshot().len(), 0);
    assert_eq!(w.stats().generation, 1);
    let mut m = Model::new(16, 1);
    m.apply(&mut w, 0..1000, &[]);
    m.apply(&mut w, 500..700, &[1, 2, 3, 999, 5000]);
    m.check(&w);
    assert_eq!(w.snapshot().version(), 1000 + 200 + 5);
    let info = w.commit(Some(b"seq-42")).unwrap();
    assert_eq!(info.generation, 2);
    assert_eq!(info.live, m.live.len() as u64);
    assert_eq!(info.version, 1205);
    drop(w);
    let w = Writer::open(dir.path(), options(300, 2)).unwrap();
    m.check(&w);
    assert_eq!(w.checkpoint().as_deref(), Some(&b"seq-42"[..]));
    assert_eq!(w.snapshot().generation(), 2);
    assert_eq!(w.snapshot().version(), 1205);
}

/// An index written in format version 1, before the store's on-disk names changed, is
/// refused by writers and readers alike.
#[test]
fn indexes_of_another_format_are_refused() {
    let dir = TempDir::new("w-format");
    let mut w = Writer::create(dir.path(), config(16), options(300, 1)).unwrap();
    Model::new(16, 2).apply(&mut w, 0..10, &[]);
    w.commit(None).unwrap();
    drop(w);
    let current = std::fs::read_to_string(dir.path().join("CURRENT")).unwrap();
    let manifest = dir.path().join(current.trim_end());
    let mut text: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest).unwrap()).unwrap();
    text["format_version"] = 1.into();
    std::fs::write(&manifest, serde_json::to_vec(&text).unwrap()).unwrap();
    for result in [
        Writer::open(dir.path(), options(300, 1)).map(drop),
        Reader::open(dir.path(), options(300, 1)).map(drop),
    ] {
        match result {
            Err(Error::UnsupportedFormat {
                found: 1,
                supported,
            }) => assert!(supported > 1),
            other => panic!("expected UnsupportedFormat, got {other:?}"),
        }
    }
}

#[test]
fn uncommitted_work_is_lost_on_reopen() {
    let dir = TempDir::new("w-lost");
    let mut w = Writer::create(dir.path(), config(8), options(100, 1)).unwrap();
    let mut m = Model::new(8, 2);
    m.apply(&mut w, 0..250, &[]);
    w.commit(None).unwrap();
    let committed = m.live.clone();
    m.apply(&mut w, 250..900, &[0, 1, 2]); // seals uncommitted segments too
    drop(w);
    let w = Writer::open(dir.path(), options(100, 1)).unwrap();
    m.live = committed;
    m.check(&w);
    assert_eq!(w.checkpoint(), None);
}

#[test]
fn refusals() {
    let dir = TempDir::new("w-refuse");
    let mut w = Writer::create(dir.path(), config(4), options(100, 1)).unwrap();
    assert!(matches!(
        Writer::create(dir.path(), config(4), options(100, 1)),
        Err(Error::AlreadyExists(_))
    ));
    assert!(matches!(
        Writer::open(dir.path(), options(100, 1)),
        Err(Error::Locked { .. })
    ));
    let missing = TempDir::new("w-missing");
    assert!(matches!(
        Writer::open(&missing.path().join("nope"), options(100, 1)),
        Err(Error::NotFound(_))
    ));
    let mut bad = config(4);
    bad.dim = 0;
    assert!(matches!(
        Writer::create(&missing.path().join("x"), bad, options(100, 1)),
        Err(Error::InvalidConfig(_))
    ));

    let good = [1.0f32, 0.0, 0.0, 0.0];
    let rec = |key, vector, path, language| {
        Op::Upsert(Record {
            key,
            vector,
            path,
            language,
        })
    };
    let long_key = "k".repeat(256);
    let long_path = "p".repeat(4097);
    let long_lang = "l".repeat(65);
    let cases: Vec<(Vec<Op>, &str)> = vec![
        (vec![rec("", &good, "a", "x")], "key"),
        (vec![rec(&long_key, &good, "a", "x")], "key"),
        (vec![rec("k", &good, &long_path, "x")], "record"),
        (vec![rec("k", &good, "a", &long_lang)], "record"),
        (vec![rec("k", &[1.0, 0.0], "a", "x")], "vector"),
        (
            vec![rec("k", &[f32::NAN, 0.0, 0.0, 0.0], "a", "x")],
            "vector",
        ),
        (vec![rec("k", &[0.0; 4], "a", "x")], "vector"),
        (
            vec![
                rec("ok", &good, "a", "x"),
                rec("k", &[2e15, 0.0, 0.0, 0.0], "a", "x"),
            ],
            "vector",
        ),
    ];
    for (ops, what) in cases {
        let err = w.apply(&ops).unwrap_err();
        let ok = match what {
            "key" => matches!(err, Error::InvalidKey(_)),
            "record" => matches!(err, Error::InvalidRecord(_)),
            _ => matches!(err, Error::InvalidVector(_)),
        };
        assert!(ok, "{what}: {err:?}");
        assert_eq!(w.snapshot().len(), 0, "nothing applied");
        assert_eq!(w.snapshot().version(), 0);
    }
    assert!(matches!(
        w.commit(Some(&[0u8; 4097])),
        Err(Error::InvalidArgument(_))
    ));
    assert!(!w.delete("absent").unwrap());
    w.upsert(Record {
        key: "k",
        vector: &good,
        path: "a",
        language: "x",
    })
    .unwrap();
    assert!(w.delete("k").unwrap());
}

#[test]
fn models_are_checked_on_open() {
    let dir = TempDir::new("w-model");
    let model = EmbeddingModel::voyage_code_4(256, "my-app/chunk-v1");
    let w = Writer::create(
        dir.path(),
        IndexConfig::for_model(model.clone()),
        options(100, 1),
    )
    .unwrap();
    drop(w);
    let mut query_only = model.clone();
    query_only.query_input = Some("search_query".into());
    drop(Writer::open_for_model(dir.path(), &query_only, options(100, 1)).unwrap());
    let other = EmbeddingModel::voyage_code_4(1024, "my-app/chunk-v1");
    match Writer::open_for_model(dir.path(), &other, options(100, 1)) {
        Err(Error::ModelMismatch { stored, requested }) => {
            assert_eq!(*stored, model);
            assert_eq!(*requested, other);
        }
        r => panic!("expected ModelMismatch, got {r:?}"),
    }
    let plain = TempDir::new("w-model-none");
    drop(Writer::create(plain.path(), config(256), options(100, 1)).unwrap());
    assert!(matches!(
        Writer::open_for_model(plain.path(), &model, options(100, 1)),
        Err(Error::ModelMismatch { .. })
    ));
}

#[test]
fn snapshots_are_isolated() {
    let dir = TempDir::new("w-iso");
    let mut w = Writer::create(dir.path(), config(8), options(50, 1)).unwrap();
    let mut m = Model::new(8, 3);
    m.apply(&mut w, 0..120, &[]);
    let before = w.snapshot();
    let keys_before = before.keys();
    m.apply(&mut w, 100..300, &(0..60).collect::<Vec<_>>());
    w.compact().unwrap();
    assert_eq!(before.keys(), keys_before, "an old snapshot never changes");
    assert_eq!(before.len(), 120);
    assert!(before.contains(&key(0)));
    m.check(&w);
}

#[test]
fn sealing_maintenance_and_compaction() {
    let dir = TempDir::new("w-maint");
    let mut w = Writer::create(dir.path(), config(16), options(200, 2)).unwrap();
    let mut m = Model::new(16, 4);
    let mut rng = Rng::new(5);
    for round in 0..10u64 {
        let deletes: Vec<u64> = (0..150).map(|_| rng.below(round * 500 + 500)).collect();
        m.apply(&mut w, round * 500..round * 500 + 500, &deletes);
    }
    assert!(
        w.stats().segments.len() > 5,
        "the delta was sealed into segments"
    );
    m.check(&w);
    let report = w.maintain().unwrap();
    assert!(report.merges > 0);
    m.check(&w);
    w.compact().unwrap();
    m.check(&w);
    let stats = w.stats();
    for s in &stats.segments {
        assert_eq!(s.vectors, s.live, "compaction purges deletions");
        if s.live >= 500 {
            assert!(s.has_graph, "large segments get graphs");
        }
    }
    assert_eq!(stats.delta, 0);
    assert_eq!(stats.live, m.live.len() as u64);
    w.commit(None).unwrap();
    drop(w);
    let w = Writer::open(dir.path(), options(200, 2)).unwrap();
    m.check(&w);
}

#[test]
fn jobs_run_elsewhere_and_keep_later_changes() {
    let dir = TempDir::new("w-jobs");
    let mut w = Writer::create(dir.path(), config(8), options(100, 1)).unwrap();
    let mut m = Model::new(8, 6);
    for c in 0..20 {
        m.apply(&mut w, c * 100..(c + 1) * 100, &[]);
    }
    let job = w.plan_compaction().expect("work to do");
    assert!(!job.describe().is_empty());
    let handle = std::thread::spawn(move || job.run());
    // Meanwhile: delete and replace records the job is rewriting, add new ones.
    m.apply(
        &mut w,
        (0..100).chain(5000..5050),
        &(100..300).collect::<Vec<_>>(),
    );
    let built = handle.join().unwrap().unwrap();
    w.install(built).unwrap();
    m.check(&w);
    w.compact().unwrap();
    m.check(&w);

    let other = TempDir::new("w-jobs-other");
    let mut w2 = Writer::create(other.path(), config(8), options(100, 1)).unwrap();
    let mut m2 = Model::new(8, 7);
    m2.apply(&mut w2, 0..500, &[]);
    let job = w2.plan_compaction().unwrap();
    let built = job.run().unwrap();
    assert!(matches!(w.install(built), Err(Error::StaleJob(_))));
}

#[test]
fn garbage_is_removed_on_open() {
    let dir = TempDir::new("w-garbage");
    let mut w = Writer::create(dir.path(), config(8), options(100, 1)).unwrap();
    let mut m = Model::new(8, 8);
    m.apply(&mut w, 0..300, &[]);
    w.commit(None).unwrap();
    drop(w);
    let stray = [
        "seg-00000000000000000999.caprock",
        "seg-00000000000000000998.caprock.tmp",
    ];
    for f in stray {
        std::fs::write(dir.path().join(f), b"junk").unwrap();
    }
    std::fs::write(dir.path().join("notes.txt"), b"mine").unwrap();
    let w = Writer::open(dir.path(), options(100, 1)).unwrap();
    for f in stray {
        assert!(!dir.path().join(f).exists(), "{f} removed");
    }
    assert!(
        dir.path().join("notes.txt").exists(),
        "unknown files are left alone"
    );
    m.check(&w);
}

#[test]
fn layouts_are_deterministic() {
    let mut images = Vec::new();
    for threads in [1, 4] {
        let dir = TempDir::new("w-det");
        let mut w = Writer::create(dir.path(), config(16), options(150, threads)).unwrap();
        let mut m = Model::new(16, 9);
        let mut rng = Rng::new(10);
        for round in 0..6u64 {
            let deletes: Vec<u64> = (0..80).map(|_| rng.below(round * 400 + 400)).collect();
            m.apply(&mut w, round * 400..round * 400 + 400, &deletes);
            if round % 2 == 1 {
                w.maintain().unwrap();
                w.commit(Some(&[round as u8])).unwrap();
            }
        }
        w.compact().unwrap();
        w.commit(None).unwrap();
        drop(w);
        let mut files: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap())
            .filter(|e| e.file_name() != "LOCK")
            .map(|e| {
                (
                    e.file_name().into_string().unwrap(),
                    std::fs::read(e.path()).unwrap(),
                )
            })
            .collect();
        files.sort();
        images.push(files);
    }
    let names = |i: &Vec<(String, Vec<u8>)>| i.iter().map(|f| f.0.clone()).collect::<Vec<_>>();
    assert_eq!(names(&images[0]), names(&images[1]));
    for (a, b) in images[0].iter().zip(&images[1]) {
        assert!(a.1 == b.1, "{} differs between thread counts", a.0);
    }
}
