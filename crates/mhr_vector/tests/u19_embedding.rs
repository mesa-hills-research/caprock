//! Acceptance tests of unit U19 (embedding models and the embedding cache).

mod common;

use std::io::Write;

use common::TempDir;
use mhr_vector::{ContentHash, EmbeddingCache, EmbeddingModel, Error, InputType};

fn hash(i: u32) -> ContentHash {
    let mut h = [0u8; 32];
    h[..4].copy_from_slice(&i.to_le_bytes());
    h[31] = 0xAB;
    h
}

fn vector(i: u32, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|j| (i as f32 + 1.0) * 0.01 + j as f32 * 0.001)
        .collect()
}

#[test]
fn models() {
    let m = EmbeddingModel::voyage_code_4(1024);
    assert_eq!(m.provider, "voyage");
    assert_eq!(m.model, "voyage-code-4");
    assert_eq!(m.dimensions, 1024);
    assert_eq!(m.input_type(InputType::Document), Some("document"));
    assert_eq!(m.input_type(InputType::Query), Some("query"));
    m.validate().unwrap();
    assert_eq!(
        m.document_fingerprint(),
        "voyage-voyage-code-4-1024-float-document-arko_index_chunk-v1"
    );
    assert!(!m.requires_reembed(&m.clone()));
    let mut q = m.clone();
    q.query_input = None;
    q.normalized = false;
    assert!(
        !m.requires_reembed(&q),
        "query side and normalization do not change documents"
    );
    assert_eq!(q.document_fingerprint(), m.document_fingerprint());
    type Change = Box<dyn Fn(&mut EmbeddingModel)>;
    let changes: Vec<Change> = vec![
        Box::new(|x| x.provider = "local".into()),
        Box::new(|x| x.model = "voyage-code-3".into()),
        Box::new(|x| x.dimensions = 512),
        Box::new(|x| x.document_input = None),
        Box::new(|x| x.output_dtype = "int8".into()),
        Box::new(|x| x.text_format = "arko_index/chunk-v2".into()),
        Box::new(|x| x.revision = Some("2026-10".into())),
    ];
    for change in changes {
        let mut other = m.clone();
        change(&mut other);
        assert!(
            m.requires_reembed(&other) && other.requires_reembed(&m),
            "{other:?}"
        );
        assert_ne!(
            other.document_fingerprint(),
            m.document_fingerprint(),
            "{other:?}"
        );
        let fp = other.document_fingerprint();
        assert!(
            fp.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b)),
            "{fp}"
        );
    }
    let mut bad = m.clone();
    bad.dimensions = 0;
    assert!(matches!(bad.validate(), Err(Error::InvalidConfig(_))));
    let mut bad = m.clone();
    bad.model = String::new();
    assert!(bad.validate().is_err());
    let json = serde_json::to_string(&m).unwrap();
    assert_eq!(serde_json::from_str::<EmbeddingModel>(&json).unwrap(), m);
}

#[test]
fn cache_round_trips_across_reopen() {
    let root = TempDir::new("cache");
    let model = EmbeddingModel::voyage_code_4(256);
    let dim = 256;
    {
        let mut c = EmbeddingCache::open(root.path(), &model).unwrap();
        assert!(c.is_empty());
        assert_eq!(c.model(), &model);
        let vs: Vec<Vec<f32>> = (0..100).map(|i| vector(i, dim)).collect();
        let items: Vec<(ContentHash, &[f32])> =
            (0..100).map(|i| (hash(i), &vs[i as usize][..])).collect();
        c.put(&items).unwrap();
        assert_eq!(c.len(), 100);
        assert!(c.contains(&hash(5)));
        assert_eq!(c.get(&hash(5)).unwrap().unwrap(), vector(5, dim));
        assert_eq!(c.get(&hash(500)).unwrap(), None);
        let asked = [hash(1), hash(200), hash(2), hash(300)];
        assert_eq!(c.missing(&asked), vec![1, 3]);
        let newer = vector(999, dim);
        c.put(&[(hash(7), &newer[..])]).unwrap();
        assert_eq!(
            c.get(&hash(7)).unwrap().unwrap(),
            newer,
            "a later record wins"
        );
        assert!(matches!(
            EmbeddingCache::open(root.path(), &model),
            Err(Error::Locked { .. })
        ));
        assert!(matches!(
            c.put(&[(hash(9000), &[1.0, 2.0][..])]),
            Err(Error::InvalidVector(_))
        ));
        let nan = vec![f32::NAN; dim];
        assert!(
            c.put(&[(hash(9001), &vector(9, dim)[..]), (hash(9002), &nan[..])])
                .is_err()
        );
        assert!(
            !c.contains(&hash(9000)) && !c.contains(&hash(9001)) && !c.contains(&hash(9002)),
            "a refused batch adds nothing"
        );
        c.flush().unwrap();
        assert_eq!(c.stats().entries, 100);
    }
    let dir = root.path().join(model.document_fingerprint());
    assert!(dir.join("model.json").exists());
    let c = EmbeddingCache::open(root.path(), &model).unwrap();
    assert_eq!(c.len(), 100);
    assert_eq!(c.get(&hash(7)).unwrap().unwrap(), vector(999, dim));
    assert_eq!(c.get(&hash(99)).unwrap().unwrap(), vector(99, dim));
}

#[test]
fn torn_tails_are_truncated_and_retain_compacts() {
    let root = TempDir::new("cache-torn");
    let model = EmbeddingModel::voyage_code_4(512);
    let dim = 512;
    {
        let mut c = EmbeddingCache::open(root.path(), &model).unwrap();
        for i in 0..50 {
            c.put(&[(hash(i), &vector(i, dim)[..])]).unwrap();
        }
        c.flush().unwrap();
    }
    let dir = root.path().join(model.document_fingerprint());
    let log = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|x| x == "log"))
        .expect("a log file");
    let mut f = std::fs::OpenOptions::new().append(true).open(&log).unwrap();
    f.write_all(&[0x42; 100]).unwrap(); // half a record
    drop(f);
    {
        let mut c = EmbeddingCache::open(root.path(), &model).unwrap();
        assert_eq!(c.len(), 50);
        assert!(c.stats().truncated_records >= 1);
        c.put(&[(hash(60), &vector(60, dim)[..])]).unwrap();
        c.flush().unwrap();
        let dropped = c.retain(&|h| h[0] % 2 == 0).unwrap();
        assert_eq!(dropped, 25);
        assert_eq!(c.len(), 26);
        assert_eq!(c.get(&hash(60)).unwrap().unwrap(), vector(60, dim));
        assert_eq!(c.get(&hash(3)).unwrap(), None);
    }
    let c = EmbeddingCache::open(root.path(), &model).unwrap();
    assert_eq!(c.len(), 26);
    assert_eq!(c.get(&hash(4)).unwrap().unwrap(), vector(4, dim));
}

#[test]
fn a_cache_directory_of_another_model_is_refused() {
    let root = TempDir::new("cache-model");
    let model = EmbeddingModel::voyage_code_4(256);
    let dir = root.path().join(model.document_fingerprint());
    std::fs::create_dir_all(&dir).unwrap();
    let mut other = model.clone();
    other.model = "something-else".into();
    std::fs::write(
        dir.join("model.json"),
        serde_json::to_string(&other).unwrap(),
    )
    .unwrap();
    assert!(matches!(
        EmbeddingCache::open(root.path(), &model),
        Err(Error::ModelMismatch { .. })
    ));
}
