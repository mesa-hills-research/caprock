use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "mhrv-embedding-{}-{}",
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

fn hash(i: u8) -> ContentHash {
    [i; 32]
}

#[test]
fn model_bounds_and_bytewise_fingerprint_safety() {
    let mut model = EmbeddingModel::voyage_code_4(1);
    for dim in [1, crate::MAX_DIM] {
        model.dimensions = dim;
        model.validate().unwrap();
    }
    for dim in [0, crate::MAX_DIM + 1, u32::MAX] {
        model.dimensions = dim;
        assert!(matches!(model.validate(), Err(Error::InvalidConfig(_))));
    }
    model.dimensions = 1;
    for field in 0..4 {
        for size in [0, 128, 129] {
            let mut changed = model.clone();
            let value = "a".repeat(size);
            match field {
                0 => changed.provider = value,
                1 => changed.model = value,
                2 => changed.output_dtype = value,
                _ => changed.text_format = value,
            }
            assert_eq!(changed.validate().is_ok(), size == 128);
        }
    }
    model.provider = "../é".into();
    model.model = "\\x\0?".into();
    model.document_input = None;
    model.revision = Some("r/2".into());
    assert_eq!(
        model.document_fingerprint(),
        "..___-_x__-1-float-none-arko_index_chunk-v1-r_2"
    );
    assert_eq!(
        Path::new(&model.document_fingerprint())
            .components()
            .count(),
        1
    );
}

#[test]
fn query_changes_reuse_documents_and_switching_back_reuses_the_old_cache() {
    let root = TestDir::new();
    let model = EmbeddingModel::voyage_code_4(2);
    let mut cache = EmbeddingCache::open(&root.0, &model).unwrap();
    cache.put(&[(hash(1), &[1.0, -0.0])]).unwrap();
    cache.flush().unwrap();
    let mut query_model = model.clone();
    query_model.query_input = None;
    query_model.normalized = false;
    assert!(matches!(
        EmbeddingCache::open(&root.0, &query_model),
        Err(Error::Locked { .. })
    ));
    let mut other = model.clone();
    other.revision = Some("new".into());
    let mut second = EmbeddingCache::open(&root.0, &other).unwrap();
    assert!(second.is_empty());
    second.put(&[(hash(1), &[2.0, 3.0])]).unwrap();
    second.flush().unwrap();
    drop(cache);
    let cache = EmbeddingCache::open(&root.0, &query_model).unwrap();
    assert_eq!(cache.model(), &query_model);
    assert_eq!(cache.get(&hash(1)).unwrap(), Some(vec![1.0, -0.0]));
    assert_eq!(second.get(&hash(1)).unwrap(), Some(vec![2.0, 3.0]));
}

#[test]
fn batch_refusal_and_float_bits_and_duplicate_hashes() {
    let root = TestDir::new();
    let model = EmbeddingModel::voyage_code_4(4);
    let mut cache = EmbeddingCache::open(&root.0, &model).unwrap();
    let vector = [0.0, -0.0, f32::from_bits(1), f32::MAX];
    cache
        .put(&[(hash(1), &[1.0; 4]), (hash(1), &vector)])
        .unwrap();
    assert_eq!(cache.len(), 1);
    assert_eq!(
        cache.stats().log_bytes,
        HEADER_LEN + 2 * cache.record_len() as u64
    );
    let stats = cache.stats();
    let bytes = fs::read(&cache.log_path).unwrap();
    for invalid in [
        vec![],
        vec![1.0; 5],
        vec![f32::NAN; 4],
        vec![f32::INFINITY; 4],
        vec![f32::NEG_INFINITY; 4],
    ] {
        assert!(matches!(
            cache.put(&[(hash(1), &[2.0; 4]), (hash(2), &invalid)]),
            Err(Error::InvalidVector(_))
        ));
        assert_eq!(cache.stats(), stats);
        assert_eq!(fs::read(&cache.log_path).unwrap(), bytes);
    }
    assert_eq!(
        cache.missing(&[hash(2), hash(1), hash(2), hash(3)]),
        vec![0, 2, 3]
    );
    cache.put(&[]).unwrap();
    cache.flush().unwrap();
    drop(cache);
    let cache = EmbeddingCache::open(&root.0, &model).unwrap();
    assert_eq!(
        cache
            .get(&hash(1))
            .unwrap()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        vector.map(f32::to_bits)
    );
}

#[test]
fn every_partial_record_tail_is_truncated_before_the_next_append() {
    let root = TestDir::new();
    let model = EmbeddingModel::voyage_code_4(1);
    let mut cache = EmbeddingCache::open(&root.0, &model).unwrap();
    cache.put(&[(hash(1), &[1.0]), (hash(2), &[2.0])]).unwrap();
    cache.flush().unwrap();
    let path = cache.log_path.clone();
    let record_len = cache.record_len();
    let good = fs::read(&path).unwrap();
    drop(cache);
    for tail in 1..record_len {
        fs::write(&path, &good[..HEADER_LEN as usize + record_len + tail]).unwrap();
        let mut cache = EmbeddingCache::open(&root.0, &model).unwrap();
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.stats().truncated_records, 1);
        assert_eq!(
            fs::metadata(&path).unwrap().len(),
            HEADER_LEN + record_len as u64
        );
        assert_eq!(cache.get(&hash(1)).unwrap(), Some(vec![1.0]));
        cache.put(&[(hash(3), &[3.0])]).unwrap();
        cache.flush().unwrap();
        drop(cache);
        let reopened = EmbeddingCache::open(&root.0, &model).unwrap();
        assert_eq!(reopened.stats().truncated_records, 0);
        assert_eq!(reopened.len(), 2);
        assert_eq!(reopened.get(&hash(3)).unwrap(), Some(vec![3.0]));
    }
}

#[test]
fn corrupt_record_stops_replay_and_reads_verify_checksums() {
    let root = TestDir::new();
    let model = EmbeddingModel::voyage_code_4(1);
    let mut cache = EmbeddingCache::open(&root.0, &model).unwrap();
    cache
        .put(&[(hash(1), &[1.0]), (hash(2), &[2.0]), (hash(1), &[3.0])])
        .unwrap();
    cache.flush().unwrap();
    let record_len = cache.record_len();
    let path = cache.log_path.clone();
    let mut bad = fs::read(&path).unwrap();
    bad[HEADER_LEN as usize + record_len + 32] ^= 1;
    fs::write(&path, &bad).unwrap();
    assert!(matches!(cache.get(&hash(2)), Err(Error::Corrupt { .. })));
    // A failed compaction must never publish a partial new log or lose the old map.
    assert!(matches!(
        cache.retain(&|_| true),
        Err(Error::Corrupt { .. })
    ));
    assert_eq!(cache.len(), 2);
    assert_eq!(log_files(&cache.dir).unwrap().len(), 1);
    drop(cache);
    let cache = EmbeddingCache::open(&root.0, &model).unwrap();
    assert_eq!(cache.stats().truncated_records, 2);
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.get(&hash(1)).unwrap(), Some(vec![1.0]));
    assert_eq!(
        fs::metadata(&path).unwrap().len(),
        HEADER_LEN + record_len as u64
    );
}

#[test]
fn invalid_headers_and_model_identity_refuse_without_rewriting_and_release_lock() {
    let root = TestDir::new();
    let model = EmbeddingModel::voyage_code_4(1);
    let cache = EmbeddingCache::open(&root.0, &model).unwrap();
    let path = cache.log_path.clone();
    let model_path = cache.dir.join("model.json");
    drop(cache);
    for bytes in [
        vec![],
        b"MHRVEMB1".to_vec(),
        header(2).to_vec(),
        vec![0; 12],
    ] {
        fs::write(&path, &bytes).unwrap();
        assert!(matches!(
            EmbeddingCache::open(&root.0, &model),
            Err(Error::Corrupt { .. })
        ));
        assert_eq!(fs::read(&path).unwrap(), bytes);
    }
    fs::write(&path, header(1)).unwrap();
    fs::write(&model_path, b"{incomplete").unwrap();
    assert!(matches!(
        EmbeddingCache::open(&root.0, &model),
        Err(Error::Corrupt { .. })
    ));
    fs::remove_file(&model_path).unwrap();
    assert!(matches!(
        EmbeddingCache::open(&root.0, &model),
        Err(Error::Corrupt { .. })
    ));
    fs::write(&model_path, serde_json::to_vec(&model).unwrap()).unwrap();
    assert!(EmbeddingCache::open(&root.0, &model).unwrap().is_empty());
}

#[test]
fn compaction_is_deterministic_and_selects_only_published_logs() {
    let root = TestDir::new();
    let model = EmbeddingModel::voyage_code_4(1);
    let mut cache = EmbeddingCache::open(&root.0, &model).unwrap();
    cache
        .put(&[(hash(2), &[2.0]), (hash(1), &[1.0]), (hash(2), &[3.0])])
        .unwrap();
    cache.flush().unwrap();
    let old_path = cache.log_path.clone();
    let old_bytes = fs::read(&old_path).unwrap();
    assert_eq!(cache.retain(&|_| true).unwrap(), 0);
    let compacted = fs::read(&cache.log_path).unwrap();
    assert_eq!(
        cache.stats().log_bytes,
        HEADER_LEN + 2 * cache.record_len() as u64
    );
    assert_eq!(&compacted[12..44], &hash(1));
    assert_eq!(cache.retain(&|_| true).unwrap(), 0);
    assert_eq!(fs::read(&cache.log_path).unwrap(), compacted);
    assert_eq!(cache.retain(&|h| *h == hash(2)).unwrap(), 1);
    let dir = cache.dir.clone();
    drop(cache);
    // Simulate a crash after publication, before removing the old log, and an
    // interrupted next compaction. Neither an old log nor a temp is authoritative.
    fs::write(&old_path, old_bytes).unwrap();
    fs::write(dir.join(format!("{}.tmp", log_name(5))), b"partial").unwrap();
    let foreign = dir.join("cache-backup.log");
    fs::write(&foreign, b"not ours").unwrap();
    let mut cache = EmbeddingCache::open(&root.0, &model).unwrap();
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.get(&hash(2)).unwrap(), Some(vec![3.0]));
    assert_eq!(cache.get(&hash(1)).unwrap(), None);
    assert!(!old_path.exists());
    assert_eq!(cache.retain(&|_| false).unwrap(), 1);
    assert_eq!(cache.stats().log_bytes, HEADER_LEN);
    cache.put(&[(hash(7), &[7.0])]).unwrap();
    cache.flush().unwrap();
    assert_eq!(fs::read(foreign).unwrap(), b"not ours");
    assert_eq!(log_files(&dir).unwrap().len(), 1);
    drop(cache);
    let cache = EmbeddingCache::open(&root.0, &model).unwrap();
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.get(&hash(7)).unwrap(), Some(vec![7.0]));
}
