//! Snapshot search and readers, through the public API, against random operation
//! sequences run on the reference implementation too.

mod common;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use caprock::{
    AnnKind, CodeKind, Error, IndexConfig, Metric, Reader, SearchParams, StoreOptions, Writer,
};
use caprock_oracle::check::{Expectation, check_query};
use caprock_oracle::generate::{OpsSpec, Step, standard_filters};
use caprock_oracle::scenario::{Scenario, run};
use caprock_oracle::{Filter as OracleFilter, Oracle, canonical};
use common::TempDir;
use common::store::{StoreUnderTest, to_filter, to_ops};

fn options() -> StoreOptions {
    StoreOptions {
        delta_max_vectors: 250,
        threads: 2,
        build_memory_budget: 256 << 20,
        ..StoreOptions::default()
    }
}

fn om(m: Metric) -> canonical::Metric {
    common::oracle_metric(m)
}

fn ops(dim: usize, seed: u64) -> OpsSpec {
    let mut o = OpsSpec::mixed(dim, seed);
    o.steps = 60;
    o.key_space = 4000;
    o.max_batch = 600;
    o
}

#[test]
fn exact_configurations_equal_the_oracle() {
    for metric in [Metric::Cosine, Metric::Dot, Metric::L2] {
        let mut store = StoreUnderTest::new(
            IndexConfig::exact(24, metric),
            options(),
            SearchParams::default(),
        );
        let scenario = Scenario {
            metric: om(metric),
            ops: ops(24, 1),
            queries: 6,
            ks: vec![1, 10, 50],
            filters: true,
            expect: Expectation::exact(),
            min_mean_recall: None,
        };
        let report = run(&mut store, &scenario).unwrap_or_else(|f| panic!("{metric:?}: {f}"));
        assert!(report.checks > 0);
    }
}

#[test]
fn exact_params_on_approximate_indexes_equal_the_oracle() {
    let mut config = IndexConfig::new(24, Metric::Cosine);
    config.ann.graph_min_vectors = 300;
    let mut store = StoreUnderTest::new(config, options(), SearchParams::exact());
    let scenario = Scenario {
        metric: canonical::Metric::Cosine,
        ops: ops(24, 2),
        queries: 6,
        ks: vec![10],
        filters: true,
        expect: Expectation::exact(),
        min_mean_recall: None,
    };
    run(&mut store, &scenario).unwrap_or_else(|f| panic!("{f}"));
}

#[test]
fn graph_indexes_meet_recall_targets() {
    for (codes, min) in [
        (CodeKind::Int8, 0.95),
        (CodeKind::Binary, 0.9),
        (CodeKind::F32, 0.95),
    ] {
        for metric in [Metric::Cosine, Metric::L2] {
            let mut config = IndexConfig::new(32, metric).with_codes(codes);
            config.ann.graph_min_vectors = 300;
            let mut store = StoreUnderTest::new(config, options(), SearchParams::default());
            let scenario = Scenario {
                metric: om(metric),
                ops: ops(32, 3),
                queries: 8,
                ks: vec![1, 10, 50],
                filters: true,
                expect: Expectation::rescored(None),
                min_mean_recall: Some(min),
            };
            let report =
                run(&mut store, &scenario).unwrap_or_else(|f| panic!("{codes:?} {metric:?}: {f}"));
            assert!(report.checks > 0);
        }
    }
}

#[test]
fn arguments_are_checked() {
    let store = StoreUnderTest::new(
        IndexConfig::new(4, Metric::Cosine),
        options(),
        SearchParams::default(),
    );
    let s = store.writer.as_ref().unwrap().snapshot();
    let q = [1.0f32, 0.0, 0.0, 0.0];
    let p = SearchParams::default();
    assert!(s.search(&q, 10, None, &p).unwrap().is_empty());
    assert!(matches!(
        s.search(&q, 0, None, &p),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        s.search(&q, caprock::MAX_K + 1, None, &p),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        s.search(&[1.0, 0.0], 10, None, &p),
        Err(Error::InvalidVector(_))
    ));
    assert!(matches!(
        s.search(&[0.0; 4], 10, None, &p),
        Err(Error::InvalidVector(_))
    ));
    assert!(matches!(
        s.search(&[f32::NAN, 0.0, 0.0, 0.0], 10, None, &p),
        Err(Error::InvalidVector(_))
    ));
}

#[test]
fn readers_see_commits_and_writer_readers_see_everything() {
    let dir = TempDir::new("readers");
    let config = IndexConfig::exact(8, Metric::Dot);
    let mut w = Writer::create(dir.path(), config, options()).unwrap();
    let steps = ops(8, 4).steps();
    let batches: Vec<_> = steps
        .iter()
        .filter_map(|s| match s {
            Step::Apply(o) => Some(o.clone()),
            _ => None,
        })
        .take(6)
        .collect();
    let follower = w.reader();
    w.apply(&to_ops(&batches[0])).unwrap();
    w.commit(None).unwrap();
    let other = Reader::open(dir.path(), StoreOptions::default()).unwrap();
    let committed = other.snapshot().len();
    assert_eq!(committed, w.snapshot().len());
    w.apply(&to_ops(&batches[1])).unwrap();
    assert_eq!(
        follower.snapshot().version(),
        w.snapshot().version(),
        "a writer's reader is current"
    );
    assert!(!follower.refresh().unwrap());
    assert_eq!(
        other.snapshot().len(),
        committed,
        "another reader sees commits only"
    );
    assert!(!other.refresh().unwrap());
    w.commit(None).unwrap();
    assert!(other.refresh().unwrap());
    assert_eq!(other.snapshot().len(), w.snapshot().len());
    assert_eq!(other.snapshot().keys(), w.snapshot().keys());
    // Old generations stay readable through a held snapshot while the writer compacts.
    let held = other.snapshot();
    for b in &batches[2..] {
        w.apply(&to_ops(b)).unwrap();
        w.compact().unwrap();
        w.commit(None).unwrap();
    }
    let q = [1.0f32; 8];
    assert_eq!(
        held.search(&q, 5, None, &SearchParams::default())
            .unwrap()
            .len(),
        5.min(held.len() as usize)
    );
    assert!(other.refresh().unwrap());
    let a = other
        .snapshot()
        .search(&q, 20, None, &SearchParams::default())
        .unwrap();
    let b = w
        .snapshot()
        .search(&q, 20, None, &SearchParams::default())
        .unwrap();
    assert_eq!(a, b);
}

/// Searches on many threads while the writer applies, compacts and commits: every
/// result is the oracle's at that snapshot's version.
#[test]
fn concurrent_readers_see_consistent_snapshots() {
    let dim = 16;
    let config = IndexConfig::exact(dim as u32, Metric::Cosine);
    let dir = TempDir::new("concurrent");
    let mut w = Writer::create(dir.path(), config, options()).unwrap();
    let states: Arc<Mutex<HashMap<u64, Oracle>>> = Arc::new(Mutex::new(HashMap::new()));
    let mut oracle = Oracle::new(canonical::Metric::Cosine, dim);
    states.lock().unwrap().insert(0, oracle.clone());
    let reader = w.reader();
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let spec = ops(dim, 5);
    let queries = spec.vectors.queries(5, 1000);
    let mut handles = Vec::new();
    for t in 0..4 {
        let (reader, states, done, queries) = (
            reader.clone(),
            states.clone(),
            done.clone(),
            queries.clone(),
        );
        handles.push(std::thread::spawn(move || {
            let filters = standard_filters();
            let mut checked = 0;
            while !done.load(std::sync::atomic::Ordering::Relaxed) || checked == 0 {
                let s = reader.snapshot();
                let oracle = loop {
                    if let Some(o) = states.lock().unwrap().get(&s.version()) {
                        break o.clone();
                    }
                    std::thread::yield_now();
                };
                let (_, f, _) = &filters[(checked + t) % filters.len()];
                for q in &queries {
                    let ours = to_filter(f);
                    let hits = s
                        .search(
                            q,
                            10,
                            if f.is_empty() { None } else { Some(&ours) },
                            &SearchParams::default(),
                        )
                        .unwrap();
                    let pairs: Vec<(String, f32)> =
                        hits.into_iter().map(|h| (h.key, h.score)).collect();
                    check_query(&oracle, q, 10, f, &pairs, Expectation::exact()).unwrap();
                }
                checked += 1;
            }
            checked
        }));
    }
    for step in spec.steps() {
        match step {
            Step::Apply(batch) => {
                for op in &batch {
                    match op {
                        caprock_oracle::generate::Op::Upsert {
                            key,
                            vector,
                            path,
                            language,
                        } => oracle.upsert(key, vector, path, language).unwrap(),
                        caprock_oracle::generate::Op::Delete { key } => {
                            oracle.delete(key);
                        }
                    }
                }
                let version = w.snapshot().version() + batch.len() as u64;
                states.lock().unwrap().insert(version, oracle.clone());
                w.apply(&to_ops(&batch)).unwrap();
            }
            Step::Commit => {
                w.commit(None).unwrap();
            }
            Step::Maintain | Step::Compact => {
                w.compact().unwrap();
            }
            _ => {}
        }
    }
    done.store(true, std::sync::atomic::Ordering::Relaxed);
    for h in handles {
        assert!(h.join().unwrap() > 0);
    }
    let _ = OracleFilter::all();
    let _ = AnnKind::Flat;
}
