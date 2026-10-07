//! Acceptance tests of unit U12 (delta): exact search, isolation of views, cheap views.
#![cfg(feature = "internals")]

mod common;

use common::{METRICS, oracle_metric};
use mhr_vector::Filter;
use mhr_vector::delta::{Delta, DeltaView};
use mhr_vector::segment::RecordSource;
use mhr_vector_oracle::check::{Expectation, check_query};
use mhr_vector_oracle::generate::{Shape, VectorSpec, key, path_for, standard_filters};
use mhr_vector_oracle::rng::Rng;
use mhr_vector_oracle::{Filter as OracleFilter, Oracle};

fn to_ours(f: &OracleFilter) -> Filter {
    Filter {
        path_prefixes: f.path_prefixes.clone(),
        languages: f.languages.clone(),
    }
}

/// Raw vectors go to the oracle, which prepares them; the delta gets the oracle's stored
/// form (normalizing twice is not bit-stable, so tests never re-prepare).
#[test]
fn searches_equal_the_oracle_through_upserts_and_deletes() {
    for metric in METRICS {
        let dim = 20;
        let spec = VectorSpec {
            dim,
            shape: Shape::Clustered {
                clusters: 20,
                spread: 0.35,
            },
            seed: 1,
        };
        let vectors = spec.vectors(3000);
        let queries = spec.queries(15, 3000);
        let mut delta = Delta::new(dim, metric);
        let mut oracle = Oracle::new(oracle_metric(metric), dim);
        let mut rng = Rng::new(2);
        for (i, v) in vectors.iter().enumerate() {
            let j = rng.below(1500); // keys repeat: replacements
            let (path, language) = path_for(j, 1);
            if rng.chance(0.15) {
                assert_eq!(delta.delete(&key(j)), oracle.delete(&key(j)));
            } else {
                oracle.upsert(&key(j), v, &path, &language).unwrap();
                let stored = oracle.get(&key(j)).unwrap().vector.clone();
                delta.upsert(&key(j), &stored, &path, &language);
            }
            if i % 1000 == 999 {
                assert_eq!(delta.len(), oracle.len());
                let view = delta.view();
                assert_eq!(view.live(), oracle.len());
                for (name, f, _) in standard_filters() {
                    for k in [1, 10, 50] {
                        for q in &queries {
                            let prepared = oracle.prepare_query(q).unwrap();
                            let hits = view.search(&prepared, k, Some(&to_ours(&f)));
                            let pairs: Vec<(String, f32)> =
                                hits.iter().map(|h| (h.key.clone(), h.score)).collect();
                            check_query(&oracle, q, k, &f, &pairs, Expectation::exact())
                                .unwrap_or_else(|v| panic!("{metric:?} {name} k {k}: {v}"));
                            for h in &hits {
                                let e = oracle.get(&h.key).unwrap();
                                assert_eq!((&h.path, &h.language), (&e.path, &e.language));
                            }
                        }
                    }
                }
            }
        }
        for j in 0..1500u64 {
            assert_eq!(delta.contains(&key(j)), oracle.get(&key(j)).is_some());
        }
    }
}

#[test]
fn views_are_isolated_and_take_empties() {
    let dim = 4;
    let mut delta = Delta::new(dim, mhr_vector::Metric::Dot);
    assert!(delta.is_empty());
    delta.upsert("a", &[1.0, 0.0, 0.0, 0.0], "x/a.rs", "rust");
    delta.upsert("b", &[0.0, 1.0, 0.0, 0.0], "x/b.rs", "rust");
    let before = delta.view();
    delta.upsert("a", &[2.0, 0.0, 0.0, 0.0], "x/a.rs", "rust");
    delta.upsert("c", &[0.0, 0.0, 1.0, 0.0], "y/c.py", "python");
    assert!(delta.delete("b"));
    assert!(!delta.delete("b"));
    assert_eq!(before.live(), 2);
    assert!(before.contains("b") && !before.contains("c"));
    assert_eq!(before.get("a").unwrap().vector, vec![1.0, 0.0, 0.0, 0.0]);
    let now = delta.view();
    assert_eq!(now.live(), 2);
    assert_eq!(now.get("a").unwrap().vector, vec![2.0, 0.0, 0.0, 0.0]);
    assert!(now.get("b").is_none());
    let taken = delta.take();
    assert!(delta.is_empty() && delta.view().live() == 0);
    assert_eq!(taken.live(), 2);
    // As a record source: live records in key order.
    assert_eq!(RecordSource::len(&taken), 2);
    assert_eq!((taken.key(0), taken.key(1)), ("a", "c"));
    assert_eq!((taken.path(1), taken.language(1)), ("y/c.py", "python"));
    let mut v = vec![0f32; dim];
    taken.read_vector(0, &mut v).unwrap();
    assert_eq!(v, vec![2.0, 0.0, 0.0, 0.0]);
    let empty = DeltaView::empty(dim, mhr_vector::Metric::Dot);
    assert_eq!(empty.live(), 0);
    assert!(empty.search(&[1.0, 0.0, 0.0, 0.0], 5, None).is_empty());
}

/// Taking a view must not copy the delta: 2,000 views of a 25 MB delta, each after one
/// upsert, would copy 50 GB.
#[test]
fn views_are_cheap() {
    let dim = 128;
    let mut delta = Delta::new(dim, mhr_vector::Metric::Dot);
    let v: Vec<f32> = (0..dim).map(|i| i as f32).collect();
    for i in 0..50_000u64 {
        delta.upsert(&key(i), &v, "p/a.rs", "rust");
    }
    let started = std::time::Instant::now();
    let mut views = Vec::new();
    for i in 0..2_000u64 {
        delta.upsert(&key(100_000 + i), &v, "p/b.rs", "rust");
        views.push(delta.view());
        if views.len() > 10 {
            views.remove(0);
        }
    }
    let took = started.elapsed();
    assert!(took.as_secs_f64() < 2.0, "2,000 views took {took:?}");
    assert!(delta.resident_bytes() >= 50_000 * dim as u64 * 4);
}
