//! Acceptance tests of unit U13 (flat search and rescoring within a segment).
#![cfg(feature = "internals")]

mod common;

use common::{METRICS, TempDir, oracle_metric};
use mhr_vector::filter::SegmentFilter;
use mhr_vector::flat::{SegmentQuery, exact_search, flat_search, hit, rescore};
use mhr_vector::segment::{MemoryRecords, Segment, SegmentBuild, write_segment};
use mhr_vector::tombstones::Tombstones;
use mhr_vector::{CodeKind, Filter, IndexConfig, Metric, VerifyMode};
use mhr_vector_oracle::check::{Expectation, RecallSummary, check_query};
use mhr_vector_oracle::generate::{Shape, VectorSpec, key, path_for, standard_filters};
use mhr_vector_oracle::rng::Rng;
use mhr_vector_oracle::{Filter as OracleFilter, Oracle};

struct Fixture {
    _dir: TempDir,
    segment: Segment,
    tombstones: Tombstones,
    oracle: Oracle,
    queries: Vec<Vec<f32>>,
}

/// A segment of `n` records (raw vectors prepared by the oracle), a fifth deleted.
fn fixture(metric: Metric, codes: CodeKind, shape: Shape, n: usize, seed: u64) -> Fixture {
    let dim = 24;
    let spec = VectorSpec { dim, shape, seed };
    let mut oracle = Oracle::new(oracle_metric(metric), dim);
    let mut records = MemoryRecords {
        dim,
        ..MemoryRecords::default()
    };
    for (i, v) in spec.vectors(n).iter().enumerate() {
        let (path, language) = path_for(i as u64 / 3, seed);
        let k = key(i as u64);
        oracle.upsert(&k, v, &path, &language).unwrap();
        records
            .vectors
            .extend_from_slice(&oracle.get(&k).unwrap().vector);
        records.keys.push(k);
        records.paths.push(path);
        records.languages.push(language);
    }
    let config = IndexConfig::new(dim as u32, metric).with_codes(codes);
    let dir = TempDir::new("flat");
    let build = SegmentBuild {
        id: 1,
        config: &config,
        graph: false,
        threads: 2,
        memory_budget: 256 << 20,
    };
    let info = write_segment(dir.path(), &records, &build).unwrap();
    let segment = Segment::open(dir.path(), &info.file, &config, VerifyMode::Headers).unwrap();
    let mut rng = Rng::new(seed ^ 77);
    let mut deleted = Vec::new();
    for i in 0..n as u64 {
        if rng.chance(0.2) {
            let k = key(i);
            deleted.push(segment.find(&k).unwrap());
            oracle.delete(&k);
        }
    }
    let tombstones = Tombstones::none(segment.len()).with_deleted(&deleted);
    let queries = spec.queries(12, n);
    Fixture {
        _dir: dir,
        segment,
        tombstones,
        oracle,
        queries,
    }
}

fn ours(f: &OracleFilter) -> Filter {
    Filter {
        path_prefixes: f.path_prefixes.clone(),
        languages: f.languages.clone(),
    }
}

/// Runs `search` for every query, k and standard filter; checks each against the oracle.
fn check_all(
    fx: &Fixture,
    expect: Expectation,
    rescore_for: impl Fn(usize) -> usize,
    exact: bool,
) -> RecallSummary {
    let mut recalls = Vec::new();
    for (name, f, _) in standard_filters() {
        let compiled = SegmentFilter::compile(&ours(&f), &fx.segment);
        for k in [1, 10, 50] {
            for q in &fx.queries {
                let prepared = fx.oracle.prepare_query(q).unwrap();
                let sq = SegmentQuery {
                    query: &prepared,
                    k,
                    rescore: rescore_for(k),
                    filter: if f.is_empty() { None } else { Some(&compiled) },
                    tombstones: &fx.tombstones,
                };
                let hits = if exact {
                    exact_search(&fx.segment, &sq).unwrap()
                } else {
                    flat_search(&fx.segment, &sq).unwrap()
                };
                let pairs: Vec<(String, f32)> =
                    hits.iter().map(|h| (h.key.clone(), h.score)).collect();
                let report = check_query(&fx.oracle, q, k, &f, &pairs, expect)
                    .unwrap_or_else(|v| panic!("filter {name} k {k}: {v}"));
                recalls.push(report.recall);
                for h in &hits {
                    let e = fx.oracle.get(&h.key).unwrap();
                    assert_eq!((&h.path, &h.language), (&e.path, &e.language));
                }
            }
        }
    }
    RecallSummary::of(&recalls)
}

#[test]
fn exact_search_equals_the_oracle() {
    for metric in METRICS {
        for shape in [
            Shape::Gaussian,
            Shape::Duplicates { distinct: 40 },
            Shape::Lattice { levels: 1 },
        ] {
            if metric == Metric::Cosine && matches!(shape, Shape::Lattice { .. }) {
                continue; // lattice vectors may be zero
            }
            let fx = fixture(metric, CodeKind::Int8, shape, 1200, 3);
            check_all(&fx, Expectation::exact(), |k| k, true);
        }
    }
}

#[test]
fn flat_search_with_f32_codes_is_exact() {
    for metric in METRICS {
        let fx = fixture(
            metric,
            CodeKind::F32,
            Shape::Duplicates { distinct: 60 },
            1200,
            4,
        );
        check_all(&fx, Expectation::exact(), |k| k, false);
    }
}

#[test]
fn flat_search_with_int8_codes_rescores() {
    let clustered = Shape::Clustered {
        clusters: 20,
        spread: 0.35,
    };
    for metric in METRICS {
        let fx = fixture(metric, CodeKind::Int8, clustered, 2000, 5);
        let summary = check_all(&fx, Expectation::rescored(None), |k| (4 * k).max(32), false);
        assert!(
            summary.mean >= 0.99,
            "{metric:?}: mean recall {}",
            summary.mean
        );
    }
}

#[test]
fn rescore_orders_by_canonical_score_then_key() {
    let fx = fixture(
        Metric::Dot,
        CodeKind::Int8,
        Shape::Duplicates { distinct: 5 },
        300,
        6,
    );
    let q = fx.queries[1].clone();
    let prepared = fx.oracle.prepare_query(&q).unwrap();
    let candidates: Vec<(f32, u32)> = (0..fx.segment.len())
        .filter(|&o| !fx.tombstones.is_deleted(o))
        .map(|o| (0.0, o))
        .collect();
    let hits = rescore(&fx.segment, &prepared, &candidates, 25).unwrap();
    let pairs: Vec<(String, f32)> = hits.iter().map(|h| (h.key.clone(), h.score)).collect();
    check_query(
        &fx.oracle,
        &q,
        25,
        &OracleFilter::all(),
        &pairs,
        Expectation::exact(),
    )
    .unwrap();
    let h = hit(&fx.segment, 0, 1.5);
    assert_eq!((h.key.as_str(), h.score), (fx.segment.key(0), 1.5));
    assert_eq!(
        (h.path.as_str(), h.language.as_str()),
        (fx.segment.path(0), fx.segment.language(0))
    );
}
