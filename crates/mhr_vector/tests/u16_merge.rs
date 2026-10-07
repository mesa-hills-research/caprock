//! Acceptance tests of unit U16 (merge planning, the memory math, live records).
#![cfg(feature = "internals")]

mod common;

use std::collections::HashSet;

use common::{TempDir, stored_vectors};
use mhr_vector::merge::{
    LiveRecords, MergeReason, SegmentSummary, max_segment_vectors, plan, plan_compaction,
};
use mhr_vector::segment::{MemoryRecords, RecordSource, Segment, SegmentBuild, write_segment};
use mhr_vector::tombstones::Tombstones;
use mhr_vector::{
    AnnConfig, AnnKind, CodeKind, Error, IndexConfig, MergePolicy, Metric, VerifyMode,
};
use mhr_vector_oracle::rng::Rng;

fn summary(id: u64, vectors: u64, live: u64, has_graph: bool) -> SegmentSummary {
    SegmentSummary {
        id,
        vectors,
        live,
        has_graph,
    }
}

fn check_plans(
    segments: &[SegmentSummary],
    plans: &[mhr_vector::merge::MergePlan],
    ann: &AnnConfig,
    max: u64,
) {
    let mut used = HashSet::new();
    let ids: Vec<u64> = segments.iter().map(|s| s.id).collect();
    for p in plans {
        assert!(!p.inputs.is_empty());
        let positions: Vec<usize> = p
            .inputs
            .iter()
            .map(|i| ids.iter().position(|x| x == i).unwrap())
            .collect();
        assert!(
            positions.windows(2).all(|w| w[0] < w[1]),
            "inputs oldest first"
        );
        for i in &p.inputs {
            assert!(used.insert(*i), "inputs are disjoint");
        }
        let live: u64 = p
            .inputs
            .iter()
            .map(|i| segments.iter().find(|s| s.id == *i).unwrap().live)
            .sum();
        assert!(live <= max, "output within the size limit");
        let graph_wanted =
            matches!(ann.kind, AnnKind::Vamana(_)) && live >= ann.graph_min_vectors as u64;
        assert_eq!(p.graph, graph_wanted, "{p:?}");
    }
}

/// Applies plans: each merge becomes one segment with its live count, no deletions.
fn apply(
    segments: &[SegmentSummary],
    plans: &[mhr_vector::merge::MergePlan],
    next_id: &mut u64,
) -> Vec<SegmentSummary> {
    let merged: HashSet<u64> = plans.iter().flat_map(|p| p.inputs.clone()).collect();
    let mut out: Vec<SegmentSummary> = segments
        .iter()
        .filter(|s| !merged.contains(&s.id))
        .cloned()
        .collect();
    for p in plans {
        let live: u64 = p
            .inputs
            .iter()
            .map(|i| segments.iter().find(|s| s.id == *i).unwrap().live)
            .sum();
        *next_id += 1;
        out.push(summary(*next_id, live, live, p.graph));
    }
    out
}

#[test]
fn tiers_deletes_and_graphs() {
    let policy = MergePolicy::default();
    let ann = AnnConfig::default();
    let max = 1_000_000;
    assert!(plan(&[], &policy, &ann, max).is_empty());
    let few: Vec<_> = (1..=3).map(|i| summary(i, 8192, 8192, false)).collect();
    assert!(
        plan(&few, &policy, &ann, max).is_empty(),
        "fewer than segments_per_tier"
    );
    let tier: Vec<_> = (1..=8).map(|i| summary(i, 8192, 8000 + i, false)).collect();
    let plans = plan(&tier, &policy, &ann, max);
    assert_eq!(plans.len(), 1);
    assert_eq!(plans[0].reason, MergeReason::Tier);
    assert_eq!(plans[0].inputs, (1..=8).collect::<Vec<_>>());
    check_plans(&tier, &plans, &ann, max);
    assert_eq!(plans, plan(&tier, &policy, &ann, max), "deterministic");

    let deleted = vec![
        summary(1, 100_000, 70_000, true),
        summary(2, 50_000, 49_000, true),
    ];
    let plans = plan(&deleted, &policy, &ann, max);
    assert_eq!(plans.len(), 1);
    assert_eq!(
        (plans[0].inputs.clone(), plans[0].reason),
        (vec![1], MergeReason::Deletes)
    );
    check_plans(&deleted, &plans, &ann, max);

    let flat_but_big = vec![summary(5, 30_000, 30_000, false)];
    let plans = plan(&flat_but_big, &policy, &ann, max);
    assert_eq!(plans.len(), 1);
    assert_eq!(plans[0].reason, MergeReason::Graph);
    assert!(plans[0].graph);
    let no_ann = AnnConfig {
        kind: AnnKind::Flat,
        ..AnnConfig::default()
    };
    assert!(plan(&flat_but_big, &policy, &no_ann, max).is_empty());
}

#[test]
fn compaction_reaches_a_fixed_point() {
    let ann = AnnConfig::default();
    let mut rng = Rng::new(3);
    for trial in 0..50 {
        let max = 10_000 + rng.below(200_000);
        let n = 1 + rng.below(40);
        let mut segments: Vec<SegmentSummary> = (0..n)
            .map(|i| {
                let vectors = 1 + rng.below(max);
                let live = rng.below(vectors + 1);
                summary(i + 1, vectors, live, rng.chance(0.3))
            })
            .collect();
        let mut next_id = 1000;
        let mut rounds = 0;
        loop {
            let plans = plan_compaction(&segments, &ann, max);
            if plans.is_empty() {
                break;
            }
            check_plans(&segments, &plans, &ann, max);
            assert!(plans.iter().all(|p| p.reason == MergeReason::Compact));
            segments = apply(&segments, &plans, &mut next_id);
            rounds += 1;
            assert!(rounds < 20, "trial {trial}: compaction does not converge");
        }
        for s in &segments {
            assert_eq!(s.vectors, s.live, "trial {trial}: deletions purged");
            if s.live >= ann.graph_min_vectors as u64 {
                assert!(s.has_graph, "trial {trial}: large segments have graphs");
            }
        }
        let nonempty: Vec<&SegmentSummary> = segments.iter().filter(|s| s.live > 0).collect();
        for (i, a) in nonempty.iter().enumerate() {
            for b in &nonempty[i + 1..] {
                assert!(
                    a.live + b.live > max,
                    "trial {trial}: two segments could still merge"
                );
            }
        }
    }
}

#[test]
fn memory_math() {
    let config = IndexConfig::new(1024, Metric::Cosine);
    let m = max_segment_vectors(&config, 1 << 30).unwrap();
    assert!((100_000..=1_050_000).contains(&m), "{m}");
    assert!(max_segment_vectors(&config, 2 << 30).unwrap() >= m);
    assert!(max_segment_vectors(&config, 1 << 29).unwrap() <= m);
    let big = IndexConfig::new(2048, Metric::Cosine);
    assert!(max_segment_vectors(&big, 1 << 30).unwrap() <= m);
    let binary = IndexConfig::new(1024, Metric::Cosine).with_codes(CodeKind::Binary);
    assert!(max_segment_vectors(&binary, 1 << 30).unwrap() >= m);
    let f32s = IndexConfig::new(4096, Metric::Cosine).with_codes(CodeKind::F32);
    assert!(matches!(
        max_segment_vectors(&f32s, 1 << 20),
        Err(Error::Budget { .. })
    ));
}

#[test]
fn live_records_skip_deletions() {
    let dim = 8;
    let config = IndexConfig::new(dim as u32, Metric::Dot);
    let dir = TempDir::new("merge-live");
    let mut segments = Vec::new();
    for (id, offset) in [(1u64, 0usize), (2, 500)] {
        let vectors = stored_vectors(Metric::Dot, dim, 500, id);
        let records = MemoryRecords {
            dim,
            keys: (0..500).map(|i| format!("k{:05}", i + offset)).collect(),
            paths: (0..500).map(|i| format!("d{}/f{}.rs", i % 7, i)).collect(),
            languages: vec!["rust".into(); 500],
            vectors: vectors.concat(),
        };
        let build = SegmentBuild {
            id,
            config: &config,
            graph: false,
            threads: 1,
            memory_budget: 64 << 20,
        };
        let info = write_segment(dir.path(), &records, &build).unwrap();
        segments.push((
            Segment::open(dir.path(), &info.file, &config, VerifyMode::Headers).unwrap(),
            records,
        ));
    }
    let t1 = Tombstones::none(500).with_deleted(&[0, 10, 499]);
    let t2 = Tombstones::none(500).with_deleted(&(0..250).collect::<Vec<u32>>());
    let live = LiveRecords::new(vec![(&segments[0].0, &t1), (&segments[1].0, &t2)]);
    assert_eq!(live.len(), 497 + 250);
    let mut keys = HashSet::new();
    let mut v = vec![0f32; dim];
    for i in 0..live.len() {
        assert!(keys.insert(live.key(i).to_string()));
        let (seg, records) = if (live.key(i)[1..].parse::<usize>().unwrap()) < 500 {
            &segments[0]
        } else {
            &segments[1]
        };
        let o = seg.find(live.key(i)).unwrap();
        assert!(
            !(if std::ptr::eq(seg, &segments[0].0) {
                &t1
            } else {
                &t2
            })
            .is_deleted(o)
        );
        let r = records.keys.iter().position(|k| k == live.key(i)).unwrap();
        live.read_vector(i, &mut v).unwrap();
        assert_eq!(&v[..], &records.vectors[r * dim..(r + 1) * dim]);
        assert_eq!(live.path(i), records.paths[r]);
        assert_eq!(live.language(i), "rust");
    }
}
