//! Vamana graph construction and graph serialization.
#![cfg(feature = "internals")]

mod common;

use std::collections::{BinaryHeap, HashSet};

use common::{oracle_metric, overlap, shaped, true_top};
use mhr_vector::quant::{Codec, F32Codec};
use mhr_vector::vamana::{Graph, NO_NEIGHBOR, build_graph};
use mhr_vector::{Metric, VamanaParams};
use mhr_vector_oracle::canonical;
use mhr_vector_oracle::generate::Shape;

const CLUSTERED: Shape = Shape::Clustered {
    clusters: 20,
    spread: 0.35,
};

fn codes_of(codec: &F32Codec, base: &[Vec<f32>]) -> Vec<u8> {
    let mut out = vec![0u8; base.len() * codec.code_len()];
    for (i, v) in base.iter().enumerate() {
        codec.encode(
            v,
            &mut out[i * codec.code_len()..(i + 1) * codec.code_len()],
        );
    }
    out
}

fn assert_valid(g: &Graph, n: usize, params: &VamanaParams) {
    assert_eq!(g.len(), n);
    assert_eq!(g.degree, params.max_degree);
    assert!((g.entry as usize) < n.max(1));
    for node in 0..n as u32 {
        let row = &g.adjacency
            [node as usize * g.degree as usize..(node as usize + 1) * g.degree as usize];
        let real = g.neighbors(node);
        assert!(
            row[real.len()..].iter().all(|&x| x == NO_NEIGHBOR),
            "padding only at the end"
        );
        let set: HashSet<u32> = real.iter().copied().collect();
        assert_eq!(set.len(), real.len(), "no repeated neighbour");
        assert!(!set.contains(&node), "no self loop");
        assert!(real.iter().all(|&x| (x as usize) < n));
    }
}

/// A plain best-first search with exact scores, written here so these tests hold
/// whatever the beam search does.
fn greedy(
    g: &Graph,
    base: &[Vec<f32>],
    metric: Metric,
    q: &[f32],
    beam: usize,
    k: usize,
) -> Vec<usize> {
    #[derive(PartialEq)]
    struct Item(f32, u32);
    impl Eq for Item {}
    impl PartialOrd for Item {
        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for Item {
        fn cmp(&self, o: &Self) -> std::cmp::Ordering {
            self.0.partial_cmp(&o.0).unwrap().then(o.1.cmp(&self.1))
        }
    }
    let score = |i: u32| canonical::score(oracle_metric(metric), q, &base[i as usize]);
    let mut visited = HashSet::new();
    let mut frontier = BinaryHeap::new();
    let mut best: Vec<(f32, u32)> = Vec::new();
    visited.insert(g.entry);
    frontier.push(Item(score(g.entry), g.entry));
    while let Some(Item(s, node)) = frontier.pop() {
        if best.len() >= beam && s < best[beam - 1].0 {
            break;
        }
        best.push((s, node));
        best.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        best.truncate(beam);
        for &nb in g.neighbors(node) {
            if visited.insert(nb) {
                frontier.push(Item(score(nb), nb));
            }
        }
    }
    best.into_iter().take(k).map(|(_, n)| n as usize).collect()
}

#[test]
fn graphs_are_valid_deterministic_and_searchable() {
    for metric in [Metric::Cosine, Metric::L2] {
        let n = 3000;
        let (base, queries) = shaped(metric, CLUSTERED, 32, n, 100, 1);
        let codec = F32Codec { metric, dim: 32 };
        let codes = codes_of(&codec, &base);
        let params = VamanaParams {
            max_degree: 32,
            build_beam: 64,
            ..VamanaParams::default()
        };
        let g1 = build_graph(&codec, &codes, &params, 7, 1).unwrap();
        assert_valid(&g1, n, &params);
        for threads in [2, 4, 7] {
            let g = build_graph(&codec, &codes, &params, 7, threads).unwrap();
            assert!(g == g1, "{metric:?}: threads {threads} gave another graph");
        }
        // Reachability from the entry.
        let mut seen = HashSet::from([g1.entry]);
        let mut stack = vec![g1.entry];
        while let Some(x) = stack.pop() {
            for &nb in g1.neighbors(x) {
                if seen.insert(nb) {
                    stack.push(nb);
                }
            }
        }
        assert!(
            seen.len() as f64 >= 0.995 * n as f64,
            "{} of {n} reachable",
            seen.len()
        );
        // Search quality.
        let mut recall = 0.0;
        for q in &queries {
            let truth = true_top(metric, &base, q, 10);
            let found = greedy(&g1, &base, metric, q, 64, 10);
            recall += overlap(&truth, &found);
        }
        recall /= queries.len() as f64;
        assert!(recall >= 0.95, "{metric:?}: recall {recall}");
        let other = build_graph(&codec, &codes, &params, 8, 1).unwrap();
        assert_valid(&other, n, &params);
    }
}

#[test]
fn small_and_degenerate_inputs() {
    let params = VamanaParams {
        max_degree: 8,
        build_beam: 16,
        ..VamanaParams::default()
    };
    let codec = F32Codec {
        metric: Metric::Dot,
        dim: 4,
    };
    for n in [1usize, 2, 3, 9, 50] {
        let (base, _) = shaped(Metric::Dot, Shape::Gaussian, 4, n, 1, 2);
        let g = build_graph(&codec, &codes_of(&codec, &base), &params, 0, 3).unwrap();
        assert_valid(&g, n, &params);
        if n > 1 {
            assert!(
                (0..n as u32).all(|x| !g.neighbors(x).is_empty()),
                "n {n}: every node has an edge"
            );
        }
    }
    // All vectors identical.
    let same = vec![vec![1.0f32, 0.0, 0.0, 0.0]; 200];
    let g = build_graph(&codec, &codes_of(&codec, &same), &params, 0, 2).unwrap();
    assert_valid(&g, 200, &params);
}

#[test]
fn serialization_round_trips_and_checks() {
    let g = Graph {
        degree: 3,
        entry: 1,
        adjacency: vec![
            1,
            2,
            NO_NEIGHBOR,
            0,
            NO_NEIGHBOR,
            NO_NEIGHBOR,
            0,
            1,
            NO_NEIGHBOR,
        ],
    };
    let bytes = g.to_bytes();
    assert_eq!(bytes.len(), 16 + 9 * 4);
    assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 3);
    assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 1);
    assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 3);
    assert_eq!(Graph::from_bytes(&bytes, 3).unwrap(), g);
    assert!(Graph::from_bytes(&bytes, 4).is_err(), "node count");
    assert!(
        Graph::from_bytes(&bytes[..bytes.len() - 1], 3).is_err(),
        "length"
    );
    let bad = |adjacency: Vec<u32>, entry: u32| Graph {
        degree: 3,
        entry,
        adjacency,
    };
    for g in [
        bad(
            vec![
                1,
                2,
                NO_NEIGHBOR,
                0,
                NO_NEIGHBOR,
                NO_NEIGHBOR,
                0,
                1,
                NO_NEIGHBOR,
            ],
            3,
        ),
        bad(
            vec![
                1,
                5,
                NO_NEIGHBOR,
                0,
                NO_NEIGHBOR,
                NO_NEIGHBOR,
                0,
                1,
                NO_NEIGHBOR,
            ],
            0,
        ),
        bad(
            vec![
                0,
                2,
                NO_NEIGHBOR,
                0,
                NO_NEIGHBOR,
                NO_NEIGHBOR,
                0,
                1,
                NO_NEIGHBOR,
            ],
            0,
        ),
        bad(
            vec![
                1,
                1,
                NO_NEIGHBOR,
                0,
                NO_NEIGHBOR,
                NO_NEIGHBOR,
                0,
                1,
                NO_NEIGHBOR,
            ],
            0,
        ),
    ] {
        assert!(Graph::from_bytes(&g.to_bytes(), 3).is_err(), "{g:?}");
    }
}
