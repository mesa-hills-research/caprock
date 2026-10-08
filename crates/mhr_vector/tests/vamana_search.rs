//! Beam search over Vamana graphs, on exact k-NN graphs built here, so these tests
//! hold whatever the graph construction does.
#![cfg(feature = "internals")]

mod common;

use common::{oracle_metric, overlap, shaped, true_top};
use mhr_vector::Metric;
use mhr_vector::quant::{Codec, F32Codec};
use mhr_vector::vamana::{Graph, NO_NEIGHBOR, SearchScratch, beam_search};
use mhr_vector_oracle::canonical;
use mhr_vector_oracle::generate::Shape;
use mhr_vector_oracle::rng::Rng;

/// Each node's `degree - 8` nearest neighbours (exact) plus 8 random long-range edges
/// (so clusters connect), entry 0.
fn knn_graph(base: &[Vec<f32>], metric: Metric, degree: usize) -> Graph {
    let n = base.len();
    let mut rng = Rng::new(11);
    let mut adjacency = vec![NO_NEIGHBOR; n * degree];
    for i in 0..n {
        let mut row: Vec<usize> = true_top(metric, base, &base[i], degree - 7)
            .into_iter()
            .filter(|&j| j != i)
            .take(degree - 8)
            .collect();
        while row.len() < degree {
            let j = rng.below(n as u64) as usize;
            if j != i && !row.contains(&j) {
                row.push(j);
            }
        }
        for (slot, j) in row.into_iter().enumerate() {
            adjacency[i * degree + slot] = j as u32;
        }
    }
    Graph {
        degree: degree as u32,
        entry: 0,
        adjacency,
    }
}

type Setup = (Vec<Vec<f32>>, Vec<Vec<f32>>, F32Codec, Vec<u8>, Graph);

fn setup(metric: Metric, n: usize) -> Setup {
    let shape = Shape::Clustered {
        clusters: 8,
        spread: 0.5,
    };
    let (base, queries) = shaped(metric, shape, 16, n, 60, 3);
    let codec = F32Codec { metric, dim: 16 };
    let mut codes = vec![0u8; n * codec.code_len()];
    for (i, v) in base.iter().enumerate() {
        codec.encode(v, &mut codes[i * 64..(i + 1) * 64]);
    }
    let graph = knn_graph(&base, metric, 24);
    (base, queries, codec, codes, graph)
}

#[test]
fn finds_the_true_neighbours_in_order() {
    for metric in [Metric::Cosine, Metric::Dot, Metric::L2] {
        let (base, queries, codec, codes, graph) = setup(metric, 2000);
        let mut scratch = SearchScratch::new();
        let mut recall = 0.0;
        for q in &queries {
            let prepared = codec.prepare(q);
            let found = beam_search(
                &graph,
                &codec,
                &codes,
                &prepared,
                64,
                10,
                &|_| true,
                &mut scratch,
            );
            assert_eq!(found.len(), 10);
            for w in found.windows(2) {
                assert!(
                    w[0].0 > w[1].0 || (w[0].0 == w[1].0 && w[0].1 < w[1].1),
                    "best first"
                );
            }
            for &(s, node) in &found {
                let want = canonical::score(oracle_metric(metric), q, &base[node as usize]);
                assert_eq!(s.to_bits(), want.to_bits(), "f32 codes estimate exactly");
            }
            let got: Vec<usize> = found.iter().map(|&(_, n)| n as usize).collect();
            recall += overlap(&true_top(metric, &base, q, 10), &got);
        }
        recall /= queries.len() as f64;
        assert!(recall >= 0.95, "{metric:?}: recall {recall}");
    }
}

#[test]
fn admission_beam_and_scratch_reuse() {
    let (_, queries, codec, codes, graph) = setup(Metric::Cosine, 1500);
    let mut shared = SearchScratch::new();
    for q in &queries {
        let p = codec.prepare(q);
        let even = beam_search(
            &graph,
            &codec,
            &codes,
            &p,
            48,
            20,
            &|o| o % 2 == 0,
            &mut shared,
        );
        assert_eq!(even.len(), 20);
        assert!(even.iter().all(|&(_, o)| o % 2 == 0), "only admitted nodes");
        let none = beam_search(&graph, &codec, &codes, &p, 48, 20, &|_| false, &mut shared);
        assert!(none.is_empty());
        let wide = beam_search(&graph, &codec, &codes, &p, 5, 40, &|_| true, &mut shared);
        assert_eq!(wide.len(), 40, "the beam is raised to `wanted`");
        let fresh = beam_search(
            &graph,
            &codec,
            &codes,
            &p,
            48,
            20,
            &|o| o % 2 == 0,
            &mut SearchScratch::new(),
        );
        assert_eq!(fresh, even, "scratch reuse changes nothing");
    }
}

#[test]
fn tiny_graphs() {
    let codec = F32Codec {
        metric: Metric::Dot,
        dim: 2,
    };
    let one = Graph {
        degree: 4,
        entry: 0,
        adjacency: vec![NO_NEIGHBOR; 4],
    };
    let codes: Vec<u8> = [1.0f32, 0.0].iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = codec.prepare(&[1.0, 1.0]);
    let mut s = SearchScratch::new();
    assert_eq!(
        beam_search(&one, &codec, &codes, &p, 8, 5, &|_| true, &mut s),
        vec![(1.0, 0)]
    );
    assert!(beam_search(&one, &codec, &codes, &p, 8, 5, &|_| false, &mut s).is_empty());
}
