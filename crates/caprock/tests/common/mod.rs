//! Helpers shared by the integration tests.
#![allow(dead_code)]

pub mod store;

use std::path::PathBuf;

use caprock_oracle::canonical;
use caprock_oracle::generate::{Shape, VectorSpec};
use caprock_oracle::rng::Rng;

/// A fresh, empty directory under the system temp dir, removed on drop.
pub struct TempDir(pub PathBuf);

impl TempDir {
    pub fn new(name: &str) -> TempDir {
        let mut rng = Rng::new(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        );
        let dir = std::env::temp_dir().join(format!(
            "caprock-{name}-{}-{:x}",
            std::process::id(),
            rng.next_u64()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }

    pub fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The oracle's metric for a store metric.
pub fn oracle_metric(m: caprock::Metric) -> canonical::Metric {
    match m {
        caprock::Metric::Cosine => canonical::Metric::Cosine,
        caprock::Metric::Dot => canonical::Metric::Dot,
        caprock::Metric::L2 => canonical::Metric::L2,
    }
}

pub const METRICS: [caprock::Metric; 3] = [
    caprock::Metric::Cosine,
    caprock::Metric::Dot,
    caprock::Metric::L2,
];

/// Random vectors with components at a given scale.
pub fn scaled_vector(rng: &mut Rng, dim: usize, scale: f32) -> Vec<f32> {
    (0..dim).map(|_| rng.gaussian() as f32 * scale).collect()
}

/// `n` clustered vectors of `dim` in stored form for `metric`.
pub fn stored_vectors(metric: caprock::Metric, dim: usize, n: usize, seed: u64) -> Vec<Vec<f32>> {
    let spec = VectorSpec {
        dim,
        shape: Shape::Clustered {
            clusters: 20,
            spread: 0.35,
        },
        seed,
    };
    spec.vectors(n)
        .iter()
        .map(|v| canonical::prepare(oracle_metric(metric), v, dim).unwrap())
        .collect()
}

/// Queries in stored form for `metric`, from the same distribution.
pub fn stored_queries(
    metric: caprock::Metric,
    dim: usize,
    n: usize,
    base_n: usize,
    seed: u64,
) -> Vec<Vec<f32>> {
    let spec = VectorSpec {
        dim,
        shape: Shape::Clustered {
            clusters: 20,
            spread: 0.35,
        },
        seed,
    };
    spec.queries(n, base_n)
        .iter()
        .map(|v| canonical::prepare(oracle_metric(metric), v, dim).unwrap())
        .collect()
}

/// Indexes of the true top `k` of `base` for `query` by canonical score (ties by index).
pub fn true_top(metric: caprock::Metric, base: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = base
        .iter()
        .enumerate()
        .map(|(i, v)| (canonical::score(oracle_metric(metric), query, v), i))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
    scored.into_iter().take(k).map(|(_, i)| i).collect()
}

/// Indexes of the top `k` by an estimate function (ties by index).
pub fn top_by(n: usize, k: usize, estimate: impl Fn(usize) -> f32) -> Vec<usize> {
    let mut scored: Vec<(f32, usize)> = (0..n).map(|i| (estimate(i), i)).collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
    scored.into_iter().take(k).map(|(_, i)| i).collect()
}

/// Share of `truth` found in `found`.
pub fn overlap(truth: &[usize], found: &[usize]) -> f64 {
    let hits = truth.iter().filter(|t| found.contains(t)).count();
    hits as f64 / truth.len().max(1) as f64
}

/// Base vectors and queries of a shape, in stored form for `metric`.
pub fn shaped(
    metric: caprock::Metric,
    shape: Shape,
    dim: usize,
    n: usize,
    queries: usize,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let spec = VectorSpec { dim, shape, seed };
    let om = oracle_metric(metric);
    let base = spec
        .vectors(n)
        .iter()
        .map(|v| canonical::prepare(om, v, dim).unwrap())
        .collect();
    let qs = spec
        .queries(queries, n)
        .iter()
        .map(|v| canonical::prepare(om, v, dim).unwrap())
        .collect();
    (base, qs)
}
