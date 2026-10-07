//! Property tests: random operation sequences on random data, any index configuration,
//! compared with the oracle (exact configurations: equal results; approximate ones: the
//! universal rules, canonical scores and a recall floor). proptest shrinks a failure to
//! a small case (fewer steps, smaller batches, lower dimension).

use mhr_vector::{CodeKind, IndexConfig, Metric, SearchParams, StoreOptions};
use mhr_vector_bench::adapter::StoreUnderTest;
use mhr_vector_bench::engine::oracle_metric;
use mhr_vector_oracle::check::Expectation;
use mhr_vector_oracle::generate::{OpsSpec, Shape};
use mhr_vector_oracle::scenario::{Scenario, run};
use proptest::prelude::*;

fn metric(i: usize) -> Metric {
    [Metric::Cosine, Metric::Dot, Metric::L2][i % 3]
}

fn shape(i: usize) -> Shape {
    match i % 5 {
        0 => Shape::Gaussian,
        1 => Shape::Clustered {
            clusters: 8,
            spread: 0.3,
        },
        2 => Shape::Duplicates { distinct: 20 },
        3 => Shape::Lattice { levels: 1 },
        _ => Shape::Anisotropic,
    }
}

#[allow(clippy::too_many_arguments)]
fn spec(
    dim: usize,
    shape_i: usize,
    seed: u64,
    steps: usize,
    batch: usize,
    key_space: u64,
) -> OpsSpec {
    let mut o = OpsSpec::mixed(dim, seed);
    o.vectors.shape = shape(shape_i);
    o.steps = steps;
    o.max_batch = batch;
    o.key_space = key_space;
    o
}

fn options(delta: u32) -> StoreOptions {
    StoreOptions {
        delta_max_vectors: delta,
        threads: 2,
        build_memory_budget: 128 << 20,
        ..StoreOptions::default()
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 24, ..ProptestConfig::default() })]

    #[test]
    fn exact_indexes_equal_the_oracle(
        m in 0usize..3, s in 0usize..5, dim in 1usize..40, seed in any::<u64>(),
        steps in 3usize..40, batch in 1usize..300, key_space in 5u64..2000, delta in 1u32..400,
    ) {
        let config = IndexConfig::exact(dim as u32, metric(m));
        let scenario = Scenario {
            metric: oracle_metric(metric(m)),
            ops: spec(dim, s, seed, steps, batch, key_space),
            queries: 4,
            ks: vec![1, 7, 50],
            filters: true,
            expect: Expectation::exact(),
            min_mean_recall: None,
        };
        let mut store = StoreUnderTest::new(config, options(delta), SearchParams::default()).unwrap();
        let r = run(&mut store, &scenario);
        prop_assert!(r.is_ok(), "{}", r.unwrap_err());
    }

    #[test]
    fn exact_search_on_approximate_indexes_equals_the_oracle(
        m in 0usize..3, s in 0usize..5, dim in 2usize..40, seed in any::<u64>(),
        steps in 3usize..30, batch in 1usize..400, codes in 0usize..3, graph_min in 2u32..500,
    ) {
        let mut config = IndexConfig::new(dim as u32, metric(m))
            .with_codes([CodeKind::F32, CodeKind::Int8, CodeKind::Binary][codes]);
        config.ann.graph_min_vectors = graph_min;
        let scenario = Scenario {
            metric: oracle_metric(metric(m)),
            ops: spec(dim, s, seed, steps, batch, 3000),
            queries: 4,
            ks: vec![10],
            filters: true,
            expect: Expectation::exact(),
            min_mean_recall: None,
        };
        let mut store = StoreUnderTest::new(config, options(200), SearchParams::exact()).unwrap();
        let r = run(&mut store, &scenario);
        prop_assert!(r.is_ok(), "{}", r.unwrap_err());
    }

    #[test]
    fn approximate_indexes_keep_the_rules_and_recall(
        m in 0usize..3, dim in 8usize..48, seed in any::<u64>(),
        steps in 10usize..40, codes in 0usize..2, graph_min in 50u32..600,
    ) {
        let codes = [CodeKind::Int8, CodeKind::F32][codes];
        let mut config = IndexConfig::new(dim as u32, metric(m)).with_codes(codes);
        config.ann.graph_min_vectors = graph_min;
        let scenario = Scenario {
            metric: oracle_metric(metric(m)),
            ops: spec(dim, 1, seed, steps, 600, 4000),
            queries: 6,
            ks: vec![1, 10],
            filters: true,
            expect: Expectation::rescored(None),
            min_mean_recall: Some(0.9),
        };
        let mut store = StoreUnderTest::new(config, options(250), SearchParams::default()).unwrap();
        let r = run(&mut store, &scenario);
        prop_assert!(r.is_ok(), "{}", r.unwrap_err());
    }
}
