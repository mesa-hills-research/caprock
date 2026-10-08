//! The binary codec.
#![cfg(feature = "internals")]

mod common;

use common::{METRICS, oracle_metric, overlap, shaped, top_by, true_top};
use mhr_vector::quant::{BinaryCodec, Codec};
use mhr_vector::{CodeKind, Metric};
use mhr_vector_oracle::canonical;
use mhr_vector_oracle::generate::Shape;

include!("codec_checks.inc");

const CLUSTERED: Shape = Shape::Clustered {
    clusters: 20,
    spread: 0.35,
};

#[test]
fn layout_and_kind() {
    for dim in [1usize, 63, 64, 65, 1024] {
        let (base, _) = shaped(Metric::Cosine, Shape::Gaussian, dim, 50, 1, 1);
        let c = BinaryCodec::train(Metric::Cosine, dim, &base.concat(), 0).unwrap();
        assert_eq!(c.kind(), CodeKind::Binary);
        let words = dim.div_ceil(64) * 8;
        assert!(
            c.code_len() >= words && c.code_len() <= words + 8,
            "dim {dim}: {} bytes",
            c.code_len()
        );
    }
}

#[test]
fn estimates_find_the_true_neighbours() {
    for (metric, shape, dim, candidates, min) in [
        (Metric::Cosine, CLUSTERED, 128, 100, 0.95),
        (Metric::Cosine, CLUSTERED, 256, 100, 0.95),
        (Metric::Dot, CLUSTERED, 128, 100, 0.9),
        (Metric::L2, CLUSTERED, 128, 100, 0.9),
        // A shared offset: plain sign bits find ~0.34 here; centring finds ~0.6.
        (Metric::Cosine, Shape::Anisotropic, 256, 200, 0.5),
    ] {
        let (base, queries) = shaped(metric, shape, dim, 2000, 50, 1);
        let codec = BinaryCodec::train(metric, dim, &base.concat(), 0).unwrap();
        let (recall, _, _) = codec_quality(&codec, metric, &base, &queries, candidates);
        assert!(
            recall >= min,
            "{metric:?} {} {dim}: recall {recall} < {min}",
            shape.name()
        );
    }
}

#[test]
fn deterministic_symmetric_and_round_trips() {
    for metric in METRICS {
        let (base, _) = shaped(metric, CLUSTERED, 96, 300, 1, 2);
        symmetric_and_deterministic(
            |s| BinaryCodec::train(metric, 96, s, 5).unwrap(),
            |b| BinaryCodec::from_bytes(metric, 96, b),
            &base,
        );
    }
}

#[test]
fn degenerate_inputs() {
    for metric in METRICS {
        degenerate_inputs_do_not_break(|s, d| BinaryCodec::train(metric, d, s, 0).unwrap());
    }
}
