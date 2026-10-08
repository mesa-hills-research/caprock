//! The int8 codec.
#![cfg(feature = "internals")]

mod common;

use caprock::quant::{Codec, Int8Codec};
use caprock::{CodeKind, Metric};
use caprock_oracle::canonical;
use caprock_oracle::generate::Shape;
use common::{METRICS, oracle_metric, overlap, shaped, top_by, true_top};

include!("codec_checks.inc");

const CLUSTERED: Shape = Shape::Clustered {
    clusters: 20,
    spread: 0.35,
};

#[test]
fn layout_and_kind() {
    for metric in METRICS {
        let (base, _) = shaped(metric, CLUSTERED, 100, 200, 1, 1);
        let c = Int8Codec::train(metric, 100, &base.concat(), 0).unwrap();
        assert_eq!(c.kind(), CodeKind::Int8);
        assert_eq!(c.dim(), 100);
        assert!(
            c.code_len() >= 100 && c.code_len() <= 108,
            "dim bytes + at most 8"
        );
    }
}

#[test]
fn estimates_find_the_true_neighbours() {
    for (metric, shape, dim) in [
        (Metric::Cosine, CLUSTERED, 128),
        (Metric::Cosine, CLUSTERED, 256),
        (Metric::Dot, CLUSTERED, 128),
        (Metric::L2, CLUSTERED, 128),
        (Metric::Cosine, Shape::Anisotropic, 256),
    ] {
        let (base, queries) = shaped(metric, shape, dim, 2000, 50, 1);
        let codec = Int8Codec::train(metric, dim, &base.concat(), 0).unwrap();
        let (recall, err, magnitude) = codec_quality(&codec, metric, &base, &queries, 40);
        assert!(
            recall >= 0.98,
            "{metric:?} {} {dim}: recall {recall}",
            shape.name()
        );
        assert!(
            err <= 0.01 * magnitude + 1e-3,
            "{metric:?} {} {dim}: mean error {err} vs mean |score| {magnitude}",
            shape.name()
        );
    }
}

#[test]
fn deterministic_symmetric_and_round_trips() {
    for metric in METRICS {
        let (base, _) = shaped(metric, CLUSTERED, 64, 300, 1, 2);
        symmetric_and_deterministic(
            |s| Int8Codec::train(metric, 64, s, 5).unwrap(),
            |b| Int8Codec::from_bytes(metric, 64, b),
            &base,
        );
    }
}

#[test]
fn degenerate_inputs() {
    for metric in METRICS {
        degenerate_inputs_do_not_break(|s, d| Int8Codec::train(metric, d, s, 0).unwrap());
    }
}
