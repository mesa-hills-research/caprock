//! Distance kernels: the canonical dot product, squared L2 distance and normalization,
//! bit for bit equal to the reference implementation's.

mod common;

use common::{METRICS, oracle_metric, scaled_vector};
use mhr_vector::VectorError;
use mhr_vector::distance;
use mhr_vector_oracle::canonical::{self, VectorProblem};
use mhr_vector_oracle::rng::Rng;

const DIMS: [usize; 24] = [
    0, 1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 256, 1000, 1023, 1024,
    2048, 4096,
];

fn same_problem(a: &VectorError, b: &VectorProblem) -> bool {
    matches!(
        (a, b),
        (
            VectorError::Dimension { .. },
            VectorProblem::Dimension { .. }
        ) | (
            VectorError::NotFinite { .. },
            VectorProblem::NotFinite { .. }
        ) | (VectorError::TooLarge { .. }, VectorProblem::TooLarge { .. })
            | (VectorError::Zero, VectorProblem::Zero)
            | (VectorError::NotNormalizable, VectorProblem::NotNormalizable)
    ) && match (a, b) {
        (VectorError::NotFinite { index: x }, VectorProblem::NotFinite { index: y })
        | (VectorError::TooLarge { index: x }, VectorProblem::TooLarge { index: y }) => x == y,
        _ => true,
    }
}

#[test]
fn dot_and_l2_are_bit_identical_to_the_oracle() {
    let mut rng = Rng::new(1);
    for &dim in &DIMS {
        for scale in [1e-3f32, 1.0, 1e3] {
            for _ in 0..20 {
                let a = scaled_vector(&mut rng, dim, scale);
                let b = scaled_vector(&mut rng, dim, scale);
                assert_eq!(
                    distance::dot(&a, &b).to_bits(),
                    canonical::dot(&a, &b).to_bits(),
                    "dot, dim {dim}"
                );
                assert_eq!(
                    distance::l2_squared(&a, &b).to_bits(),
                    canonical::l2_squared(&a, &b).to_bits(),
                    "l2, dim {dim}"
                );
            }
        }
    }
}

#[test]
fn normalize_prepare_and_score_match_the_oracle() {
    let mut rng = Rng::new(2);
    for &dim in &DIMS[1..] {
        for _ in 0..10 {
            let v = scaled_vector(&mut rng, dim, 0.7);
            let q = scaled_vector(&mut rng, dim, 0.7);
            let ours = distance::normalize(&v).unwrap();
            let theirs = canonical::normalize(&v).unwrap();
            assert_eq!(
                ours.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                theirs.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "normalize, dim {dim}"
            );
            for metric in METRICS {
                let om = oracle_metric(metric);
                let pv = distance::prepare(metric, &v, dim).unwrap();
                let pq = distance::prepare(metric, &q, dim).unwrap();
                assert_eq!(pv, canonical::prepare(om, &v, dim).unwrap());
                assert_eq!(
                    distance::score(metric, &pq, &pv).to_bits(),
                    canonical::score(om, &pq, &pv).to_bits(),
                    "{metric:?}, dim {dim}"
                );
            }
        }
    }
}

#[test]
fn bad_vectors_are_refused_like_the_oracle() {
    let cases: Vec<(Vec<f32>, usize)> = vec![
        (vec![1.0, 2.0], 3),
        (vec![1.0, f32::NAN, 0.0], 3),
        (vec![f32::INFINITY, 0.0, 0.0], 3),
        (vec![0.0, f32::NEG_INFINITY, 1.0], 3),
        (vec![0.0, 0.0, 2e15], 3),
        (vec![-1.5e15, 0.0, 0.0], 3),
        (vec![0.0, 0.0, 0.0], 3),
        (vec![1e-30, 0.0, 0.0], 3),
    ];
    for (v, dim) in &cases {
        let ours = distance::validate(v, *dim);
        let theirs = canonical::validate(v, *dim);
        assert_eq!(ours.is_ok(), theirs.is_ok(), "validate {v:?}");
        if let (Err(a), Err(b)) = (&ours, &theirs) {
            assert!(same_problem(a, b), "{a:?} vs {b:?}");
        }
        for metric in METRICS {
            let ours = distance::prepare(metric, v, *dim);
            let theirs = canonical::prepare(oracle_metric(metric), v, *dim);
            assert_eq!(ours.is_ok(), theirs.is_ok(), "prepare {metric:?} {v:?}");
            match (&ours, &theirs) {
                (Err(a), Err(b)) => assert!(same_problem(a, b), "{a:?} vs {b:?}"),
                (Ok(a), Ok(b)) => assert_eq!(a, b),
                _ => unreachable!(),
            }
        }
    }
    assert_eq!(distance::normalize(&[0.0, 0.0]), Err(VectorError::Zero));
}

#[test]
fn fast_kernels_are_close() {
    let mut rng = Rng::new(3);
    for &dim in &DIMS {
        for _ in 0..10 {
            let a = scaled_vector(&mut rng, dim, 1.0);
            let b = scaled_vector(&mut rng, dim, 1.0);
            let magnitude: f32 = a.iter().zip(&b).map(|(x, y)| (x * y).abs()).sum();
            let exact = canonical::dot(&a, &b);
            let fast = distance::dot_fast(&a, &b);
            assert!(
                (fast - exact).abs() <= 1e-5 * magnitude + 1e-30,
                "dot_fast {fast} vs {exact}, dim {dim}"
            );
            let l2 = canonical::l2_squared(&a, &b);
            let l2_fast = distance::l2_squared_fast(&a, &b);
            assert!(
                (l2_fast - l2).abs() <= 1e-5 * l2.abs() + 1e-30,
                "l2_fast, dim {dim}"
            );
            for metric in METRICS {
                let s = canonical::score(oracle_metric(metric), &a, &b);
                let f = distance::score_fast(metric, &a, &b);
                assert!((f - s).abs() <= 1e-5 * (magnitude + l2.abs()) + 1e-30);
            }
        }
    }
}

#[test]
fn length_mismatch_panics() {
    panics_for_real(|| {
        distance::dot(&[1.0, 2.0], &[1.0]);
    });
}

/// Runs `f`, which must panic for a reason other than an unimplemented stub.
fn panics_for_real(f: impl FnOnce() + std::panic::UnwindSafe) {
    let err = std::panic::catch_unwind(f).expect_err("expected a panic");
    let msg = err
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(!msg.contains("not yet implemented"), "{msg}");
}
