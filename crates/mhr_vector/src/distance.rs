//! Canonical score arithmetic (unit U01; REQUIREMENTS.md, "Scores").
//!
//! The **canonical dot product** keeps 16 `f32` lane accumulators. Element `i` is added
//! to lane `i % 16`, in increasing `i`, as `acc = acc + (a[i] * b[i])`, the product and
//! the sum each rounded to `f32` (no fused multiply-add). The lanes are reduced in a fixed
//! tree: for `w` in 8, 4, 2, 1, `acc[l] = acc[l] + acc[l + w]` for each `l < w`; the
//! result is `acc[0]`. The **canonical squared L2** is the same with `d = a[i] - b[i]`
//! and `d * d`. **Normalization** computes `norm = sqrt(dot(v, v))` in `f32` and divides
//! each component by it.
//!
//! Final scores must be computed with these functions: they equal the oracle's bit for
//! bit, which is what makes exact search checkable with `==`. A SIMD implementation that
//! keeps this lane order produces the same bits. Candidate generation may use any faster
//! approximation (see [`dot_fast`]).

use crate::config::Metric;
use crate::error::VectorError;

/// Lane accumulators of the canonical kernels.
pub const LANES: usize = 16;

/// Largest absolute value of a vector component (so no score can overflow).
pub const MAX_ABS_COMPONENT: f32 = 1e15;

// Keep the same tree even for short vectors: unused lanes contain positive zero.
#[inline]
fn reduce(mut acc: [f32; LANES]) -> f32 {
    for width in [8, 4, 2, 1] {
        for lane in 0..width {
            acc[lane] += acc[lane + width];
        }
    }
    acc[0]
}

/// The canonical dot product. Panics if the lengths differ.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot: length mismatch");
    let mut acc = [0.0f32; LANES];
    let (a_chunks, a_tail) = a.as_chunks::<LANES>();
    let (b_chunks, b_tail) = b.as_chunks::<LANES>();
    // Independent lanes let LLVM vectorize without reassociating any additions.
    // Ordinary multiplication and addition must remain separate (no mul_add).
    for (a, b) in a_chunks.iter().zip(b_chunks) {
        for lane in 0..LANES {
            let product = a[lane] * b[lane];
            acc[lane] += product;
        }
    }
    for (lane, (&a, &b)) in a_tail.iter().zip(b_tail).enumerate() {
        let product = a * b;
        acc[lane] += product;
    }
    reduce(acc)
}

/// The canonical squared Euclidean distance. Panics if the lengths differ.
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "l2_squared: length mismatch");
    let mut acc = [0.0f32; LANES];
    let (a_chunks, a_tail) = a.as_chunks::<LANES>();
    let (b_chunks, b_tail) = b.as_chunks::<LANES>();
    for (a, b) in a_chunks.iter().zip(b_chunks) {
        for lane in 0..LANES {
            let difference = a[lane] - b[lane];
            let square = difference * difference;
            acc[lane] += square;
        }
    }
    for (lane, (&a, &b)) in a_tail.iter().zip(b_tail).enumerate() {
        let difference = a - b;
        let square = difference * difference;
        acc[lane] += square;
    }
    reduce(acc)
}

/// Checks length, finiteness and magnitude (components at most [`MAX_ABS_COMPONENT`]).
pub fn validate(v: &[f32], dim: usize) -> Result<(), VectorError> {
    if v.len() != dim {
        return Err(VectorError::Dimension {
            expected: dim,
            found: v.len(),
        });
    }
    for (index, &x) in v.iter().enumerate() {
        if !x.is_finite() {
            return Err(VectorError::NotFinite { index });
        }
        if x.abs() > MAX_ABS_COMPONENT {
            return Err(VectorError::TooLarge { index });
        }
    }
    Ok(())
}

/// The canonical normalization (see the module docs). Errors: [`VectorError::Zero`] when
/// the squared norm is 0, [`VectorError::NotNormalizable`] when it or a result component
/// is not finite.
pub fn normalize(v: &[f32]) -> Result<Vec<f32>, VectorError> {
    let squared = dot(v, v);
    if squared == 0.0 {
        return Err(VectorError::Zero);
    }
    if !squared.is_finite() {
        return Err(VectorError::NotNormalizable);
    }
    let norm = squared.sqrt();
    let normalized: Vec<f32> = v.iter().map(|x| x / norm).collect();
    if normalized.iter().any(|x| !x.is_finite()) {
        return Err(VectorError::NotNormalizable);
    }
    Ok(normalized)
}

/// Validates `v` and returns its stored form: normalized for cosine, as given otherwise.
/// Queries are prepared the same way.
pub fn prepare(metric: Metric, v: &[f32], dim: usize) -> Result<Vec<f32>, VectorError> {
    validate(v, dim)?;
    match metric {
        Metric::Cosine => normalize(v),
        Metric::Dot | Metric::L2 => Ok(v.to_vec()),
    }
}

/// The canonical score of a prepared query against a stored vector: `dot` for cosine and
/// dot, `-l2_squared` for L2.
pub fn score(metric: Metric, query: &[f32], stored: &[f32]) -> f32 {
    match metric {
        Metric::Cosine | Metric::Dot => dot(query, stored),
        Metric::L2 => -l2_squared(query, stored),
    }
}

/// A dot product in any summation order (for candidate generation only; its result may
/// differ from [`dot`] in the last bits). Panics if the lengths differ.
#[inline]
pub fn dot_fast(a: &[f32], b: &[f32]) -> f32 {
    // The canonical kernel already permits SIMD across its independent lanes.
    dot(a, b)
}

/// A squared L2 distance in any summation order (candidate generation only).
#[inline]
pub fn l2_squared_fast(a: &[f32], b: &[f32]) -> f32 {
    l2_squared(a, b)
}

/// [`score`] with the fast kernels (candidate generation only).
pub fn score_fast(metric: Metric, query: &[f32], stored: &[f32]) -> f32 {
    match metric {
        Metric::Cosine | Metric::Dot => dot_fast(query, stored),
        Metric::L2 => -l2_squared_fast(query, stored),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulation_rounds_products_and_uses_the_fixed_tree() {
        let mut a = [0.0; 17];
        let mut b = [0.0; 17];
        a[0] = -1.0;
        b[0] = 1.0;
        a[16] = 1.0 + f32::EPSILON;
        b[16] = 1.0 - f32::EPSILON;
        // The product at element 16 rounds to 1 before addition. Fusing would
        // instead retain a small negative residual in lane 0.
        assert_eq!(dot(&a, &b).to_bits(), 0.0f32.to_bits());

        let mut a = [0.0; LANES];
        a[0] = 1e20;
        a[1] = 3.0;
        a[8] = -1e20;
        // The width-8 step cancels the large terms before adding lane 1.
        assert_eq!(dot(&a, &[1.0; LANES]), 3.0);
    }

    #[test]
    fn validation_boundaries_and_error_precedence() {
        assert_eq!(
            validate(&[MAX_ABS_COMPONENT, -MAX_ABS_COMPONENT], 2),
            Ok(())
        );
        assert_eq!(validate(&[], 0), Ok(()));
        assert_eq!(
            validate(&[f32::NAN], 2),
            Err(VectorError::Dimension {
                expected: 2,
                found: 1
            })
        );
        assert_eq!(
            validate(&[2e15, f32::NAN], 2),
            Err(VectorError::TooLarge { index: 0 })
        );
        assert_eq!(
            validate(&[f32::INFINITY, 2e15], 2),
            Err(VectorError::NotFinite { index: 0 })
        );
        let a = vec![MAX_ABS_COMPONENT; 4096];
        let b = vec![-MAX_ABS_COMPONENT; 4096];
        assert!(dot(&a, &b).is_finite());
        assert!(l2_squared(&a, &b).is_finite());
    }

    #[test]
    fn normalization_handles_empty_underflow_nonfinite_and_signed_zero() {
        assert_eq!(normalize(&[]), Err(VectorError::Zero));
        assert_eq!(normalize(&[-0.0]), Err(VectorError::Zero));
        assert_eq!(normalize(&[f32::MIN_POSITIVE]), Err(VectorError::Zero));
        for value in [f32::MAX, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            assert_eq!(normalize(&[value]), Err(VectorError::NotNormalizable));
        }
        let normalized = normalize(&[-0.0, 1.0]).unwrap();
        assert_eq!(normalized[0].to_bits(), (-0.0f32).to_bits());
        assert_eq!(normalized[1], 1.0);
    }

    #[test]
    fn all_score_paths_reject_different_lengths() {
        for metric in [Metric::Cosine, Metric::Dot, Metric::L2] {
            assert!(std::panic::catch_unwind(|| score(metric, &[1.0], &[])).is_err());
            assert!(std::panic::catch_unwind(|| score(metric, &[], &[1.0])).is_err());
            assert!(std::panic::catch_unwind(|| score_fast(metric, &[1.0], &[])).is_err());
            assert!(std::panic::catch_unwind(|| score_fast(metric, &[], &[1.0])).is_err());
        }
    }
}
