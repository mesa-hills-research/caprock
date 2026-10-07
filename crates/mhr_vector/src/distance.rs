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

/// The canonical dot product. Panics if the lengths differ.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    todo!("U01 distance: dot")
}

/// The canonical squared Euclidean distance. Panics if the lengths differ.
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    todo!("U01 distance: l2_squared")
}

/// Checks length, finiteness and magnitude (components at most [`MAX_ABS_COMPONENT`]).
pub fn validate(v: &[f32], dim: usize) -> Result<(), VectorError> {
    todo!("U01 distance: validate")
}

/// The canonical normalization (see the module docs). Errors: [`VectorError::Zero`] when
/// the squared norm is 0, [`VectorError::NotNormalizable`] when it or a result component
/// is not finite.
pub fn normalize(v: &[f32]) -> Result<Vec<f32>, VectorError> {
    todo!("U01 distance: normalize")
}

/// Validates `v` and returns its stored form: normalized for cosine, as given otherwise.
/// Queries are prepared the same way.
pub fn prepare(metric: Metric, v: &[f32], dim: usize) -> Result<Vec<f32>, VectorError> {
    todo!("U01 distance: prepare")
}

/// The canonical score of a prepared query against a stored vector: `dot` for cosine and
/// dot, `-l2_squared` for L2.
pub fn score(metric: Metric, query: &[f32], stored: &[f32]) -> f32 {
    todo!("U01 distance: score")
}

/// A dot product in any summation order (for candidate generation only; its result may
/// differ from [`dot`] in the last bits). Panics if the lengths differ.
pub fn dot_fast(a: &[f32], b: &[f32]) -> f32 {
    todo!("U01 distance: dot_fast")
}

/// A squared L2 distance in any summation order (candidate generation only).
pub fn l2_squared_fast(a: &[f32], b: &[f32]) -> f32 {
    todo!("U01 distance: l2_squared_fast")
}

/// [`score`] with the fast kernels (candidate generation only).
pub fn score_fast(metric: Metric, query: &[f32], stored: &[f32]) -> f32 {
    todo!("U01 distance: score_fast")
}
