//! Canonical score arithmetic (REQUIREMENTS.md, "Scores").
//!
//! The canonical dot product keeps 16 `f32` lane accumulators. Element `i` is added to
//! lane `i % 16`, in increasing `i`, as `acc = acc + (a[i] * b[i])` with the product and
//! the sum each rounded to `f32` (no fused multiply-add). The lanes are then reduced in a
//! fixed tree: for `w` in 8, 4, 2, 1, `acc[l] = acc[l] + acc[l + w]` for `l < w`; the
//! result is `acc[0]`. Squared L2 is the same with `d = a[i] - b[i]` and `d * d`.
//!
//! A SIMD implementation that processes 16 elements per step (one AVX-512 register, two
//! AVX2 registers, four SSE or NEON registers) in this lane order produces the same bits,
//! which is why the order was chosen: fast and exactly reproducible.

use serde::{Deserialize, Serialize};

/// Number of lane accumulators in the canonical kernels.
pub const LANES: usize = 16;

/// Largest absolute value a vector component may have. With it no score can overflow:
/// `(2e15)^2 * 4096 < f32::MAX`.
pub const MAX_ABS_COMPONENT: f32 = 1e15;

/// The similarity a store ranks by. Scores are "higher is better" for every metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    /// Cosine similarity: vectors (and queries) are normalized with [`normalize`]; the
    /// score is the canonical dot product of the normalized vectors.
    Cosine,
    /// Inner product of the vectors as given.
    Dot,
    /// Negated squared Euclidean distance: `-l2_squared(q, x)`.
    L2,
}

impl Metric {
    /// The name used in dataset files and reports: `cosine`, `dot`, `l2`.
    pub fn name(self) -> &'static str {
        match self {
            Metric::Cosine => "cosine",
            Metric::Dot => "dot",
            Metric::L2 => "l2",
        }
    }

    /// Parses [`Metric::name`] (and ann-benchmarks' `angular` and `euclidean`).
    pub fn parse(text: &str) -> Option<Metric> {
        match text {
            "cosine" | "angular" => Some(Metric::Cosine),
            "dot" | "ip" => Some(Metric::Dot),
            "l2" | "euclidean" => Some(Metric::L2),
            _ => None,
        }
    }
}

/// Why a vector was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VectorProblem {
    /// Wrong number of components.
    Dimension { expected: usize, found: usize },
    /// A component is NaN or infinite.
    NotFinite { index: usize },
    /// A component's magnitude exceeds [`MAX_ABS_COMPONENT`].
    TooLarge { index: usize },
    /// Cosine: the vector's canonical squared norm is zero.
    Zero,
    /// Cosine: the normalized vector has a non-finite component.
    NotNormalizable,
}

impl std::fmt::Display for VectorProblem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorProblem::Dimension { expected, found } => {
                write!(f, "expected {expected} dimensions, found {found}")
            }
            VectorProblem::NotFinite { index } => write!(f, "component {index} is not finite"),
            VectorProblem::TooLarge { index } => {
                write!(
                    f,
                    "component {index} exceeds {MAX_ABS_COMPONENT:e} in magnitude"
                )
            }
            VectorProblem::Zero => write!(f, "zero vector (cosine needs a direction)"),
            VectorProblem::NotNormalizable => write!(f, "vector cannot be normalized"),
        }
    }
}

fn reduce(mut acc: [f32; LANES]) -> f32 {
    let mut width = LANES / 2;
    while width > 0 {
        for lane in 0..width {
            acc[lane] += acc[lane + width];
        }
        width /= 2;
    }
    acc[0]
}

/// The canonical dot product. Panics if the lengths differ.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot: length mismatch");
    let mut acc = [0.0f32; LANES];
    for i in 0..a.len() {
        let product = a[i] * b[i];
        acc[i % LANES] += product;
    }
    reduce(acc)
}

/// The canonical squared Euclidean distance. Panics if the lengths differ.
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "l2_squared: length mismatch");
    let mut acc = [0.0f32; LANES];
    for i in 0..a.len() {
        let difference = a[i] - b[i];
        let square = difference * difference;
        acc[i % LANES] += square;
    }
    reduce(acc)
}

/// Checks the length, finiteness and magnitude of a vector.
pub fn validate(v: &[f32], dim: usize) -> Result<(), VectorProblem> {
    if v.len() != dim {
        return Err(VectorProblem::Dimension {
            expected: dim,
            found: v.len(),
        });
    }
    for (index, x) in v.iter().enumerate() {
        if !x.is_finite() {
            return Err(VectorProblem::NotFinite { index });
        }
        if x.abs() > MAX_ABS_COMPONENT {
            return Err(VectorProblem::TooLarge { index });
        }
    }
    Ok(())
}

/// The canonical normalization: `norm = sqrt(dot(v, v))` (in `f32`), then each component
/// is `v[i] / norm`.
pub fn normalize(v: &[f32]) -> Result<Vec<f32>, VectorProblem> {
    let squared = dot(v, v);
    if squared == 0.0 {
        return Err(VectorProblem::Zero);
    }
    if !squared.is_finite() {
        return Err(VectorProblem::NotNormalizable);
    }
    let norm = squared.sqrt();
    let out: Vec<f32> = v.iter().map(|x| x / norm).collect();
    if out.iter().any(|x| !x.is_finite()) {
        return Err(VectorProblem::NotNormalizable);
    }
    Ok(out)
}

/// Validates `v` and puts it in stored form: normalized for cosine, as given otherwise.
/// Queries are prepared the same way.
pub fn prepare(metric: Metric, v: &[f32], dim: usize) -> Result<Vec<f32>, VectorProblem> {
    validate(v, dim)?;
    match metric {
        Metric::Cosine => normalize(v),
        Metric::Dot | Metric::L2 => Ok(v.to_vec()),
    }
}

/// The canonical score of a prepared query against a stored (prepared) vector.
pub fn score(metric: Metric, query: &[f32], stored: &[f32]) -> f32 {
    match metric {
        Metric::Cosine | Metric::Dot => dot(query, stored),
        Metric::L2 => -l2_squared(query, stored),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_cases() {
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
        assert_eq!(l2_squared(&[1.0, 2.0], &[3.0, 5.0]), 13.0);
        assert_eq!(dot(&[], &[]), 0.0);
        let n = normalize(&[3.0, 4.0]).unwrap();
        assert_eq!(n, vec![0.6, 0.8]);
        assert_eq!(normalize(&[0.0, 0.0]), Err(VectorProblem::Zero));
        assert_eq!(
            validate(&[1.0, f32::NAN], 2),
            Err(VectorProblem::NotFinite { index: 1 })
        );
        assert_eq!(
            validate(&[1.0], 2),
            Err(VectorProblem::Dimension {
                expected: 2,
                found: 1
            })
        );
        assert_eq!(
            validate(&[2e15], 1),
            Err(VectorProblem::TooLarge { index: 0 })
        );
    }

    #[test]
    fn lane_order_is_the_documented_one() {
        // 17 elements: lane 0 gets elements 0 and 16, lanes 1..15 one each.
        let a: Vec<f32> = (0..17).map(|i| 1.0 + i as f32 * 1e-3).collect();
        let b: Vec<f32> = (0..17).map(|i| 1.0 - i as f32 * 7e-4).collect();
        let mut acc = [0.0f32; LANES];
        for i in 0..17 {
            acc[i % LANES] += a[i] * b[i];
        }
        let mut w = 8;
        while w > 0 {
            for l in 0..w {
                acc[l] += acc[l + w];
            }
            w /= 2;
        }
        assert_eq!(dot(&a, &b).to_bits(), acc[0].to_bits());
    }

    #[test]
    fn large_components_never_overflow() {
        let dim = crate::MAX_DIM;
        let a = vec![MAX_ABS_COMPONENT; dim];
        let b = vec![-MAX_ABS_COMPONENT; dim];
        assert!(dot(&a, &b).is_finite());
        assert!(l2_squared(&a, &b).is_finite());
    }
}
