//! Int8 scalar quantization (unit U08).
//!
//! Each dimension has a trained minimum and a step size spanning its observed range
//! with 256 unsigned byte values. Codes contain exactly one byte per dimension.
//! Queries are not quantized: dot/cosine queries fold the affine decoder into their
//! weights, while L2 queries subtract the decoder's offset. Graph construction uses
//! symmetric scores between decoded codes. All kernels are allocation-free.

use crate::config::{CodeKind, Metric};
use crate::error::{Error, Result, VectorError};
use crate::format::{f32s_from_le, f32s_to_le};
use crate::quant::Codec;

const MAGIC: [u8; 8] = *b"MHRVI8\0\x01";
const HEADER_LEN: usize = 16;
const MAX_COMPONENT: f32 = 1e15;

/// Int8 codec parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct Int8Codec {
    pub metric: Metric,
    pub dim: usize,
    /// `dim` minima followed by `dim` nonnegative step sizes. A zero step denotes
    /// a constant dimension, which decodes to its minimum for every byte value.
    pub params: Vec<f32>,
}

/// An int8-prepared query.
#[derive(Debug, Clone)]
pub struct Int8Query {
    /// Dot/cosine: query times step. L2: query minus minimum.
    pub values: Vec<f32>,
    /// Dot/cosine: dot(query, minima). L2: zero (distances are evaluated directly).
    pub constant: f32,
}

impl Int8Codec {
    /// Trains on `sample` (flattened stored-form vectors). Deterministic.
    pub fn train(metric: Metric, dim: usize, sample: &[f32], seed: u64) -> Result<Int8Codec> {
        // There is no stochastic fitting or sampling here. The caller has already
        // selected the training rows; every row contributes to the range.
        let _ = seed;
        if dim == 0 || dim > crate::MAX_DIM as usize {
            return Err(Error::InvalidArgument(format!(
                "int8 dimension must be in 1..={}",
                crate::MAX_DIM
            )));
        }
        if !sample.len().is_multiple_of(dim) {
            return Err(Error::InvalidArgument(
                "int8 training sample contains an incomplete vector".into(),
            ));
        }
        let mut minima = vec![f32::INFINITY; dim];
        let mut maxima = vec![f32::NEG_INFINITY; dim];
        for row in sample.chunks_exact(dim) {
            for (i, &value) in row.iter().enumerate() {
                if !value.is_finite() {
                    return Err(VectorError::NotFinite { index: i }.into());
                }
                if value.abs() > MAX_COMPONENT {
                    return Err(VectorError::TooLarge { index: i }.into());
                }
                minima[i] = minima[i].min(value);
                maxima[i] = maxima[i].max(value);
            }
        }
        // An empty segment has no distribution to learn. Keep a well-defined
        // default range rather than saving infinities in its codec parameters.
        if sample.is_empty() {
            minima.fill(-1.0);
            maxima.fill(1.0);
        }
        let mut params = minima;
        for i in 0..dim {
            // The wider intermediate also handles extremely narrow ranges. A step
            // that underflows to zero is safely treated as a constant dimension.
            params.push(((maxima[i] as f64 - params[i] as f64) / 255.0) as f32);
        }
        Ok(Int8Codec {
            metric,
            dim,
            params,
        })
    }

    /// The inverse of `to_bytes`.
    pub fn from_bytes(metric: Metric, dim: usize, bytes: &[u8]) -> Result<Int8Codec> {
        let corrupt = || Error::corrupt("", "invalid int8 codec parameters");
        // Bound dimensions before doing arithmetic or allocating from file data.
        if dim == 0 || dim > crate::MAX_DIM as usize || bytes.len() != HEADER_LEN + 8 * dim {
            return Err(corrupt());
        }
        if bytes[..8] != MAGIC
            || bytes[8..12] != (dim as u32).to_le_bytes()
            || bytes[12..16] != metric_tag(metric).to_le_bytes()
        {
            return Err(corrupt());
        }
        let params = f32s_from_le(&bytes[HEADER_LEN..]).ok_or_else(corrupt)?;
        for (&minimum, &step) in params[..dim].iter().zip(&params[dim..]) {
            // Allow a few ulps of endpoint rounding from the trained f32 step.
            let limit = MAX_COMPONENT as f64 * (1.0 + 4.0 * f32::EPSILON as f64);
            let maximum = minimum as f64 + 255.0 * step as f64;
            if !minimum.is_finite()
                || minimum.abs() > MAX_COMPONENT
                || !step.is_finite()
                || step < 0.0
                || maximum.abs() > limit
            {
                return Err(corrupt());
            }
        }
        Ok(Int8Codec {
            metric,
            dim,
            params,
        })
    }
}

impl Codec for Int8Codec {
    type Query = Int8Query;

    fn kind(&self) -> CodeKind {
        CodeKind::Int8
    }

    fn metric(&self) -> Metric {
        self.metric
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn code_len(&self) -> usize {
        self.dim
    }

    fn encode(&self, vector: &[f32], out: &mut [u8]) {
        assert_eq!(vector.len(), self.dim, "int8 vector dimension mismatch");
        assert_eq!(out.len(), self.code_len(), "int8 code length mismatch");
        let (minima, steps) = self.params.split_at(self.dim);
        for i in 0..self.dim {
            out[i] = if steps[i] == 0.0 {
                0
            } else {
                // Float-to-u8 casts saturate, including for values outside the
                // training range. Do not wrap a negative or outlying component.
                ((vector[i] - minima[i]) / steps[i]).round() as u8
            };
        }
    }

    fn prepare(&self, query: &[f32]) -> Int8Query {
        assert_eq!(query.len(), self.dim, "int8 query dimension mismatch");
        let (minima, steps) = self.params.split_at(self.dim);
        match self.metric {
            Metric::Cosine | Metric::Dot => Int8Query {
                values: query.iter().zip(steps).map(|(q, step)| q * step).collect(),
                constant: sum_terms(self.dim, |i| query[i] * minima[i]),
            },
            Metric::L2 => Int8Query {
                values: query.iter().zip(minima).map(|(q, min)| q - min).collect(),
                constant: 0.0,
            },
        }
    }

    fn score(&self, query: &Int8Query, code: &[u8]) -> f32 {
        assert_eq!(
            query.values.len(),
            self.dim,
            "int8 query dimension mismatch"
        );
        assert_eq!(code.len(), self.code_len(), "int8 code length mismatch");
        match self.metric {
            Metric::Cosine | Metric::Dot => {
                query.constant + sum_terms(self.dim, |i| query.values[i] * code[i] as f32)
            }
            Metric::L2 => {
                let steps = &self.params[self.dim..];
                // Compute squared differences, not q^2 + x^2 - 2*q*x, to avoid
                // cancellation when the two vectors are very close.
                -sum_terms(self.dim, |i| {
                    let delta = query.values[i] - steps[i] * code[i] as f32;
                    delta * delta
                })
            }
        }
    }

    fn score_codes(&self, a: &[u8], b: &[u8]) -> f32 {
        assert_eq!(a.len(), self.code_len(), "int8 code length mismatch");
        assert_eq!(b.len(), self.code_len(), "int8 code length mismatch");
        let (minima, steps) = self.params.split_at(self.dim);
        match self.metric {
            Metric::Cosine | Metric::Dot => sum_terms(self.dim, |i| {
                let x = minima[i] + steps[i] * a[i] as f32;
                let y = minima[i] + steps[i] * b[i] as f32;
                x * y
            }),
            Metric::L2 => -sum_terms(self.dim, |i| {
                // Offsets cancel. Subtract bytes before scaling to preserve small
                // distances even when the offset is much larger than the step.
                let delta = (a[i] as f32 - b[i] as f32) * steps[i];
                delta * delta
            }),
        }
    }

    fn to_bytes(&self) -> Vec<u8> {
        // Versioned header, dimension and metric, then minima and steps as LE f32.
        // Section-file checksums protect this payload when saved in a segment.
        let mut bytes = Vec::with_capacity(HEADER_LEN + 8 * self.dim);
        bytes.extend_from_slice(&MAGIC);
        bytes.extend_from_slice(&(self.dim as u32).to_le_bytes());
        bytes.extend_from_slice(&metric_tag(self.metric).to_le_bytes());
        bytes.extend_from_slice(&f32s_to_le(&self.params));
        bytes
    }
}

fn metric_tag(metric: Metric) -> u32 {
    match metric {
        Metric::Cosine => 0,
        Metric::Dot => 1,
        Metric::L2 => 2,
    }
}

/// Independent lanes make the small byte-to-f32 kernels amenable to vectorization,
/// while a fixed reduction order keeps estimates deterministic on every thread.
#[inline]
fn sum_terms(dim: usize, term: impl Fn(usize) -> f32) -> f32 {
    let mut lanes = [0.0f32; 16];
    let end = dim / 16 * 16;
    for start in (0..end).step_by(16) {
        for (lane, acc) in lanes.iter_mut().enumerate() {
            *acc += term(start + lane);
        }
    }
    for (lane, acc) in lanes.iter_mut().take(dim - end).enumerate() {
        *acc += term(end + lane);
    }
    for width in [8, 4, 2, 1] {
        for lane in 0..width {
            lanes[lane] += lanes[lane + width];
        }
    }
    lanes[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    const METRICS: [Metric; 3] = [Metric::Cosine, Metric::Dot, Metric::L2];

    fn encode(codec: &Int8Codec, vector: &[f32]) -> Vec<u8> {
        let mut code = vec![0; codec.code_len()];
        codec.encode(vector, &mut code);
        code
    }

    #[test]
    fn ranges_rounding_clipping_and_constant_dimensions() {
        let sample = [-5.0, 7.0, -2.0, 250.0, 7.0, 508.0];
        let codec = Int8Codec::train(Metric::Dot, 3, &sample, 0).unwrap();
        assert_eq!(codec.params, [-5.0, 7.0, -2.0, 1.0, 0.0, 2.0]);
        assert_eq!(encode(&codec, &sample[..3]), [0, 0, 0]);
        assert_eq!(encode(&codec, &sample[3..]), [255, 0, 255]);
        assert_eq!(encode(&codec, &[-4.5, 200.0, 1.0]), [1, 0, 2]);
        assert_eq!(encode(&codec, &[-10.0, -200.0, 1000.0]), [0, 0, 255]);
        // No random fitting: changes to seed or training row order do not matter.
        assert_eq!(
            codec,
            Int8Codec::train(Metric::Dot, 3, &sample, 999).unwrap()
        );
        let reversed = [&sample[3..], &sample[..3]].concat();
        assert_eq!(
            codec,
            Int8Codec::train(Metric::Dot, 3, &reversed, 0).unwrap()
        );
    }

    #[test]
    fn score_scale_and_tail_dimensions() {
        for dim in [1, 3, 15, 16, 17, 31, 32, 33, crate::MAX_DIM as usize] {
            let a = vec![-2.0; dim];
            let b = vec![253.0; dim];
            for metric in METRICS {
                let codec =
                    Int8Codec::train(metric, dim, &[a.clone(), b.clone()].concat(), 0).unwrap();
                let ca = encode(&codec, &a);
                let cb = encode(&codec, &b);
                let expected = match metric {
                    Metric::Cosine | Metric::Dot => -506.0 * dim as f32,
                    Metric::L2 => -65025.0 * dim as f32,
                };
                assert_eq!(codec.score(&codec.prepare(&a), &cb), expected);
                assert_eq!(codec.score(&codec.prepare(&b), &ca), expected);
                assert_eq!(codec.score_codes(&ca, &cb), expected);
                assert_eq!(codec.score_codes(&cb, &ca), expected);
            }
        }
    }

    #[test]
    fn l2_small_distances_with_large_offsets() {
        let sample = [10_000_000.0, 10_000_255.0];
        let codec = Int8Codec::train(Metric::L2, 1, &sample, 0).unwrap();
        let query = codec.prepare(&[10_000_001.0]);
        let code = encode(&codec, &[10_000_002.0]);
        assert_eq!(codec.score(&query, &code), -1.0);
        assert_eq!(codec.score_codes(&[1], &code), -1.0);
        assert_eq!(codec.score_codes(&code, &code), 0.0);
    }

    #[test]
    fn empty_zero_subnormal_and_extreme_samples_stay_finite() {
        let tiny = f32::from_bits(1);
        for metric in METRICS {
            for sample in [
                vec![],
                vec![0.0; 6],
                vec![-tiny, 0.0, tiny, tiny, 0.0, -tiny],
                vec![-MAX_COMPONENT; 3],
                vec![MAX_COMPONENT; 3],
                vec![
                    -MAX_COMPONENT,
                    MAX_COMPONENT,
                    0.0,
                    MAX_COMPONENT,
                    -MAX_COMPONENT,
                    MAX_COMPONENT,
                ],
            ] {
                let codec = Int8Codec::train(metric, 3, &sample, 0).unwrap();
                let restored = Int8Codec::from_bytes(metric, 3, &codec.to_bytes()).unwrap();
                assert_eq!(restored, codec);
                for vector in sample
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .chain(std::iter::once(&[0.0; 3]))
                {
                    let code = encode(&codec, vector);
                    let prepared = codec.prepare(vector);
                    assert!(codec.score(&prepared, &code).is_finite());
                    assert!(codec.score_codes(&code, &code).is_finite());
                    assert_eq!(
                        codec.score(&prepared, &code),
                        restored.score(&prepared, &code)
                    );
                }
            }
        }
    }

    #[test]
    fn training_rejects_invalid_inputs() {
        for dim in [0, crate::MAX_DIM as usize + 1, usize::MAX] {
            assert!(Int8Codec::train(Metric::Dot, dim, &[], 0).is_err());
        }
        assert!(Int8Codec::train(Metric::Dot, 2, &[1.0], 0).is_err());
        for value in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            2.0 * MAX_COMPONENT,
        ] {
            assert!(matches!(
                Int8Codec::train(Metric::Dot, 1, &[value], 0),
                Err(Error::InvalidVector(_))
            ));
        }
    }

    #[test]
    fn serialized_parameters_reject_malformed_data() {
        let codec = Int8Codec::train(Metric::Dot, 2, &[1.0, 2.0, 3.0, 4.0], 0).unwrap();
        let bytes = codec.to_bytes();
        let rejects = |bytes: &[u8]| {
            assert!(matches!(
                Int8Codec::from_bytes(Metric::Dot, 2, bytes),
                Err(Error::Corrupt { path, .. }) if path.as_os_str().is_empty()
            ));
        };
        for end in 0..bytes.len() {
            rejects(&bytes[..end]);
        }
        let mut appended = bytes.clone();
        appended.push(0);
        rejects(&appended);
        for i in 0..HEADER_LEN {
            let mut damaged = bytes.clone();
            damaged[i] ^= 0xff;
            rejects(&damaged);
        }
        for dim in [0, 1, 3, usize::MAX] {
            assert!(Int8Codec::from_bytes(Metric::Dot, dim, &bytes).is_err());
        }
        for metric in [Metric::Cosine, Metric::L2] {
            assert!(Int8Codec::from_bytes(metric, 2, &bytes).is_err());
        }
        for offset in (HEADER_LEN..bytes.len()).step_by(4) {
            for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
                let mut damaged = bytes.clone();
                damaged[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
                rejects(&damaged);
            }
        }
        let mut negative_step = bytes;
        negative_step[HEADER_LEN + 8..HEADER_LEN + 12].copy_from_slice(&(-1.0f32).to_le_bytes());
        rejects(&negative_step);
    }
}
