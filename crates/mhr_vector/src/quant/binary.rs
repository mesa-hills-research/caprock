//! Centred binary quantization (unit U09).
//!
//! Codes contain little-endian sign words followed by two little-endian `f32`s:
//! the centred vector's norm and its inner product with the training centre. These
//! corrections retain both radial information and the offset needed by dot/cosine.
//! Queries use an asymmetric sign projection; graph construction uses the angular
//! estimate `cos(pi * Hamming / dim)`. Both estimates are on the metric's scale.

use crate::config::{CodeKind, Metric};
use crate::error::{Error, Result, VectorError};
use crate::format::{f32s_from_le, f32s_to_le};
use crate::quant::Codec;

const MAGIC: &[u8; 8] = b"MHRVBIN1";
const HEADER_LEN: usize = 16;

/// Binary codec parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryCodec {
    pub metric: Metric,
    pub dim: usize,
    /// Per-dimension centre, followed by the trained sign-projection scale.
    pub params: Vec<f32>,
    centre_norm2: f32,
    /// Cached Hamming-to-angle conversion, rebuilt when loading parameters.
    correlations: Vec<f32>,
}

/// A binary-prepared query.
#[derive(Debug, Clone)]
pub struct BinaryQuery {
    /// The query's sign bits, packed like codes.
    pub bits: Vec<u64>,
    /// Sixteen signed sums per four dimensions, for asymmetric lookup scoring.
    pub values: Vec<f32>,
    /// `q . centre` for dot/cosine, or the negated centred squared norm for L2.
    pub constant: f32,
}

impl BinaryCodec {
    /// Trains on `sample` (flattened stored-form vectors). Deterministic.
    pub fn train(metric: Metric, dim: usize, sample: &[f32], seed: u64) -> Result<BinaryCodec> {
        // No random choices: the entire supplied sample is reduced in row order.
        let _ = seed;
        if dim == 0 || dim > crate::MAX_DIM as usize {
            return Err(Error::InvalidArgument(format!(
                "binary dimension must be in 1..={}",
                crate::MAX_DIM
            )));
        }
        if !sample.len().is_multiple_of(dim) {
            return Err(Error::InvalidArgument(
                "binary training sample has a partial vector".into(),
            ));
        }
        let mut sums = vec![0.0f64; dim];
        for (index, &value) in sample.iter().enumerate() {
            if !value.is_finite() {
                return Err(VectorError::NotFinite { index }.into());
            }
            if value.abs() > 1e15 {
                return Err(VectorError::TooLarge { index }.into());
            }
            sums[index % dim] += f64::from(value);
        }
        let n = sample.len() / dim;
        let mut params: Vec<f32> = sums
            .into_iter()
            .map(|sum| (sum / n.max(1) as f64) as f32)
            .collect();

        // A sign projection loses magnitude. Calibrate its scale using the mean
        // ||r||_2 / ||r||_1: for an isotropic Gaussian this is approximately
        // sqrt(pi / (2 * dim)), but training also handles non-Gaussian residuals
        // and small dimensions. Accumulate in f64 so a constant sample centres
        // exactly and large valid inputs cannot overflow intermediate sums.
        let mut scale = 0.0f64;
        let mut nonzero = 0usize;
        for vector in sample.chunks_exact(dim) {
            let mut norm2 = 0.0;
            let mut l1 = 0.0;
            for (&x, &c) in vector.iter().zip(&params) {
                let r = f64::from(x) - f64::from(c);
                norm2 += r * r;
                l1 += r.abs();
            }
            if l1 > 0.0 {
                scale += norm2.sqrt() / l1;
                nonzero += 1;
            }
        }
        let scale = if nonzero == 0 {
            (std::f64::consts::FRAC_PI_2 / dim as f64)
                .sqrt()
                .min(1.0)
        } else {
            scale / nonzero as f64
        };
        params.push(scale as f32);
        Ok(Self::with_params(metric, dim, params))
    }

    /// The inverse of `to_bytes`.
    pub fn from_bytes(metric: Metric, dim: usize, bytes: &[u8]) -> Result<BinaryCodec> {
        let corrupt = || Error::corrupt("", "invalid binary codec parameters");
        if dim == 0
            || dim > crate::MAX_DIM as usize
            || bytes.len() != HEADER_LEN + (dim + 1) * 4
            || &bytes[..8] != MAGIC
            || bytes[8..12] != (dim as u32).to_le_bytes()
            || bytes[12..16] != metric_tag(metric).to_le_bytes()
        {
            return Err(corrupt());
        }
        let params = f32s_from_le(&bytes[HEADER_LEN..]).ok_or_else(corrupt)?;
        if params[..dim]
            .iter()
            .any(|x| !x.is_finite() || x.abs() > 1e15)
            || !params[dim].is_finite()
            || params[dim] <= 0.0
            || params[dim] > 1.0
        {
            return Err(corrupt());
        }
        Ok(Self::with_params(metric, dim, params))
    }

    fn with_params(metric: Metric, dim: usize, params: Vec<f32>) -> Self {
        let centre_norm2 = params[..dim]
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum::<f64>() as f32;
        let correlations = (0..=dim)
            .map(|h| (std::f64::consts::PI * h as f64 / dim as f64).cos() as f32)
            .collect();
        Self {
            metric,
            dim,
            params,
            centre_norm2,
            correlations,
        }
    }

    fn bit_bytes(&self) -> usize {
        self.dim.div_ceil(64) * 8
    }

    fn corrections(&self, code: &[u8]) -> (f32, f32) {
        let offset = self.bit_bytes();
        let norm = f32::from_le_bytes(code[offset..offset + 4].try_into().unwrap());
        let centre_dot = f32::from_le_bytes(code[offset + 4..offset + 8].try_into().unwrap());
        (norm, centre_dot)
    }
}

impl Codec for BinaryCodec {
    type Query = BinaryQuery;

    fn kind(&self) -> CodeKind {
        CodeKind::Binary
    }

    fn metric(&self) -> Metric {
        self.metric
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn code_len(&self) -> usize {
        self.bit_bytes() + 8
    }

    fn encode(&self, vector: &[f32], out: &mut [u8]) {
        assert_eq!(vector.len(), self.dim);
        assert_eq!(out.len(), self.code_len());
        out.fill(0);
        let mut norm2 = 0.0f64;
        let mut centre_dot = 0.0f64;
        for (j, (&x, &c)) in vector.iter().zip(&self.params).enumerate() {
            let r = f64::from(x) - f64::from(c);
            // Bit j is bit j % 64 of little-endian word j / 64. Padding is zero.
            if r >= 0.0 {
                out[j / 8] |= 1 << (j % 8);
            }
            norm2 += r * r;
            centre_dot += f64::from(c) * r;
        }
        let offset = self.bit_bytes();
        out[offset..offset + 4].copy_from_slice(&(norm2.sqrt() as f32).to_le_bytes());
        out[offset + 4..].copy_from_slice(&(centre_dot as f32).to_le_bytes());
    }

    fn prepare(&self, query: &[f32]) -> BinaryQuery {
        assert_eq!(query.len(), self.dim);
        let mut bits = vec![0; self.dim.div_ceil(64)];
        let mut residual = vec![0.0; self.dim.div_ceil(4) * 4];
        let mut norm2 = 0.0f64;
        let mut centre_dot = 0.0f64;
        for (j, (&x, &c)) in query.iter().zip(&self.params).enumerate() {
            let r = f64::from(x) - f64::from(c);
            residual[j] = r as f32;
            if r >= 0.0 {
                bits[j / 64] |= 1 << (j % 64);
            }
            norm2 += r * r;
            centre_dot += f64::from(x) * f64::from(c);
        }
        // Four-dimensional tables avoid a multiply and bit extraction per
        // dimension in the hot path, without a 256-entry table per byte.
        let mut values = Vec::with_capacity(residual.len() * 4);
        for chunk in residual.as_chunks::<4>().0 {
            for nibble in 0..16 {
                let mut sum = 0.0;
                for (j, &r) in chunk.iter().enumerate() {
                    sum += if nibble & (1 << j) == 0 { -r } else { r };
                }
                values.push(sum);
            }
        }
        BinaryQuery {
            bits,
            values,
            constant: match self.metric {
                Metric::Cosine | Metric::Dot => centre_dot as f32,
                Metric::L2 => -norm2 as f32,
            },
        }
    }

    fn score(&self, query: &BinaryQuery, code: &[u8]) -> f32 {
        let mut projection = 0.0;
        for (j, table) in query.values.as_chunks::<16>().0.iter().enumerate() {
            let nibble = (code[j / 2] >> ((j % 2) * 4)) & 15;
            projection += table[usize::from(nibble)];
        }
        let (norm, centre_dot) = self.corrections(code);
        let dot = projection * (norm * self.params[self.dim]);
        match self.metric {
            Metric::Cosine | Metric::Dot => query.constant + centre_dot + dot,
            Metric::L2 => query.constant - norm * norm + 2.0 * dot,
        }
    }

    fn score_codes(&self, a: &[u8], b: &[u8]) -> f32 {
        let bit_bytes = self.bit_bytes();
        let mut hamming = 0usize;
        for (j, (a, b)) in a[..bit_bytes]
            .as_chunks::<8>()
            .0
            .iter()
            .zip(b[..bit_bytes].as_chunks::<8>().0)
            .enumerate()
        {
            let mut different = u64::from_le_bytes(*a) ^ u64::from_le_bytes(*b);
            if j + 1 == bit_bytes / 8 && !self.dim.is_multiple_of(64) {
                different &= (1u64 << (self.dim % 64)) - 1;
            }
            hamming += different.count_ones() as usize;
        }
        let (norm_a, centre_a) = self.corrections(a);
        let (norm_b, centre_b) = self.corrections(b);
        let dot = (norm_a * norm_b) * self.correlations[hamming];
        // Group symmetric terms before adding the others, so swapping operands
        // cannot change the rounding (including the last bit).
        match self.metric {
            Metric::Cosine | Metric::Dot => {
                self.centre_norm2 + (centre_a + centre_b) + dot
            }
            Metric::L2 => -(norm_a * norm_a + norm_b * norm_b) + 2.0 * dot,
        }
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(HEADER_LEN + self.params.len() * 4);
        bytes.extend_from_slice(MAGIC);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(codec: &BinaryCodec, vector: &[f32]) -> Vec<u8> {
        // Nonzero output verifies that encode clears padding and old sign bits.
        let mut code = vec![0xff; codec.code_len()];
        codec.encode(vector, &mut code);
        code
    }

    #[test]
    fn signs_corrections_and_partial_words() {
        for dim in [1, 3, 4, 7, 8, 63, 64, 65, 127, 128, 129] {
            let centre: Vec<f32> = (0..dim).map(|j| j as f32 * 0.25).collect();
            let codec = BinaryCodec::train(Metric::Dot, dim, &centre, 0).unwrap();
            let vector: Vec<f32> = centre
                .iter()
                .enumerate()
                .map(|(j, c)| c + if j % 3 == 0 { -1.0 } else { 1.0 })
                .collect();
            let code = encode(&codec, &vector);
            for j in 0..codec.bit_bytes() * 8 {
                assert_eq!(code[j / 8] >> (j % 8) & 1 != 0, j < dim && j % 3 != 0);
            }
            let (norm, centre_dot) = codec.corrections(&code);
            assert_eq!(norm, (dim as f32).sqrt());
            let expected_dot: f32 = vector
                .iter()
                .zip(&centre)
                .map(|(x, c)| (x - c) * c)
                .sum();
            assert_eq!(centre_dot, expected_dot);

            // Ignoring the padding must also hold for the Hamming kernel.
            let mut dirty = code.clone();
            for j in dim..codec.bit_bytes() * 8 {
                dirty[j / 8] |= 1 << (j % 8);
            }
            assert_eq!(codec.score_codes(&code, &code), codec.score_codes(&code, &dirty));
            let q = codec.prepare(&vector);
            assert_eq!(codec.score(&q, &code), codec.score(&q, &dirty));
            for (j, &word) in q.bits.iter().enumerate() {
                assert_eq!(word, u64::from_le_bytes(code[j * 8..j * 8 + 8].try_into().unwrap()));
            }
        }
    }

    #[test]
    fn lookup_matches_scalar_asymmetric_estimate() {
        let dim = 65;
        let centre: Vec<f32> = (0..dim).map(|j| (j as f32 - 30.0) * 0.01).collect();
        let vector: Vec<f32> = (0..dim).map(|j| (j as f32 * 1.7).sin()).collect();
        let query: Vec<f32> = (0..dim).map(|j| (j as f32 * 2.3).cos()).collect();
        for metric in [Metric::Cosine, Metric::Dot, Metric::L2] {
            let codec = BinaryCodec::train(metric, dim, &centre, 11).unwrap();
            let code = encode(&codec, &vector);
            let prepared = codec.prepare(&query);
            let (norm, centre_dot) = codec.corrections(&code);
            let projection: f32 = (0..dim)
                .map(|j| {
                    let r = query[j] - centre[j];
                    if vector[j] >= centre[j] { r } else { -r }
                })
                .sum();
            let dot = projection * (norm * codec.params[dim]);
            let expected = match metric {
                Metric::Cosine | Metric::Dot => prepared.constant + centre_dot + dot,
                Metric::L2 => prepared.constant - norm * norm + 2.0 * dot,
            };
            assert!((codec.score(&prepared, &code) - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn metric_scale_in_one_dimension() {
        for metric in [Metric::Cosine, Metric::Dot, Metric::L2] {
            let codec = BinaryCodec::train(metric, 1, &[-2.0, 4.0], 0).unwrap();
            assert_eq!(codec.params, [1.0, 1.0]);
            for x in [-4.0, 0.0, 1.0, 7.0] {
                for y in [-3.0, 0.0, 1.0, 9.0] {
                    let expected = match metric {
                        Metric::Cosine | Metric::Dot => x * y,
                        Metric::L2 => -(x - y) * (x - y),
                    };
                    let a = encode(&codec, &[x]);
                    let b = encode(&codec, &[y]);
                    assert_eq!(codec.score(&codec.prepare(&[x]), &b), expected);
                    assert_eq!(codec.score_codes(&a, &b), expected);
                }
            }
        }
    }

    #[test]
    fn training_refuses_invalid_inputs() {
        assert!(BinaryCodec::train(Metric::Dot, 0, &[], 0).is_err());
        assert!(BinaryCodec::train(Metric::Dot, usize::MAX, &[], 0).is_err());
        assert!(BinaryCodec::train(Metric::Dot, 2, &[0.0], 0).is_err());
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 2e15] {
            assert!(BinaryCodec::train(Metric::Dot, 1, &[bad], 0).is_err());
        }
    }

    #[test]
    fn loading_checks_header_length_and_parameters() {
        let codec = BinaryCodec::train(Metric::Dot, 2, &[1.0, 2.0, 3.0, 4.0], 0).unwrap();
        let bytes = codec.to_bytes();
        assert_eq!(BinaryCodec::from_bytes(Metric::Dot, 2, &bytes).unwrap(), codec);
        for end in 0..bytes.len() {
            assert!(matches!(
                BinaryCodec::from_bytes(Metric::Dot, 2, &bytes[..end]),
                Err(Error::Corrupt { .. })
            ));
        }
        assert!(BinaryCodec::from_bytes(Metric::Cosine, 2, &bytes).is_err());
        assert!(BinaryCodec::from_bytes(Metric::Dot, 1, &bytes).is_err());
        assert!(BinaryCodec::from_bytes(Metric::Dot, 0, &bytes).is_err());
        assert!(BinaryCodec::from_bytes(Metric::Dot, usize::MAX, &bytes).is_err());
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(BinaryCodec::from_bytes(Metric::Dot, 2, &extra).is_err());
        for offset in 0..HEADER_LEN {
            let mut bad = bytes.clone();
            bad[offset] ^= 1;
            assert!(BinaryCodec::from_bytes(Metric::Dot, 2, &bad).is_err());
        }
        for (offset, values) in [
            (HEADER_LEN, vec![f32::NAN, f32::INFINITY, 2e15]),
            (HEADER_LEN + 8, vec![f32::NAN, f32::INFINITY, 0.0, -1.0, 2.0]),
        ] {
            for value in values {
                let mut bad = bytes.clone();
                bad[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
                assert!(BinaryCodec::from_bytes(Metric::Dot, 2, &bad).is_err());
            }
        }
    }

    #[test]
    fn empty_zero_constant_and_large_inputs_stay_finite() {
        for metric in [Metric::Cosine, Metric::Dot, Metric::L2] {
            for dim in [1, 65, crate::MAX_DIM as usize] {
                for value in [0.0, 0.25, 1e15, -1e15] {
                    let vector = vec![value; dim];
                    for sample in [&[][..], &vector[..]] {
                        let codec = BinaryCodec::train(metric, dim, sample, 0).unwrap();
                        let code = encode(&codec, &vector);
                        let q = codec.prepare(&vector);
                        assert!(codec.score(&q, &code).is_finite());
                        assert!(codec.score_codes(&code, &code).is_finite());
                        assert_eq!(BinaryCodec::from_bytes(metric, dim, &codec.to_bytes()).unwrap(), codec);
                    }
                }
            }
        }
    }
}
