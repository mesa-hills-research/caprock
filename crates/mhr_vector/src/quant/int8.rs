//! Int8 scalar quantization (unit U08).
//!
//! One signed or unsigned byte per dimension, with whatever per-dimension or per-vector
//! parameters the implementation keeps (ranges trained on the sample, clipping at
//! quantiles, per-vector norms or correction terms). Requirements (REQUIREMENTS.md,
//! "Quantization"): deterministic training; `score` within the accuracy bound of the
//! acceptance tests; `score_codes` symmetric; codes of `dim` bytes plus at most 8 bytes
//! of per-vector terms.

use crate::config::{CodeKind, Metric};
use crate::error::Result;
use crate::quant::Codec;

/// Int8 codec parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct Int8Codec {
    pub metric: Metric,
    pub dim: usize,
    /// Implementation-defined parameters (for example per-dimension offset and scale).
    pub params: Vec<f32>,
}

/// An int8-prepared query.
#[derive(Debug, Clone)]
pub struct Int8Query {
    /// Implementation-defined (for example the scaled query and a constant term).
    pub values: Vec<f32>,
    pub quantized: Vec<i8>,
    pub constant: f32,
}

impl Int8Codec {
    /// Trains on `sample` (flattened stored-form vectors). Deterministic.
    pub fn train(metric: Metric, dim: usize, sample: &[f32], seed: u64) -> Result<Int8Codec> {
        todo!("U08 int8: Int8Codec::train")
    }

    /// The inverse of `to_bytes`.
    pub fn from_bytes(metric: Metric, dim: usize, bytes: &[u8]) -> Result<Int8Codec> {
        todo!("U08 int8: Int8Codec::from_bytes")
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
        todo!("U08 int8: code_len")
    }

    fn encode(&self, vector: &[f32], out: &mut [u8]) {
        todo!("U08 int8: encode")
    }

    fn prepare(&self, query: &[f32]) -> Int8Query {
        todo!("U08 int8: prepare")
    }

    fn score(&self, query: &Int8Query, code: &[u8]) -> f32 {
        todo!("U08 int8: score")
    }

    fn score_codes(&self, a: &[u8], b: &[u8]) -> f32 {
        todo!("U08 int8: score_codes")
    }

    fn to_bytes(&self) -> Vec<u8> {
        todo!("U08 int8: to_bytes")
    }
}
