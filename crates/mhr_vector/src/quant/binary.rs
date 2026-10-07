//! Binary quantization (unit U09).
//!
//! One bit per dimension (the sign of the stored vector's component after an optional
//! trained centring), packed little-endian into `u64` words, plus at most 8 bytes of
//! per-vector correction terms (for example the vector's norm, or RaBitQ-style factors).
//! Requirements (REQUIREMENTS.md, "Quantization"): deterministic training; `score`
//! monotone enough for the acceptance tests' rank-correlation bound; `score_codes`
//! symmetric; Hamming-style kernels over `u64` words (`count_ones`).

use crate::config::{CodeKind, Metric};
use crate::error::Result;
use crate::quant::Codec;

/// Binary codec parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryCodec {
    pub metric: Metric,
    pub dim: usize,
    /// Implementation-defined parameters (for example a per-dimension centre).
    pub params: Vec<f32>,
}

/// A binary-prepared query.
#[derive(Debug, Clone)]
pub struct BinaryQuery {
    /// The query's sign bits, packed like codes.
    pub bits: Vec<u64>,
    /// Implementation-defined (for example the query itself for asymmetric estimates).
    pub values: Vec<f32>,
    pub constant: f32,
}

impl BinaryCodec {
    /// Trains on `sample` (flattened stored-form vectors). Deterministic.
    pub fn train(metric: Metric, dim: usize, sample: &[f32], seed: u64) -> Result<BinaryCodec> {
        todo!("U09 binary: BinaryCodec::train")
    }

    /// The inverse of `to_bytes`.
    pub fn from_bytes(metric: Metric, dim: usize, bytes: &[u8]) -> Result<BinaryCodec> {
        todo!("U09 binary: BinaryCodec::from_bytes")
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
        todo!("U09 binary: code_len")
    }

    fn encode(&self, vector: &[f32], out: &mut [u8]) {
        todo!("U09 binary: encode")
    }

    fn prepare(&self, query: &[f32]) -> BinaryQuery {
        todo!("U09 binary: prepare")
    }

    fn score(&self, query: &BinaryQuery, code: &[u8]) -> f32 {
        todo!("U09 binary: score")
    }

    fn score_codes(&self, a: &[u8], b: &[u8]) -> f32 {
        todo!("U09 binary: score_codes")
    }

    fn to_bytes(&self) -> Vec<u8> {
        todo!("U09 binary: to_bytes")
    }
}
