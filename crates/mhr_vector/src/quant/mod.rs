//! Codecs: the compact codes segments keep in memory to find candidates (unit U07 for this
//! file; U08 int8; U09 binary; REQUIREMENTS.md, "Quantization").
//!
//! A codec is trained once per segment, deterministically, on a sample of the segment's
//! stored vectors ([`training_sample`]); its parameters are saved in the segment. Codes
//! give **estimates** of the canonical score (higher is better, same scale as the
//! metric's score); final results are rescored with full-precision vectors when the index
//! keeps them.
//!
//! Hot loops are generic over [`Codec`]; [`AnyCodec`] holds a segment's codec and
//! [`with_codec!`] dispatches once per segment search, not once per vector.

pub mod binary;
pub mod int8;

pub use binary::BinaryCodec;
pub use int8::Int8Codec;

use crate::config::{CodeKind, Metric};
use crate::error::Result;

/// Most vectors a codec trains on.
pub const MAX_TRAINING_VECTORS: usize = 65_536;

/// A segment's code format.
pub trait Codec: Send + Sync {
    /// A query prepared for scoring codes (quantized, rotated, tables, ...).
    type Query: Send + Sync;

    fn kind(&self) -> CodeKind;
    fn metric(&self) -> Metric;
    fn dim(&self) -> usize;
    /// Bytes per code.
    fn code_len(&self) -> usize;
    /// Encodes a stored (prepared) vector; `out.len() == code_len()`.
    fn encode(&self, vector: &[f32], out: &mut [u8]);
    /// Prepares a stored-form query.
    fn prepare(&self, query: &[f32]) -> Self::Query;
    /// The estimated score of a code for a prepared query.
    fn score(&self, query: &Self::Query, code: &[u8]) -> f32;
    /// The estimated score between two codes (graph construction).
    fn score_codes(&self, a: &[u8], b: &[u8]) -> f32;
    /// The codec's parameters, as saved in a segment.
    fn to_bytes(&self) -> Vec<u8>;
}

/// Codes that are the stored vectors themselves (little-endian f32). Scores are canonical.
#[derive(Debug, Clone, PartialEq)]
pub struct F32Codec {
    pub metric: Metric,
    pub dim: usize,
}

impl Codec for F32Codec {
    type Query = Vec<f32>;

    fn kind(&self) -> CodeKind {
        CodeKind::F32
    }

    fn metric(&self) -> Metric {
        self.metric
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn code_len(&self) -> usize {
        todo!("U07 quant: F32Codec::code_len")
    }

    fn encode(&self, vector: &[f32], out: &mut [u8]) {
        todo!("U07 quant: F32Codec::encode")
    }

    fn prepare(&self, query: &[f32]) -> Vec<f32> {
        todo!("U07 quant: F32Codec::prepare")
    }

    /// The canonical score of the decoded vector.
    fn score(&self, query: &Vec<f32>, code: &[u8]) -> f32 {
        todo!("U07 quant: F32Codec::score")
    }

    fn score_codes(&self, a: &[u8], b: &[u8]) -> f32 {
        todo!("U07 quant: F32Codec::score_codes")
    }

    /// Empty: the metric and dimension come from the index configuration.
    fn to_bytes(&self) -> Vec<u8> {
        todo!("U07 quant: F32Codec::to_bytes")
    }
}

/// A segment's codec.
#[derive(Debug, Clone, PartialEq)]
pub enum AnyCodec {
    F32(F32Codec),
    Int8(Int8Codec),
    Binary(BinaryCodec),
}

/// Runs `$body` with `$c` bound to the concrete codec inside an [`AnyCodec`], so generic
/// code is monomorphized per codec: `with_codec!(&any, c => scan(c, ...))`.
macro_rules! with_codec {
    ($codec:expr, $c:ident => $body:expr) => {
        match $codec {
            $crate::quant::AnyCodec::F32($c) => $body,
            $crate::quant::AnyCodec::Int8($c) => $body,
            $crate::quant::AnyCodec::Binary($c) => $body,
        }
    };
}
pub(crate) use with_codec;

impl AnyCodec {
    /// Trains a codec of `kind` on `sample` (flattened stored-form vectors, `dim` each,
    /// at most [`MAX_TRAINING_VECTORS`]). Deterministic in its inputs and `seed`.
    pub fn train(
        kind: CodeKind,
        metric: Metric,
        dim: usize,
        sample: &[f32],
        seed: u64,
    ) -> Result<AnyCodec> {
        todo!("U07 quant: AnyCodec::train")
    }

    /// The codec saved as `bytes` by [`Codec::to_bytes`]. `Corrupt`-style errors are
    /// reported as `Error::Corrupt` with an empty path (the caller adds context).
    pub fn from_bytes(kind: CodeKind, metric: Metric, dim: usize, bytes: &[u8]) -> Result<AnyCodec> {
        todo!("U07 quant: AnyCodec::from_bytes")
    }

    pub fn kind(&self) -> CodeKind {
        todo!("U07 quant: AnyCodec::kind")
    }

    pub fn code_len(&self) -> usize {
        todo!("U07 quant: AnyCodec::code_len")
    }

    pub fn encode(&self, vector: &[f32], out: &mut [u8]) {
        todo!("U07 quant: AnyCodec::encode")
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        todo!("U07 quant: AnyCodec::to_bytes")
    }
}

/// The rows a segment of `n` vectors trains its codec on: all of them when
/// `n <= MAX_TRAINING_VECTORS`, otherwise `MAX_TRAINING_VECTORS` rows chosen by
/// `SplitMix64::new(seed).sample(n, MAX_TRAINING_VECTORS)`. Sorted.
pub fn training_sample(n: usize, seed: u64) -> Vec<usize> {
    todo!("U07 quant: training_sample")
}
