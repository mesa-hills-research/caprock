//! Codecs: the compact codes segments keep in memory to find candidates (unit U07 for this
//! file; U08 int8; U09 binary; docs/requirements.md, "Quantization").
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
use crate::distance;
use crate::error::{Error, Result};
use crate::format::f32s_from_le;
use crate::util::SplitMix64;

/// Most vectors a codec trains on.
pub const MAX_TRAINING_VECTORS: usize = 65_536;

/// Runs a kernel compiled for the best instruction set this CPU offers (AVX2 on
/// x86-64, NEON on aarch64), detected once. Kernels keep their scalar operation
/// order lane for lane and Rust never fuses a multiply and an add, so every
/// instruction set gives the same bits.
pub(crate) fn dispatch<Op: pulp::WithSimd>(kernel: Op) -> Op::Output {
    static ARCH: std::sync::OnceLock<pulp::Arch> = std::sync::OnceLock::new();
    ARCH.get_or_init(pulp::Arch::new).dispatch(kernel)
}

/// AVX2 (with FMA, which no kernel uses), when this CPU has it; detected once.
#[cfg(target_arch = "x86_64")]
pub(crate) fn x86_v3() -> Option<pulp::x86::V3> {
    static V3: std::sync::OnceLock<Option<pulp::x86::V3>> = std::sync::OnceLock::new();
    *V3.get_or_init(pulp::x86::V3::try_new)
}

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
    /// Appends `score_codes(a, code)` for the code of each id in `ids`, bit for bit;
    /// `codes` holds `code_len()` bytes per id. Codecs may prepare `a` once.
    fn score_codes_many(&self, a: &[u8], codes: &[u8], ids: &[u32], scores: &mut Vec<f32>) {
        let len = self.code_len();
        scores.extend(ids.iter().map(|&id| {
            let start = id as usize * len;
            self.score_codes(a, &codes[start..start + len])
        }));
    }
    /// The codec's parameters, as saved in a segment.
    fn to_bytes(&self) -> Vec<u8>;
}

/// Codes that are the stored vectors themselves (little-endian f32). Scores are canonical.
#[derive(Debug, Clone, PartialEq)]
pub struct F32Codec {
    pub metric: Metric,
    pub dim: usize,
}

impl F32Codec {
    fn decode(&self, code: &[u8]) -> Vec<f32> {
        assert_eq!(code.len(), self.code_len(), "f32 code length mismatch");
        f32s_from_le(code).expect("f32 code length is a multiple of four")
    }
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
        self.dim.checked_mul(4).expect("f32 code length overflow")
    }

    fn encode(&self, vector: &[f32], out: &mut [u8]) {
        assert_eq!(vector.len(), self.dim, "f32 vector dimension mismatch");
        assert_eq!(out.len(), self.code_len(), "f32 code length mismatch");
        for (&value, bytes) in vector.iter().zip(out.as_chunks_mut::<4>().0) {
            *bytes = value.to_le_bytes();
        }
    }

    fn prepare(&self, query: &[f32]) -> Vec<f32> {
        assert_eq!(query.len(), self.dim, "f32 query dimension mismatch");
        // Queries are already in stored form; normalizing again would change bits.
        query.to_vec()
    }

    /// The canonical score of the decoded vector.
    fn score(&self, query: &Vec<f32>, code: &[u8]) -> f32 {
        assert_eq!(query.len(), self.dim, "f32 query dimension mismatch");
        distance::score(self.metric, query, &self.decode(code))
    }

    fn score_codes(&self, a: &[u8], b: &[u8]) -> f32 {
        distance::score(self.metric, &self.decode(a), &self.decode(b))
    }

    fn score_codes_many(&self, a: &[u8], codes: &[u8], ids: &[u32], scores: &mut Vec<f32>) {
        let a = self.decode(a);
        let mut b = vec![0.0; self.dim];
        let len = self.code_len();
        for &id in ids {
            let start = id as usize * len;
            let code = &codes[start..start + len];
            for (value, bytes) in b.iter_mut().zip(code.as_chunks::<4>().0) {
                *value = f32::from_le_bytes(*bytes);
            }
            scores.push(distance::score(self.metric, &a, &b));
        }
    }

    /// Empty: the metric and dimension come from the index configuration.
    fn to_bytes(&self) -> Vec<u8> {
        Vec::new()
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
        if !(1..=crate::MAX_DIM as usize).contains(&dim) {
            return Err(Error::InvalidArgument(format!(
                "codec dimension must be in 1..={}",
                crate::MAX_DIM
            )));
        }
        if !sample.len().is_multiple_of(dim) || sample.len() / dim > MAX_TRAINING_VECTORS {
            return Err(Error::InvalidArgument(format!(
                "codec sample must contain at most {MAX_TRAINING_VECTORS} complete vectors"
            )));
        }
        match kind {
            CodeKind::F32 => Ok(AnyCodec::F32(F32Codec { metric, dim })),
            CodeKind::Int8 => Int8Codec::train(metric, dim, sample, seed).map(AnyCodec::Int8),
            CodeKind::Binary => BinaryCodec::train(metric, dim, sample, seed).map(AnyCodec::Binary),
        }
    }

    /// The codec saved as `bytes` by [`Codec::to_bytes`]. `Corrupt`-style errors are
    /// reported as `Error::Corrupt` with an empty path (the caller adds context).
    pub fn from_bytes(kind: CodeKind, metric: Metric, dim: usize, bytes: &[u8]) -> Result<AnyCodec> {
        if !(1..=crate::MAX_DIM as usize).contains(&dim) {
            return Err(Error::corrupt("", "codec dimension is out of range"));
        }
        match kind {
            CodeKind::F32 => {
                if !bytes.is_empty() {
                    return Err(Error::corrupt("", "f32 codec parameters must be empty"));
                }
                Ok(AnyCodec::F32(F32Codec { metric, dim }))
            }
            CodeKind::Int8 => Int8Codec::from_bytes(metric, dim, bytes).map(AnyCodec::Int8),
            CodeKind::Binary => BinaryCodec::from_bytes(metric, dim, bytes).map(AnyCodec::Binary),
        }
    }

    pub fn kind(&self) -> CodeKind {
        with_codec!(self, codec => codec.kind())
    }

    pub fn code_len(&self) -> usize {
        with_codec!(self, codec => codec.code_len())
    }

    pub fn encode(&self, vector: &[f32], out: &mut [u8]) {
        with_codec!(self, codec => codec.encode(vector, out))
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        with_codec!(self, codec => codec.to_bytes())
    }
}

/// The rows a segment of `n` vectors trains its codec on: all of them when
/// `n <= MAX_TRAINING_VECTORS`, otherwise `MAX_TRAINING_VECTORS` rows chosen by
/// `SplitMix64::new(seed).sample(n, MAX_TRAINING_VECTORS)`. Sorted.
pub fn training_sample(n: usize, seed: u64) -> Vec<usize> {
    if n <= MAX_TRAINING_VECTORS {
        return (0..n).collect();
    }
    SplitMix64::new(seed)
        .sample(n as u64, MAX_TRAINING_VECTORS)
        .into_iter()
        .map(|row| row as usize)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_preserves_stored_bits_and_does_not_normalize_queries_again() {
        let vector = [-0.0, f32::MIN_POSITIVE, f32::from_bits(1), -1.25, 1e15];
        let codec = F32Codec {
            metric: Metric::Cosine,
            dim: vector.len(),
        };
        let query = codec.prepare(&vector);
        assert_eq!(
            query.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            vector.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        let mut code = vec![0; codec.code_len()];
        codec.encode(&vector, &mut code);
        assert_eq!(code, crate::format::f32s_to_le(&vector));
        assert_eq!(
            codec.score(&query, &code).to_bits(),
            distance::score(codec.metric, &vector, &vector).to_bits()
        );
    }

    #[test]
    fn f32_dispatch_round_trips_configuration_and_rejects_bad_parameters() {
        for metric in [Metric::Cosine, Metric::Dot, Metric::L2] {
            for dim in [1, 15, 16, 17, crate::MAX_DIM as usize] {
                // F32 has no trained parameters and can use an empty sample.
                let codec = AnyCodec::train(CodeKind::F32, metric, dim, &[], 99).unwrap();
                assert_eq!(
                    AnyCodec::from_bytes(CodeKind::F32, metric, dim, &codec.to_bytes()).unwrap(),
                    codec
                );
                assert!(matches!(
                    AnyCodec::from_bytes(CodeKind::F32, metric, dim, &[0]),
                    Err(Error::Corrupt { path, .. }) if path.as_os_str().is_empty()
                ));
            }
        }
        for dim in [0, crate::MAX_DIM as usize + 1, usize::MAX] {
            assert!(matches!(
                AnyCodec::train(CodeKind::F32, Metric::Dot, dim, &[], 0),
                Err(Error::InvalidArgument(_))
            ));
            assert!(matches!(
                AnyCodec::from_bytes(CodeKind::F32, Metric::Dot, dim, &[]),
                Err(Error::Corrupt { path, .. }) if path.as_os_str().is_empty()
            ));
        }
        for (dim, sample) in [(2, vec![1.0; 3]), (1, vec![1.0; MAX_TRAINING_VECTORS + 1])] {
            assert!(matches!(
                AnyCodec::train(CodeKind::F32, Metric::Dot, dim, &sample, 0),
                Err(Error::InvalidArgument(_))
            ));
        }
    }

    #[test]
    fn batched_code_scores_equal_pairwise_scores() {
        let mut rng = SplitMix64::new(3);
        for metric in [Metric::Cosine, Metric::Dot, Metric::L2] {
            for dim in [1, 7, 16, 33, 64, 65, 130] {
                let n = 20;
                let mut vectors: Vec<f32> = (0..n * dim)
                    .map(|_| (rng.below(2001) as f32 - 1000.0) / 250.0)
                    .collect();
                if metric == Metric::Cosine {
                    for v in vectors.chunks_exact_mut(dim) {
                        let norm = distance::dot(v, v).sqrt();
                        v.iter_mut().for_each(|x| *x /= norm);
                    }
                }
                for kind in [CodeKind::F32, CodeKind::Int8, CodeKind::Binary] {
                    let codec = AnyCodec::train(kind, metric, dim, &vectors, 0).unwrap();
                    let len = codec.code_len();
                    let mut codes = vec![0; n * len];
                    for (v, code) in vectors.chunks_exact(dim).zip(codes.chunks_exact_mut(len)) {
                        codec.encode(v, code);
                    }
                    let ids: Vec<u32> = (0..n as u32).rev().collect();
                    with_codec!(&codec, c => {
                        for a in codes.chunks_exact(len) {
                            let mut many = vec![f32::NAN];
                            c.score_codes_many(a, &codes, &ids, &mut many);
                            assert!(many[0].is_nan(), "appends");
                            for (&id, score) in ids.iter().zip(&many[1..]) {
                                let b = &codes[id as usize * len..(id as usize + 1) * len];
                                assert_eq!(score.to_bits(), c.score_codes(a, b).to_bits());
                            }
                        }
                    });
                }
            }
        }
    }

    #[test]
    fn f32_kernels_reject_mismatched_lengths() {
        let codec = F32Codec {
            metric: Metric::Dot,
            dim: 2,
        };
        assert!(std::panic::catch_unwind(|| codec.prepare(&[1.0])).is_err());
        assert!(std::panic::catch_unwind(|| codec.encode(&[1.0], &mut [0; 8])).is_err());
        assert!(std::panic::catch_unwind(|| codec.encode(&[1.0; 2], &mut [0; 7])).is_err());
        assert!(std::panic::catch_unwind(|| codec.score(&vec![1.0], &[0; 8])).is_err());
        for len in [0, 4, 7, 9, 12] {
            let code = vec![0; len];
            assert!(std::panic::catch_unwind(|| codec.score(&vec![1.0; 2], &code)).is_err());
            assert!(std::panic::catch_unwind(|| codec.score_codes(&code, &[0; 8])).is_err());
            assert!(std::panic::catch_unwind(|| codec.score_codes(&[0; 8], &code)).is_err());
        }
    }
}
