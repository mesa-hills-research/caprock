//! Centred binary quantization.
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

    /// The angular estimate between two codes whose sign bits differ in `hamming`
    /// places.
    #[inline(always)]
    fn score_hamming(&self, a: &[u8], b: &[u8], hamming: usize) -> f32 {
        let (norm_a, centre_a) = self.corrections(a);
        let (norm_b, centre_b) = self.corrections(b);
        let dot = (norm_a * norm_b) * self.correlations[hamming];
        // Group symmetric terms before adding the others, so swapping operands
        // cannot change the rounding (including the last bit).
        match self.metric {
            Metric::Cosine | Metric::Dot => self.centre_norm2 + (centre_a + centre_b) + dot,
            Metric::L2 => -(norm_a * norm_a + norm_b * norm_b) + 2.0 * dot,
        }
    }

    /// The asymmetric estimate of a code with sign projection `projection`.
    #[inline(always)]
    fn estimate(&self, query: &BinaryQuery, projection: f32, code: &[u8]) -> f32 {
        let (norm, centre_dot) = self.corrections(code);
        let dot = projection * (norm * self.params[self.dim]);
        match self.metric {
            Metric::Cosine | Metric::Dot => query.constant + centre_dot + dot,
            Metric::L2 => query.constant - norm * norm + 2.0 * dot,
        }
    }

    #[inline(always)]
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
        let mut residual = vec![0.0; self.dim.div_ceil(4) * 4];
        let mut norm2 = 0.0f64;
        let mut centre_dot = 0.0f64;
        for (j, (&x, &c)) in query.iter().zip(&self.params).enumerate() {
            let r = f64::from(x) - f64::from(c);
            residual[j] = r as f32;
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
            values,
            constant: match self.metric {
                Metric::Cosine | Metric::Dot => centre_dot as f32,
                Metric::L2 => -norm2 as f32,
            },
        }
    }

    fn score(&self, query: &BinaryQuery, code: &[u8]) -> f32 {
        let [projection] = projections(&query.values, [code]);
        self.estimate(query, projection, code)
    }

    fn score_many(&self, query: &BinaryQuery, codes: &[u8], ids: &[u32], scores: &mut Vec<f32>) {
        #[cfg(target_arch = "x86_64")]
        if let Some(simd) = crate::quant::x86_v3() {
            return pulp::Simd::vectorize(
                simd,
                avx2::QueryCodes {
                    simd,
                    codec: self,
                    query,
                    codes,
                    ids,
                    scores,
                },
            );
        }
        let len = self.code_len();
        let code = |id: u32| {
            let start = id as usize * len;
            &codes[start..start + len]
        };
        let (groups, rest) = ids.as_chunks::<SCORE_GROUP>();
        for group in groups {
            let group = group.map(code);
            let projections = projections(&query.values, group);
            for (projection, code) in projections.into_iter().zip(group) {
                scores.push(self.estimate(query, projection, code));
            }
        }
        scores.extend(rest.iter().map(|&id| self.score(query, code(id))));
    }

    fn score_codes(&self, a: &[u8], b: &[u8]) -> f32 {
        let bit_bytes = self.bit_bytes();
        let hamming = crate::quant::dispatch(Hamming {
            a: &a[..bit_bytes],
            b: &b[..bit_bytes],
            dim: self.dim,
        });
        self.score_hamming(a, b, hamming)
    }

    fn score_codes_many(&self, a: &[u8], codes: &[u8], ids: &[u32], scores: &mut Vec<f32>) {
        // One dispatch for the whole batch.
        crate::quant::dispatch(HammingMany {
            codec: self,
            a,
            codes,
            ids,
            scores,
        })
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

/// Codes [`BinaryCodec::score_many`] scores at once.
const SCORE_GROUP: usize = 8;

/// The sign projections of `N` codes on a prepared query: for each code, the sum of
/// table `j`'s entry for nibble `j` (the low nibble of byte `j / 2` for even `j`, the
/// high one for odd `j`), added in increasing `j`. Each code's sum is one sequential
/// chain, so scoring several codes together lets the chains overlap without changing
/// any code's bits.
#[inline(always)]
fn projections<const N: usize>(values: &[f32], codes: [&[u8]; N]) -> [f32; N] {
    resume_projections(values.as_chunks::<16>().0, codes, 0, [0.0; N])
}

/// [`projections`] from table `first` (even) on, continuing the sums `projections`
/// of the tables before it.
#[inline(always)]
fn resume_projections<const N: usize>(
    tables: &[[f32; 16]],
    codes: [&[u8]; N],
    first: usize,
    mut projections: [f32; N],
) -> [f32; N] {
    let (pairs, last) = tables[first..].as_chunks::<2>();
    let codes = codes.map(|code| &code[first / 2..tables.len().div_ceil(2)]);
    let paired = codes.map(|code| &code[..pairs.len()]);
    // Checked once here, so the loop indexes the codes without checks.
    assert!(paired.iter().all(|code| code.len() == pairs.len()));
    for (j, [low, high]) in pairs.iter().enumerate() {
        for (projection, code) in projections.iter_mut().zip(paired) {
            let byte = code[j];
            *projection += low[usize::from(byte & 15)];
            *projection += high[usize::from(byte >> 4)];
        }
    }
    if let [table] = last {
        for (projection, code) in projections.iter_mut().zip(codes) {
            *projection += table[usize::from(code[pairs.len()] & 15)];
        }
    }
    projections
}

/// [`BinaryCodec::score_many`] with eight codes in the eight lanes of an AVX2
/// register: 32 bytes of each code are transposed so that lane `c` holds code `c`'s
/// nibbles, and each nibble's table entry is looked up for all eight lanes at once. Each
/// lane adds its own code's entries in increasing `j`, so the sums are
/// [`projections`]' bits.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use core::arch::x86_64::{__m256, __m256i};

    use pulp::x86::V3;

    use super::{BinaryCodec, BinaryQuery, SCORE_GROUP};
    use crate::quant::Codec;

    pub(super) struct QueryCodes<'a> {
        pub simd: V3,
        pub codec: &'a BinaryCodec,
        pub query: &'a BinaryQuery,
        pub codes: &'a [u8],
        pub ids: &'a [u32],
        pub scores: &'a mut Vec<f32>,
    }

    impl pulp::WithSimd for QueryCodes<'_> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, _: S) {
            let (codec, query) = (self.codec, self.query);
            let tables = query.values.as_chunks::<16>().0;
            let len = codec.code_len();
            let code = |id: u32| {
                let start = id as usize * len;
                &self.codes[start..start + len]
            };
            let (groups, rest) = self.ids.as_chunks::<SCORE_GROUP>();
            for group in groups {
                let codes = group.map(code);
                let (sums, done) = projections(self.simd, tables, codes);
                let projections = super::resume_projections(tables, codes, done, sums);
                for (projection, code) in projections.into_iter().zip(codes) {
                    self.scores.push(codec.estimate(query, projection, code));
                }
            }
            for &id in rest {
                let code = code(id);
                let [projection] = super::projections(&query.values, [code]);
                self.scores.push(codec.estimate(query, projection, code));
            }
        }
    }

    /// The projections of eight codes over the first `64 * n` tables (32 bytes of each
    /// code per 64 tables), and that number of tables.
    #[inline(always)]
    fn projections(simd: V3, tables: &[[f32; 16]], codes: [&[u8]; 8]) -> ([f32; 8], usize) {
        let chunks = tables.as_chunks::<64>().0;
        let codes = codes.map(|code| &code.as_chunks::<32>().0[..chunks.len()]);
        // Checked once here, so the loop indexes the codes without checks.
        assert!(codes.iter().all(|code| code.len() == chunks.len()));
        let mut sums = simd.avx._mm256_setzero_ps();
        for (i, tables) in chunks.iter().enumerate() {
            let rows = [0, 1, 2, 3, 4, 5, 6, 7].map(|c| pulp::cast(codes[c][i]));
            // Column t holds bytes 4t..4t + 4 of the chunk, tables 8t..8t + 8.
            for (column, tables) in transpose(simd, rows).into_iter().zip(tables.as_chunks::<8>().0) {
                sums = lookup::<0, 28>(simd, sums, column, &tables[0]);
                sums = lookup::<4, 24>(simd, sums, column, &tables[1]);
                sums = lookup::<8, 20>(simd, sums, column, &tables[2]);
                sums = lookup::<12, 16>(simd, sums, column, &tables[3]);
                sums = lookup::<16, 12>(simd, sums, column, &tables[4]);
                sums = lookup::<20, 8>(simd, sums, column, &tables[5]);
                sums = lookup::<24, 4>(simd, sums, column, &tables[6]);
                sums = lookup::<28, 0>(simd, sums, column, &tables[7]);
            }
        }
        (pulp::cast(sums), chunks.len() * 64)
    }

    /// `sums` plus, in each lane, `table`'s entry for the nibble at bits
    /// `RIGHT..RIGHT + 4` of the lane (`LEFT` = 28 - `RIGHT` moves its top bit to the
    /// sign). The permutes use an index's low three bits, the sign bit picks the half.
    #[inline(always)]
    fn lookup<const RIGHT: i32, const LEFT: i32>(
        simd: V3,
        sums: __m256,
        column: __m256i,
        table: &[f32; 16],
    ) -> __m256 {
        let (avx, avx2) = (simd.avx, simd.avx2);
        let [low, high]: [__m256; 2] = pulp::cast(*table);
        let index = avx2._mm256_srli_epi32::<RIGHT>(column);
        let upper = avx._mm256_castsi256_ps(avx2._mm256_slli_epi32::<LEFT>(column));
        let low = avx2._mm256_permutevar8x32_ps(low, index);
        let high = avx2._mm256_permutevar8x32_ps(high, index);
        avx._mm256_add_ps(sums, avx._mm256_blendv_ps(low, high, upper))
    }

    /// Rows of eight 32-bit words to columns: word `t` of row `c` becomes lane `c` of
    /// column `t`.
    #[inline(always)]
    fn transpose(simd: V3, rows: [__m256i; 8]) -> [__m256i; 8] {
        let a = simd.avx2;
        // Pairs of rows: words 0, 1 (low) and 2, 3 (high) of each 128-bit half.
        let low01 = a._mm256_unpacklo_epi32(rows[0], rows[1]);
        let high01 = a._mm256_unpackhi_epi32(rows[0], rows[1]);
        let low23 = a._mm256_unpacklo_epi32(rows[2], rows[3]);
        let high23 = a._mm256_unpackhi_epi32(rows[2], rows[3]);
        let low45 = a._mm256_unpacklo_epi32(rows[4], rows[5]);
        let high45 = a._mm256_unpackhi_epi32(rows[4], rows[5]);
        let low67 = a._mm256_unpacklo_epi32(rows[6], rows[7]);
        let high67 = a._mm256_unpackhi_epi32(rows[6], rows[7]);
        // Words t and t + 4 of rows 0-3 (and of rows 4-7).
        let w04 = a._mm256_unpacklo_epi64(low01, low23);
        let w15 = a._mm256_unpackhi_epi64(low01, low23);
        let w26 = a._mm256_unpacklo_epi64(high01, high23);
        let w37 = a._mm256_unpackhi_epi64(high01, high23);
        let x04 = a._mm256_unpacklo_epi64(low45, low67);
        let x15 = a._mm256_unpackhi_epi64(low45, low67);
        let x26 = a._mm256_unpacklo_epi64(high45, high67);
        let x37 = a._mm256_unpackhi_epi64(high45, high67);
        [
            a._mm256_permute2x128_si256::<0x20>(w04, x04),
            a._mm256_permute2x128_si256::<0x20>(w15, x15),
            a._mm256_permute2x128_si256::<0x20>(w26, x26),
            a._mm256_permute2x128_si256::<0x20>(w37, x37),
            a._mm256_permute2x128_si256::<0x31>(w04, x04),
            a._mm256_permute2x128_si256::<0x31>(w15, x15),
            a._mm256_permute2x128_si256::<0x31>(w26, x26),
            a._mm256_permute2x128_si256::<0x31>(w37, x37),
        ]
    }
}

/// The Hamming distance of two codes' first `dim` sign bits (whole words; padding
/// bits are masked, so a code's padding never counts). Integer arithmetic: exact in
/// any order, with or without a population-count instruction.
#[inline(always)]
fn hamming(a: &[u8], b: &[u8], dim: usize) -> usize {
    let words = a.as_chunks::<8>().0.iter().zip(b.as_chunks::<8>().0);
    let last = dim.div_ceil(64) - 1;
    let mut hamming = 0;
    for (j, (a, b)) in words.enumerate() {
        let mut different = u64::from_le_bytes(*a) ^ u64::from_le_bytes(*b);
        if j == last && !dim.is_multiple_of(64) {
            different &= (1u64 << (dim % 64)) - 1;
        }
        hamming += different.count_ones() as usize;
    }
    hamming
}

struct Hamming<'a> {
    a: &'a [u8],
    b: &'a [u8],
    dim: usize,
}

impl pulp::WithSimd for Hamming<'_> {
    type Output = usize;

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, _: S) -> usize {
        hamming(self.a, self.b, self.dim)
    }
}

struct HammingMany<'a> {
    codec: &'a BinaryCodec,
    a: &'a [u8],
    codes: &'a [u8],
    ids: &'a [u32],
    scores: &'a mut Vec<f32>,
}

impl pulp::WithSimd for HammingMany<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: pulp::Simd>(self, _: S) {
        let codec = self.codec;
        let len = codec.code_len();
        let bit_bytes = codec.bit_bytes();
        for &id in self.ids {
            let start = id as usize * len;
            let b = &self.codes[start..start + len];
            let hamming = hamming(&self.a[..bit_bytes], &b[..bit_bytes], codec.dim);
            self.scores.push(codec.score_hamming(self.a, b, hamming));
        }
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
        }
    }

    #[test]
    fn batched_and_interleaved_lookups_equal_the_sequential_sum_bit_for_bit() {
        // The estimate before interleaving: one sequential sum over the tables.
        let reference = |codec: &BinaryCodec, query: &BinaryQuery, code: &[u8]| {
            let mut projection = 0.0;
            for (j, table) in query.values.as_chunks::<16>().0.iter().enumerate() {
                let nibble = (code[j / 2] >> ((j % 2) * 4)) & 15;
                projection += table[usize::from(nibble)];
            }
            let (norm, centre_dot) = codec.corrections(code);
            let dot = projection * (norm * codec.params[codec.dim]);
            match codec.metric {
                Metric::Cosine | Metric::Dot => query.constant + centre_dot + dot,
                Metric::L2 => query.constant - norm * norm + 2.0 * dot,
            }
        };
        let mut rng = crate::util::SplitMix64::new(5);
        let mut value = || (rng.below(2001) as f32 - 1000.0) / 300.0;
        for dim in [1, 2, 3, 4, 5, 7, 8, 9, 63, 64, 65, 127, 128, 129, 255, 256, 257, 300, 511, 1024, 1027] {
            let n = 21;
            let sample: Vec<f32> = (0..n * dim).map(|_| value()).collect();
            let query: Vec<f32> = (0..dim).map(|_| value()).collect();
            for metric in [Metric::Cosine, Metric::Dot, Metric::L2] {
                let codec = BinaryCodec::train(metric, dim, &sample, 0).unwrap();
                let len = codec.code_len();
                let codes: Vec<u8> = sample.chunks_exact(dim).flat_map(|v| encode(&codec, v)).collect();
                let prepared = codec.prepare(&query);
                let code = |i: u32| &codes[i as usize * len..(i as usize + 1) * len];
                let ids: Vec<u32> = (0..n as u32).rev().chain([2, 2]).collect();
                for count in [0, 1, 7, 8, 9, 16, ids.len()] {
                    let mut many = vec![f32::NAN];
                    codec.score_many(&prepared, &codes, &ids[..count], &mut many);
                    assert!(many[0].is_nan(), "appends");
                    assert_eq!(many.len(), count + 1);
                    for (&id, batched) in ids.iter().zip(&many[1..]) {
                        let expected = reference(&codec, &prepared, code(id)).to_bits();
                        assert_eq!(batched.to_bits(), expected, "{metric:?} {dim}");
                        assert_eq!(codec.score(&prepared, code(id)).to_bits(), expected);
                    }
                }
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
