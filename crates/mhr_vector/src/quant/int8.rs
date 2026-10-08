//! Int8 scalar quantization.
//!
//! Each dimension has a trained minimum and a step size spanning its observed range
//! with 256 unsigned byte values. Codes contain exactly one byte per dimension.
//! Queries are not quantized: dot/cosine queries fold the affine decoder into their
//! weights, while L2 queries subtract the decoder's offset. Graph construction uses
//! symmetric scores between decoded codes. The per-code kernels are allocation-free;
//! batch scoring decodes the shared code once.

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
        let steps = &self.params[self.dim..];
        #[cfg(target_arch = "x86_64")]
        if let Some(simd) = crate::quant::x86_v3() {
            return pulp::Simd::vectorize(
                simd,
                avx2::QueryCode {
                    simd,
                    metric: self.metric,
                    query,
                    steps,
                    code,
                },
            );
        }
        portable_query(self.metric, query, steps, code)
    }

    fn score_many(&self, query: &Int8Query, codes: &[u8], ids: &[u32], scores: &mut Vec<f32>) {
        assert_eq!(
            query.values.len(),
            self.dim,
            "int8 query dimension mismatch"
        );
        let steps = &self.params[self.dim..];
        #[cfg(target_arch = "x86_64")]
        if let Some(simd) = crate::quant::x86_v3() {
            return pulp::Simd::vectorize(
                simd,
                avx2::QueryCodes {
                    simd,
                    metric: self.metric,
                    query,
                    steps,
                    codes,
                    ids,
                    scores,
                },
            );
        }
        let len = self.code_len();
        scores.extend(ids.iter().map(|&id| {
            let start = id as usize * len;
            portable_query(self.metric, query, steps, &codes[start..start + len])
        }));
    }

    fn score_codes(&self, a: &[u8], b: &[u8]) -> f32 {
        assert_eq!(a.len(), self.code_len(), "int8 code length mismatch");
        assert_eq!(b.len(), self.code_len(), "int8 code length mismatch");
        let (minima, steps) = self.params.split_at(self.dim);
        code_pair(self.metric, minima, &steps[..self.dim], a, b)
    }

    fn score_codes_many(&self, a: &[u8], codes: &[u8], ids: &[u32], scores: &mut Vec<f32>) {
        assert_eq!(a.len(), self.code_len(), "int8 code length mismatch");
        let (minima, steps) = self.params.split_at(self.dim);
        let steps = &steps[..self.dim];
        #[cfg(target_arch = "x86_64")]
        if let Some(simd) = crate::quant::x86_v3() {
            return pulp::Simd::vectorize(
                simd,
                avx2::CodesMany {
                    simd,
                    metric: self.metric,
                    minima,
                    steps,
                    a,
                    codes,
                    ids,
                    scores,
                },
            );
        }
        let len = self.code_len();
        scores.extend(ids.iter().map(|&id| {
            let start = id as usize * len;
            code_pair(self.metric, minima, steps, a, &codes[start..start + len])
        }));
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

/// The symmetric score of two codes, with [`sum_terms`]'s lanes and reduction:
/// dot/cosine `sum((min + step * a) * (min + step * b))`, L2 the negated sum of
/// `((a - b) * step)^2` (offsets cancel; subtracting bytes before scaling preserves
/// small distances even when the offset is much larger than the step). Element `i`
/// goes to lane `i % 16`, and each lane adds its terms in increasing `i`, so the
/// portable and the AVX2 kernels give the same bits.
fn code_pair(metric: Metric, minima: &[f32], steps: &[f32], a: &[u8], b: &[u8]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if let Some(simd) = crate::quant::x86_v3() {
        return pulp::Simd::vectorize(
            simd,
            avx2::CodePair {
                simd,
                metric,
                minima,
                steps,
                a,
                b,
            },
        );
    }
    match metric {
        Metric::Cosine | Metric::Dot => portable_dot(minima, steps, a, b),
        Metric::L2 => -portable_l2(steps, a, b),
    }
}

/// The estimated score of a code for a prepared query, with [`sum_terms`]'s lanes and
/// reduction.
fn portable_query(metric: Metric, query: &Int8Query, steps: &[f32], code: &[u8]) -> f32 {
    let dim = code.len();
    match metric {
        Metric::Cosine | Metric::Dot => {
            query.constant + sum_terms(dim, |i| query.values[i] * code[i] as f32)
        }
        // Compute squared differences, not q^2 + x^2 - 2*q*x, to avoid cancellation
        // when the two vectors are very close.
        Metric::L2 => -sum_terms(dim, |i| {
            let delta = query.values[i] - steps[i] * code[i] as f32;
            delta * delta
        }),
    }
}

// Two 8-lane halves vectorize well with or without wide registers.
fn portable_dot(minima: &[f32], steps: &[f32], a: &[u8], b: &[u8]) -> f32 {
    let mut halves = [[0.0f32; 8]; 2];
    let (minima, minima_tail) = minima.as_chunks::<16>();
    let (steps, steps_tail) = steps.as_chunks::<16>();
    let (a, a_tail) = a.as_chunks::<16>();
    let (b, b_tail) = b.as_chunks::<16>();
    let term = |min: f32, step: f32, a: u8, b: u8| {
        let x = min + step * a as f32;
        let y = min + step * b as f32;
        x * y
    };
    for (((minima, steps), a), b) in minima.iter().zip(steps).zip(a).zip(b) {
        for (half, lanes) in halves.iter_mut().enumerate() {
            let mut terms = [0.0f32; 8];
            for (lane, value) in terms.iter_mut().enumerate() {
                let i = half * 8 + lane;
                *value = term(minima[i], steps[i], a[i], b[i]);
            }
            for (lane, value) in lanes.iter_mut().zip(terms) {
                *lane += value;
            }
        }
    }
    let mut lanes: [f32; 16] = pulp::cast(halves);
    let tail = minima_tail.iter().zip(steps_tail).zip(a_tail).zip(b_tail);
    for (lane, (((&min, &step), &a), &b)) in lanes.iter_mut().zip(tail) {
        *lane += term(min, step, a, b);
    }
    reduce(lanes)
}

fn portable_l2(steps: &[f32], a: &[u8], b: &[u8]) -> f32 {
    let mut halves = [[0.0f32; 8]; 2];
    let (steps, steps_tail) = steps.as_chunks::<16>();
    let (a, a_tail) = a.as_chunks::<16>();
    let (b, b_tail) = b.as_chunks::<16>();
    let term = |step: f32, a: u8, b: u8| {
        let delta = (a as f32 - b as f32) * step;
        delta * delta
    };
    for ((steps, a), b) in steps.iter().zip(a).zip(b) {
        for (half, lanes) in halves.iter_mut().enumerate() {
            let mut terms = [0.0f32; 8];
            for (lane, value) in terms.iter_mut().enumerate() {
                let i = half * 8 + lane;
                *value = term(steps[i], a[i], b[i]);
            }
            for (lane, value) in lanes.iter_mut().zip(terms) {
                *lane += value;
            }
        }
    }
    let mut lanes: [f32; 16] = pulp::cast(halves);
    for (lane, ((&step, &a), &b)) in lanes.iter_mut().zip(steps_tail.iter().zip(a_tail).zip(b_tail)) {
        *lane += term(step, a, b);
    }
    reduce(lanes)
}

/// The same kernels with explicit AVX2 operations: lanes 0-7 and 8-15 of each
/// 16-element chunk in two registers, no fused multiply-add.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use core::arch::x86_64::{__m128i, __m256, __m256i};

    use pulp::x86::V3;

    use super::Int8Query;
    use crate::config::Metric;

    pub(super) struct CodePair<'a> {
        pub simd: V3,
        pub metric: Metric,
        pub minima: &'a [f32],
        pub steps: &'a [f32],
        pub a: &'a [u8],
        pub b: &'a [u8],
    }

    impl pulp::WithSimd for CodePair<'_> {
        type Output = f32;

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, _: S) -> f32 {
            match self.metric {
                Metric::Cosine | Metric::Dot => dot(self.simd, self.minima, self.steps, self.a, self.b),
                Metric::L2 => -l2(self.simd, self.steps, self.a, self.b),
            }
        }
    }

    /// A prepared query against one code ([`super::portable_query`]).
    pub(super) struct QueryCode<'a> {
        pub simd: V3,
        pub metric: Metric,
        pub query: &'a Int8Query,
        pub steps: &'a [f32],
        pub code: &'a [u8],
    }

    impl pulp::WithSimd for QueryCode<'_> {
        type Output = f32;

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, _: S) -> f32 {
            let one = [self.code];
            let [score] = query_scores(self.simd, self.metric, self.query, self.steps, one, None);
            score
        }
    }

    /// [`CodePair`] for one code against many, decoding the shared code once.
    pub(super) struct CodesMany<'a> {
        pub simd: V3,
        pub metric: Metric,
        pub minima: &'a [f32],
        pub steps: &'a [f32],
        pub a: &'a [u8],
        pub codes: &'a [u8],
        pub ids: &'a [u32],
        pub scores: &'a mut Vec<f32>,
    }

    impl pulp::WithSimd for CodesMany<'_> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, _: S) {
            let len = self.a.len();
            let code = |id: u32| {
                let start = id as usize * len;
                &self.codes[start..start + len]
            };
            match self.metric {
                Metric::Cosine | Metric::Dot => {
                    // The same operations as the pair kernel's `x`.
                    let decoded: Vec<f32> = (self.minima.iter().zip(self.steps).zip(self.a))
                        .map(|((&min, &step), &a)| min + step * a as f32)
                        .collect();
                    for &id in self.ids {
                        let b = code(id);
                        let score = dot_decoded(self.simd, &decoded, self.minima, self.steps, b);
                        self.scores.push(score);
                    }
                }
                Metric::L2 => {
                    for &id in self.ids {
                        self.scores.push(-l2(self.simd, self.steps, self.a, code(id)));
                    }
                }
            }
        }
    }

    /// [`QueryCode`] for many codes, four at a time: each code keeps its own
    /// accumulators, and the four chains overlap.
    pub(super) struct QueryCodes<'a> {
        pub simd: V3,
        pub metric: Metric,
        pub query: &'a Int8Query,
        pub steps: &'a [f32],
        pub codes: &'a [u8],
        pub ids: &'a [u32],
        pub scores: &'a mut Vec<f32>,
    }

    impl pulp::WithSimd for QueryCodes<'_> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, _: S) {
            let len = self.steps.len();
            let code = |id: u32| {
                let start = id as usize * len;
                &self.codes[start..start + len]
            };
            let (groups, rest) = self.ids.as_chunks::<4>();
            for (g, group) in groups.iter().enumerate() {
                let codes = group.map(code);
                // Prefetch the codes of a later group: four codes of a contiguous scan
                // span a page, and hardware prefetchers stop at page boundaries.
                let ahead = groups.get(g + PREFETCH_GROUPS).map(|ids| ids.map(code));
                let group = query_scores(self.simd, self.metric, self.query, self.steps, codes, ahead);
                self.scores.extend(group);
            }
            for &id in rest {
                let one = [code(id)];
                let one = query_scores(self.simd, self.metric, self.query, self.steps, one, None);
                self.scores.extend(one);
            }
        }
    }

    /// How many groups ahead [`QueryCodes`] prefetches (8 KiB of 1024-byte codes).
    const PREFETCH_GROUPS: usize = 2;

    #[inline(always)]
    fn query_scores<const N: usize>(
        simd: V3,
        metric: Metric,
        query: &Int8Query,
        steps: &[f32],
        codes: [&[u8]; N],
        ahead: Option<[&[u8]; N]>,
    ) -> [f32; N] {
        match metric {
            Metric::Cosine | Metric::Dot => {
                query_dot(simd, &query.values, codes, ahead).map(|sum| query.constant + sum)
            }
            Metric::L2 => query_l2(simd, &query.values, steps, codes, ahead).map(|sum| -sum),
        }
    }

    /// `sum(value * code)` for `N` codes, in [`super::sum_terms`]'s lanes, prefetching
    /// the codes `ahead`, if any.
    #[inline(always)]
    fn query_dot<const N: usize>(
        simd: V3,
        values: &[f32],
        codes: [&[u8]; N],
        ahead: Option<[&[u8]; N]>,
    ) -> [f32; N] {
        let avx = simd.avx;
        let (values, values_tail) = values.as_chunks::<16>();
        let codes = codes.map(|code| split(code, values.len()));
        // Checked once here, so the loop indexes the chunks without checks.
        assert!(codes.iter().all(|(body, _)| body.len() == values.len()));
        let mut low = [avx._mm256_setzero_ps(); N];
        let mut high = [avx._mm256_setzero_ps(); N];
        for (j, values) in values.iter().enumerate() {
            if let Some(ahead) = ahead
                && j % 4 == 0
            {
                prefetch(simd, ahead, j * 16);
            }
            let (value_low, value_high) = halves(values);
            for c in 0..N {
                let (code_low, code_high) = widen(simd, &codes[c].0[j]);
                let term = |value, code| avx._mm256_mul_ps(value, avx._mm256_cvtepi32_ps(code));
                low[c] = avx._mm256_add_ps(low[c], term(value_low, code_low));
                high[c] = avx._mm256_add_ps(high[c], term(value_high, code_high));
            }
        }
        let mut sums = [0.0; N];
        for c in 0..N {
            let mut lanes = lanes(low[c], high[c]);
            for (lane, (&value, &code)) in lanes.iter_mut().zip(values_tail.iter().zip(codes[c].1)) {
                *lane += value * code as f32;
            }
            sums[c] = super::reduce(lanes);
        }
        sums
    }

    /// `sum((value - step * code)^2)` for `N` codes, in [`super::sum_terms`]'s lanes,
    /// prefetching the codes `ahead`, if any.
    #[inline(always)]
    fn query_l2<const N: usize>(
        simd: V3,
        values: &[f32],
        steps: &[f32],
        codes: [&[u8]; N],
        ahead: Option<[&[u8]; N]>,
    ) -> [f32; N] {
        let avx = simd.avx;
        let (values, values_tail) = values.as_chunks::<16>();
        let (steps, steps_tail) = steps.as_chunks::<16>();
        let steps = &steps[..values.len()];
        let codes = codes.map(|code| split(code, values.len()));
        // Checked once here, so the loop indexes the chunks without checks.
        assert!(codes.iter().all(|(body, _)| body.len() == values.len()));
        let mut low = [avx._mm256_setzero_ps(); N];
        let mut high = [avx._mm256_setzero_ps(); N];
        for (j, values) in values.iter().enumerate() {
            if let Some(ahead) = ahead
                && j % 4 == 0
            {
                prefetch(simd, ahead, j * 16);
            }
            let (value_low, value_high) = halves(values);
            let (step_low, step_high) = halves(&steps[j]);
            for c in 0..N {
                let (code_low, code_high) = widen(simd, &codes[c].0[j]);
                let term = |value, step, code| {
                    let decoded = avx._mm256_mul_ps(step, avx._mm256_cvtepi32_ps(code));
                    let delta = avx._mm256_sub_ps(value, decoded);
                    avx._mm256_mul_ps(delta, delta)
                };
                low[c] = avx._mm256_add_ps(low[c], term(value_low, step_low, code_low));
                high[c] = avx._mm256_add_ps(high[c], term(value_high, step_high, code_high));
            }
        }
        let mut sums = [0.0; N];
        for c in 0..N {
            let mut lanes = lanes(low[c], high[c]);
            let tail = values_tail.iter().zip(steps_tail).zip(codes[c].1);
            for (lane, ((&value, &step), &code)) in lanes.iter_mut().zip(tail) {
                let delta = value - step * code as f32;
                *lane += delta * delta;
            }
            sums[c] = super::reduce(lanes);
        }
        sums
    }

    /// Prefetches the cache line at `offset` of each code (a hint: it never faults).
    #[inline(always)]
    fn prefetch<const N: usize>(simd: V3, codes: [&[u8]; N], offset: usize) {
        for code in codes {
            let line = code.as_ptr().wrapping_add(offset).cast::<i8>();
            simd.sse._mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(line);
        }
    }

    /// A code's whole 16-byte chunks (exactly `chunks` of them) and its tail.
    #[inline(always)]
    fn split(code: &[u8], chunks: usize) -> (&[[u8; 16]], &[u8]) {
        let (body, tail) = code.as_chunks::<16>();
        (&body[..chunks], tail)
    }

    /// Bytes `0..8` and `8..16` of a chunk as 32-bit integers.
    #[inline(always)]
    fn widen(simd: V3, chunk: &[u8; 16]) -> (__m256i, __m256i) {
        let bytes: __m128i = pulp::cast(*chunk);
        let high = simd.sse2._mm_unpackhi_epi64(bytes, bytes);
        (
            simd.avx2._mm256_cvtepu8_epi32(bytes),
            simd.avx2._mm256_cvtepu8_epi32(high),
        )
    }

    #[inline(always)]
    fn halves(chunk: &[f32; 16]) -> (__m256, __m256) {
        let [low, high]: [[f32; 8]; 2] = pulp::cast(*chunk);
        (pulp::cast(low), pulp::cast(high))
    }

    #[inline(always)]
    fn lanes(low: __m256, high: __m256) -> [f32; 16] {
        pulp::cast([
            pulp::cast::<__m256, [f32; 8]>(low),
            pulp::cast::<__m256, [f32; 8]>(high),
        ])
    }

    #[inline(always)]
    fn dot(simd: V3, minima: &[f32], steps: &[f32], a: &[u8], b: &[u8]) -> f32 {
        let avx = simd.avx;
        let (minima, minima_tail) = minima.as_chunks::<16>();
        let (steps, steps_tail) = steps.as_chunks::<16>();
        let (a, a_tail) = a.as_chunks::<16>();
        let (b, b_tail) = b.as_chunks::<16>();
        let mut low = avx._mm256_setzero_ps();
        let mut high = avx._mm256_setzero_ps();
        for (((minima, steps), a), b) in minima.iter().zip(steps).zip(a).zip(b) {
            let (min_low, min_high) = halves(minima);
            let (step_low, step_high) = halves(steps);
            let (a_low, a_high) = widen(simd, a);
            let (b_low, b_high) = widen(simd, b);
            let term = |min, step, a, b| {
                let x = avx._mm256_add_ps(min, avx._mm256_mul_ps(step, avx._mm256_cvtepi32_ps(a)));
                let y = avx._mm256_add_ps(min, avx._mm256_mul_ps(step, avx._mm256_cvtepi32_ps(b)));
                avx._mm256_mul_ps(x, y)
            };
            low = avx._mm256_add_ps(low, term(min_low, step_low, a_low, b_low));
            high = avx._mm256_add_ps(high, term(min_high, step_high, a_high, b_high));
        }
        let mut lanes = lanes(low, high);
        let tail = minima_tail.iter().zip(steps_tail).zip(a_tail).zip(b_tail);
        for (lane, (((&min, &step), &a), &b)) in lanes.iter_mut().zip(tail) {
            *lane += (min + step * a as f32) * (min + step * b as f32);
        }
        super::reduce(lanes)
    }

    /// [`dot`] with `a` already decoded (`min + step * a`).
    #[inline(always)]
    fn dot_decoded(simd: V3, decoded: &[f32], minima: &[f32], steps: &[f32], b: &[u8]) -> f32 {
        let avx = simd.avx;
        let (decoded, decoded_tail) = decoded.as_chunks::<16>();
        let (minima, minima_tail) = minima.as_chunks::<16>();
        let (steps, steps_tail) = steps.as_chunks::<16>();
        let (b, b_tail) = b.as_chunks::<16>();
        let mut low = avx._mm256_setzero_ps();
        let mut high = avx._mm256_setzero_ps();
        for (((decoded, minima), steps), b) in decoded.iter().zip(minima).zip(steps).zip(b) {
            let (x_low, x_high) = halves(decoded);
            let (min_low, min_high) = halves(minima);
            let (step_low, step_high) = halves(steps);
            let (b_low, b_high) = widen(simd, b);
            let term = |x, min, step, b| {
                let y = avx._mm256_add_ps(min, avx._mm256_mul_ps(step, avx._mm256_cvtepi32_ps(b)));
                avx._mm256_mul_ps(x, y)
            };
            low = avx._mm256_add_ps(low, term(x_low, min_low, step_low, b_low));
            high = avx._mm256_add_ps(high, term(x_high, min_high, step_high, b_high));
        }
        let mut lanes = lanes(low, high);
        let tail = decoded_tail.iter().zip(minima_tail).zip(steps_tail).zip(b_tail);
        for (lane, (((&x, &min), &step), &b)) in lanes.iter_mut().zip(tail) {
            *lane += x * (min + step * b as f32);
        }
        super::reduce(lanes)
    }

    #[inline(always)]
    fn l2(simd: V3, steps: &[f32], a: &[u8], b: &[u8]) -> f32 {
        let avx = simd.avx;
        let (steps, steps_tail) = steps.as_chunks::<16>();
        let (a, a_tail) = a.as_chunks::<16>();
        let (b, b_tail) = b.as_chunks::<16>();
        let mut low = avx._mm256_setzero_ps();
        let mut high = avx._mm256_setzero_ps();
        for ((steps, a), b) in steps.iter().zip(a).zip(b) {
            let (step_low, step_high) = halves(steps);
            let (a_low, a_high) = widen(simd, a);
            let (b_low, b_high) = widen(simd, b);
            // Byte differences are small integers: subtracting them as integers
            // equals subtracting their exact f32 conversions.
            let term = |step, a, b| {
                let difference = avx._mm256_cvtepi32_ps(simd.avx2._mm256_sub_epi32(a, b));
                let delta = avx._mm256_mul_ps(difference, step);
                avx._mm256_mul_ps(delta, delta)
            };
            low = avx._mm256_add_ps(low, term(step_low, a_low, b_low));
            high = avx._mm256_add_ps(high, term(step_high, a_high, b_high));
        }
        let mut lanes = lanes(low, high);
        for (lane, ((&step, &a), &b)) in lanes.iter_mut().zip(steps_tail.iter().zip(a_tail).zip(b_tail)) {
            let delta = (a as f32 - b as f32) * step;
            *lane += delta * delta;
        }
        super::reduce(lanes)
    }
}

#[inline(always)]
fn reduce(mut lanes: [f32; 16]) -> f32 {
    for width in [8, 4, 2, 1] {
        for lane in 0..width {
            lanes[lane] += lanes[lane + width];
        }
    }
    lanes[0]
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
    reduce(lanes)
}

#[cfg(test)]
mod tests {
    use super::*;

    const METRICS: [Metric; 3] = [Metric::Cosine, Metric::Dot, Metric::L2];

    #[test]
    fn every_code_kernel_equals_the_scalar_formula_bit_for_bit() {
        // The formula before vectorization: one `sum_terms` per score.
        let reference = |metric, minima: &[f32], steps: &[f32], a: &[u8], b: &[u8]| match metric {
            Metric::Cosine | Metric::Dot => sum_terms(minima.len(), |i| {
                let x = minima[i] + steps[i] * a[i] as f32;
                let y = minima[i] + steps[i] * b[i] as f32;
                x * y
            }),
            Metric::L2 => -sum_terms(minima.len(), |i| {
                let delta = (a[i] as f32 - b[i] as f32) * steps[i];
                delta * delta
            }),
        };
        let mut rng = crate::util::SplitMix64::new(7);
        for dim in (1..=40).chain([63, 64, 100, 255, 256, 1024, 1027]) {
            let n = 24;
            let codes: Vec<u8> = (0..n * dim).map(|_| rng.next_u64() as u8).collect();
            let mut params: Vec<f32> = (0..dim)
                .map(|_| (rng.below(2001) as f32 - 1000.0) / 997.0)
                .collect();
            params.extend((0..dim).map(|i| {
                // Include constant dimensions (zero steps) and tiny steps.
                [0.0, 1e-7, 0.013, 0.5][(i + rng.below(4) as usize) % 4]
            }));
            for metric in METRICS {
                let codec = Int8Codec {
                    metric,
                    dim,
                    params: params.clone(),
                };
                let (minima, steps) = params.split_at(dim);
                let code = |i: usize| &codes[i * dim..(i + 1) * dim];
                let ids: Vec<u32> = (0..n as u32).rev().chain([3, 3]).collect();
                for i in 0..n {
                    let mut many = Vec::new();
                    codec.score_codes_many(code(i), &codes, &ids, &mut many);
                    for (&j, &batched) in ids.iter().zip(&many) {
                        let (a, b) = (code(i), code(j as usize));
                        let expected = reference(metric, minima, steps, a, b).to_bits();
                        assert_eq!(codec.score_codes(a, b).to_bits(), expected);
                        assert_eq!(batched.to_bits(), expected);
                        let portable = match metric {
                            Metric::L2 => -portable_l2(steps, a, b),
                            _ => portable_dot(minima, steps, a, b),
                        };
                        assert_eq!(portable.to_bits(), expected, "{metric:?} {dim}");
                    }
                    // A query against each code: the dispatched kernel, the portable
                    // one and the formula agree.
                    let vector: Vec<f32> = (0..dim)
                        .map(|_| (rng.below(4001) as f32 - 2000.0) / 1013.0)
                        .collect();
                    let query = codec.prepare(&vector);
                    let b = code(i);
                    let expected = match metric {
                        Metric::L2 => -sum_terms(dim, |d| {
                            let delta = query.values[d] - steps[d] * b[d] as f32;
                            delta * delta
                        }),
                        _ => query.constant + sum_terms(dim, |d| query.values[d] * b[d] as f32),
                    };
                    assert_eq!(codec.score(&query, b).to_bits(), expected.to_bits());
                    let portable = portable_query(metric, &query, steps, b);
                    assert_eq!(portable.to_bits(), expected.to_bits(), "{metric:?} {dim}");
                }
            }
        }
    }

    #[test]
    fn batched_query_scores_equal_the_scalar_formula_bit_for_bit() {
        // The formula before vectorization: one `sum_terms` per score.
        let reference = |codec: &Int8Codec, query: &Int8Query, code: &[u8]| {
            let (dim, values) = (codec.dim, &query.values);
            let steps = &codec.params[dim..];
            match codec.metric {
                Metric::Cosine | Metric::Dot => {
                    query.constant + sum_terms(dim, |i| values[i] * code[i] as f32)
                }
                Metric::L2 => -sum_terms(dim, |i| {
                    let delta = values[i] - steps[i] * code[i] as f32;
                    delta * delta
                }),
            }
        };
        let mut rng = crate::util::SplitMix64::new(11);
        for dim in (1..=40).chain([63, 64, 100, 255, 256, 1024, 1027]) {
            let n = 23;
            let codes: Vec<u8> = (0..n * dim).map(|_| rng.next_u64() as u8).collect();
            let mut params: Vec<f32> = (0..dim)
                .map(|_| (rng.below(2001) as f32 - 1000.0) / 997.0)
                .collect();
            params.extend((0..dim).map(|i| {
                // Include constant dimensions (zero steps) and tiny steps.
                [0.0, 1e-7, 0.013, 0.5][(i + rng.below(4) as usize) % 4]
            }));
            let vector: Vec<f32> = (0..dim)
                .map(|_| (rng.below(2001) as f32 - 1000.0) / 300.0)
                .collect();
            for metric in METRICS {
                let codec = Int8Codec {
                    metric,
                    dim,
                    params: params.clone(),
                };
                let query = codec.prepare(&vector);
                let steps = &params[dim..];
                let code = |i: u32| &codes[i as usize * dim..(i as usize + 1) * dim];
                // Groups of four and a remainder, out of order, with repeats.
                let ids: Vec<u32> = (0..n as u32).rev().chain([3, 3, 0]).collect();
                for len in [0, 1, 3, 4, 5, 8, ids.len()] {
                    let mut many = vec![f32::NAN];
                    codec.score_many(&query, &codes, &ids[..len], &mut many);
                    assert!(many[0].is_nan(), "appends");
                    assert_eq!(many.len(), len + 1);
                    for (&id, batched) in ids.iter().zip(&many[1..]) {
                        let expected = reference(&codec, &query, code(id)).to_bits();
                        assert_eq!(batched.to_bits(), expected, "{metric:?} {dim}");
                        assert_eq!(codec.score(&query, code(id)).to_bits(), expected);
                        let portable = portable_query(metric, &query, steps, code(id));
                        assert_eq!(portable.to_bits(), expected);
                    }
                }
            }
        }
    }

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
