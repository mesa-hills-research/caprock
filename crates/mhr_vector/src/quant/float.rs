//! Canonical scores of f32 codes read in place. A code is the stored vector as
//! little-endian `f32`s; these kernels give [`distance::score`]'s bits without decoding
//! it first: element `i` goes to lane `i % 16`, each lane adds its terms in increasing
//! `i`, and the lanes are reduced in the fixed tree. The AVX2 kernel scores four codes
//! at once, so their independent accumulators overlap.

use crate::config::Metric;
use crate::distance::{self, LANES};

/// `distance::score(metric, query, stored)` where `code` holds `stored`.
pub(super) fn score(metric: Metric, query: &[f32], code: &[u8]) -> f32 {
    assert_eq!(code.len(), query.len() * 4, "f32 code length mismatch");
    #[cfg(target_arch = "x86_64")]
    if let Some(simd) = super::x86_v3() {
        return pulp::Simd::vectorize(
            simd,
            avx2::One {
                simd,
                metric,
                query,
                code,
            },
        );
    }
    portable(metric, query, code)
}

/// Appends [`score`] for the code of each id in `ids` (`codes` holds `4 * dim` bytes
/// per id).
pub(super) fn score_many(
    metric: Metric,
    query: &[f32],
    codes: &[u8],
    ids: &[u32],
    scores: &mut Vec<f32>,
) {
    #[cfg(target_arch = "x86_64")]
    if let Some(simd) = super::x86_v3() {
        return pulp::Simd::vectorize(
            simd,
            avx2::Many {
                simd,
                metric,
                query,
                codes,
                ids,
                scores,
            },
        );
    }
    let len = query.len() * 4;
    scores.extend(ids.iter().map(|&id| {
        let start = id as usize * len;
        portable(metric, query, &codes[start..start + len])
    }));
}

fn portable(metric: Metric, query: &[f32], code: &[u8]) -> f32 {
    let elements = code.as_chunks::<4>().0;
    assert_eq!(elements.len(), query.len(), "f32 code length mismatch");
    let (query, query_tail) = query.as_chunks::<LANES>();
    let (stored, stored_tail) = elements.as_chunks::<LANES>();
    let mut acc = [0.0f32; LANES];
    // The terms of `distance::dot` and `distance::l2_squared`, query first.
    match metric {
        Metric::Cosine | Metric::Dot => {
            let term = |q: f32, x: &[u8; 4]| q * f32::from_le_bytes(*x);
            for (query, stored) in query.iter().zip(stored) {
                for lane in 0..LANES {
                    acc[lane] += term(query[lane], &stored[lane]);
                }
            }
            for (lane, (&q, x)) in query_tail.iter().zip(stored_tail).enumerate() {
                acc[lane] += term(q, x);
            }
            distance::reduce(acc)
        }
        Metric::L2 => {
            let term = |q: f32, x: &[u8; 4]| {
                let difference = q - f32::from_le_bytes(*x);
                difference * difference
            };
            for (query, stored) in query.iter().zip(stored) {
                for lane in 0..LANES {
                    acc[lane] += term(query[lane], &stored[lane]);
                }
            }
            for (lane, (&q, x)) in query_tail.iter().zip(stored_tail).enumerate() {
                acc[lane] += term(q, x);
            }
            -distance::reduce(acc)
        }
    }
}

/// The same kernels with explicit AVX2 operations: lanes 0-7 and 8-15 of each
/// 16-element chunk in two registers, no fused multiply-add. x86-64 is little-endian,
/// so a code's bytes are its `f32`s.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use core::arch::x86_64::__m256;

    use pulp::x86::V3;

    use crate::config::Metric;
    use crate::distance;

    pub(super) struct One<'a> {
        pub simd: V3,
        pub metric: Metric,
        pub query: &'a [f32],
        pub code: &'a [u8],
    }

    impl pulp::WithSimd for One<'_> {
        type Output = f32;

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, _: S) -> f32 {
            let [score] = scores(self.simd, self.metric, self.query, [self.code], None);
            score
        }
    }

    pub(super) struct Many<'a> {
        pub simd: V3,
        pub metric: Metric,
        pub query: &'a [f32],
        pub codes: &'a [u8],
        pub ids: &'a [u32],
        pub scores: &'a mut Vec<f32>,
    }

    impl pulp::WithSimd for Many<'_> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, _: S) {
            let len = self.query.len() * 4;
            let code = |id: u32| {
                let start = id as usize * len;
                &self.codes[start..start + len]
            };
            let (groups, rest) = self.ids.as_chunks::<4>();
            for (g, group) in groups.iter().enumerate() {
                // Each code spans a page, where hardware prefetchers stop.
                let ahead = groups.get(g + PREFETCH_GROUPS).map(|ids| ids.map(code));
                let group = scores(self.simd, self.metric, self.query, group.map(code), ahead);
                self.scores.extend(group);
            }
            for &id in rest {
                let one = scores(self.simd, self.metric, self.query, [code(id)], None);
                self.scores.extend(one);
            }
        }
    }

    /// How many groups ahead [`Many`] prefetches.
    const PREFETCH_GROUPS: usize = 1;

    #[inline(always)]
    fn scores<const N: usize>(
        simd: V3,
        metric: Metric,
        query: &[f32],
        codes: [&[u8]; N],
        ahead: Option<[&[u8]; N]>,
    ) -> [f32; N] {
        match metric {
            Metric::Cosine | Metric::Dot => kernel(simd, query, codes, ahead, false),
            Metric::L2 => kernel(simd, query, codes, ahead, true).map(|sum| -sum),
        }
    }

    /// The canonical dot products (or squared distances, `l2`) of `query` and `N` codes,
    /// prefetching the codes `ahead`, if any.
    #[inline(always)]
    fn kernel<const N: usize>(
        simd: V3,
        query: &[f32],
        codes: [&[u8]; N],
        ahead: Option<[&[u8]; N]>,
        l2: bool,
    ) -> [f32; N] {
        let avx = simd.avx;
        let (query, query_tail) = query.as_chunks::<16>();
        let codes = codes.map(|code| {
            assert_eq!(code.len(), (query.len() * 16 + query_tail.len()) * 4);
            let (body, tail) = code.as_chunks::<64>();
            (&body[..query.len()], tail.as_chunks::<4>().0)
        });
        // Checked once here, so the loop indexes the chunks without checks.
        assert!(codes.iter().all(|(body, _)| body.len() == query.len()));
        let term = |q, x| {
            if l2 {
                let difference = avx._mm256_sub_ps(q, x);
                avx._mm256_mul_ps(difference, difference)
            } else {
                avx._mm256_mul_ps(q, x)
            }
        };
        let mut low = [avx._mm256_setzero_ps(); N];
        let mut high = [avx._mm256_setzero_ps(); N];
        for (j, q) in query.iter().enumerate() {
            if let Some(ahead) = ahead {
                // One 64-byte line per code and chunk (a hint: it never faults).
                for code in ahead {
                    let line = code.as_ptr().wrapping_add(j * 64).cast::<i8>();
                    simd.sse._mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(line);
                }
            }
            let [q_low, q_high]: [__m256; 2] = pulp::cast(*q);
            for c in 0..N {
                let [x_low, x_high]: [__m256; 2] = pulp::cast(codes[c].0[j]);
                low[c] = avx._mm256_add_ps(low[c], term(q_low, x_low));
                high[c] = avx._mm256_add_ps(high[c], term(q_high, x_high));
            }
        }
        let mut sums = [0.0; N];
        for c in 0..N {
            let mut lanes: [f32; 16] = pulp::cast([low[c], high[c]]);
            for (lane, (&q, x)) in lanes.iter_mut().zip(query_tail.iter().zip(codes[c].1)) {
                let x = f32::from_le_bytes(*x);
                *lane += if l2 {
                    let difference = q - x;
                    difference * difference
                } else {
                    q * x
                };
            }
            sums[c] = distance::reduce(lanes);
        }
        sums
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::f32s_to_le;
    use crate::util::SplitMix64;

    #[test]
    fn in_place_scores_equal_the_canonical_scores_bit_for_bit() {
        let mut rng = SplitMix64::new(13);
        for dim in (1..=40).chain([63, 64, 65, 255, 1024, 1027]) {
            let n = 11;
            // Mixed magnitudes so rounding differs between summation orders.
            let mut value = || {
                let scale = [1e-3, 1.0, 1e4][rng.below(3) as usize];
                (rng.below(2001) as f32 - 1000.0) / 997.0 * scale
            };
            let vectors: Vec<f32> = (0..n * dim).map(|_| value()).collect();
            let query: Vec<f32> = (0..dim).map(|_| value()).collect();
            let codes = f32s_to_le(&vectors);
            let ids: Vec<u32> = (0..n as u32).rev().chain([1, 1]).collect();
            for metric in [Metric::Cosine, Metric::Dot, Metric::L2] {
                for count in [0, 1, 3, 4, 5, ids.len()] {
                    let mut many = vec![f32::NAN];
                    score_many(metric, &query, &codes, &ids[..count], &mut many);
                    assert!(many[0].is_nan(), "appends");
                    assert_eq!(many.len(), count + 1);
                    for (&id, batched) in ids.iter().zip(&many[1..]) {
                        let id = id as usize;
                        let vector = &vectors[id * dim..(id + 1) * dim];
                        let code = &codes[id * dim * 4..(id + 1) * dim * 4];
                        let expected = distance::score(metric, &query, vector).to_bits();
                        assert_eq!(batched.to_bits(), expected, "{metric:?} {dim}");
                        assert_eq!(score(metric, &query, code).to_bits(), expected);
                        assert_eq!(portable(metric, &query, code).to_bits(), expected);
                    }
                }
            }
        }
    }
}
