//! Small shared tools (unit U02): a stable hash, a seeded PRNG and a bitset.
//!
//! Everything here is persisted or determines persisted bytes, so the algorithms are fixed
//! by this documentation and must never change within a format version (std's
//! `DefaultHasher` is unspecified and may change between Rust releases, so it must not be
//! used for anything persisted or for any ordering).

use std::collections::BTreeSet;

/// A stable 64-bit hash of `bytes`: FNV-1a 64 (offset basis `0xcbf29ce484222325` xor
/// `seed`, prime `0x100000001b3`, one byte at a time), then the SplitMix64 finalizer
/// ([`mix64`]).
pub fn stable_hash(bytes: &[u8], seed: u64) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325 ^ seed;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    mix64(hash)
}

/// The SplitMix64 finalizer: `z ^= z >> 30; z *= 0xBF58476D1CE4E5B9; z ^= z >> 27;
/// z *= 0x94D049BB133111EB; z ^= z >> 31` (wrapping).
pub fn mix64(z: u64) -> u64 {
    let mut z = z;
    z ^= z >> 30;
    z = z.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// SplitMix64: `state += 0x9E3779B97F4A7C15` (wrapping), output [`mix64`]`(state)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitMix64 {
    pub state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> SplitMix64 {
        SplitMix64 { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix64(self.state)
    }

    /// Uniform in `0..n` (`n > 0`): `next_u64() % n`.
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "SplitMix64::below requires a nonzero bound");
        self.next_u64() % n
    }

    /// A deterministic sample of `m` distinct values from `0..n` (all of them, in order,
    /// when `m >= n`), returned sorted.
    ///
    /// Uses Floyd's algorithm: for each `j` in `n-m..n`, draw `below(j+1)`;
    /// insert the draw if absent, otherwise insert `j`. This takes exactly `m`
    /// draws and O(m) storage. When `m >= n`, no random values are consumed.
    pub fn sample(&mut self, n: u64, m: usize) -> Vec<u64> {
        let m = m as u64;
        if m >= n {
            return (0..n).collect();
        }
        let mut selected = BTreeSet::new();
        for j in n - m..n {
            let value = self.below(j + 1);
            if !selected.insert(value) {
                selected.insert(j);
            }
        }
        selected.into_iter().collect()
    }
}

/// A fixed-length set of bits.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Bitset {
    words: Vec<u64>,
    len: usize,
}

impl Bitset {
    /// `len` bits, all clear.
    pub fn new(len: usize) -> Bitset {
        Bitset {
            words: vec![0; len.div_ceil(64)],
            len,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Panics if `i >= len`.
    pub fn get(&self, i: usize) -> bool {
        assert!(i < self.len, "bit index {i} out of bounds for {} bits", self.len);
        self.words[i / 64] & (1u64 << (i % 64)) != 0
    }

    /// Sets bit `i`; returns whether it was clear. Panics if `i >= len`.
    pub fn set(&mut self, i: usize) -> bool {
        assert!(i < self.len, "bit index {i} out of bounds for {} bits", self.len);
        let word = &mut self.words[i / 64];
        let mask = 1u64 << (i % 64);
        let was_clear = *word & mask == 0;
        *word |= mask;
        was_clear
    }

    /// Clears bit `i`. Panics if `i >= len`.
    pub fn clear(&mut self, i: usize) {
        assert!(i < self.len, "bit index {i} out of bounds for {} bits", self.len);
        self.words[i / 64] &= !(1u64 << (i % 64));
    }

    /// Number of set bits.
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|word| word.count_ones() as usize).sum()
    }

    /// The set bits, ascending.
    pub fn ones(&self) -> Vec<usize> {
        let mut ones = Vec::with_capacity(self.count_ones());
        for (i, &word) in self.words.iter().enumerate() {
            let mut remaining = word;
            while remaining != 0 {
                ones.push(i * 64 + remaining.trailing_zeros() as usize);
                remaining &= remaining - 1;
            }
        }
        ones
    }

    /// Little-endian 64-bit words (`len.div_ceil(64)` of them; bits past `len` clear).
    pub fn to_bytes(&self) -> Vec<u8> {
        self.words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    /// The inverse of [`Bitset::to_bytes`]; `None` if the byte count does not fit `len`
    /// or a bit past `len` is set.
    pub fn from_bytes(bytes: &[u8], len: usize) -> Option<Bitset> {
        let word_count = len.div_ceil(64);
        if bytes.len() != word_count * 8 {
            return None;
        }
        let mut words = Vec::with_capacity(word_count);
        for &chunk in bytes.as_chunks::<8>().0 {
            words.push(u64::from_le_bytes(chunk));
        }
        let tail_bits = len % 64;
        if tail_bits != 0 && words.last()? >> tail_bits != 0 {
            return None;
        }
        Some(Bitset { words, len })
    }
}

#[cfg(test)]
mod tests {
    use super::{Bitset, SplitMix64};

    #[test]
    fn samples_handle_empty_full_and_maximum_ranges() {
        let mut rng = SplitMix64::new(u64::MAX);
        assert!(rng.sample(0, 10).is_empty());
        assert!(rng.sample(u64::MAX, 0).is_empty());
        assert_eq!(rng.sample(4, usize::MAX), vec![0, 1, 2, 3]);
        assert_eq!(rng.state, u64::MAX, "empty and full samples use no draws");

        let sample = rng.sample(u64::MAX, 128);
        assert_eq!(sample.len(), 128);
        assert!(sample.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(sample.iter().all(|&value| value < u64::MAX));
        let mut reference = SplitMix64::new(u64::MAX);
        for _ in 0..128 {
            reference.next_u64();
        }
        assert_eq!(rng.state, reference.state, "one draw per sampled value");
    }

    #[test]
    fn bitset_encoding_is_little_endian_and_strict() {
        let mut bits = Bitset::new(70);
        for i in [0, 7, 8, 31, 63, 64, 69] {
            bits.set(i);
        }
        let bytes = [0x81, 1, 0, 0x80, 0, 0, 0, 0x80, 0x21, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(bits.to_bytes(), bytes);
        assert_eq!(Bitset::from_bytes(&bytes, 70), Some(bits));
        assert_eq!(Bitset::from_bytes(&[], 0), Some(Bitset::default()));
        assert_eq!(Bitset::from_bytes(&[0; 8], 0), None);
        assert_eq!(Bitset::from_bytes(&[], usize::MAX), None);
        assert_eq!(Bitset::from_bytes(&bytes[..15], 70), None);

        for len in 1usize..130 {
            let mut bytes = Bitset::new(len).to_bytes();
            for bit in len..len.div_ceil(64) * 64 {
                bytes[bit / 8] = 1 << (bit % 8);
                assert_eq!(Bitset::from_bytes(&bytes, len), None);
                bytes[bit / 8] = 0;
            }
        }
    }

    #[test]
    fn all_bit_accessors_reject_out_of_bounds_indices() {
        for len in [0usize, 1, 63, 64, 65] {
            for i in [len, len + 1, usize::MAX] {
                assert!(std::panic::catch_unwind(|| Bitset::new(len).get(i)).is_err());
                assert!(std::panic::catch_unwind(|| Bitset::new(len).set(i)).is_err());
                assert!(std::panic::catch_unwind(|| Bitset::new(len).clear(i)).is_err());
            }
        }
    }

    #[test]
    #[should_panic(expected = "nonzero bound")]
    fn zero_random_bound_is_rejected() {
        SplitMix64::new(0).below(0);
    }
}
