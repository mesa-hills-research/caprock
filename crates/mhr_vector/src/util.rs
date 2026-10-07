//! Small shared tools (unit U02): a stable hash, a seeded PRNG and a bitset.
//!
//! Everything here is persisted or determines persisted bytes, so the algorithms are fixed
//! by this documentation and must never change within a format version (std's
//! `DefaultHasher` is unspecified and may change between Rust releases, so it must not be
//! used for anything persisted or for any ordering).

/// A stable 64-bit hash of `bytes`: FNV-1a 64 (offset basis `0xcbf29ce484222325` xor
/// `seed`, prime `0x100000001b3`, one byte at a time), then the SplitMix64 finalizer
/// ([`mix64`]).
pub fn stable_hash(bytes: &[u8], seed: u64) -> u64 {
    todo!("U02 util: stable_hash")
}

/// The SplitMix64 finalizer: `z ^= z >> 30; z *= 0xBF58476D1CE4E5B9; z ^= z >> 27;
/// z *= 0x94D049BB133111EB; z ^= z >> 31` (wrapping).
pub fn mix64(z: u64) -> u64 {
    todo!("U02 util: mix64")
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
        todo!("U02 util: SplitMix64::next_u64")
    }

    /// Uniform in `0..n` (`n > 0`): `next_u64() % n`.
    pub fn below(&mut self, n: u64) -> u64 {
        todo!("U02 util: SplitMix64::below")
    }

    /// A deterministic sample of `m` distinct values from `0..n` (all of them, in order,
    /// when `m >= n`), returned sorted.
    pub fn sample(&mut self, n: u64, m: usize) -> Vec<u64> {
        todo!("U02 util: SplitMix64::sample")
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
        todo!("U02 util: Bitset::new")
    }

    pub fn len(&self) -> usize {
        todo!("U02 util: Bitset::len")
    }

    pub fn is_empty(&self) -> bool {
        todo!("U02 util: Bitset::is_empty")
    }

    /// Panics if `i >= len`.
    pub fn get(&self, i: usize) -> bool {
        todo!("U02 util: Bitset::get")
    }

    /// Sets bit `i`; returns whether it was clear. Panics if `i >= len`.
    pub fn set(&mut self, i: usize) -> bool {
        todo!("U02 util: Bitset::set")
    }

    /// Clears bit `i`. Panics if `i >= len`.
    pub fn clear(&mut self, i: usize) {
        todo!("U02 util: Bitset::clear")
    }

    /// Number of set bits.
    pub fn count_ones(&self) -> usize {
        todo!("U02 util: Bitset::count_ones")
    }

    /// The set bits, ascending.
    pub fn ones(&self) -> Vec<usize> {
        todo!("U02 util: Bitset::ones")
    }

    /// Little-endian 64-bit words (`len.div_ceil(64)` of them; bits past `len` clear).
    pub fn to_bytes(&self) -> Vec<u8> {
        todo!("U02 util: Bitset::to_bytes")
    }

    /// The inverse of [`Bitset::to_bytes`]; `None` if the byte count does not fit `len`
    /// or a bit past `len` is set.
    pub fn from_bytes(bytes: &[u8], len: usize) -> Option<Bitset> {
        todo!("U02 util: Bitset::from_bytes")
    }
}
