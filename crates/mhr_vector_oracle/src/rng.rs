//! A small, fixed PRNG (SplitMix64) so generated data never changes with a dependency
//! upgrade. Not for cryptography.

/// SplitMix64 (Steele, Lea, Flood 2014).
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng { state: seed }
    }

    /// A generator for a sub-stream: `seed` mixed with `stream`.
    pub fn derive(seed: u64, stream: u64) -> Rng {
        let mut r = Rng::new(seed ^ stream.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        r.next_u64();
        r
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix(self.state)
    }

    /// Uniform in `[0, 1)` with 53 bits.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in `0..n` (`n > 0`); the slight modulo bias is irrelevant here.
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0);
        self.next_u64() % n
    }

    /// A standard normal sample (Box-Muller, in `f64`).
    pub fn gaussian(&mut self) -> f64 {
        let u1 = 1.0 - self.next_f64(); // (0, 1]
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Whether an event of probability `p` happens.
    pub fn chance(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }
}

/// The SplitMix64 finalizer: a good 64-bit mixing function.
pub fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_values() {
        // SplitMix64 with seed 0: published first outputs.
        let mut r = Rng::new(0);
        assert_eq!(r.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(r.next_u64(), 0x6E78_9E6A_A1B9_65F4);
    }
}
