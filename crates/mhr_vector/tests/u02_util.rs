//! Acceptance tests of unit U02 (util): stable hash, PRNG, bitset.
#![cfg(feature = "internals")]

use mhr_vector::util::{Bitset, SplitMix64, mix64, stable_hash};
use mhr_vector_oracle::rng::Rng;

/// The specification, written out independently.
fn reference_hash(bytes: &[u8], seed: u64) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ seed;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    mhr_vector_oracle::rng::mix(h)
}

#[test]
fn stable_hash_follows_the_specification() {
    let mut rng = Rng::new(9);
    for len in 0..200usize {
        let bytes: Vec<u8> = (0..len).map(|_| rng.next_u64() as u8).collect();
        for seed in [0u64, 1, 0xDEAD_BEEF, u64::MAX] {
            assert_eq!(stable_hash(&bytes, seed), reference_hash(&bytes, seed));
        }
    }
    assert_eq!(mix64(0), 0);
    assert_eq!(mix64(0x9E37_79B9_7F4A_7C15), 0xE220_A839_7B1D_CDAF);
}

#[test]
fn splitmix_matches_the_reference_sequence() {
    for seed in [0u64, 42, u64::MAX] {
        let mut ours = SplitMix64::new(seed);
        let mut theirs = Rng::new(seed);
        for _ in 0..1000 {
            assert_eq!(ours.next_u64(), theirs.next_u64());
        }
        let mut a = SplitMix64::new(seed);
        let mut b = Rng::new(seed);
        for n in 1..500u64 {
            assert_eq!(a.below(n), b.next_u64() % n);
        }
    }
}

#[test]
fn samples_are_distinct_sorted_and_deterministic() {
    for (n, m) in [
        (0u64, 5usize),
        (5, 0),
        (5, 5),
        (5, 10),
        (1000, 10),
        (100_000, 65_536),
        (70_000, 65_536),
    ] {
        let s = SplitMix64::new(7).sample(n, m);
        assert_eq!(s, SplitMix64::new(7).sample(n, m), "deterministic");
        assert_eq!(s.len(), (m as u64).min(n) as usize, "n {n} m {m}");
        assert!(s.windows(2).all(|w| w[0] < w[1]), "sorted and distinct");
        assert!(s.iter().all(|&x| x < n));
        if m as u64 >= n {
            assert_eq!(s, (0..n).collect::<Vec<_>>());
        }
    }
    assert_ne!(
        SplitMix64::new(1).sample(1_000_000, 100),
        SplitMix64::new(2).sample(1_000_000, 100)
    );
}

#[test]
fn bitset_operations() {
    for len in [0usize, 1, 63, 64, 65, 1000] {
        let mut b = Bitset::new(len);
        assert_eq!(b.len(), len);
        assert_eq!(b.is_empty(), len == 0);
        assert_eq!(b.count_ones(), 0);
        let picks: Vec<usize> = (0..len).filter(|i| i % 3 == 0 || i % 7 == 1).collect();
        for &i in &picks {
            assert!(b.set(i));
            assert!(!b.set(i), "second set reports already set");
        }
        assert_eq!(b.count_ones(), picks.len());
        assert_eq!(b.ones(), picks);
        for i in 0..len {
            assert_eq!(b.get(i), picks.contains(&i));
        }
        let bytes = b.to_bytes();
        assert_eq!(bytes.len(), len.div_ceil(64) * 8);
        assert_eq!(Bitset::from_bytes(&bytes, len), Some(b.clone()));
        if let Some(&first) = picks.first() {
            b.clear(first);
            assert!(!b.get(first));
            assert_eq!(b.count_ones(), picks.len() - 1);
        }
    }
    // Wrong length, or a bit set past `len`, is refused.
    assert_eq!(Bitset::from_bytes(&[0u8; 7], 10), None);
    let mut word = [0u8; 8];
    word[0] = 0b0000_0100; // bit 2 set
    assert_eq!(Bitset::from_bytes(&word, 2), None);
    assert!(Bitset::from_bytes(&word, 3).unwrap().get(2));
}

#[test]
fn bitset_out_of_range_panics() {
    panics_for_real(|| {
        Bitset::new(10).get(10);
    });
}

/// Runs `f`, which must panic for a reason other than an unimplemented stub.
fn panics_for_real(f: impl FnOnce() + std::panic::UnwindSafe) {
    let err = std::panic::catch_unwind(f).expect_err("expected a panic");
    let msg = err
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(!msg.contains("not yet implemented"), "{msg}");
}
