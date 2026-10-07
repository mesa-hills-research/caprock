//! Top-k selection with deterministic ties (unit U03).
//!
//! The order of results everywhere: score descending, then key ascending (bytes). Scores
//! are never NaN; `-0.0` and `0.0` are equal.

use std::cmp::Ordering;

use crate::types::Hit;

/// Resolves a candidate id (an ordinal in a segment) to its key, for tie-breaking.
pub trait KeyLookup {
    fn key(&self, id: u32) -> &str;
}

/// Keys held in a vector (tests and small callers).
impl KeyLookup for Vec<String> {
    fn key(&self, id: u32) -> &str {
        &self[id as usize]
    }
}

/// The result order of two (score, key) pairs: `Less` when `a` comes first.
pub fn order(a_score: f32, a_key: &str, b_score: f32, b_key: &str) -> Ordering {
    todo!("U03 topk: order")
}

/// [`order`] for hits.
pub fn hit_order(a: &Hit, b: &Hit) -> Ordering {
    todo!("U03 topk: hit_order")
}

/// The best `k` candidates seen, by [`order`].
#[derive(Debug, Clone)]
pub struct TopK {
    k: usize,
    items: Vec<(f32, u32)>,
}

impl TopK {
    /// Panics if `k == 0`.
    pub fn new(k: usize) -> TopK {
        todo!("U03 topk: TopK::new")
    }

    pub fn k(&self) -> usize {
        todo!("U03 topk: TopK::k")
    }

    pub fn len(&self) -> usize {
        todo!("U03 topk: TopK::len")
    }

    pub fn is_empty(&self) -> bool {
        todo!("U03 topk: TopK::is_empty")
    }

    /// While full, the worst kept score: a candidate scoring below it can never enter
    /// (one scoring equal enters only if its key sorts before the worst's). `None` while
    /// fewer than `k` are kept.
    pub fn threshold(&self) -> Option<f32> {
        todo!("U03 topk: TopK::threshold")
    }

    /// Offers a candidate; returns whether it was kept. Offering the same id twice is a
    /// caller error (debug builds may panic).
    pub fn offer(&mut self, score: f32, id: u32, keys: &dyn KeyLookup) -> bool {
        todo!("U03 topk: TopK::offer")
    }

    /// The kept candidates in result order.
    pub fn into_sorted(self, keys: &dyn KeyLookup) -> Vec<(f32, u32)> {
        todo!("U03 topk: TopK::into_sorted")
    }
}

/// Merges result lists that are each in result order and hold distinct keys (no key in
/// two lists) into the best `k`, in result order.
pub fn merge_hits(lists: Vec<Vec<Hit>>, k: usize) -> Vec<Hit> {
    todo!("U03 topk: merge_hits")
}
