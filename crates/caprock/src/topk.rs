//! Top-k selection with deterministic ties.
//!
//! The order of results everywhere: score descending, then key ascending (bytes). Scores
//! are never NaN; `-0.0` and `0.0` are equal.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, binary_heap::PeekMut};

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

// Unlike total_cmp, numeric comparison treats the two signed zeros as equal.
fn score_order(a: f32, b: f32) -> Ordering {
    b.partial_cmp(&a).expect("top-k scores must not be NaN")
}

/// The result order of two (score, key) pairs: `Less` when `a` comes first.
pub fn order(a_score: f32, a_key: &str, b_score: f32, b_key: &str) -> Ordering {
    score_order(a_score, b_score).then_with(|| a_key.as_bytes().cmp(b_key.as_bytes()))
}

/// [`order`] for hits.
pub fn hit_order(a: &Hit, b: &Hit) -> Ordering {
    order(a.score, &a.key, b.score, &b.key)
}

// Resolve keys only on score ties, including during heap maintenance and sorting.
fn candidate_order(a: &(f32, u32), b: &(f32, u32), keys: &dyn KeyLookup) -> Ordering {
    score_order(a.0, b.0).then_with(|| keys.key(a.1).as_bytes().cmp(keys.key(b.1).as_bytes()))
}

/// The best `k` candidates seen, by [`order`].
#[derive(Debug, Clone)]
pub struct TopK {
    k: usize,
    // A max-heap in result order: the worst retained candidate is at the root.
    items: Vec<(f32, u32)>,
}

impl TopK {
    /// Panics if `k == 0`.
    pub fn new(k: usize) -> TopK {
        assert!(k > 0, "top-k capacity must be positive");
        TopK {
            k,
            items: Vec::with_capacity(k),
        }
    }

    #[cfg(feature = "internals")]
    pub fn k(&self) -> usize {
        self.k
    }

    #[cfg(feature = "internals")]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    #[cfg(feature = "internals")]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// While full, the worst kept score: a candidate scoring below it can never enter
    /// (one scoring equal enters only if its key sorts before the worst's). `None` while
    /// fewer than `k` are kept.
    pub fn threshold(&self) -> Option<f32> {
        (self.items.len() == self.k).then(|| self.items[0].0)
    }

    /// Offers a candidate; returns whether it was kept. Offering the same id twice is a
    /// caller error (debug builds may panic).
    pub fn offer(&mut self, score: f32, id: u32, keys: &dyn KeyLookup) -> bool {
        debug_assert!(!score.is_nan(), "top-k scores must not be NaN");
        let candidate = (score, id);
        if self.items.len() < self.k {
            let mut child = self.items.len();
            self.items.push(candidate);
            // Sift the new leaf up until its parent is at least as bad.
            while child > 0 {
                let parent = (child - 1) / 2;
                if candidate_order(&self.items[child], &self.items[parent], keys)
                    != Ordering::Greater
                {
                    break;
                }
                self.items.swap(child, parent);
                child = parent;
            }
            return true;
        }

        if candidate_order(&candidate, &self.items[0], keys) != Ordering::Less {
            return false;
        }
        self.items[0] = candidate;
        self.sift_down(keys);
        true
    }

    fn sift_down(&mut self, keys: &dyn KeyLookup) {
        let len = self.items.len();
        let mut parent = 0;
        while parent < len / 2 {
            let mut child = parent * 2 + 1;
            let right = child + 1;
            if right < len
                && candidate_order(&self.items[right], &self.items[child], keys)
                    == Ordering::Greater
            {
                child = right;
            }
            if candidate_order(&self.items[child], &self.items[parent], keys) != Ordering::Greater {
                break;
            }
            self.items.swap(parent, child);
            parent = child;
        }
    }

    /// The kept candidates in result order.
    pub fn into_sorted(self, keys: &dyn KeyLookup) -> Vec<(f32, u32)> {
        let mut items = self.items;
        items.sort_unstable_by(|a, b| candidate_order(a, b, keys));
        items
    }
}

// One cursor per nonempty list. Reverse result order so BinaryHeap selects the best
// head first. Equality concerns the head's result rank, not its metadata or tail.
struct MergeCursor {
    head: Hit,
    tail: std::vec::IntoIter<Hit>,
}

impl PartialEq for MergeCursor {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for MergeCursor {}

impl PartialOrd for MergeCursor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeCursor {
    fn cmp(&self, other: &Self) -> Ordering {
        hit_order(&other.head, &self.head)
    }
}

/// Merges result lists that are each in result order and hold distinct keys (no key in
/// two lists) into the best `k`, in result order.
pub fn merge_hits(lists: Vec<Vec<Hit>>, k: usize) -> Vec<Hit> {
    if k == 0 {
        return Vec::new();
    }
    let cursors: Vec<_> = lists
        .into_iter()
        .filter_map(|list| {
            let mut tail = list.into_iter();
            tail.next().map(|head| MergeCursor { head, tail })
        })
        .collect();
    let mut heap = BinaryHeap::from(cursors);
    let mut hits = Vec::new();
    while hits.len() < k {
        let Some(mut cursor) = heap.peek_mut() else {
            break;
        };
        if let Some(next) = cursor.tail.next() {
            hits.push(std::mem::replace(&mut cursor.head, next));
            // Dropping PeekMut restores the heap after advancing this list.
        } else {
            hits.push(PeekMut::pop(cursor).head);
        }
    }
    hits
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(score: f32, key: &str) -> Hit {
        Hit {
            score,
            key: key.into(),
            path: format!("path/{key}"),
            language: "rust".into(),
        }
    }

    #[test]
    #[should_panic(expected = "top-k capacity must be positive")]
    fn zero_capacity_panics() {
        TopK::new(0);
    }

    #[test]
    fn empty_and_partial_thresholds() {
        let keys = vec!["a".into(), "b".into()];
        let mut top = TopK::new(3);
        assert_eq!(top.threshold(), None);
        assert!(top.clone().into_sorted(&keys).is_empty());
        assert!(top.offer(2.0, 0, &keys));
        assert_eq!(top.threshold(), None);
        assert!(top.offer(1.0, 1, &keys));
        assert_eq!(top.threshold(), None);
        assert_eq!(top.into_sorted(&keys), vec![(2.0, 0), (1.0, 1)]);
    }

    #[test]
    fn singleton_signed_zero_and_infinities() {
        let keys = vec!["z".into(), "é".into(), "a".into(), "b".into()];
        let mut top = TopK::new(1);
        assert!(top.offer(f32::NEG_INFINITY, 1, &keys));
        assert!(top.offer(-0.0, 0, &keys));
        assert!(!top.offer(0.0, 1, &keys));
        assert!(top.offer(0.0, 2, &keys));
        assert_eq!(top.threshold(), Some(0.0));
        assert!(top.offer(f32::INFINITY, 3, &keys));
        assert_eq!(top.into_sorted(&keys), vec![(f32::INFINITY, 3)]);
    }

    #[test]
    fn unequal_scores_do_not_resolve_keys() {
        struct NoKeys;
        impl KeyLookup for NoKeys {
            fn key(&self, _: u32) -> &str {
                panic!("keys are only needed for ties");
            }
        }
        let mut top = TopK::new(2);
        assert!(top.offer(2.0, 0, &NoKeys));
        assert!(top.offer(1.0, 1, &NoKeys));
        assert!(!top.offer(0.0, 2, &NoKeys));
        assert!(top.offer(3.0, 3, &NoKeys));
        assert_eq!(top.into_sorted(&NoKeys), vec![(3.0, 3), (2.0, 0)]);
    }

    #[test]
    fn merge_handles_empty_lists_zero_limit_and_signed_zero() {
        let lists = vec![
            vec![],
            vec![hit(f32::INFINITY, "z"), hit(-0.0, "b")],
            vec![],
            vec![hit(0.0, "a"), hit(f32::NEG_INFINITY, "c")],
        ];
        assert!(merge_hits(lists.clone(), 0).is_empty());
        assert_eq!(merge_hits(lists.clone(), 1), vec![hit(f32::INFINITY, "z")]);
        assert_eq!(
            merge_hits(lists, 10),
            vec![
                hit(f32::INFINITY, "z"),
                hit(0.0, "a"),
                hit(-0.0, "b"),
                hit(f32::NEG_INFINITY, "c"),
            ]
        );
        assert!(merge_hits(vec![vec![], vec![]], 3).is_empty());
    }
}
