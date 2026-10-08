//! Top-k selection and merging with deterministic ties.
#![cfg(feature = "internals")]

use std::cmp::Ordering;

use caprock::Hit;
use caprock::topk::{TopK, hit_order, merge_hits, order};
use caprock_oracle::generate::key;
use caprock_oracle::rng::Rng;

fn reference(scores: &[f32], keys: &[String], k: usize) -> Vec<(f32, u32)> {
    let mut all: Vec<(f32, u32)> = scores
        .iter()
        .enumerate()
        .map(|(i, &s)| (s, i as u32))
        .collect();
    all.sort_by(|a, b| {
        b.0.partial_cmp(&a.0).unwrap().then_with(|| {
            keys[a.1 as usize]
                .as_bytes()
                .cmp(keys[b.1 as usize].as_bytes())
        })
    });
    all.truncate(k);
    all
}

#[test]
fn order_is_score_desc_then_key_asc() {
    assert_eq!(order(2.0, "b", 1.0, "a"), Ordering::Less);
    assert_eq!(order(1.0, "a", 1.0, "b"), Ordering::Less);
    assert_eq!(order(1.0, "b", 1.0, "a"), Ordering::Greater);
    assert_eq!(order(1.0, "a", 1.0, "a"), Ordering::Equal);
    assert_eq!(
        order(0.0, "a", -0.0, "b"),
        Ordering::Less,
        "-0.0 equals 0.0"
    );
    assert_eq!(order(-0.0, "b", 0.0, "a"), Ordering::Greater);
    let h = |s: f32, k: &str| Hit {
        key: k.into(),
        score: s,
        path: String::new(),
        language: String::new(),
    };
    assert_eq!(hit_order(&h(3.0, "z"), &h(2.0, "a")), Ordering::Less);
}

#[test]
fn topk_equals_a_full_sort() {
    let mut rng = Rng::new(4);
    for trial in 0..200 {
        let n = rng.below(300) as usize;
        let k = 1 + rng.below(60) as usize;
        let levels = 1 + rng.below(8); // few distinct scores: many ties
        let keys: Vec<String> = (0..n as u64).map(|i| key(i + trial * 1000)).collect();
        let scores: Vec<f32> = (0..n)
            .map(|_| rng.below(levels) as f32 * 0.5 - 1.0)
            .collect();
        let mut order_in: Vec<u32> = (0..n as u32).collect();
        for i in (1..order_in.len()).rev() {
            order_in.swap(i, rng.below(i as u64 + 1) as usize);
        }
        let mut top = TopK::new(k);
        assert!(top.is_empty());
        for &id in &order_in {
            top.offer(scores[id as usize], id, &keys);
            assert!(top.len() <= k);
        }
        assert_eq!(top.k(), k);
        let want = reference(&scores, &keys, k);
        if n >= k {
            assert_eq!(top.threshold(), want.last().map(|x| x.0));
        } else {
            assert_eq!(top.threshold(), None);
        }
        assert_eq!(top.into_sorted(&keys), want, "trial {trial}");
    }
}

#[test]
fn offer_reports_whether_kept() {
    let keys: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
    let mut top = TopK::new(2);
    assert!(top.offer(1.0, 1, &keys));
    assert!(top.offer(2.0, 2, &keys));
    assert!(!top.offer(0.5, 3, &keys), "worse than the worst kept");
    assert!(
        top.offer(1.0, 0, &keys),
        "ties the worst with a smaller key"
    );
    assert_eq!(top.into_sorted(&keys), vec![(2.0, 2), (1.0, 0)]);
}

#[test]
fn merge_equals_top_of_the_union() {
    let mut rng = Rng::new(5);
    for trial in 0..100 {
        let n = rng.below(200) as usize;
        let k = 1 + rng.below(50) as usize;
        let mut all: Vec<Hit> = (0..n as u64)
            .map(|i| Hit {
                key: key(i + trial * 1000),
                score: rng.below(5) as f32,
                path: format!("p{i}"),
                language: "x".into(),
            })
            .collect();
        all.sort_by(hit_order);
        let parts = 1 + rng.below(5) as usize;
        let mut lists: Vec<Vec<Hit>> = vec![Vec::new(); parts];
        for h in &all {
            lists[rng.below(parts as u64) as usize].push(h.clone());
        }
        let merged = merge_hits(lists, k);
        let mut want = all.clone();
        want.truncate(k);
        assert_eq!(merged, want, "trial {trial}");
    }
    assert!(merge_hits(Vec::new(), 3).is_empty());
}
