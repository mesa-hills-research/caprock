//! Checks an index's results for one query against the oracle.
//!
//! Every result list must:
//! - hold at most `k` hits, with no key twice;
//! - hold only live keys whose records match the filter;
//! - hold exactly `min(k, live records matching the filter)` hits (an index may not
//!   return fewer because its approximate search found fewer);
//! - be sorted by its own reported scores, descending, ties by key ascending.
//!
//! Depending on the [`Expectation`]:
//! - `canonical_scores`: each reported score equals the canonical score (with `==`; this
//!   holds whenever the index rescored with full-precision vectors);
//! - `exact`: the list equals the oracle's top `k` (keys and scores, rank by rank);
//! - `min_recall`: the query's recall (below) is at least this.
//!
//! **Recall@k.** Let `T` be the oracle's top `k` and `s_k` the score of its last hit.
//! Hits scoring above `s_k` are all in `T`; hits scoring exactly `s_k` are
//! interchangeable (ties at the boundary). Recall is
//! `(|hits ∩ {t in T: s(t) > s_k}| + min(|{t in T: s(t) = s_k}|, |{h: s(h) = s_k}|)) / |T|`,
//! with canonical scores, and 1 when `T` is empty.

use std::collections::HashSet;

use crate::filter::Filter;
use crate::oracle::{Hit, Oracle, OracleError, hit_order};

/// What a query's results must satisfy beyond the universal rules.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Expectation {
    pub exact: bool,
    pub canonical_scores: bool,
    pub min_recall: Option<f64>,
}

impl Expectation {
    /// Exact search: the oracle's list, bit for bit.
    pub fn exact() -> Expectation {
        Expectation {
            exact: true,
            canonical_scores: true,
            min_recall: None,
        }
    }

    /// Approximate search with full-precision rescoring.
    pub fn rescored(min_recall: Option<f64>) -> Expectation {
        Expectation {
            exact: false,
            canonical_scores: true,
            min_recall,
        }
    }

    /// Approximate search without rescoring: scores are estimates.
    pub fn approximate(min_recall: Option<f64>) -> Expectation {
        Expectation {
            exact: false,
            canonical_scores: false,
            min_recall,
        }
    }
}

/// A broken rule. The `Display` text names the rule first, so the first word or two
/// make a stable signature for bucketing.
#[derive(Debug, Clone, PartialEq)]
pub enum Violation {
    Oracle(OracleError),
    TooMany {
        got: usize,
        k: usize,
    },
    TooFew {
        got: usize,
        expected: usize,
    },
    Duplicate {
        key: String,
    },
    NotLive {
        key: String,
    },
    FilterViolated {
        key: String,
        path: String,
        language: String,
    },
    OutOfOrder {
        rank: usize,
    },
    ScoreMismatch {
        key: String,
        got: f32,
        canonical: f32,
    },
    ExactMismatch {
        rank: usize,
        got: Option<(String, f32)>,
        expected: Option<(String, f32)>,
    },
    RecallBelow {
        recall: f64,
        min: f64,
    },
}

impl Violation {
    /// A short stable name for the rule: `too-many`, `recall-below`, ...
    pub fn kind(&self) -> &'static str {
        match self {
            Violation::Oracle(_) => "oracle-refused",
            Violation::TooMany { .. } => "too-many",
            Violation::TooFew { .. } => "too-few",
            Violation::Duplicate { .. } => "duplicate",
            Violation::NotLive { .. } => "not-live",
            Violation::FilterViolated { .. } => "filter-violated",
            Violation::OutOfOrder { .. } => "out-of-order",
            Violation::ScoreMismatch { .. } => "score-mismatch",
            Violation::ExactMismatch { .. } => "exact-mismatch",
            Violation::RecallBelow { .. } => "recall-below",
        }
    }
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: ", self.kind())?;
        match self {
            Violation::Oracle(e) => write!(f, "{e}"),
            Violation::TooMany { got, k } => write!(f, "{got} hits for k = {k}"),
            Violation::TooFew { got, expected } => {
                write!(f, "{got} hits, expected {expected}")
            }
            Violation::Duplicate { key } => write!(f, "key {key:?} returned twice"),
            Violation::NotLive { key } => write!(f, "key {key:?} is not live"),
            Violation::FilterViolated {
                key,
                path,
                language,
            } => write!(
                f,
                "key {key:?} ({path:?}, {language:?}) does not match the filter"
            ),
            Violation::OutOfOrder { rank } => {
                write!(f, "hit {rank} is out of (score desc, key asc) order")
            }
            Violation::ScoreMismatch {
                key,
                got,
                canonical,
            } => write!(
                f,
                "key {key:?} scored {got:e} (bits {:08x}), canonical {canonical:e} (bits {:08x})",
                got.to_bits(),
                canonical.to_bits()
            ),
            Violation::ExactMismatch {
                rank,
                got,
                expected,
            } => write!(f, "rank {rank}: got {got:?}, expected {expected:?}"),
            Violation::RecallBelow { recall, min } => {
                write!(f, "recall {recall:.4} below {min:.4}")
            }
        }
    }
}

/// What a query's check measured.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueryReport {
    pub recall: f64,
    pub returned: usize,
    pub expected: usize,
}

/// Recall of `hits` (keys with their canonical scores) against the oracle's top list
/// `truth` (see the module docs).
pub fn recall(truth: &[Hit], hits_canonical: &[(String, f32)]) -> f64 {
    let Some(last) = truth.last() else {
        return 1.0;
    };
    let boundary = last.score;
    let above: HashSet<&str> = truth
        .iter()
        .filter(|t| t.score > boundary)
        .map(|t| t.key.as_str())
        .collect();
    let at_boundary = truth.iter().filter(|t| t.score == boundary).count();
    let found_above = hits_canonical
        .iter()
        .filter(|(k, _)| above.contains(k.as_str()))
        .count();
    let found_at = hits_canonical
        .iter()
        .filter(|(_, s)| *s == boundary)
        .count()
        .min(at_boundary);
    (found_above + found_at) as f64 / truth.len() as f64
}

/// Checks one query's `hits` (key, reported score) against the oracle.
pub fn check_query(
    oracle: &Oracle,
    query: &[f32],
    k: usize,
    filter: &Filter,
    hits: &[(String, f32)],
    expect: Expectation,
) -> Result<QueryReport, Violation> {
    let truth = oracle.search(query, k, filter).map_err(Violation::Oracle)?;
    let q = oracle.prepare_query(query).map_err(Violation::Oracle)?;
    let expected = truth.len();
    if hits.len() > k {
        return Err(Violation::TooMany { got: hits.len(), k });
    }
    let mut seen = HashSet::new();
    let mut canonical = Vec::with_capacity(hits.len());
    for (key, reported) in hits {
        if !seen.insert(key.as_str()) {
            return Err(Violation::Duplicate { key: key.clone() });
        }
        let Some(entry) = oracle.get(key) else {
            return Err(Violation::NotLive { key: key.clone() });
        };
        if !filter.matches(&entry.path, &entry.language) {
            return Err(Violation::FilterViolated {
                key: key.clone(),
                path: entry.path.clone(),
                language: entry.language.clone(),
            });
        }
        let score = crate::canonical::score(oracle.metric(), &q, &entry.vector);
        if expect.canonical_scores && *reported != score {
            return Err(Violation::ScoreMismatch {
                key: key.clone(),
                got: *reported,
                canonical: score,
            });
        }
        canonical.push((key.clone(), score));
    }
    if hits.len() < expected {
        return Err(Violation::TooFew {
            got: hits.len(),
            expected,
        });
    }
    for rank in 1..hits.len() {
        let (ka, sa) = &hits[rank - 1];
        let (kb, sb) = &hits[rank];
        if hit_order(*sa, ka, *sb, kb) != std::cmp::Ordering::Less {
            return Err(Violation::OutOfOrder { rank });
        }
    }
    if expect.exact {
        for rank in 0..expected.max(hits.len()) {
            let got = hits.get(rank).cloned();
            let want = truth.get(rank).map(|h| (h.key.clone(), h.score));
            let same = match (&got, &want) {
                (Some((gk, gs)), Some((wk, ws))) => gk == wk && gs == ws,
                _ => false,
            };
            if !same {
                return Err(Violation::ExactMismatch {
                    rank,
                    got,
                    expected: want,
                });
            }
        }
    }
    let r = recall(&truth, &canonical);
    if let Some(min) = expect.min_recall
        && r < min
    {
        return Err(Violation::RecallBelow { recall: r, min });
    }
    Ok(QueryReport {
        recall: r,
        returned: hits.len(),
        expected,
    })
}

/// Mean, minimum and count of per-query recalls.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RecallSummary {
    pub mean: f64,
    pub min: f64,
    pub queries: usize,
}

impl RecallSummary {
    pub fn of(recalls: &[f64]) -> RecallSummary {
        if recalls.is_empty() {
            return RecallSummary {
                mean: 1.0,
                min: 1.0,
                queries: 0,
            };
        }
        RecallSummary {
            mean: recalls.iter().sum::<f64>() / recalls.len() as f64,
            min: recalls.iter().copied().fold(f64::INFINITY, f64::min),
            queries: recalls.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::Metric;

    fn oracle() -> Oracle {
        let mut o = Oracle::new(Metric::Dot, 1);
        for (key, v, path) in [
            ("a", 3.0, "x/a"),
            ("b", 2.0, "x/b"),
            ("c", 2.0, "y/c"),
            ("d", 1.0, "y/d"),
        ] {
            o.upsert(key, &[v], path, "rust").unwrap();
        }
        o
    }

    fn hits(list: &[(&str, f32)]) -> Vec<(String, f32)> {
        list.iter().map(|(k, s)| (k.to_string(), *s)).collect()
    }

    #[test]
    fn boundary_ties_are_interchangeable() {
        let o = oracle();
        let f = Filter::all();
        let e = Expectation::rescored(Some(1.0));
        // Truth for k=2 is [a, b]; c ties b at the boundary.
        assert!(check_query(&o, &[1.0], 2, &f, &hits(&[("a", 3.0), ("c", 2.0)]), e).is_ok());
        // Missing a is a real miss even though b and c both tie the boundary.
        let r = check_query(&o, &[1.0], 2, &f, &hits(&[("b", 2.0), ("c", 2.0)]), e);
        assert_eq!(r.unwrap_err().kind(), "recall-below");
        // Exact demands the key order too.
        let r = check_query(
            &o,
            &[1.0],
            2,
            &f,
            &hits(&[("a", 3.0), ("c", 2.0)]),
            Expectation::exact(),
        );
        assert_eq!(r.unwrap_err().kind(), "exact-mismatch");
    }

    #[test]
    fn universal_rules() {
        let o = oracle();
        let f = Filter::all();
        let e = Expectation::approximate(None);
        let kind = |list: &[(&str, f32)], k| {
            check_query(&o, &[1.0], k, &f, &hits(list), e)
                .err()
                .map(|v| v.kind())
        };
        assert_eq!(kind(&[("a", 3.0)], 2), Some("too-few"));
        assert_eq!(kind(&[("a", 3.0), ("a", 3.0)], 2), Some("duplicate"));
        assert_eq!(kind(&[("a", 3.0), ("zz", 1.0)], 2), Some("not-live"));
        assert_eq!(kind(&[("b", 2.0), ("a", 3.0)], 2), Some("out-of-order"));
        assert_eq!(
            kind(&[("a", 3.0), ("b", 2.0), ("c", 2.0)], 2),
            Some("too-many")
        );
        let only_y = Filter {
            path_prefixes: vec!["y".into()],
            languages: vec![],
        };
        let r = check_query(&o, &[1.0], 2, &only_y, &hits(&[("a", 3.0), ("c", 2.0)]), e);
        assert_eq!(r.unwrap_err().kind(), "filter-violated");
        let r = check_query(
            &o,
            &[1.0],
            1,
            &f,
            &hits(&[("a", 2.5)]),
            Expectation::rescored(None),
        );
        assert_eq!(r.unwrap_err().kind(), "score-mismatch");
    }

    #[test]
    fn recall_counts() {
        let o = oracle();
        let truth = o.search(&[1.0], 3, &Filter::all()).unwrap();
        assert_eq!(recall(&truth, &hits(&[("a", 3.0), ("d", 1.0)])), 1.0 / 3.0);
        assert_eq!(recall(&[], &[]), 1.0);
    }
}
