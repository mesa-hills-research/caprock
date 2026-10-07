//! The brute-force oracle: every live record in a `BTreeMap`, every search a full scan
//! and a full sort.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use crate::canonical::{self, Metric, VectorProblem};
use crate::filter::Filter;

/// A live record, with its vector in stored (prepared) form.
#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub vector: Vec<f32>,
    pub path: String,
    pub language: String,
}

/// One result: a key and its canonical score.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub key: String,
    pub score: f32,
}

/// Why the oracle refused an operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OracleError {
    Vector(VectorProblem),
    Key(String),
    Path(String),
    Language(String),
    /// `k` must be at least 1.
    ZeroK,
}

impl std::fmt::Display for OracleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OracleError::Vector(p) => write!(f, "invalid vector: {p}"),
            OracleError::Key(why) => write!(f, "invalid key: {why}"),
            OracleError::Path(why) => write!(f, "invalid path: {why}"),
            OracleError::Language(why) => write!(f, "invalid language: {why}"),
            OracleError::ZeroK => write!(f, "k must be at least 1"),
        }
    }
}

impl std::error::Error for OracleError {}

/// The order of results: score descending, then key ascending (bytes). Scores are never
/// NaN (inputs are validated), so `partial_cmp` is total here; `-0.0` equals `0.0`.
pub fn hit_order(a_score: f32, a_key: &str, b_score: f32, b_key: &str) -> Ordering {
    b_score
        .partial_cmp(&a_score)
        .expect("scores are never NaN")
        .then_with(|| a_key.as_bytes().cmp(b_key.as_bytes()))
}

/// Checks a key: 1..=255 bytes.
pub fn validate_key(key: &str) -> Result<(), OracleError> {
    if key.is_empty() {
        return Err(OracleError::Key("empty".into()));
    }
    if key.len() > crate::MAX_KEY_BYTES {
        return Err(OracleError::Key(format!(
            "{} bytes, the limit is {}",
            key.len(),
            crate::MAX_KEY_BYTES
        )));
    }
    Ok(())
}

/// Checks a record's path and language lengths.
pub fn validate_meta(path: &str, language: &str) -> Result<(), OracleError> {
    if path.len() > crate::MAX_PATH_BYTES {
        return Err(OracleError::Path(format!("{} bytes", path.len())));
    }
    if language.len() > crate::MAX_LANGUAGE_BYTES {
        return Err(OracleError::Language(format!("{} bytes", language.len())));
    }
    Ok(())
}

/// The oracle: a map from key to live record.
#[derive(Debug, Clone)]
pub struct Oracle {
    metric: Metric,
    dim: usize,
    entries: BTreeMap<String, Entry>,
}

impl Oracle {
    pub fn new(metric: Metric, dim: usize) -> Oracle {
        Oracle {
            metric,
            dim,
            entries: BTreeMap::new(),
        }
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Inserts or replaces a record. A refused record leaves the oracle unchanged.
    pub fn upsert(
        &mut self,
        key: &str,
        vector: &[f32],
        path: &str,
        language: &str,
    ) -> Result<(), OracleError> {
        validate_key(key)?;
        validate_meta(path, language)?;
        let vector =
            canonical::prepare(self.metric, vector, self.dim).map_err(OracleError::Vector)?;
        self.entries.insert(
            key.to_string(),
            Entry {
                vector,
                path: path.to_string(),
                language: language.to_string(),
            },
        );
        Ok(())
    }

    /// Removes a record; returns whether it was live.
    pub fn delete(&mut self, key: &str) -> bool {
        self.entries.remove(key).is_some()
    }

    pub fn get(&self, key: &str) -> Option<&Entry> {
        self.entries.get(key)
    }

    /// Live records in key order.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Entry)> {
        self.entries.iter()
    }

    /// Prepares a query (validated; normalized for cosine).
    pub fn prepare_query(&self, query: &[f32]) -> Result<Vec<f32>, OracleError> {
        canonical::prepare(self.metric, query, self.dim).map_err(OracleError::Vector)
    }

    /// The canonical score of `key` for a raw query, if the key is live.
    pub fn score(&self, query: &[f32], key: &str) -> Result<Option<f32>, OracleError> {
        let q = self.prepare_query(query)?;
        Ok(self
            .entries
            .get(key)
            .map(|e| canonical::score(self.metric, &q, &e.vector)))
    }

    /// Number of live records the filter matches.
    pub fn count_matching(&self, filter: &Filter) -> usize {
        self.entries
            .values()
            .filter(|e| filter.matches(&e.path, &e.language))
            .count()
    }

    /// The exact top `k`: every matching live record scored, sorted by
    /// [`hit_order`], truncated to `k`.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<Hit>, OracleError> {
        if k == 0 {
            return Err(OracleError::ZeroK);
        }
        let q = self.prepare_query(query)?;
        let mut all: Vec<Hit> = self
            .entries
            .iter()
            .filter(|(_, e)| filter.matches(&e.path, &e.language))
            .map(|(key, e)| Hit {
                key: key.clone(),
                score: canonical::score(self.metric, &q, &e.vector),
            })
            .collect();
        all.sort_by(|a, b| hit_order(a.score, &a.key, b.score, &b.key));
        all.truncate(k);
        Ok(all)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ties_break_by_key_and_upsert_replaces() {
        let mut o = Oracle::new(Metric::Dot, 2);
        o.upsert("b", &[1.0, 0.0], "p/b.rs", "rust").unwrap();
        o.upsert("a", &[1.0, 0.0], "p/a.rs", "rust").unwrap();
        o.upsert("c", &[0.5, 0.0], "q/c.py", "python").unwrap();
        let hits = o.search(&[1.0, 0.0], 2, &Filter::all()).unwrap();
        assert_eq!(
            hits.iter().map(|h| h.key.as_str()).collect::<Vec<_>>(),
            ["a", "b"]
        );
        o.upsert("c", &[2.0, 0.0], "q/c.py", "python").unwrap();
        let hits = o.search(&[1.0, 0.0], 1, &Filter::all()).unwrap();
        assert_eq!(hits[0].key, "c");
        assert!(o.delete("c"));
        assert!(!o.delete("c"));
        let only_q = Filter {
            path_prefixes: vec!["q".into()],
            languages: vec![],
        };
        assert!(o.search(&[1.0, 0.0], 5, &only_q).unwrap().is_empty());
    }

    #[test]
    fn refusals_leave_state_unchanged() {
        let mut o = Oracle::new(Metric::Cosine, 2);
        assert!(o.upsert("a", &[0.0, 0.0], "", "").is_err());
        assert!(o.upsert("", &[1.0, 0.0], "", "").is_err());
        assert!(o.upsert(&"x".repeat(256), &[1.0, 0.0], "", "").is_err());
        assert!(o.is_empty());
        assert_eq!(
            o.search(&[1.0, 0.0], 0, &Filter::all()),
            Err(OracleError::ZeroK)
        );
    }
}
