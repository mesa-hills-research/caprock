//! Dataset slices, record metadata, cached ground truth, and the recall judge.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use mhr_vector::Filter;
use mhr_vector_oracle::canonical::{self, Metric};
use mhr_vector_oracle::check::recall;
use mhr_vector_oracle::dataset::{Dataset, PATH_SEED, RecordMeta, index_key};
use mhr_vector_oracle::generate::{path_for, standard_filters};
use mhr_vector_oracle::gt::{self, TruthList};
use mhr_vector_oracle::{Hit, hit_order};

/// A filter by name: one of the standard filters (see
/// `mhr_vector_oracle::generate::standard_filters`), or `path:<prefix>` or
/// `lang:<language>` (for datasets with real paths, such as the code sets).
pub fn named_filter(name: &str) -> Result<Filter, String> {
    if let Some(prefix) = name.strip_prefix("path:") {
        return Ok(Filter::path(prefix));
    }
    if let Some(language) = name.strip_prefix("lang:") {
        return Ok(Filter::language(language));
    }
    standard_filters()
        .into_iter()
        .find(|(n, _, _)| *n == name)
        .map(|(_, f, _)| Filter {
            path_prefixes: f.path_prefixes,
            languages: f.languages,
        })
        .ok_or_else(|| format!("unknown filter {name:?}"))
}

/// A dataset restricted to its first `base` rows and `queries` queries.
pub struct Slice {
    pub ds: Dataset,
    pub base: usize,
    pub queries: Vec<Vec<f32>>,
    meta: Option<Vec<RecordMeta>>,
    key_rows: OnceLock<HashMap<String, usize>>,
}

impl Slice {
    pub fn open(root: &Path, name: &str, base: usize, queries: usize) -> Result<Slice, String> {
        let ds = Dataset::open(&root.join(name)).map_err(|e| format!("dataset {name}: {e}"))?;
        let base = if base == 0 { ds.info.base_count } else { base };
        if base > ds.info.base_count {
            return Err(format!(
                "dataset {name} has {} base rows, not {base}",
                ds.info.base_count
            ));
        }
        let queries = ds.read_queries(queries).map_err(|e| e.to_string())?;
        let meta = match ds.info.meta {
            Some(_) => Some(ds.read_meta(base).map_err(|e| e.to_string())?),
            None => None,
        };
        Ok(Slice {
            ds,
            base,
            queries,
            meta,
            key_rows: OnceLock::new(),
        })
    }

    pub fn metric(&self) -> Metric {
        self.ds.metric()
    }

    pub fn store_metric(&self) -> mhr_vector::Metric {
        match self.metric() {
            Metric::Cosine => mhr_vector::Metric::Cosine,
            Metric::Dot => mhr_vector::Metric::Dot,
            Metric::L2 => mhr_vector::Metric::L2,
        }
    }

    pub fn dim(&self) -> usize {
        self.ds.dim()
    }

    pub fn name(&self) -> &str {
        &self.ds.info.name
    }

    /// Row `i`'s key, path and language.
    pub fn record(&self, i: usize) -> (String, String, String) {
        match &self.meta {
            Some(m) => (m[i].key.clone(), m[i].path.clone(), m[i].language.clone()),
            None => {
                let (p, l) = path_for(i as u64, PATH_SEED);
                (index_key(i as u64), p, l)
            }
        }
    }

    /// The row holding `key`, if any.
    pub fn row_of(&self, key: &str) -> Option<usize> {
        match &self.meta {
            None => {
                if key.len() != 10 {
                    return None;
                }
                key.parse::<usize>().ok()
            }
            Some(m) => self
                .key_rows
                .get_or_init(|| {
                    m.iter()
                        .enumerate()
                        .map(|(i, r)| (r.key.clone(), i))
                        .collect()
                })
                .get(key)
                .copied(),
        }
    }

    pub fn read_rows(&self, start: usize, count: usize) -> Result<Vec<f32>, String> {
        self.ds.read_base(start, count).map_err(|e| e.to_string())
    }

    /// Which of rows `0..rows` a filter keeps (and an exclusion set removes).
    pub fn mask(&self, rows: usize, filter: &Filter, excluded: &HashSet<usize>) -> Vec<bool> {
        let f = crate::engine::oracle_filter(filter);
        (0..rows)
            .map(|i| {
                let (_, p, l) = self.record(i);
                !excluded.contains(&i) && f.matches(&p, &l)
            })
            .collect()
    }

    fn gt_path(&self, tag: &str, k: usize) -> PathBuf {
        let tag: String = tag
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        self.ds.dir.join("gt").join(format!(
            "k{k}-b{}-q{}-{tag}.bin",
            self.base,
            self.queries.len()
        ))
    }

    /// Ground truth of the queries over rows `0..rows` minus `excluded`, filtered by
    /// `filter` (named `tag` in the cache file name), top `k`. Computed in chunks of
    /// rows (bounded memory) and cached under `<dataset>/gt/`.
    pub fn ground_truth(
        &self,
        tag: &str,
        filter: &Filter,
        rows: usize,
        excluded: &HashSet<usize>,
        k: usize,
        threads: usize,
    ) -> Result<Vec<TruthList>, String> {
        let path = self.gt_path(tag, k);
        if let Ok((kk, base_rows, lists)) = gt::read_truth(&path)
            && kk == k
            && base_rows == rows as u64
            && lists.len() == self.queries.len()
        {
            return Ok(lists);
        }
        let dim = self.dim();
        let metric = self.metric();
        let mask = self.mask(rows, filter, excluded);
        let chunk = (256usize << 20) / (dim * 4).max(1);
        let mut merged: Vec<TruthList> = vec![Vec::new(); self.queries.len()];
        let mut start = 0;
        while start < rows {
            let count = chunk.min(rows - start);
            let mut flat = self.read_rows(start, count)?;
            gt::prepare_rows(metric, dim, &mut flat)
                .map_err(|(i, p)| format!("row {}: {p}", start + i))?;
            let keys: Vec<String> = (start..start + count).map(|i| self.record(i).0).collect();
            let lists = gt::ground_truth(
                metric,
                dim,
                &flat,
                &keys,
                Some(&mask[start..start + count]),
                &self.queries,
                k,
                threads,
            )
            .map_err(|p| format!("query: {p}"))?;
            for (m, list) in merged.iter_mut().zip(lists) {
                let shifted = list.into_iter().map(|(r, s)| (r + start as u32, s));
                let mut keyed: Vec<(f32, String, u32)> = m
                    .iter()
                    .copied()
                    .chain(shifted)
                    .map(|(r, s)| (s, self.record(r as usize).0, r))
                    .collect();
                keyed.sort_by(|a, b| hit_order(a.0, &a.1, b.0, &b.1));
                keyed.truncate(k);
                *m = keyed.into_iter().map(|(s, _, r)| (r, s)).collect();
            }
            start += count;
        }
        gt::write_truth(&path, k, rows as u64, &merged).map_err(|e| e.to_string())?;
        Ok(merged)
    }
}

/// Judges result lists against ground truth: the universal rules and recall.
pub struct Judge<'a> {
    pub slice: &'a Slice,
    pub truth: Vec<TruthList>,
    pub filter: Filter,
    /// Rows live in the measured index (others are "not live").
    pub rows: usize,
    pub excluded: HashSet<usize>,
    /// Matching live rows (for the exact-count rule).
    pub matching: usize,
    prepared: Vec<Vec<f32>>,
}

/// One query's judgement.
#[derive(Debug, Clone, PartialEq)]
pub struct Verdict {
    pub recall: f64,
    pub violation: Option<String>,
}

impl<'a> Judge<'a> {
    pub fn new(
        slice: &'a Slice,
        truth: Vec<TruthList>,
        filter: Filter,
        rows: usize,
        excluded: HashSet<usize>,
    ) -> Result<Judge<'a>, String> {
        let matching = slice
            .mask(rows, &filter, &excluded)
            .iter()
            .filter(|m| **m)
            .count();
        let prepared = slice
            .queries
            .iter()
            .map(|q| canonical::prepare(slice.metric(), q, slice.dim()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|p| p.to_string())?;
        Ok(Judge {
            slice,
            truth,
            filter,
            rows,
            excluded,
            matching,
            prepared,
        })
    }

    fn canonical_score(&self, query: usize, row: usize) -> Result<f32, String> {
        let raw = self.slice.read_rows(row, 1)?;
        let stored = canonical::prepare(self.slice.metric(), &raw, self.slice.dim())
            .map_err(|p| p.to_string())?;
        Ok(canonical::score(
            self.slice.metric(),
            &self.prepared[query],
            &stored,
        ))
    }

    /// Checks `hits` for query `query` at `k` and measures recall@k.
    pub fn judge(&self, query: usize, hits: &[(String, f32)], k: usize) -> Verdict {
        match self.judge_inner(query, hits, k) {
            Ok(recall) => Verdict {
                recall,
                violation: None,
            },
            Err((recall, v)) => Verdict {
                recall,
                violation: Some(v),
            },
        }
    }

    fn judge_inner(
        &self,
        query: usize,
        hits: &[(String, f32)],
        k: usize,
    ) -> Result<f64, (f64, String)> {
        let truth: Vec<Hit> = self.truth[query]
            .iter()
            .take(k)
            .map(|&(r, s)| Hit {
                key: self.slice.record(r as usize).0,
                score: s,
            })
            .collect();
        let known: HashMap<&str, f32> = truth.iter().map(|h| (h.key.as_str(), h.score)).collect();
        let fail = |v: String| (0.0, v);
        if hits.len() > k {
            return Err(fail(format!("too-many: {} hits for k {k}", hits.len())));
        }
        let mut seen = HashSet::new();
        let mut canonical_hits = Vec::with_capacity(hits.len());
        let f = crate::engine::oracle_filter(&self.filter);
        for (key, reported) in hits {
            if !seen.insert(key.as_str()) {
                return Err(fail(format!("duplicate: {key}")));
            }
            let Some(row) = self
                .slice
                .row_of(key)
                .filter(|&r| r < self.rows && !self.excluded.contains(&r))
            else {
                return Err(fail(format!("not-live: {key}")));
            };
            let (_, path, language) = self.slice.record(row);
            if !f.matches(&path, &language) {
                return Err(fail(format!("filter-violated: {key} ({path}, {language})")));
            }
            let score = match known.get(key.as_str()) {
                Some(s) => *s,
                None => self.canonical_score(query, row).map_err(fail)?,
            };
            if *reported != score {
                return Err(fail(format!(
                    "score-mismatch: {key} reported {reported:e}, canonical {score:e}"
                )));
            }
            canonical_hits.push((key.clone(), score));
        }
        for w in hits.windows(2) {
            if hit_order(w[0].1, &w[0].0, w[1].1, &w[1].0) != std::cmp::Ordering::Less {
                return Err(fail("out-of-order".to_string()));
            }
        }
        let r = recall(&truth, &canonical_hits);
        let expected = k.min(self.matching);
        if hits.len() < expected {
            return Err((
                r,
                format!("too-few: {} hits, expected {expected}", hits.len()),
            ));
        }
        Ok(r)
    }
}
