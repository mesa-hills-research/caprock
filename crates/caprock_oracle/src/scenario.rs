//! Operation-sequence scenarios run against any index through [`IndexUnderTest`], with
//! the oracle replaying the same sequence. The property tests and the conformance
//! runner both use these.
//!
//! The oracle models durability: `Reopen` drops everything applied since the last
//! `Commit`, so an index must keep exactly its committed state across a reopen.

use crate::canonical::Metric;
use crate::check::{self, Expectation, RecallSummary};
use crate::filter::Filter;
use crate::generate::{self, Op, OpsSpec, Step};
use crate::oracle::Oracle;

/// The operations a scenario drives. Implemented by adapters around the index under test.
pub trait IndexUnderTest {
    /// Applies a batch of upserts and deletes, in order. The changes must be visible to
    /// the next `search` (read-your-writes).
    fn apply(&mut self, ops: &[Op]) -> Result<(), String>;
    /// Makes everything applied so far durable.
    fn commit(&mut self) -> Result<(), String>;
    /// Drops the index (without committing) and opens it again from disk.
    fn reopen(&mut self) -> Result<(), String>;
    /// Runs maintenance (merges, graph builds) to completion.
    fn maintain(&mut self) -> Result<(), String>;
    /// Merges everything into as few segments as the memory budget allows.
    fn compact(&mut self) -> Result<(), String>;
    /// The top `k` for a raw query: `(key, score)` in result order.
    fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<(String, f32)>, String>;
    /// Number of live records.
    fn live_count(&self) -> Result<usize, String>;
}

/// A scenario: an operation sequence and what each check expects.
#[derive(Debug, Clone)]
pub struct Scenario {
    pub metric: Metric,
    pub ops: OpsSpec,
    pub queries: usize,
    pub ks: Vec<usize>,
    /// Check with the standard filters too (see [`generate::standard_filters`]).
    pub filters: bool,
    pub expect: Expectation,
    /// The mean recall over every checked query must reach this.
    pub min_mean_recall: Option<f64>,
}

/// Where and how a scenario failed. `kind` is a stable short name for bucketing.
#[derive(Debug, Clone, PartialEq)]
pub struct Failure {
    pub step: usize,
    pub kind: String,
    pub detail: String,
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "step {}: {}: {}", self.step, self.kind, self.detail)
    }
}

/// What a passing scenario measured.
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    pub recall: RecallSummary,
    pub checks: usize,
    pub live: usize,
}

fn fail(step: usize, kind: &str, detail: impl Into<String>) -> Failure {
    Failure {
        step,
        kind: kind.to_string(),
        detail: detail.into(),
    }
}

/// Runs `scenario` on `index` (which must start empty) and the oracle side by side.
pub fn run<I: IndexUnderTest>(index: &mut I, scenario: &Scenario) -> Result<Report, Failure> {
    run_steps(index, scenario, &scenario.ops.steps())
}

/// Runs explicit `steps` (instead of `scenario.ops.steps()`), with `scenario`'s
/// metric, queries, filters and expectations.
pub fn run_steps<I: IndexUnderTest>(
    index: &mut I,
    scenario: &Scenario,
    steps: &[Step],
) -> Result<Report, Failure> {
    let dim = scenario.ops.vectors.dim;
    let mut live = Oracle::new(scenario.metric, dim);
    let mut committed = live.clone();
    let queries = scenario.ops.vectors.queries(scenario.queries, 1000);
    let filters: Vec<(&str, Filter)> = if scenario.filters {
        generate::standard_filters()
            .into_iter()
            .map(|(n, f, _)| (n, f))
            .collect()
    } else {
        vec![("none", Filter::all())]
    };
    let mut recalls = Vec::new();
    let mut checks = 0;
    for (n, step) in steps.iter().enumerate() {
        match step {
            Step::Apply(ops) => {
                for op in ops {
                    match op {
                        Op::Upsert {
                            key,
                            vector,
                            path,
                            language,
                        } => live
                            .upsert(key, vector, path, language)
                            .map_err(|e| fail(n, "generator", e.to_string()))?,
                        Op::Delete { key } => {
                            live.delete(key);
                        }
                    }
                }
                index.apply(ops).map_err(|e| fail(n, "apply-error", e))?;
            }
            Step::Commit => {
                index.commit().map_err(|e| fail(n, "commit-error", e))?;
                committed = live.clone();
            }
            Step::Reopen => {
                index.reopen().map_err(|e| fail(n, "reopen-error", e))?;
                live = committed.clone();
            }
            Step::Maintain => index.maintain().map_err(|e| fail(n, "maintain-error", e))?,
            Step::Compact => index.compact().map_err(|e| fail(n, "compact-error", e))?,
            Step::Check => {
                let got = index.live_count().map_err(|e| fail(n, "len-error", e))?;
                if got != live.len() {
                    return Err(fail(
                        n,
                        "live-count",
                        format!("index has {got} live records, oracle {}", live.len()),
                    ));
                }
                for (name, filter) in &filters {
                    for &k in &scenario.ks {
                        for (qi, q) in queries.iter().enumerate() {
                            let hits = index
                                .search(q, k, filter)
                                .map_err(|e| fail(n, "search-error", e))?;
                            let report =
                                check::check_query(&live, q, k, filter, &hits, scenario.expect)
                                    .map_err(|v| {
                                        fail(
                                            n,
                                            v.kind(),
                                            format!("filter {name}, k {k}, query {qi}: {v}"),
                                        )
                                    })?;
                            recalls.push(report.recall);
                            checks += 1;
                        }
                    }
                }
            }
        }
    }
    let recall = RecallSummary::of(&recalls);
    if let Some(min) = scenario.min_mean_recall
        && recall.mean < min
    {
        return Err(fail(
            usize::MAX,
            "mean-recall-below",
            format!(
                "mean recall {:.4} below {min:.4} over {checks} checks",
                recall.mean
            ),
        ));
    }
    Ok(Report {
        recall,
        checks,
        live: live.len(),
    })
}

/// An in-memory reference implementation of [`IndexUnderTest`] backed by the oracle
/// itself (with a committed copy for `reopen`). Scenarios must pass on it with exact
/// expectations; the harness uses it to check itself.
#[derive(Debug, Clone)]
pub struct OracleIndex {
    live: Oracle,
    committed: Oracle,
}

impl OracleIndex {
    pub fn new(metric: Metric, dim: usize) -> OracleIndex {
        let o = Oracle::new(metric, dim);
        OracleIndex {
            live: o.clone(),
            committed: o,
        }
    }
}

impl IndexUnderTest for OracleIndex {
    fn apply(&mut self, ops: &[Op]) -> Result<(), String> {
        for op in ops {
            match op {
                Op::Upsert {
                    key,
                    vector,
                    path,
                    language,
                } => self
                    .live
                    .upsert(key, vector, path, language)
                    .map_err(|e| e.to_string())?,
                Op::Delete { key } => {
                    self.live.delete(key);
                }
            }
        }
        Ok(())
    }
    fn commit(&mut self) -> Result<(), String> {
        self.committed = self.live.clone();
        Ok(())
    }
    fn reopen(&mut self) -> Result<(), String> {
        self.live = self.committed.clone();
        Ok(())
    }
    fn maintain(&mut self) -> Result<(), String> {
        Ok(())
    }
    fn compact(&mut self) -> Result<(), String> {
        Ok(())
    }
    fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<(String, f32)>, String> {
        Ok(self
            .live
            .search(query, k, filter)
            .map_err(|e| e.to_string())?
            .into_iter()
            .map(|h| (h.key, h.score))
            .collect())
    }
    fn live_count(&self) -> Result<usize, String> {
        Ok(self.live.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::key as gen_key;

    #[test]
    fn the_oracle_passes_its_own_scenarios() {
        for metric in [Metric::Cosine, Metric::Dot, Metric::L2] {
            let mut ops = OpsSpec::mixed(12, 5);
            ops.steps = 25;
            ops.max_batch = 120;
            ops.key_space = 500;
            let scenario = Scenario {
                metric,
                ops,
                queries: 8,
                ks: vec![1, 10],
                filters: true,
                expect: Expectation::exact(),
                min_mean_recall: Some(1.0),
            };
            let mut index = OracleIndex::new(metric, 12);
            let report = run(&mut index, &scenario).unwrap();
            assert!(report.checks > 0);
        }
    }

    /// An index that loses one committed record on reopen must fail the next check.
    #[test]
    fn a_broken_index_fails() {
        struct Lossy(OracleIndex);
        impl IndexUnderTest for Lossy {
            fn apply(&mut self, ops: &[Op]) -> Result<(), String> {
                self.0.apply(ops)
            }
            fn commit(&mut self) -> Result<(), String> {
                self.0.commit()
            }
            fn reopen(&mut self) -> Result<(), String> {
                self.0.reopen()?;
                let first = self.0.live.iter().next().map(|(k, _)| k.clone());
                if let Some(k) = first {
                    self.0.live.delete(&k);
                }
                Ok(())
            }
            fn maintain(&mut self) -> Result<(), String> {
                Ok(())
            }
            fn compact(&mut self) -> Result<(), String> {
                Ok(())
            }
            fn search(
                &self,
                q: &[f32],
                k: usize,
                f: &Filter,
            ) -> Result<Vec<(String, f32)>, String> {
                self.0.search(q, k, f)
            }
            fn live_count(&self) -> Result<usize, String> {
                self.0.live_count()
            }
        }
        let ops = OpsSpec::mixed(8, 11);
        let scenario = Scenario {
            metric: Metric::Cosine,
            ops,
            queries: 4,
            ks: vec![5],
            filters: false,
            expect: Expectation::exact(),
            min_mean_recall: None,
        };
        let batch: Vec<Op> = (0..20)
            .map(|i| Op::Upsert {
                key: gen_key(i),
                vector: ops.vectors.vector(i),
                path: "a/b.rs".into(),
                language: "rust".into(),
            })
            .collect();
        let steps = vec![
            Step::Apply(batch.clone()),
            Step::Commit,
            Step::Reopen,
            Step::Check,
        ];
        let mut good = OracleIndex::new(Metric::Cosine, 8);
        assert!(run_steps(&mut good, &scenario, &steps).is_ok());
        let mut index = Lossy(OracleIndex::new(Metric::Cosine, 8));
        let failure = run_steps(&mut index, &scenario, &steps).unwrap_err();
        assert_eq!(failure.kind, "live-count");
        // Uncommitted work is dropped by a reopen: the oracle expects 0 records.
        let steps = vec![Step::Apply(batch), Step::Reopen, Step::Check];
        let mut good = OracleIndex::new(Metric::Cosine, 8);
        assert_eq!(run_steps(&mut good, &scenario, &steps).unwrap().live, 0);
    }
}
