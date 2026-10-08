//! The index under measurement: `caprock` through its public API, or the oracle
//! itself (to check the harness: its recall is 1 by construction).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use caprock::{Filter, IndexConfig, Metric, Op, Record, SearchParams, StoreOptions};
use caprock_oracle::Oracle;
use serde::{Deserialize, Serialize};

pub type EResult<T> = Result<T, String>;

/// A record to insert.
#[derive(Debug, Clone, Copy)]
pub struct Rec<'a> {
    pub key: &'a str,
    pub vector: &'a [f32],
    pub path: &'a str,
    pub language: &'a str,
}

/// A writable index.
pub trait EngineIndex: Send {
    fn apply(&mut self, upserts: &[Rec<'_>], deletes: &[&str]) -> EResult<()>;
    fn commit(&mut self) -> EResult<()>;
    fn maintain(&mut self) -> EResult<()>;
    fn compact(&mut self) -> EResult<()>;
    /// A searcher over the latest state (read-your-writes).
    fn searcher(&self) -> Arc<dyn Searcher>;
    /// Engine statistics for reports.
    fn stats(&self) -> serde_json::Value;
}

/// A read-only view, searched from many threads.
pub trait Searcher: Send + Sync {
    /// `beam`: the candidate list size (`None`: the default); `exact`: exact search.
    fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
        beam: Option<u32>,
        exact: bool,
    ) -> EResult<Vec<(String, f32)>>;
    fn live(&self) -> u64;
    fn contains(&self, key: &str) -> bool;
}

/// Which engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineKind {
    Caprock,
    Oracle,
}

impl EngineKind {
    pub fn parse(s: &str) -> EResult<EngineKind> {
        match s {
            "caprock" => Ok(EngineKind::Caprock),
            "oracle" => Ok(EngineKind::Oracle),
            _ => Err(format!("unknown engine {s:?} (caprock, oracle)")),
        }
    }
}

pub fn create(
    kind: EngineKind,
    dir: &Path,
    config: IndexConfig,
    options: StoreOptions,
) -> EResult<Box<dyn EngineIndex>> {
    match kind {
        EngineKind::Caprock => Ok(Box::new(CaprockIndex {
            writer: caprock::Writer::create(dir, config, options).map_err(|e| e.to_string())?,
        })),
        EngineKind::Oracle => Ok(Box::new(OracleEngine::create(dir, config)?)),
    }
}

pub fn open_writer(
    kind: EngineKind,
    dir: &Path,
    options: StoreOptions,
) -> EResult<Box<dyn EngineIndex>> {
    match kind {
        EngineKind::Caprock => Ok(Box::new(CaprockIndex {
            writer: caprock::Writer::open(dir, options).map_err(|e| e.to_string())?,
        })),
        EngineKind::Oracle => Ok(Box::new(OracleEngine::open(dir)?)),
    }
}

pub fn open_reader(
    kind: EngineKind,
    dir: &Path,
    options: StoreOptions,
) -> EResult<Arc<dyn Searcher>> {
    match kind {
        EngineKind::Caprock => {
            let reader = caprock::Reader::open(dir, options).map_err(|e| e.to_string())?;
            Ok(Arc::new(CaprockSearcher {
                snapshot: reader.snapshot(),
            }))
        }
        EngineKind::Oracle => Ok(OracleEngine::open(dir)?.searcher()),
    }
}

struct CaprockIndex {
    writer: caprock::Writer,
}

struct CaprockSearcher {
    snapshot: caprock::Snapshot,
}

impl EngineIndex for CaprockIndex {
    fn apply(&mut self, upserts: &[Rec<'_>], deletes: &[&str]) -> EResult<()> {
        let mut ops: Vec<Op<'_>> = upserts
            .iter()
            .map(|r| {
                Op::Upsert(Record {
                    key: r.key,
                    vector: r.vector,
                    path: r.path,
                    language: r.language,
                })
            })
            .collect();
        ops.extend(deletes.iter().map(|k| Op::Delete(k)));
        self.writer
            .apply(&ops)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    fn commit(&mut self) -> EResult<()> {
        self.writer
            .commit(None)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    fn maintain(&mut self) -> EResult<()> {
        self.writer
            .maintain()
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    fn compact(&mut self) -> EResult<()> {
        self.writer.compact().map(|_| ()).map_err(|e| e.to_string())
    }

    fn searcher(&self) -> Arc<dyn Searcher> {
        Arc::new(CaprockSearcher {
            snapshot: self.writer.snapshot(),
        })
    }

    fn stats(&self) -> serde_json::Value {
        let s = self.writer.stats();
        serde_json::json!({
            "live": s.live,
            "delta": s.delta,
            "segments": s.segments.len(),
            "graph_segments": s.segments.iter().filter(|x| x.has_graph).count(),
            "resident_bytes": s.resident_bytes,
            "disk_bytes": s.disk_bytes,
            "generation": s.generation,
        })
    }
}

impl Searcher for CaprockSearcher {
    fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
        beam: Option<u32>,
        exact: bool,
    ) -> EResult<Vec<(String, f32)>> {
        let params = SearchParams {
            beam,
            rescore: None,
            exact,
        };
        self.snapshot
            .search(query, k, filter, &params)
            .map(|hits| hits.into_iter().map(|h| (h.key, h.score)).collect())
            .map_err(|e| e.to_string())
    }

    fn live(&self) -> u64 {
        self.snapshot.len()
    }

    fn contains(&self, key: &str) -> bool {
        self.snapshot.contains(key)
    }
}

/// The oracle as an engine. Persists raw vectors as JSON on commit (small sets only).
pub struct OracleEngine {
    dir: PathBuf,
    metric: Metric,
    dim: usize,
    raw: BTreeMap<String, (Vec<f32>, String, String)>,
    oracle: Arc<Oracle>,
}

#[derive(Serialize, Deserialize)]
struct OracleFile {
    metric: Metric,
    dim: usize,
    records: Vec<(String, Vec<f32>, String, String)>,
}

pub fn oracle_metric(m: Metric) -> caprock_oracle::Metric {
    match m {
        Metric::Cosine => caprock_oracle::Metric::Cosine,
        Metric::Dot => caprock_oracle::Metric::Dot,
        Metric::L2 => caprock_oracle::Metric::L2,
    }
}

pub fn oracle_filter(f: &Filter) -> caprock_oracle::Filter {
    caprock_oracle::Filter {
        path_prefixes: f.path_prefixes.clone(),
        languages: f.languages.clone(),
    }
}

impl OracleEngine {
    fn file(dir: &Path) -> PathBuf {
        dir.join("oracle-index.json")
    }

    pub fn create(dir: &Path, config: IndexConfig) -> EResult<OracleEngine> {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        if Self::file(dir).exists() {
            return Err(format!("an index already exists at {}", dir.display()));
        }
        let mut e = OracleEngine {
            dir: dir.to_path_buf(),
            metric: config.metric,
            dim: config.dim as usize,
            raw: BTreeMap::new(),
            oracle: Arc::new(Oracle::new(
                oracle_metric(config.metric),
                config.dim as usize,
            )),
        };
        e.commit()?;
        Ok(e)
    }

    pub fn open(dir: &Path) -> EResult<OracleEngine> {
        let text = std::fs::read_to_string(Self::file(dir)).map_err(|e| e.to_string())?;
        let f: OracleFile = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        let mut oracle = Oracle::new(oracle_metric(f.metric), f.dim);
        let mut raw = BTreeMap::new();
        for (k, v, p, l) in f.records {
            oracle.upsert(&k, &v, &p, &l).map_err(|e| e.to_string())?;
            raw.insert(k, (v, p, l));
        }
        Ok(OracleEngine {
            dir: dir.to_path_buf(),
            metric: f.metric,
            dim: f.dim,
            raw,
            oracle: Arc::new(oracle),
        })
    }
}

impl EngineIndex for OracleEngine {
    fn apply(&mut self, upserts: &[Rec<'_>], deletes: &[&str]) -> EResult<()> {
        let oracle = Arc::make_mut(&mut self.oracle);
        for r in upserts {
            oracle
                .upsert(r.key, r.vector, r.path, r.language)
                .map_err(|e| e.to_string())?;
            self.raw.insert(
                r.key.to_string(),
                (
                    r.vector.to_vec(),
                    r.path.to_string(),
                    r.language.to_string(),
                ),
            );
        }
        for k in deletes {
            oracle.delete(k);
            self.raw.remove(*k);
        }
        Ok(())
    }

    fn commit(&mut self) -> EResult<()> {
        let f = OracleFile {
            metric: self.metric,
            dim: self.dim,
            records: self
                .raw
                .iter()
                .map(|(k, (v, p, l))| (k.clone(), v.clone(), p.clone(), l.clone()))
                .collect(),
        };
        let text = serde_json::to_string(&f).map_err(|e| e.to_string())?;
        let tmp = self.dir.join("oracle-index.json.tmp");
        std::fs::write(&tmp, text).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, Self::file(&self.dir)).map_err(|e| e.to_string())
    }

    fn maintain(&mut self) -> EResult<()> {
        Ok(())
    }

    fn compact(&mut self) -> EResult<()> {
        Ok(())
    }

    fn searcher(&self) -> Arc<dyn Searcher> {
        Arc::new(OracleSearcher(self.oracle.clone()))
    }

    fn stats(&self) -> serde_json::Value {
        serde_json::json!({ "live": self.oracle.len() })
    }
}

struct OracleSearcher(Arc<Oracle>);

impl Searcher for OracleSearcher {
    fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
        _beam: Option<u32>,
        _exact: bool,
    ) -> EResult<Vec<(String, f32)>> {
        let f = filter.map(oracle_filter).unwrap_or_default();
        self.0
            .search(query, k, &f)
            .map(|hits| hits.into_iter().map(|h| (h.key, h.score)).collect())
            .map_err(|e| e.to_string())
    }

    fn live(&self) -> u64 {
        self.0.len() as u64
    }

    fn contains(&self, key: &str) -> bool {
        self.0.get(key).is_some()
    }
}
