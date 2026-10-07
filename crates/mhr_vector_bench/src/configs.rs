//! Benchmark configurations, suites and workloads (the JSON files in `workloads/`).

use std::path::{Path, PathBuf};

use mhr_vector::{AnnKind, CodeKind, IndexConfig, Metric, StoreOptions, VamanaParams};
use serde::{Deserialize, Serialize};

/// A named index configuration. Unset fields keep `IndexConfig::new`'s defaults (after
/// `with_codes`), so a configuration tracks the crate's defaults unless it pins them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ConfigSpec {
    pub name: String,
    /// `f32`, `int8` or `binary`.
    #[serde(default)]
    pub codes: Option<String>,
    /// `vamana` or `flat`.
    #[serde(default)]
    pub ann: Option<String>,
    #[serde(default)]
    pub max_degree: Option<u32>,
    #[serde(default)]
    pub build_beam: Option<u32>,
    #[serde(default)]
    pub alpha: Option<f32>,
    #[serde(default)]
    pub passes: Option<u32>,
    #[serde(default)]
    pub search_beam: Option<u32>,
    #[serde(default)]
    pub graph_min_vectors: Option<u32>,
    #[serde(default)]
    pub filter_flat_below: Option<u32>,
    #[serde(default)]
    pub rescore_factor: Option<u32>,
    #[serde(default)]
    pub rescore_min: Option<u32>,
    /// `exact`: `IndexConfig::exact` (f32 codes, no graphs).
    #[serde(default)]
    pub exact: bool,
}

/// The built-in configurations, by name: `exact`, `flat-int8`, `flat-binary`,
/// `vamana-int8` (the default ANN configuration), `vamana-binary`, `vamana-f32`.
pub fn builtin(name: &str) -> Option<ConfigSpec> {
    let spec = |codes: &str, ann: &str| ConfigSpec {
        name: name.to_string(),
        codes: Some(codes.to_string()),
        ann: Some(ann.to_string()),
        ..ConfigSpec::default()
    };
    Some(match name {
        "exact" => ConfigSpec {
            name: name.into(),
            exact: true,
            ..ConfigSpec::default()
        },
        "flat-int8" => spec("int8", "flat"),
        "flat-binary" => spec("binary", "flat"),
        "vamana-int8" => spec("int8", "vamana"),
        "vamana-binary" => spec("binary", "vamana"),
        "vamana-f32" => spec("f32", "vamana"),
        _ => return None,
    })
}

impl ConfigSpec {
    /// The index configuration for vectors of `dim` under `metric`.
    pub fn index_config(&self, dim: u32, metric: Metric) -> Result<IndexConfig, String> {
        if self.exact {
            return Ok(IndexConfig::exact(dim, metric));
        }
        let codes = match self.codes.as_deref().unwrap_or("int8") {
            "f32" => CodeKind::F32,
            "int8" => CodeKind::Int8,
            "binary" => CodeKind::Binary,
            other => return Err(format!("config {}: unknown codes {other:?}", self.name)),
        };
        let mut c = IndexConfig::new(dim, metric).with_codes(codes);
        match self.ann.as_deref().unwrap_or("vamana") {
            "flat" => c.ann.kind = AnnKind::Flat,
            "vamana" => {
                let mut p = match c.ann.kind {
                    AnnKind::Vamana(p) => p,
                    AnnKind::Flat => VamanaParams::default(),
                };
                if let Some(v) = self.max_degree {
                    p.max_degree = v;
                }
                if let Some(v) = self.build_beam {
                    p.build_beam = v;
                }
                if let Some(v) = self.alpha {
                    p.alpha = v;
                }
                if let Some(v) = self.passes {
                    p.passes = v;
                }
                if let Some(v) = self.search_beam {
                    p.search_beam = v;
                }
                c.ann.kind = AnnKind::Vamana(p);
            }
            other => return Err(format!("config {}: unknown ann {other:?}", self.name)),
        }
        if let Some(v) = self.graph_min_vectors {
            c.ann.graph_min_vectors = v;
        }
        if let Some(v) = self.filter_flat_below {
            c.ann.filter_flat_below = v;
        }
        if let Some(v) = self.rescore_factor {
            c.rescore_factor = v;
        }
        if let Some(v) = self.rescore_min {
            c.rescore_min = v;
        }
        c.validate()
            .map_err(|e| format!("config {}: {e}", self.name))?;
        Ok(c)
    }
}

/// A configuration given by name (built-in) or inline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ConfigRef {
    Name(String),
    Spec(ConfigSpec),
}

impl ConfigRef {
    pub fn resolve(&self) -> Result<ConfigSpec, String> {
        match self {
            ConfigRef::Name(n) => builtin(n).ok_or_else(|| format!("unknown configuration {n:?}")),
            ConfigRef::Spec(s) => Ok(s.clone()),
        }
    }
}

/// Build options shared by suites and workloads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildOptions {
    #[serde(default = "default_threads")]
    pub threads: usize,
    #[serde(default = "default_budget")]
    pub budget: u64,
    /// Records per `apply` while loading.
    #[serde(default = "default_batch")]
    pub batch: usize,
    /// Commit after this many records while loading.
    #[serde(default = "default_commit_every")]
    pub commit_every: usize,
    #[serde(default = "default_delta")]
    pub delta_max_vectors: u32,
}

fn default_threads() -> usize {
    16
}
fn default_budget() -> u64 {
    2 << 30
}
fn default_batch() -> usize {
    10_000
}
fn default_commit_every() -> usize {
    200_000
}
fn default_delta() -> u32 {
    StoreOptions::default().delta_max_vectors
}

impl Default for BuildOptions {
    fn default() -> Self {
        BuildOptions {
            threads: default_threads(),
            budget: default_budget(),
            batch: default_batch(),
            commit_every: default_commit_every(),
            delta_max_vectors: default_delta(),
        }
    }
}

impl BuildOptions {
    pub fn store_options(&self) -> StoreOptions {
        StoreOptions {
            build_memory_budget: self.budget,
            threads: self.threads,
            delta_max_vectors: self.delta_max_vectors,
            ..StoreOptions::default()
        }
    }
}

/// One measured part of a workload or suite: a dataset slice and a configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Part {
    /// Dataset directory name under the datasets root.
    pub dataset: String,
    /// Base vectors used (0 or absent: all).
    #[serde(default)]
    pub base: usize,
    /// Queries used.
    #[serde(default = "default_queries")]
    pub queries: usize,
    pub config: ConfigRef,
    #[serde(default = "default_k")]
    pub k: usize,
    /// Filters to measure besides "none" (names from the standard set).
    #[serde(default)]
    pub filters: Vec<String>,
    /// Recall threshold of the score (QPS at the smallest beam reaching it).
    #[serde(default = "default_target")]
    pub recall_target: f64,
    /// Measure updates (inserts, deletes, recall afterwards).
    #[serde(default)]
    pub updates: bool,
    /// Measure opening in a fresh process.
    #[serde(default)]
    pub open: bool,
}

fn default_queries() -> usize {
    2000
}
fn default_k() -> usize {
    10
}
fn default_target() -> f64 {
    0.95
}

/// The beam ladder searched for the score's operating point.
pub const BEAMS: [u32; 38] = [
    10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96,
    104, 112, 128, 144, 160, 176, 192, 224, 256, 288, 320, 384, 448, 512,
];

/// A workload (the speed gate's, or a suite): parts and how to run them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Workload {
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub parts: Vec<Part>,
    #[serde(default)]
    pub build: BuildOptions,
    /// Timed passes over the queries (the median pass counts).
    #[serde(default = "default_repeats")]
    pub timed_repeats: usize,
    /// Threads of the untimed recall passes.
    #[serde(default = "default_recall_threads")]
    pub recall_threads: usize,
    /// Thread counts of the QPS measurements (1 is the score's).
    #[serde(default = "default_qps_threads")]
    pub qps_threads: Vec<usize>,
    /// Beams to report in suites (the score always searches `BEAMS`).
    #[serde(default)]
    pub beams: Vec<u32>,
}

fn default_repeats() -> usize {
    3
}
fn default_recall_threads() -> usize {
    4
}
fn default_qps_threads() -> Vec<usize> {
    vec![1]
}

impl Workload {
    pub fn load(path: &Path) -> Result<Workload, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let w: Workload =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        for p in &w.parts {
            p.config.resolve()?;
        }
        Ok(w)
    }

    /// A stable identity of the workload file (for caches): its bytes' FNV-1a hash.
    pub fn fingerprint(path: &Path) -> Result<String, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        Ok(format!("{h:016x}"))
    }
}

/// Where datasets live: `--datasets DIR`, else `$MHRV_DATASETS`, else `./datasets`.
pub fn datasets_root(arg: Option<&str>) -> PathBuf {
    if let Some(a) = arg {
        return PathBuf::from(a);
    }
    if let Ok(e) = std::env::var("MHRV_DATASETS") {
        return PathBuf::from(e);
    }
    PathBuf::from("datasets")
}
