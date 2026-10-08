//! Configuration. [`IndexConfig`] is persisted with the index and fixed at creation;
//! [`StoreOptions`] apply to one open handle; [`SearchParams`] to one search.

use serde::{Deserialize, Serialize};

use crate::embedding::EmbeddingModel;
use crate::error::{Error, Result};

/// The similarity results are ranked by. Scores are "higher is better" for every metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    /// Cosine similarity. Vectors and queries are normalized with
    /// [`crate::distance::normalize`]; the score is their canonical dot product.
    Cosine,
    /// Inner product of the vectors as given.
    Dot,
    /// Negated squared Euclidean distance.
    L2,
}

/// The codes a segment keeps in memory to find candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodeKind {
    /// The vectors themselves (4 bytes per dimension). Candidate scores are exact.
    F32,
    /// Scalar quantization to one byte per dimension.
    Int8,
    /// One bit per dimension (sign), with whatever per-vector corrections the codec keeps.
    Binary,
}

impl CodeKind {
    /// `f32`, `int8`, `binary`.
    pub fn name(self) -> &'static str {
        match self {
            CodeKind::F32 => "f32",
            CodeKind::Int8 => "int8",
            CodeKind::Binary => "binary",
        }
    }
}

/// Parameters of a Vamana graph (DiskANN's in-memory index).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VamanaParams {
    /// Out-degree bound `R`.
    pub max_degree: u32,
    /// Candidate list size `L` while building.
    pub build_beam: u32,
    /// Pruning parameter `alpha` of the last pass (the first pass uses 1.0).
    pub alpha: f32,
    /// Build passes over the points (1 or 2).
    pub passes: u32,
    /// Default candidate list size `L` while searching (raised to at least `k`).
    pub search_beam: u32,
}

impl Default for VamanaParams {
    fn default() -> Self {
        VamanaParams {
            max_degree: 64,
            build_beam: 128,
            alpha: 1.2,
            passes: 2,
            search_beam: 64,
        }
    }
}

/// The approximate index family of large segments.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum AnnKind {
    /// No graphs: every segment is searched flat (a scan of its codes, then rescoring).
    Flat,
    /// A Vamana graph per segment of at least [`AnnConfig::graph_min_vectors`] vectors.
    Vamana(VamanaParams),
}

/// When and how segments are searched approximately.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AnnConfig {
    pub kind: AnnKind,
    /// Segments with fewer vectors are searched flat.
    pub graph_min_vectors: u32,
    /// A filtered search scans a segment flat (over exactly the matching records) when at
    /// most this many of its live records match.
    pub filter_flat_below: u32,
}

impl Default for AnnConfig {
    fn default() -> Self {
        AnnConfig {
            kind: AnnKind::Vamana(VamanaParams::default()),
            graph_min_vectors: 20_000,
            filter_flat_below: 2_048,
        }
    }
}

/// An index's fixed configuration, saved in its manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexConfig {
    pub dim: u32,
    pub metric: Metric,
    /// The model the vectors come from, if recorded. When set, `dim` equals its
    /// `dimensions`.
    #[serde(default)]
    pub embedding: Option<EmbeddingModel>,
    /// Codes kept in memory for candidate search. Full-precision vectors are always kept
    /// too: on disk for int8 and binary codes (read for rescoring, exact search and
    /// merges), and as the codes themselves for f32.
    pub codes: CodeKind,
    /// A search rescores `max(k * rescore_factor, rescore_min)` candidates (unless
    /// [`SearchParams::rescore`] says otherwise).
    pub rescore_factor: u32,
    pub rescore_min: u32,
    pub ann: AnnConfig,
    /// Seeds every random choice (training samples, graph construction).
    #[serde(default)]
    pub seed: u64,
}

impl IndexConfig {
    /// The default configuration for `dim` and `metric`: int8 codes, rescoring at full
    /// precision, Vamana graphs on segments of 20,000 vectors or more.
    pub fn new(dim: u32, metric: Metric) -> IndexConfig {
        IndexConfig {
            dim,
            metric,
            embedding: None,
            codes: CodeKind::Int8,
            rescore_factor: 4,
            rescore_min: 32,
            ann: AnnConfig::default(),
            seed: 0,
        }
    }

    /// The default configuration for vectors from `model` (cosine).
    pub fn for_model(model: EmbeddingModel) -> IndexConfig {
        let mut config = IndexConfig::new(model.dimensions, Metric::Cosine);
        config.embedding = Some(model);
        config
    }

    /// An exact index: f32 codes, no graphs. Its results are exact: the keys, scores and
    /// order of a brute-force scan.
    pub fn exact(dim: u32, metric: Metric) -> IndexConfig {
        let mut config = IndexConfig::new(dim, metric).with_codes(CodeKind::F32);
        config.ann.kind = AnnKind::Flat;
        config
    }

    /// This configuration with other resident codes (and their default rescoring).
    pub fn with_codes(mut self, codes: CodeKind) -> IndexConfig {
        self.codes = codes;
        self.rescore_factor = match codes {
            CodeKind::F32 => 1,
            CodeKind::Int8 => 4,
            CodeKind::Binary => 16,
        };
        self.rescore_min = match codes {
            CodeKind::F32 => 0,
            CodeKind::Int8 => 32,
            CodeKind::Binary => 128,
        };
        self
    }

    /// Checks every field's range.
    pub fn validate(&self) -> Result<()> {
        let bad = |msg: String| Err(Error::InvalidConfig(msg));
        if self.dim == 0 || self.dim > crate::MAX_DIM {
            return bad(format!("dim {} not in 1..={}", self.dim, crate::MAX_DIM));
        }
        if let Some(model) = &self.embedding
            && model.dimensions != self.dim
        {
            return bad(format!(
                "dim {} differs from the model's {} dimensions",
                self.dim, model.dimensions
            ));
        }
        if self.rescore_factor == 0 {
            return bad("rescore_factor must be at least 1".into());
        }
        if self.ann.filter_flat_below == 0 {
            return bad("filter_flat_below must be at least 1".into());
        }
        if let AnnKind::Vamana(p) = self.ann.kind {
            if !(4..=256).contains(&p.max_degree) {
                return bad(format!("max_degree {} not in 4..=256", p.max_degree));
            }
            if p.build_beam < p.max_degree || p.build_beam > 4096 {
                return bad(format!(
                    "build_beam {} not in max_degree..=4096",
                    p.build_beam
                ));
            }
            if !(1.0..=2.0).contains(&p.alpha) {
                return bad(format!("alpha {} not in 1.0..=2.0", p.alpha));
            }
            if !(1..=2).contains(&p.passes) {
                return bad(format!("passes {} not in 1..=2", p.passes));
            }
            if p.search_beam == 0 || p.search_beam > 4096 {
                return bad(format!("search_beam {} not in 1..=4096", p.search_beam));
            }
            if self.ann.graph_min_vectors < 2 {
                return bad("graph_min_vectors must be at least 2".into());
            }
        }
        Ok(())
    }
}

/// When background-free maintenance merges segments (a tiered policy).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MergePolicy {
    /// Merge when this many segments of one size tier exist.
    pub segments_per_tier: u32,
    /// A segment whose deleted share reaches this is rewritten.
    pub max_deleted_fraction: f32,
}

impl Default for MergePolicy {
    fn default() -> Self {
        MergePolicy {
            segments_per_tier: 8,
            max_deleted_fraction: 0.25,
        }
    }
}

/// How much of each file `open` checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerifyMode {
    /// Headers, section tables and every section loaded into memory; large sections read
    /// on demand (full-precision vectors) are checked by [`VerifyMode::Full`] only.
    Headers,
    /// Every section's checksum (reads every byte).
    Full,
}

/// Options of one open handle. Not persisted.
#[derive(Debug, Clone, PartialEq)]
pub struct StoreOptions {
    /// Peak extra memory a seal, merge or graph build may use. Also bounds the size of a
    /// segment.
    pub build_memory_budget: u64,
    /// Threads for builds. Results never depend on it.
    pub threads: usize,
    /// The in-memory delta is sealed into a segment when it holds this many vectors.
    pub delta_max_vectors: u32,
    pub merge: MergePolicy,
    pub verify: VerifyMode,
}

impl Default for StoreOptions {
    fn default() -> Self {
        StoreOptions {
            build_memory_budget: 1 << 30,
            threads: std::thread::available_parallelism()
                .map(|n| n.get().min(8))
                .unwrap_or(1),
            delta_max_vectors: 8_192,
            merge: MergePolicy::default(),
            verify: VerifyMode::Headers,
        }
    }
}

impl StoreOptions {
    /// Checks every field's range.
    pub fn validate(&self) -> Result<()> {
        let bad = |msg: String| Err(Error::InvalidConfig(msg));
        if self.threads == 0 {
            return bad("threads must be at least 1".into());
        }
        if self.delta_max_vectors == 0 {
            return bad("delta_max_vectors must be at least 1".into());
        }
        if self.merge.segments_per_tier < 2 {
            return bad("segments_per_tier must be at least 2".into());
        }
        if !(0.0..=1.0).contains(&self.merge.max_deleted_fraction)
            || self.merge.max_deleted_fraction == 0.0
        {
            return bad("max_deleted_fraction must be in (0, 1]".into());
        }
        Ok(())
    }
}

/// Per-search overrides. The default uses the index's configuration.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SearchParams {
    /// Vamana candidate list size `L` (raised to at least `k`).
    pub beam: Option<u32>,
    /// Number of candidates rescored at full precision (raised to at least `k`).
    pub rescore: Option<u32>,
    /// Exact search over the full-precision vectors: the keys, scores and order of a
    /// brute-force scan.
    pub exact: bool,
}

impl SearchParams {
    /// Exact search.
    pub fn exact() -> SearchParams {
        SearchParams {
            exact: true,
            ..SearchParams::default()
        }
    }

    /// The default parameters with candidate list size `beam`.
    pub fn with_beam(beam: u32) -> SearchParams {
        SearchParams {
            beam: Some(beam),
            ..SearchParams::default()
        }
    }
}
