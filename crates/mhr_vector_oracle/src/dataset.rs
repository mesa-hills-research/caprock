//! The raw dataset format shared by the benchmark harness and the conversion scripts.
//!
//! A dataset is a directory:
//!
//! ```text
//! dataset.json   the description below (format "mhr-vector-dataset/1")
//! base.f32       base vectors, row-major little-endian f32, base_count x dim
//! queries.f32    query vectors, query_count x dim
//! meta.jsonl     optional: one {"key", "path", "language"} object per base vector
//! queries.jsonl  optional: one {"text", ...} object per query (code sets)
//! neighbors.i32  optional: the source's own ground truth, query_count x neighbors_k
//!                little-endian i32 base indexes (ann-benchmarks' "neighbors"), used
//!                only to cross-check the oracle
//! gt/            ground truth computed by the harness (cached; see `gt`)
//! ```
//!
//! Without `meta.jsonl`, record `i` has key [`index_key`]`(i)` (zero-padded, so key
//! order is index order) and the synthetic path and language
//! [`crate::generate::path_for`]`(i, PATH_SEED)`, which give the standard filters their
//! documented selectivities.

use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::canonical::Metric;

/// The `format` value this version reads and writes.
pub const FORMAT: &str = "mhr-vector-dataset/1";

/// The seed of synthetic paths for datasets without `meta.jsonl`.
pub const PATH_SEED: u64 = 0x5EED;

/// The key of base vector `i` in datasets without `meta.jsonl`.
pub fn index_key(i: u64) -> String {
    format!("{i:010}")
}

/// `dataset.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetInfo {
    pub format: String,
    pub name: String,
    /// `cosine`, `dot` or `l2` (see [`Metric::name`]).
    pub metric: String,
    pub dim: usize,
    pub base_count: usize,
    pub query_count: usize,
    #[serde(default = "default_base")]
    pub base: String,
    #[serde(default = "default_queries")]
    pub queries: String,
    #[serde(default)]
    pub meta: Option<String>,
    #[serde(default)]
    pub query_texts: Option<String>,
    #[serde(default)]
    pub neighbors: Option<String>,
    #[serde(default)]
    pub neighbors_k: usize,
    /// Free-form provenance: URL, checksum, slicing, tool versions, cost.
    #[serde(default)]
    pub source: serde_json::Value,
}

fn default_base() -> String {
    "base.f32".into()
}

fn default_queries() -> String {
    "queries.f32".into()
}

/// The key, path and language of a base vector.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordMeta {
    pub key: String,
    pub path: String,
    pub language: String,
}

/// An opened dataset directory.
#[derive(Debug, Clone)]
pub struct Dataset {
    pub dir: PathBuf,
    pub info: DatasetInfo,
}

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// Decodes little-endian f32s.
pub fn f32s_from_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Encodes f32s as little-endian bytes.
pub fn f32s_to_le(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

impl Dataset {
    pub fn open(dir: &Path) -> io::Result<Dataset> {
        let text = std::fs::read_to_string(dir.join("dataset.json"))?;
        let info: DatasetInfo = serde_json::from_str(&text)
            .map_err(|e| invalid(format!("{}: {e}", dir.join("dataset.json").display())))?;
        if info.format != FORMAT {
            return Err(invalid(format!(
                "unsupported dataset format {:?} (expected {FORMAT})",
                info.format
            )));
        }
        if Metric::parse(&info.metric).is_none() {
            return Err(invalid(format!("unknown metric {:?}", info.metric)));
        }
        let ds = Dataset {
            dir: dir.to_path_buf(),
            info,
        };
        let expect = |name: &str, rows: usize| -> io::Result<()> {
            let len = std::fs::metadata(ds.dir.join(name))?.len();
            let want = (rows * ds.info.dim * 4) as u64;
            if len != want {
                return Err(invalid(format!("{name}: {len} bytes, expected {want}")));
            }
            Ok(())
        };
        expect(&ds.info.base, ds.info.base_count)?;
        expect(&ds.info.queries, ds.info.query_count)?;
        Ok(ds)
    }

    pub fn metric(&self) -> Metric {
        Metric::parse(&self.info.metric).expect("checked at open")
    }

    pub fn dim(&self) -> usize {
        self.info.dim
    }

    fn read_rows(&self, file: &str, start: usize, count: usize) -> io::Result<Vec<f32>> {
        let dim = self.info.dim;
        let mut f = File::open(self.dir.join(file))?;
        f.seek(SeekFrom::Start((start * dim * 4) as u64))?;
        let mut bytes = vec![0u8; count * dim * 4];
        f.read_exact(&mut bytes)?;
        Ok(f32s_from_le(&bytes))
    }

    /// Base rows `start..start + count`, flattened.
    pub fn read_base(&self, start: usize, count: usize) -> io::Result<Vec<f32>> {
        if start + count > self.info.base_count {
            return Err(invalid(format!(
                "base rows {start}..{} beyond {}",
                start + count,
                self.info.base_count
            )));
        }
        self.read_rows(&self.info.base, start, count)
    }

    /// The first `limit` queries (all when `limit` is larger).
    pub fn read_queries(&self, limit: usize) -> io::Result<Vec<Vec<f32>>> {
        let n = limit.min(self.info.query_count);
        let flat = self.read_rows(&self.info.queries, 0, n)?;
        Ok(flat
            .chunks_exact(self.info.dim)
            .map(|c| c.to_vec())
            .collect())
    }

    /// Keys, paths and languages of the first `limit` base vectors.
    pub fn read_meta(&self, limit: usize) -> io::Result<Vec<RecordMeta>> {
        let n = limit.min(self.info.base_count);
        match &self.info.meta {
            None => Ok((0..n as u64)
                .map(|i| {
                    let (path, language) = crate::generate::path_for(i, PATH_SEED);
                    RecordMeta {
                        key: index_key(i),
                        path,
                        language,
                    }
                })
                .collect()),
            Some(file) => {
                let reader = BufReader::new(File::open(self.dir.join(file))?);
                let mut out = Vec::with_capacity(n);
                for line in reader.lines() {
                    if out.len() == n {
                        break;
                    }
                    let line = line?;
                    let m: RecordMeta =
                        serde_json::from_str(&line).map_err(|e| invalid(format!("{file}: {e}")))?;
                    out.push(m);
                }
                if out.len() != n {
                    return Err(invalid(format!("{file}: {} records, need {n}", out.len())));
                }
                Ok(out)
            }
        }
    }

    /// The source's ground truth (`neighbors.i32`), if present: per query, base indexes.
    pub fn source_neighbors(&self, limit: usize) -> io::Result<Option<Vec<Vec<i32>>>> {
        let Some(file) = &self.info.neighbors else {
            return Ok(None);
        };
        let k = self.info.neighbors_k;
        let n = limit.min(self.info.query_count);
        let mut f = File::open(self.dir.join(file))?;
        let mut bytes = vec![0u8; n * k * 4];
        f.read_exact(&mut bytes)?;
        let all: Vec<i32> = bytes
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        Ok(Some(all.chunks_exact(k).map(|c| c.to_vec()).collect()))
    }
}

/// Writes a dataset directory (used by the synthetic generator and tests).
pub fn write_dataset(
    dir: &Path,
    info: &DatasetInfo,
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    meta: Option<&[RecordMeta]>,
) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let write_rows = |name: &str, rows: &[Vec<f32>]| -> io::Result<()> {
        let mut w = io::BufWriter::new(File::create(dir.join(name))?);
        for r in rows {
            if r.len() != info.dim {
                return Err(invalid(format!("{name}: row of {} dims", r.len())));
            }
            w.write_all(&f32s_to_le(r))?;
        }
        w.flush()
    };
    write_rows(&info.base, base)?;
    write_rows(&info.queries, queries)?;
    if let (Some(file), Some(meta)) = (&info.meta, meta) {
        let mut w = io::BufWriter::new(File::create(dir.join(file))?);
        for m in meta {
            serde_json::to_writer(&mut w, m).map_err(|e| invalid(e.to_string()))?;
            w.write_all(b"\n")?;
        }
        w.flush()?;
    }
    let mut info = info.clone();
    info.base_count = base.len();
    info.query_count = queries.len();
    let text = serde_json::to_string_pretty(&info).map_err(|e| invalid(e.to_string()))?;
    std::fs::write(dir.join("dataset.json"), text + "\n")
}
