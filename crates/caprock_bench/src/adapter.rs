//! `IndexUnderTest` over the public store API (for oracle scenarios), with a scratch
//! directory that is removed on drop.

use std::path::{Path, PathBuf};

use caprock::{Filter, IndexConfig, Op, Record, SearchParams, StoreOptions, Writer};
use caprock_oracle::filter::Filter as OracleFilter;
use caprock_oracle::generate;
use caprock_oracle::rng::Rng;
use caprock_oracle::scenario::IndexUnderTest;

/// A scratch directory removed on drop.
pub struct Scratch(pub PathBuf);

impl Scratch {
    pub fn new(name: &str) -> Scratch {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let base = std::env::var("MHRV_SCRATCH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir());
        let dir = base.join(format!(
            "mhrv-{name}-{}-{:x}",
            std::process::id(),
            Rng::new(nanos).next_u64()
        ));
        std::fs::create_dir_all(&dir).expect("scratch directory");
        Scratch(dir)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub fn to_filter(f: &OracleFilter) -> Filter {
    Filter {
        path_prefixes: f.path_prefixes.clone(),
        languages: f.languages.clone(),
    }
}

pub fn to_ops(ops: &[generate::Op]) -> Vec<Op<'_>> {
    ops.iter()
        .map(|op| match op {
            generate::Op::Upsert {
                key,
                vector,
                path,
                language,
            } => Op::Upsert(Record {
                key,
                vector,
                path,
                language,
            }),
            generate::Op::Delete { key } => Op::Delete(key),
        })
        .collect()
}

/// The store under an oracle scenario.
pub struct StoreUnderTest {
    pub dir: Scratch,
    pub options: StoreOptions,
    pub writer: Option<Writer>,
    pub params: SearchParams,
}

impl StoreUnderTest {
    pub fn new(
        config: IndexConfig,
        options: StoreOptions,
        params: SearchParams,
    ) -> Result<StoreUnderTest, String> {
        let dir = Scratch::new("conform");
        let writer =
            Writer::create(dir.path(), config, options.clone()).map_err(|e| e.to_string())?;
        Ok(StoreUnderTest {
            dir,
            options,
            writer: Some(writer),
            params,
        })
    }

    pub fn writer(&mut self) -> &mut Writer {
        self.writer.as_mut().expect("open writer")
    }
}

impl IndexUnderTest for StoreUnderTest {
    fn apply(&mut self, ops: &[generate::Op]) -> Result<(), String> {
        let ops = to_ops(ops);
        self.writer()
            .apply(&ops)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    fn commit(&mut self) -> Result<(), String> {
        self.writer()
            .commit(None)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    fn reopen(&mut self) -> Result<(), String> {
        self.writer = None;
        self.writer =
            Some(Writer::open(self.dir.path(), self.options.clone()).map_err(|e| e.to_string())?);
        Ok(())
    }

    fn maintain(&mut self) -> Result<(), String> {
        self.writer()
            .maintain()
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    fn compact(&mut self) -> Result<(), String> {
        self.writer()
            .compact()
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: &OracleFilter,
    ) -> Result<Vec<(String, f32)>, String> {
        let f = to_filter(filter);
        let snapshot = self.writer.as_ref().expect("open writer").snapshot();
        snapshot
            .search(
                query,
                k,
                if filter.is_empty() { None } else { Some(&f) },
                &self.params,
            )
            .map(|hits| hits.into_iter().map(|h| (h.key, h.score)).collect())
            .map_err(|e| e.to_string())
    }

    fn live_count(&self) -> Result<usize, String> {
        Ok(self.writer.as_ref().expect("open writer").snapshot().len() as usize)
    }
}
