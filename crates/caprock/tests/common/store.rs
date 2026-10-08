//! An `IndexUnderTest` adapter over the public store API, for oracle scenarios.

use caprock::{Filter, IndexConfig, Op, Record, SearchParams, StoreOptions, Writer};
use caprock_oracle::filter::Filter as OracleFilter;
use caprock_oracle::generate;
use caprock_oracle::scenario::IndexUnderTest;

use super::TempDir;

pub struct StoreUnderTest {
    pub dir: TempDir,
    pub options: StoreOptions,
    pub writer: Option<Writer>,
    pub params: SearchParams,
}

impl StoreUnderTest {
    pub fn new(config: IndexConfig, options: StoreOptions, params: SearchParams) -> StoreUnderTest {
        let dir = TempDir::new("store");
        let writer = Writer::create(dir.path(), config, options.clone()).unwrap();
        StoreUnderTest {
            dir,
            options,
            writer: Some(writer),
            params,
        }
    }

    pub fn writer(&mut self) -> &mut Writer {
        self.writer.as_mut().unwrap()
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
        let snapshot = self.writer.as_ref().unwrap().snapshot();
        let hits = snapshot
            .search(
                query,
                k,
                if filter.is_empty() { None } else { Some(&f) },
                &self.params,
            )
            .map_err(|e| e.to_string())?;
        Ok(hits.into_iter().map(|h| (h.key, h.score)).collect())
    }

    fn live_count(&self) -> Result<usize, String> {
        Ok(self.writer.as_ref().unwrap().snapshot().len() as usize)
    }
}
