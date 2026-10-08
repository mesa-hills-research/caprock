//! The embedding cache.
//!
//! One directory per model under a root: `<root>/<document fingerprint>/`, holding a
//! `model.json` (the [`EmbeddingModel`]) and an append-only log of records:
//!
//! ```text
//! log file "cache-<20 digits>.log": magic b"CAPREMB1", dimension u32, then records:
//!   content hash [u8; 32], dimension x f32 (little-endian), CRC-32 of the hash and
//!   vector bytes (u32)
//! ```
//!
//! Opening replays the newest log into an in-memory map from hash to file offset
//! (memory: O(entries), about 48 bytes each), truncating a torn or corrupt tail (the
//! records before it stay). A later record for the same hash wins. Only one process may
//! hold a model's cache open for writing (a lock file); [`EmbeddingCache::retain`]
//! rewrites the live records into the next log and switches atomically.

use std::collections::HashMap;
use std::fs::{self, File, TryLockError};
use std::io::{self, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::embedding::{ContentHash, EmbeddingModel};
use crate::error::{Error, Result, VectorError};
use crate::format::{atomic_write, f32s_from_le, read_exact_at, sync_dir};

const MAGIC: &[u8; 8] = b"CAPREMB1";
const HEADER_LEN: u64 = 12;

/// Cache counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CacheStats {
    pub entries: u64,
    pub log_bytes: u64,
    /// Records dropped at open (a torn tail).
    pub truncated_records: u64,
}

/// A model's embedding cache.
#[derive(Debug)]
pub struct EmbeddingCache {
    model: EmbeddingModel,
    dir: PathBuf,
    generation: u64,
    log_path: PathBuf,
    log: File,
    offsets: HashMap<ContentHash, u64>,
    log_bytes: u64,
    truncated_records: u64,
    // Keep the inode (and its OS lock) alive until all other fields are dropped.
    // Never unlink LOCK: another opener might already have a handle to that inode.
    _lock: File,
}

impl EmbeddingCache {
    /// Opens (or creates) `root/<model.document_fingerprint()>/`. `ModelMismatch` if the
    /// directory's `model.json` describes a model that `requires_reembed` relative to
    /// `model`; `Locked` if another handle has it open.
    pub fn open(root: &Path, model: &EmbeddingModel) -> Result<EmbeddingCache> {
        model.validate()?;
        let dir = root.join(model.document_fingerprint());
        fs::create_dir_all(&dir).map_err(|e| Error::io(&dir, e))?;
        let lock_path = dir.join("LOCK");
        let lock = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .map_err(|e| Error::io(&lock_path, e))?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => return Err(Error::Locked { path: lock_path }),
            Err(TryLockError::Error(e)) => return Err(Error::io(&lock_path, e)),
        }

        let logs = log_files(&dir)?;
        let model_path = dir.join("model.json");
        match fs::read(&model_path) {
            Ok(bytes) => {
                let stored: EmbeddingModel = serde_json::from_slice(&bytes)
                    .map_err(|e| Error::corrupt(&model_path, e.to_string()))?;
                if model.requires_reembed(&stored) {
                    return Err(Error::ModelMismatch {
                        stored: Box::new(stored),
                        requested: Box::new(model.clone()),
                    });
                }
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                if !logs.is_empty() {
                    return Err(Error::corrupt(
                        &model_path,
                        "cache log has no model identity",
                    ));
                }
                let bytes =
                    serde_json::to_vec(model).map_err(|e| Error::InvalidConfig(e.to_string()))?;
                atomic_write(&dir, "model.json", &bytes)?;
            }
            Err(e) => return Err(Error::io(&model_path, e)),
        }

        let (generation, log_path) = match logs.into_iter().max_by_key(|(id, _)| *id) {
            Some(latest) => latest,
            None => {
                let name = log_name(1);
                atomic_write(&dir, &name, &header(model.dimensions))?;
                (1, dir.join(name))
            }
        };
        let log = File::options()
            .read(true)
            .write(true)
            .open(&log_path)
            .map_err(|e| Error::io(&log_path, e))?;
        let mut cache = EmbeddingCache {
            model: model.clone(),
            dir,
            generation,
            log_path,
            log,
            offsets: HashMap::new(),
            log_bytes: HEADER_LEN,
            truncated_records: 0,
            _lock: lock,
        };
        cache.replay()?;
        // Persist the model directory and any recovered log before removing older logs.
        sync_dir(&cache.dir)?;
        sync_dir(root)?;
        cache.remove_old_logs()?;
        Ok(cache)
    }

    pub fn model(&self) -> &EmbeddingModel {
        &self.model
    }

    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    pub fn contains(&self, hash: &ContentHash) -> bool {
        self.offsets.contains_key(hash)
    }

    /// The cached vector, if any (read from the log; checksum verified).
    pub fn get(&self, hash: &ContentHash) -> Result<Option<Vec<f32>>> {
        let Some(&offset) = self.offsets.get(hash) else {
            return Ok(None);
        };
        let mut record = vec![0; self.record_len()];
        self.read_record(hash, offset, &mut record)?;
        let vector = f32s_from_le(&record[32..record.len() - 4])
            .ok_or_else(|| Error::corrupt(&self.log_path, "invalid vector encoding"))?;
        Ok(Some(vector))
    }

    /// The positions in `hashes` with no cached vector, ascending (what to embed).
    pub fn missing(&self, hashes: &[ContentHash]) -> Vec<usize> {
        hashes
            .iter()
            .enumerate()
            .filter_map(|(i, hash)| (!self.contains(hash)).then_some(i))
            .collect()
    }

    /// Adds vectors (`InvalidVector` on a wrong dimension or a non-finite value; nothing
    /// is added then). Buffered: durable after [`flush`](Self::flush).
    pub fn put(&mut self, items: &[(ContentHash, &[f32])]) -> Result<()> {
        let dim = self.model.dimensions as usize;
        // Validate the entire batch before changing either the log or the index.
        for (_, vector) in items {
            if vector.len() != dim {
                return Err(VectorError::Dimension {
                    expected: dim,
                    found: vector.len(),
                }
                .into());
            }
            if let Some(index) = vector.iter().position(|v| !v.is_finite()) {
                return Err(VectorError::NotFinite { index }.into());
            }
        }
        if items.is_empty() {
            return Ok(());
        }
        let record_len = self.record_len();
        let end = (items.len() as u64)
            .checked_mul(record_len as u64)
            .and_then(|bytes| self.log_bytes.checked_add(bytes))
            .ok_or_else(|| Error::InvalidArgument("cache log length overflow".into()))?;
        self.log
            .seek(SeekFrom::Start(self.log_bytes))
            .map_err(|e| Error::io(&self.log_path, e))?;
        let mut record = Vec::with_capacity(record_len);
        // Writes go to the OS cache immediately, so get(&self) needs no mutable cursor
        // or vector-sized resident cache. Only flush makes them durable.
        for (hash, vector) in items {
            record.clear();
            record.extend_from_slice(hash);
            for value in *vector {
                record.extend_from_slice(&value.to_le_bytes());
            }
            let crc = crc32fast::hash(&record);
            record.extend_from_slice(&crc.to_le_bytes());
            if let Err(e) = self.log.write_all(&record) {
                // Keep the old index on an I/O failure too; discard partial appends.
                self.log
                    .set_len(self.log_bytes)
                    .map_err(|e| Error::io(&self.log_path, e))?;
                return Err(Error::io(&self.log_path, e));
            }
        }
        for (i, (hash, _)) in items.iter().enumerate() {
            self.offsets
                .insert(*hash, self.log_bytes + i as u64 * record_len as u64);
        }
        self.log_bytes = end;
        Ok(())
    }

    /// Writes buffered records and fsyncs the log.
    pub fn flush(&mut self) -> Result<()> {
        self.log
            .sync_all()
            .map_err(|e| Error::io(&self.log_path, e))?;
        sync_dir(&self.dir)
    }

    /// Keeps only the entries `keep` accepts, rewriting the log; returns how many were
    /// dropped.
    pub fn retain(&mut self, keep: &dyn Fn(&ContentHash) -> bool) -> Result<usize> {
        let generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| Error::InvalidArgument("cache generation exhausted".into()))?;
        let mut hashes: Vec<_> = self.offsets.keys().copied().collect();
        hashes.sort_unstable();
        hashes.retain(keep);
        let dropped = self.len() - hashes.len();
        self.flush()?;

        let log_path = self.dir.join(log_name(generation));
        let tmp_path = self.dir.join(format!("{}.tmp", log_name(generation)));
        // Unlink instead of truncating a stale temp: it could be a link to a live file.
        match fs::remove_file(&tmp_path) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(Error::io(&tmp_path, e)),
        }
        let mut log = File::options()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&tmp_path)
            .map_err(|e| Error::io(&tmp_path, e))?;
        log.write_all(&header(self.model.dimensions))
            .map_err(|e| Error::io(&tmp_path, e))?;
        let mut offsets = HashMap::with_capacity(hashes.len());
        let mut log_bytes = HEADER_LEN;
        let mut record = vec![0; self.record_len()];
        for hash in hashes {
            self.read_record(&hash, self.offsets[&hash], &mut record)?;
            log.write_all(&record)
                .map_err(|e| Error::io(&tmp_path, e))?;
            offsets.insert(hash, log_bytes);
            log_bytes += record.len() as u64;
        }
        log.sync_all().map_err(|e| Error::io(&tmp_path, e))?;
        // Only a complete, synced log ever has a .log name. Until this rename, open
        // selects the old log; afterwards it selects the new generation in full.
        fs::rename(&tmp_path, &log_path).map_err(|e| Error::io(&log_path, e))?;
        // Switch memory immediately: even if directory sync fails we must not append
        // to an older generation that the next open will no longer use.
        self.log = log;
        self.log_path = log_path;
        self.generation = generation;
        self.offsets = offsets;
        self.log_bytes = log_bytes;
        sync_dir(&self.dir)?;
        self.remove_old_logs()?;
        Ok(dropped)
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.len() as u64,
            log_bytes: self.log_bytes,
            truncated_records: self.truncated_records,
        }
    }

    fn record_len(&self) -> usize {
        32 + self.model.dimensions as usize * 4 + 4
    }

    fn replay(&mut self) -> Result<()> {
        let file_len = self
            .log
            .metadata()
            .map_err(|e| Error::io(&self.log_path, e))?
            .len();
        let mut reader = BufReader::new(&self.log);
        let mut found_header = [0; HEADER_LEN as usize];
        reader
            .read_exact(&mut found_header)
            .map_err(|e| read_error(&self.log_path, e))?;
        if found_header != header(self.model.dimensions) {
            return Err(Error::corrupt(
                &self.log_path,
                "bad cache magic or dimension",
            ));
        }
        let record_len = self.record_len() as u64;
        let mut record = vec![0; record_len as usize];
        while file_len.saturating_sub(self.log_bytes) >= record_len {
            match reader.read_exact(&mut record) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(Error::io(&self.log_path, e)),
            }
            if !valid_record(&record) {
                break;
            }
            let mut hash = [0; 32];
            hash.copy_from_slice(&record[..32]);
            self.offsets.insert(hash, self.log_bytes);
            self.log_bytes += record_len;
        }
        if self.log_bytes < file_len {
            self.truncated_records = (file_len - self.log_bytes).div_ceil(record_len);
            self.log
                .set_len(self.log_bytes)
                .and_then(|_| self.log.sync_all())
                .map_err(|e| Error::io(&self.log_path, e))?;
        }
        Ok(())
    }

    fn read_record(&self, hash: &ContentHash, offset: u64, record: &mut [u8]) -> Result<()> {
        read_exact_at(&self.log, record, offset).map_err(|e| read_error(&self.log_path, e))?;
        if &record[..32] != hash || !valid_record(record) {
            return Err(Error::corrupt(
                &self.log_path,
                "bad cache record hash, vector or CRC",
            ));
        }
        Ok(())
    }

    fn remove_old_logs(&self) -> Result<()> {
        let mut removed = false;
        for (generation, path) in log_files(&self.dir)? {
            if generation < self.generation {
                fs::remove_file(&path).map_err(|e| Error::io(&path, e))?;
                removed = true;
            }
        }
        if removed {
            sync_dir(&self.dir)?;
        }
        Ok(())
    }
}

fn log_name(generation: u64) -> String {
    format!("cache-{generation:020}.log")
}

fn log_files(dir: &Path) -> Result<Vec<(u64, PathBuf)>> {
    let mut logs = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| Error::io(dir, e))? {
        let entry = entry.map_err(|e| Error::io(dir, e))?;
        let name = entry.file_name();
        let Some(digits) = name
            .to_str()
            .and_then(|name| name.strip_prefix("cache-"))
            .and_then(|name| name.strip_suffix(".log"))
        else {
            continue;
        };
        if digits.len() == 20
            && digits.bytes().all(|b| b.is_ascii_digit())
            && let Ok(generation) = digits.parse::<u64>()
        {
            logs.push((generation, entry.path()));
        }
    }
    Ok(logs)
}

fn header(dim: u32) -> [u8; HEADER_LEN as usize] {
    let mut bytes = [0; HEADER_LEN as usize];
    bytes[..8].copy_from_slice(MAGIC);
    bytes[8..].copy_from_slice(&dim.to_le_bytes());
    bytes
}

// Called only on a whole record of the validated model's fixed record length.
fn valid_record(record: &[u8]) -> bool {
    let data_len = record.len() - 4;
    record[data_len..] == crc32fast::hash(&record[..data_len]).to_le_bytes()
        && record[32..data_len]
            .as_chunks::<4>()
            .0
            .iter()
            .all(|&bytes| f32::from_le_bytes(bytes).is_finite())
}

fn read_error(path: &Path, error: io::Error) -> Error {
    if error.kind() == io::ErrorKind::UnexpectedEof {
        Error::corrupt(path, "truncated cache file")
    } else {
        Error::io(path, error)
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
