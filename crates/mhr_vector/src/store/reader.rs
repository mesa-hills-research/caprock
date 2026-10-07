//! Readers (unit U18).

use std::fs;
use std::io::ErrorKind;
use std::path::Path;
use std::sync::{Arc, Mutex};

use super::{DirectoryReader, Reader, ReaderSource, Snapshot};
use crate::config::{StoreOptions, VerifyMode};
use crate::error::{Error, Result};
use crate::manifest;

impl Reader {
    /// A reader of the last committed generation of `dir` (no lock; another process may
    /// be writing). Errors: `NotFound`, `Corrupt`, `UnsupportedFormat`, `Io`.
    pub fn open(dir: &Path, options: StoreOptions) -> Result<Reader> {
        options.validate()?;
        let snapshot = load_newer(dir, options.verify, None)?
            .expect("loading without a generation limit always returns a snapshot");
        Ok(Reader {
            source: ReaderSource::Directory(Arc::new(DirectoryReader {
                dir: dir.to_owned(),
                options,
                current: Mutex::new(snapshot),
            })),
        })
    }

    /// Picks up a newer committed generation, if any; returns whether it changed. A
    /// writer's reader always has the latest snapshot (returns `false`). Retries when a
    /// file named by the manifest it read was removed meanwhile.
    pub fn refresh(&self) -> Result<bool> {
        let ReaderSource::Directory(reader) = &self.source else {
            return Ok(false);
        };
        let generation = reader
            .current
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .generation();
        // No file I/O or verification under the snapshot lock: snapshot() only
        // waits for an Arc clone/swap, even during a large refresh.
        let Some(snapshot) = load_newer(&reader.dir, reader.options.verify, Some(generation))?
        else {
            return Ok(false);
        };
        let mut current = reader.current.lock().unwrap_or_else(|e| e.into_inner());
        if snapshot.generation() <= current.generation() {
            // A concurrent refresh already published this generation or a newer one.
            return Ok(false);
        }
        let old = std::mem::replace(&mut *current, snapshot);
        drop(current);
        // Releasing the last reference can free a large index. Do it outside the lock.
        drop(old);
        Ok(true)
    }

    /// The current snapshot.
    pub fn snapshot(&self) -> Snapshot {
        let slot = match &self.source {
            ReaderSource::Writer(slot) => slot.as_ref(),
            ReaderSource::Directory(reader) => &reader.current,
        };
        slot.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

fn load_newer(dir: &Path, verify: VerifyMode, generation: Option<u64>) -> Result<Option<Snapshot>> {
    retry_vanished(dir, || {
        let manifest =
            manifest::read_current(dir)?.ok_or_else(|| Error::NotFound(dir.to_owned()))?;
        if generation.is_some_and(|known| manifest.generation <= known) {
            return Ok(None);
        }
        Snapshot::from_manifest(dir, &manifest, verify).map(Some)
    })
}

// GC retains the preceding generation, but a reader can lag behind multiple
// commits while opening files. Restart from CURRENT on a missing-file error only
// when CURRENT advanced: permanent damage must return an error, not spin forever.
fn retry_vanished<T>(dir: &Path, mut load: impl FnMut() -> Result<T>) -> Result<T> {
    let path = dir.join(manifest::CURRENT);
    let mut current = read_current_bytes(dir)?;
    loop {
        match load() {
            Err(error) if vanished(&error) => {
                let next = fs::read(&path).map_err(|e| Error::io(&path, e))?;
                if next == current {
                    return Err(error);
                }
                current = next;
            }
            result => return result,
        }
    }
}

fn read_current_bytes(dir: &Path) -> Result<Vec<u8>> {
    let path = dir.join(manifest::CURRENT);
    fs::read(&path).map_err(|e| {
        if e.kind() == ErrorKind::NotFound {
            Error::NotFound(dir.to_owned())
        } else {
            Error::io(path, e)
        }
    })
}

fn vanished(error: &Error) -> bool {
    match error {
        Error::Io { source, .. } => source.kind() == ErrorKind::NotFound,
        // read_current reports an unreadable manifest as Corrupt, not Io. Check
        // existence rather than depending on its human-readable error detail.
        Error::Corrupt { path, .. } => matches!(path.try_exists(), Ok(false)),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{IndexConfig, Metric};
    use crate::format::atomic_write;
    use crate::store::Writer;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Directory(PathBuf);

    impl Directory {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "mhrv-reader-unit-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn vanished_manifests_and_segments_retry_only_when_current_changes() {
        let dir = Directory::new();
        let missing = dir.0.join("missing");
        for manifest_error in [false, true] {
            atomic_write(&dir.0, manifest::CURRENT, b"old\n").unwrap();
            let error = || {
                if manifest_error {
                    Error::corrupt(&missing, "cannot read current manifest")
                } else {
                    Error::io(&missing, std::io::Error::from(ErrorKind::NotFound))
                }
            };
            let mut attempts = 0;
            let value = retry_vanished(&dir.0, || {
                attempts += 1;
                if attempts == 1 {
                    atomic_write(&dir.0, manifest::CURRENT, b"new\n").unwrap();
                    return Err(error());
                }
                Ok(42)
            })
            .unwrap();
            assert_eq!(value, 42);
            assert_eq!(attempts, 2);

            let mut attempts = 0;
            let result: Result<()> = retry_vanished(&dir.0, || {
                attempts += 1;
                assert_eq!(attempts, 1, "permanent damage must not loop");
                Err(error())
            });
            assert!(result.is_err());
        }
    }

    #[test]
    fn real_corruption_is_not_hidden_by_a_commit() {
        let dir = Directory::new();
        let damaged = dir.0.join("damaged");
        fs::write(&damaged, b"bad checksum").unwrap();
        atomic_write(&dir.0, manifest::CURRENT, b"old\n").unwrap();
        let mut attempts = 0;
        let result: Result<()> = retry_vanished(&dir.0, || {
            attempts += 1;
            assert_eq!(attempts, 1);
            atomic_write(&dir.0, manifest::CURRENT, b"new\n").unwrap();
            Err(Error::corrupt(&damaged, "checksum mismatch"))
        });
        assert!(matches!(result, Err(Error::Corrupt { .. })));
    }

    #[test]
    fn open_errors_and_failed_refresh_leave_the_last_snapshot_intact() {
        let dir = Directory::new();
        assert!(matches!(
            Reader::open(&dir.0, StoreOptions::default()),
            Err(Error::NotFound(_))
        ));
        let mut writer = Writer::create(
            &dir.0,
            IndexConfig::exact(3, Metric::Dot),
            StoreOptions::default(),
        )
        .unwrap();
        let reader = Reader::open(&dir.0, StoreOptions::default()).unwrap();
        let cloned = reader.clone();
        let held = reader.snapshot();
        let current = read_current_bytes(&dir.0).unwrap();
        atomic_write(&dir.0, manifest::CURRENT, b"malformed").unwrap();
        assert!(matches!(reader.refresh(), Err(Error::Corrupt { .. })));
        assert!(Arc::ptr_eq(&held.inner, &reader.snapshot().inner));
        atomic_write(&dir.0, manifest::CURRENT, &current).unwrap();
        assert!(!reader.refresh().unwrap());
        writer.commit(None).unwrap();
        assert!(cloned.refresh().unwrap());
        assert!(reader.snapshot().generation() > held.generation());
        assert_eq!(
            reader.snapshot().generation(),
            cloned.snapshot().generation()
        );
        assert!(!reader.refresh().unwrap());
    }
}
