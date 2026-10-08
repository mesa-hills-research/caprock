//! Manifests, the `CURRENT` switch and the writer lock (unit U06; docs/requirements.md,
//! "Persistence").
//!
//! An index directory holds:
//!
//! ```text
//! LOCK                         the writer lock (an exclusive std file lock)
//! CURRENT                      the name of the current manifest, one line
//! gen-<20 digits>.json         one manifest per committed generation
//! seg-<20 digits>.mhrv         segment files (section files, immutable)
//! del-<seg 20>-<gen 20>.bits   tombstones of a segment as of a generation (immutable)
//! *.tmp                        unfinished writes (removed by the next writer)
//! ```
//!
//! A commit writes new segment and tombstone files, then the new manifest (atomically),
//! then replaces `CURRENT` (atomically). A crash at any point leaves the last published
//! generation intact; the files of an unfinished commit are garbage the next writer
//! removes. Files referenced by the previous generation are kept one more commit, so a
//! reader that just read the old `CURRENT` can still open them.

use std::collections::HashSet;
use std::fs::{self, File, TryLockError};
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::IndexConfig;
use crate::error::{Error, Result};
use crate::format::{FORMAT_VERSION, atomic_write};

/// The `CURRENT` file's name.
pub const CURRENT: &str = "CURRENT";
/// The lock file's name.
pub const LOCK: &str = "LOCK";

/// One segment in a manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentEntry {
    pub id: u64,
    /// File name within the index directory.
    pub file: String,
    pub vectors: u64,
    pub live: u64,
    /// The tombstone file in force, if any record is deleted.
    #[serde(default)]
    pub tombstones: Option<String>,
    pub has_graph: bool,
    pub bytes: u64,
}

/// A committed generation's description (`gen-<n>.json`). No timestamps: the same
/// operations give the same manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub format_version: u32,
    pub generation: u64,
    pub config: IndexConfig,
    /// Oldest first.
    pub segments: Vec<SegmentEntry>,
    pub next_segment_id: u64,
    /// The caller's checkpoint token, hex-encoded.
    #[serde(default)]
    pub checkpoint: Option<String>,
    /// Operations applied since creation, as of this generation.
    pub version: u64,
}

/// `gen-<20 digits>.json`.
pub fn manifest_file_name(generation: u64) -> String {
    format!("gen-{generation:020}.json")
}

/// `seg-<20 digits>.mhrv`.
pub fn segment_file_name(id: u64) -> String {
    format!("seg-{id:020}.mhrv")
}

/// `del-<segment 20 digits>-<generation 20 digits>.bits`.
pub fn tombstone_file_name(segment: u64, generation: u64) -> String {
    format!("del-{segment:020}-{generation:020}.bits")
}

/// Lower-case hex, and its inverse (`None` on bad input).
pub fn to_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        text.push(DIGITS[(byte >> 4) as usize] as char);
        text.push(DIGITS[(byte & 15) as usize] as char);
    }
    text
}

pub fn from_hex(text: &str) -> Option<Vec<u8>> {
    fn digit(byte: u8) -> Option<u8> {
        match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            b'A'..=b'F' => Some(byte - b'A' + 10),
            _ => None,
        }
    }

    let (pairs, remainder) = text.as_bytes().as_chunks::<2>();
    if !remainder.is_empty() {
        return None;
    }
    pairs
        .iter()
        .map(|&[high, low]| Some((digit(high)? << 4) | digit(low)?))
        .collect()
}

/// Reads `CURRENT` and the manifest it names. `Ok(None)` when there is no `CURRENT`.
/// `Corrupt` when `CURRENT` names a missing or unreadable manifest;
/// `UnsupportedFormat` for a newer format version.
pub fn read_current(dir: &Path) -> Result<Option<Manifest>> {
    let current_path = dir.join(CURRENT);
    let bytes = match fs::read(&current_path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(Error::io(&current_path, e)),
    };
    let text = std::str::from_utf8(&bytes)
        .map_err(|e| Error::corrupt(&current_path, format!("invalid CURRENT text: {e}")))?;
    // Accept a name with or without its terminating line ending, but never paths,
    // multiple lines, or other names that could escape the index directory.
    let name = text
        .strip_suffix("\r\n")
        .or_else(|| text.strip_suffix('\n'))
        .unwrap_or(text);
    let generation = parse_file_id(name, "gen-", ".json")
        .ok_or_else(|| Error::corrupt(&current_path, "invalid manifest name"))?;
    let path = dir.join(name);
    let bytes = fs::read(&path)
        .map_err(|e| Error::corrupt(&path, format!("cannot read current manifest: {e}")))?;

    // Check the version independently: a newer version may not even have the
    // fields needed to deserialize this build's Manifest.
    #[derive(Deserialize)]
    struct Version {
        format_version: u32,
    }
    let version: Version = serde_json::from_slice(&bytes)
        .map_err(|e| Error::corrupt(&path, format!("invalid manifest: {e}")))?;
    if version.format_version != FORMAT_VERSION {
        return Err(Error::UnsupportedFormat {
            found: version.format_version,
            supported: FORMAT_VERSION,
        });
    }
    let manifest: Manifest = serde_json::from_slice(&bytes)
        .map_err(|e| Error::corrupt(&path, format!("invalid manifest: {e}")))?;
    if manifest.generation != generation {
        return Err(Error::corrupt(
            &path,
            "manifest generation differs from its name",
        ));
    }
    for segment in &manifest.segments {
        if segment.file != segment_file_name(segment.id) {
            return Err(Error::corrupt(&path, "invalid segment file name"));
        }
        if let Some(name) = &segment.tombstones {
            match parse_tombstone_name(name) {
                Some((id, deleted_at)) if id == segment.id && deleted_at <= generation => {}
                _ => return Err(Error::corrupt(&path, "invalid tombstone file name")),
            }
        }
    }
    if let Some(checkpoint) = &manifest.checkpoint
        && from_hex(checkpoint).is_none()
    {
        return Err(Error::corrupt(&path, "invalid checkpoint hex"));
    }
    Ok(Some(manifest))
}

/// Publishes `manifest`: writes `gen-<n>.json` atomically, then `CURRENT` atomically.
pub fn publish(dir: &Path, manifest: &Manifest) -> Result<()> {
    let name = manifest_file_name(manifest.generation);
    let bytes = serde_json::to_vec(manifest)
        .map_err(|e| Error::InvalidArgument(format!("cannot serialize manifest: {e}")))?;
    // The caller has already made the referenced files durable. Each atomic_write
    // includes both the file fsync and the directory fsync, in this order.
    atomic_write(dir, &name, &bytes)?;
    atomic_write(dir, CURRENT, format!("{name}\n").as_bytes())
}

/// The files of `dir` that neither `current` nor `previous` references (segments,
/// tombstones, manifests, `*.tmp`), sorted. Never `LOCK` or `CURRENT`, never anything
/// whose name this module does not recognise.
pub fn garbage(dir: &Path, current: &Manifest, previous: Option<&Manifest>) -> Result<Vec<PathBuf>> {
    let mut referenced = HashSet::new();
    for manifest in std::iter::once(current).chain(previous) {
        referenced.insert(manifest_file_name(manifest.generation));
        for segment in &manifest.segments {
            referenced.insert(segment.file.clone());
            if let Some(name) = &segment.tombstones {
                referenced.insert(name.clone());
            }
        }
    }

    let mut garbage = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| Error::io(dir, e))? {
        let entry = entry.map_err(|e| Error::io(dir, e))?;
        let name = entry.file_name();
        let text = name.to_str();
        if text.is_some_and(|name| name == CURRENT || name == LOCK || referenced.contains(name)) {
            continue;
        }
        let known = name.as_encoded_bytes().ends_with(b".tmp")
            || text.is_some_and(|name| {
                parse_file_id(name, "gen-", ".json").is_some()
                    || parse_file_id(name, "seg-", ".mhrv").is_some()
                    || parse_tombstone_name(name).is_some()
            });
        if known {
            let path = entry.path();
            let kind = entry.file_type().map_err(|e| Error::io(&path, e))?;
            // Never recurse into directories. A recognized symlink can be unlinked
            // as garbage without following it or touching its target.
            if kind.is_file() || kind.is_symlink() {
                garbage.push(path);
            }
        }
    }
    garbage.sort();
    Ok(garbage)
}

fn parse_id(text: &str) -> Option<u64> {
    if text.len() != 20 || !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    text.parse().ok()
}

fn parse_file_id(name: &str, prefix: &str, suffix: &str) -> Option<u64> {
    parse_id(name.strip_prefix(prefix)?.strip_suffix(suffix)?)
}

fn parse_tombstone_name(name: &str) -> Option<(u64, u64)> {
    let (segment, generation) = name
        .strip_prefix("del-")?
        .strip_suffix(".bits")?
        .split_once('-')?;
    Some((parse_id(segment)?, parse_id(generation)?))
}

/// The exclusive writer lock on `dir/LOCK`, held until dropped.
#[derive(Debug)]
pub struct IndexLock {
    // Dropping the handle releases the OS lock; no process-global lock registry.
    _file: File,
}

impl IndexLock {
    /// Takes the lock without waiting (`File::try_lock`). `Locked` if another holder has
    /// it, in this process or another.
    pub fn acquire(dir: &Path) -> Result<IndexLock> {
        let path = dir.join(LOCK);
        let file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|e| Error::io(&path, e))?;
        match file.try_lock() {
            Ok(()) => Ok(IndexLock { _file: file }),
            Err(TryLockError::WouldBlock) => Err(Error::Locked { path }),
            Err(TryLockError::Error(e)) => Err(Error::io(path, e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Metric;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            loop {
                let path = std::env::temp_dir().join(format!(
                    "mhrv-manifest-{}-{}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed)
                ));
                match fs::create_dir(&path) {
                    Ok(()) => return Self(path),
                    Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                    Err(e) => panic!("cannot create test directory: {e}"),
                }
            }
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn empty_manifest(generation: u64) -> Manifest {
        Manifest {
            format_version: FORMAT_VERSION,
            generation,
            config: IndexConfig::new(8, Metric::Cosine),
            segments: Vec::new(),
            next_segment_id: 1,
            checkpoint: None,
            version: 0,
        }
    }

    #[test]
    fn hex_all_bytes_and_malformed_utf8_text() {
        let bytes: Vec<_> = (0..=255).collect();
        let text = to_hex(&bytes);
        assert_eq!(from_hex(&text), Some(bytes.clone()));
        assert_eq!(from_hex(&text.to_uppercase()), Some(bytes));
        for text in ["0", "abc", "0g", "\0\0", " 0", "0\n", "é", "１２"] {
            assert_eq!(from_hex(text), None, "{text:?}");
        }
    }

    #[test]
    fn rejects_bad_current_names_and_manifest_metadata() {
        let dir = TestDir::new();
        let current_path = dir.0.join(CURRENT);
        for name in [
            "",
            "../gen-00000000000000000001.json",
            "/gen-00000000000000000001.json",
            "gen-1.json",
            "gen-99999999999999999999.json",
            "gen-00000000000000000001.json\nextra",
            "gen-00000000000000000001.json.tmp",
            "gen-é0000000000000000001.json",
        ] {
            fs::write(&current_path, name).unwrap();
            assert!(
                matches!(
                    read_current(&dir.0),
                    Err(Error::Corrupt { path, .. }) if path == current_path
                ),
                "{name:?}"
            );
        }
        fs::write(&current_path, [0xff]).unwrap();
        assert!(matches!(read_current(&dir.0), Err(Error::Corrupt { .. })));

        let name = manifest_file_name(1);
        let path = dir.0.join(&name);
        fs::write(&current_path, format!("{name}\r\n")).unwrap();
        fs::write(&path, br#"{"format_version":2,"future_schema":true}"#).unwrap();
        assert!(matches!(
            read_current(&dir.0),
            Err(Error::UnsupportedFormat {
                found: 2,
                supported: FORMAT_VERSION
            })
        ));
        fs::write(&path, serde_json::to_vec(&empty_manifest(2)).unwrap()).unwrap();
        assert!(matches!(
            read_current(&dir.0),
            Err(Error::Corrupt { path: found, .. }) if found == path
        ));
        let mut manifest = empty_manifest(1);
        manifest.checkpoint = Some("not hex".into());
        fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert!(matches!(read_current(&dir.0), Err(Error::Corrupt { .. })));
        manifest.checkpoint = None;
        fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        assert_eq!(read_current(&dir.0).unwrap(), Some(manifest));
    }

    #[test]
    fn failed_publications_do_not_switch_current() {
        let dir = TestDir::new();
        let first = empty_manifest(1);
        let second = empty_manifest(2);
        publish(&dir.0, &first).unwrap();
        let second_name = manifest_file_name(2);
        let blocked = dir.0.join(format!("{second_name}.tmp"));
        fs::create_dir(&blocked).unwrap();
        assert!(publish(&dir.0, &second).is_err());
        assert_eq!(read_current(&dir.0).unwrap(), Some(first.clone()));
        assert!(!dir.0.join(&second_name).exists());
        fs::remove_dir(blocked).unwrap();

        let blocked = dir.0.join("CURRENT.tmp");
        fs::create_dir(&blocked).unwrap();
        assert!(publish(&dir.0, &second).is_err());
        assert_eq!(read_current(&dir.0).unwrap(), Some(first.clone()));
        assert!(dir.0.join(&second_name).exists());
        assert_eq!(
            garbage(&dir.0, &first, None).unwrap(),
            vec![dir.0.join(&second_name)]
        );
        fs::remove_dir(blocked).unwrap();
        publish(&dir.0, &second).unwrap();
        assert_eq!(read_current(&dir.0).unwrap(), Some(second));
    }

    #[test]
    fn garbage_does_not_claim_near_misses_or_directories() {
        let dir = TestDir::new();
        let current = empty_manifest(7);
        publish(&dir.0, &current).unwrap();
        for name in [
            "notes.txt",
            "LOCK",
            "seg-1.mhrv",
            "gen-1.json",
            "seg-00000000000000000001.mhrv.backup",
            "seg-99999999999999999999.mhrv",
            "del-00000000000000000001-00000000000000000001.bits.extra",
            "del-00000000000000000001-1.bits",
            "del-00000000000000000001-+0000000000000000001.bits",
        ] {
            fs::write(dir.0.join(name), b"leave me").unwrap();
        }
        fs::create_dir(dir.0.join("directory.tmp")).unwrap();
        fs::create_dir(dir.0.join(segment_file_name(2))).unwrap();
        let mut wanted: Vec<_> = [
            manifest_file_name(0),
            segment_file_name(u64::MAX),
            tombstone_file_name(0, u64::MAX),
            "notes.tmp".into(),
            "CURRENT.tmp".into(),
        ]
        .into_iter()
        .map(|name| dir.0.join(name))
        .collect();
        for path in &wanted {
            fs::write(path, b"garbage").unwrap();
        }
        wanted.sort();
        assert_eq!(garbage(&dir.0, &current, None).unwrap(), wanted);
    }

    #[test]
    fn lock_preserves_file_and_reports_io_errors() {
        let dir = TestDir::new();
        let path = dir.0.join(LOCK);
        fs::write(&path, b"existing lock file").unwrap();
        let first = IndexLock::acquire(&dir.0).unwrap();
        std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    assert!(matches!(
                        IndexLock::acquire(&dir.0),
                        Err(Error::Locked { .. })
                    ));
                })
                .join()
                .unwrap();
        });
        assert_eq!(fs::read(&path).unwrap(), b"existing lock file");
        drop(first);
        let _again = IndexLock::acquire(&dir.0).unwrap();
        let missing = dir.0.join("missing");
        assert!(matches!(
            IndexLock::acquire(&missing),
            Err(Error::Io { path, .. }) if path == missing.join(LOCK)
        ));
    }
}
