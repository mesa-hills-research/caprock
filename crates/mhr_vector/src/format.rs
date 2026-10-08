//! Section files and file-system helpers (unit U05; docs/requirements.md, "On-disk format").
//!
//! A section file is a header, aligned sections, and a section table:
//!
//! ```text
//! offset 0, 64 bytes, the header:
//!   0..8    magic (b"MHRVSEG1" for segments)
//!   8..12   format version, u32 = FORMAT_VERSION
//!   12..16  section count, u32
//!   16..24  table offset, u64
//!   24..32  table length in bytes, u64 (= count * 32)
//!   32..36  CRC-32 (IEEE, crc32fast) of the table, u32
//!   36..60  zero
//!   60..64  CRC-32 of bytes 0..60, u32
//! then the sections, each starting at a multiple of its kind's alignment
//! ([`SectionKind::alignment`]), zero padding between them;
//! then the table: per section 32 bytes:
//!   kind u32, flags u32 (0), offset u64, length u64, CRC-32 of the section u32, zero u32
//! ```
//!
//! All integers are little-endian; `f32` values are IEEE 754 little-endian. A writer
//! writes to `<name>.tmp`, fsyncs, renames to `<name>` and fsyncs the directory, so a
//! section file either exists complete or not at all. Large arrays are aligned to 4096
//! bytes so a future reader may map them; this crate reads with positional reads (no
//! `unsafe`, no mmap).

use std::fs::{self, File};
use std::io::{self, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};

use crate::config::VerifyMode;
use crate::error::{Error, Result};

/// The on-disk format version this build writes and reads.
pub const FORMAT_VERSION: u32 = 1;
/// The magic of segment files.
pub const SEGMENT_MAGIC: [u8; 8] = *b"MHRVSEG1";
/// Size of the header.
pub const HEADER_LEN: u64 = 64;
/// Size of one section-table entry.
pub const TABLE_ENTRY_LEN: u64 = 32;

/// The sections of a segment file. Ids are part of the format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SectionKind {
    /// JSON: the segment's own description (see `segment`).
    Info = 1,
    /// Full-precision vectors in stored form, `len x dim` f32, ordinal order.
    Vectors = 2,
    /// Resident codes, `len x code_len` bytes, ordinal order.
    Codes = 3,
    /// The codec's parameters (`AnyCodec::to_bytes`).
    CodecParams = 4,
    /// Key bytes, concatenated in ordinal order.
    KeyBytes = 5,
    /// `len + 1` u32 offsets into `KeyBytes`.
    KeyOffsets = 6,
    /// The key lookup index (layout owned by `segment`).
    KeyIndex = 7,
    /// Distinct paths, sorted, concatenated.
    PathBytes = 8,
    /// `paths + 1` u32 offsets into `PathBytes`.
    PathOffsets = 9,
    /// Per ordinal, a u32 index into the distinct paths (non-decreasing).
    PathOfOrdinal = 10,
    /// JSON array of the distinct language names; a language id is a position in it.
    Languages = 11,
    /// Per ordinal, a u16 language id.
    LanguageOfOrdinal = 12,
    /// The Vamana graph (`Graph::to_bytes`).
    Graph = 13,
}

impl SectionKind {
    /// The kind with this id.
    pub fn from_id(id: u32) -> Option<SectionKind> {
        match id {
            1 => Some(Self::Info),
            2 => Some(Self::Vectors),
            3 => Some(Self::Codes),
            4 => Some(Self::CodecParams),
            5 => Some(Self::KeyBytes),
            6 => Some(Self::KeyOffsets),
            7 => Some(Self::KeyIndex),
            8 => Some(Self::PathBytes),
            9 => Some(Self::PathOffsets),
            10 => Some(Self::PathOfOrdinal),
            11 => Some(Self::Languages),
            12 => Some(Self::LanguageOfOrdinal),
            13 => Some(Self::Graph),
            _ => None,
        }
    }

    pub fn id(self) -> u32 {
        self as u32
    }

    /// 4096 for `Vectors`, `Codes` and `Graph`; 64 otherwise.
    pub fn alignment(self) -> u64 {
        match self {
            Self::Vectors | Self::Codes | Self::Graph => 4096,
            _ => 64,
        }
    }
}

/// One section-table entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SectionEntry {
    pub kind: SectionKind,
    pub offset: u64,
    pub len: u64,
    pub crc32: u32,
}

/// Writes a section file (see the module docs).
#[derive(Debug)]
pub struct SectionWriter {
    file: File,
    tmp_path: PathBuf,
    final_path: PathBuf,
    magic: [u8; 8],
    offset: u64,
    entries: Vec<SectionEntry>,
    open_section: Option<(SectionKind, u64, crc32fast::Hasher)>,
    // After a partial write, the cursor and checksum can no longer be trusted.
    failed: bool,
}

impl SectionWriter {
    /// Starts `<final_path>.tmp` (replacing a stale one). Fails if `final_path` exists.
    pub fn create(final_path: &Path, magic: [u8; 8]) -> Result<SectionWriter> {
        require_absent(final_path)?;
        let tmp_path = temporary_path(final_path);
        let mut file = create_temporary(&tmp_path)?;
        file.write_all(&[0; HEADER_LEN as usize])
            .map_err(|e| Error::io(&tmp_path, e))?;
        Ok(Self {
            file,
            tmp_path,
            final_path: final_path.to_owned(),
            magic,
            offset: HEADER_LEN,
            entries: Vec::new(),
            open_section: None,
            failed: false,
        })
    }

    /// Writes a whole section. Each kind at most once.
    pub fn write_section(&mut self, kind: SectionKind, bytes: &[u8]) -> Result<()> {
        self.begin_section(kind)?;
        self.write_chunk(bytes)?;
        self.end_section()
    }

    /// Starts a section written in chunks (for sections too large to hold in memory).
    pub fn begin_section(&mut self, kind: SectionKind) -> Result<()> {
        self.check_writable()?;
        if self.open_section.is_some() {
            return Err(Error::InvalidArgument("a section is already open".into()));
        }
        if self.entries.iter().any(|entry| entry.kind == kind) {
            return Err(Error::InvalidArgument(format!("duplicate section {kind:?}")));
        }
        let padding = (kind.alignment() - self.offset % kind.alignment()) % kind.alignment();
        self.append(&[0; 4096][..padding as usize])?;
        self.open_section = Some((kind, self.offset, crc32fast::Hasher::new()));
        Ok(())
    }

    /// Appends to the section begun last.
    pub fn write_chunk(&mut self, bytes: &[u8]) -> Result<()> {
        self.check_writable()?;
        if self.open_section.is_none() {
            return Err(Error::InvalidArgument("no section is open".into()));
        }
        self.append(bytes)?;
        if let Some((_, _, hasher)) = &mut self.open_section {
            hasher.update(bytes);
        }
        Ok(())
    }

    /// Ends the section begun last.
    pub fn end_section(&mut self) -> Result<()> {
        self.check_writable()?;
        let (kind, offset, hasher) = self
            .open_section
            .take()
            .ok_or_else(|| Error::InvalidArgument("no section is open".into()))?;
        self.entries.push(SectionEntry {
            kind,
            offset,
            len: self.offset - offset,
            crc32: hasher.finalize(),
        });
        Ok(())
    }

    /// Writes the table and the header, fsyncs, renames to the final path and fsyncs the
    /// directory. Returns the file's length.
    pub fn finish(self) -> Result<u64> {
        let mut writer = self;
        writer.check_writable()?;
        if writer.open_section.is_some() {
            return Err(Error::InvalidArgument("a section is still open".into()));
        }
        let table_offset = writer.offset;
        let mut table = Vec::with_capacity(writer.entries.len() * TABLE_ENTRY_LEN as usize);
        for entry in &writer.entries {
            table.extend_from_slice(&entry.kind.id().to_le_bytes());
            table.extend_from_slice(&0u32.to_le_bytes());
            table.extend_from_slice(&entry.offset.to_le_bytes());
            table.extend_from_slice(&entry.len.to_le_bytes());
            table.extend_from_slice(&entry.crc32.to_le_bytes());
            table.extend_from_slice(&0u32.to_le_bytes());
        }
        writer.append(&table)?;

        let mut header = [0u8; HEADER_LEN as usize];
        header[..8].copy_from_slice(&writer.magic);
        header[8..12].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        header[12..16].copy_from_slice(&(writer.entries.len() as u32).to_le_bytes());
        header[16..24].copy_from_slice(&table_offset.to_le_bytes());
        header[24..32].copy_from_slice(&(table.len() as u64).to_le_bytes());
        header[32..36].copy_from_slice(&crc32fast::hash(&table).to_le_bytes());
        let header_crc = crc32fast::hash(&header[..60]);
        header[60..64].copy_from_slice(&header_crc.to_le_bytes());
        writer
            .file
            .seek(SeekFrom::Start(0))
            .and_then(|_| writer.file.write_all(&header))
            .and_then(|_| writer.file.sync_all())
            .map_err(|e| Error::io(&writer.tmp_path, e))?;
        // Close before rename as well, for platforms that restrict renaming open files.
        drop(writer.file);
        require_absent(&writer.final_path)?;
        fs::rename(&writer.tmp_path, &writer.final_path)
            .map_err(|e| Error::io(&writer.final_path, e))?;
        sync_dir(parent_dir(&writer.final_path))?;
        Ok(writer.offset)
    }

    fn check_writable(&self) -> Result<()> {
        if self.failed {
            return Err(Error::InvalidArgument("writer had a failed write".into()));
        }
        Ok(())
    }

    fn append(&mut self, bytes: &[u8]) -> Result<()> {
        let end = self
            .offset
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| Error::InvalidArgument("section file is too large".into()))?;
        if let Err(e) = self.file.write_all(bytes) {
            self.failed = true;
            return Err(Error::io(&self.tmp_path, e));
        }
        self.offset = end;
        Ok(())
    }
}

/// An open section file.
#[derive(Debug)]
pub struct SectionFile {
    file: File,
    path: PathBuf,
    len: u64,
    entries: Vec<SectionEntry>,
}

impl SectionFile {
    /// Opens and checks the header and table ([`VerifyMode::Headers`]) or every
    /// section's checksum too ([`VerifyMode::Full`]). Errors: `Corrupt` for any
    /// inconsistency (bad magic or CRC, a section outside the file, an unknown kind),
    /// `UnsupportedFormat` for another version.
    pub fn open(path: &Path, magic: [u8; 8], verify: VerifyMode) -> Result<SectionFile> {
        let file = File::open(path).map_err(|e| Error::io(path, e))?;
        let len = file.metadata().map_err(|e| Error::io(path, e))?.len();
        if len < HEADER_LEN {
            return Err(Error::corrupt(path, "truncated header"));
        }
        let mut header = [0u8; HEADER_LEN as usize];
        read_checked(&file, path, &mut header, 0)?;
        if header[..8] != magic {
            return Err(Error::corrupt(path, "wrong magic"));
        }
        if crc32fast::hash(&header[..60]) != le_u32(&header[60..64]) {
            return Err(Error::corrupt(path, "header checksum mismatch"));
        }
        let version = le_u32(&header[8..12]);
        if version != FORMAT_VERSION {
            return Err(Error::UnsupportedFormat {
                found: version,
                supported: FORMAT_VERSION,
            });
        }
        if header[36..60].iter().any(|&byte| byte != 0) {
            return Err(Error::corrupt(path, "nonzero reserved header bytes"));
        }
        let count = le_u32(&header[12..16]);
        let table_offset = le_u64(&header[16..24]);
        let table_len = le_u64(&header[24..32]);
        // There are only thirteen distinct kinds in v1. Bound allocations before
        // trusting anything from disk, including a checksummed but hostile header.
        if count > 13 || table_len != u64::from(count) * TABLE_ENTRY_LEN {
            return Err(Error::corrupt(path, "invalid section count or table length"));
        }
        if table_offset < HEADER_LEN || table_offset.checked_add(table_len) != Some(len) {
            return Err(Error::corrupt(path, "section table is not at the end of the file"));
        }
        let mut table = vec![0u8; table_len as usize];
        read_checked(&file, path, &mut table, table_offset)?;
        if crc32fast::hash(&table) != le_u32(&header[32..36]) {
            return Err(Error::corrupt(path, "table checksum mismatch"));
        }
        let mut entries: Vec<SectionEntry> = Vec::with_capacity(count as usize);
        for bytes in table.as_chunks::<{ TABLE_ENTRY_LEN as usize }>().0 {
            let kind = SectionKind::from_id(le_u32(&bytes[..4]))
                .ok_or_else(|| Error::corrupt(path, "unknown section kind"))?;
            if le_u32(&bytes[4..8]) != 0 || le_u32(&bytes[28..32]) != 0 {
                return Err(Error::corrupt(path, "nonzero section flags or reserved bytes"));
            }
            if entries.iter().any(|entry| entry.kind == kind) {
                return Err(Error::corrupt(path, "duplicate section kind"));
            }
            let offset = le_u64(&bytes[8..16]);
            let section_len = le_u64(&bytes[16..24]);
            if offset < HEADER_LEN
                || !offset.is_multiple_of(kind.alignment())
                || offset > table_offset
                || section_len > table_offset - offset
            {
                return Err(Error::corrupt(path, "misaligned section or section outside data area"));
            }
            entries.push(SectionEntry {
                kind,
                offset,
                len: section_len,
                crc32: le_u32(&bytes[24..28]),
            });
        }
        // Table order need not be physical order. Empty sections can share the
        // starting offset of a following section, so sort them first at equal offsets.
        let mut physical = entries.clone();
        physical.sort_unstable_by_key(|entry| (entry.offset, entry.len));
        let mut end = HEADER_LEN;
        for entry in physical {
            if entry.offset < end {
                return Err(Error::corrupt(path, "overlapping sections"));
            }
            end = entry.offset + entry.len; // bounded by table_offset above
        }
        let result = Self {
            file,
            path: path.to_owned(),
            len,
            entries,
        };
        if verify == VerifyMode::Full {
            result.verify_all()?;
        }
        Ok(result)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The file's length in bytes.
    pub fn file_len(&self) -> u64 {
        self.len
    }

    pub fn entries(&self) -> &[SectionEntry] {
        &self.entries
    }

    pub fn entry(&self, kind: SectionKind) -> Option<&SectionEntry> {
        self.entries.iter().find(|entry| entry.kind == kind)
    }

    /// A whole section, its checksum verified. `Corrupt` if absent or damaged.
    pub fn read_section(&self, kind: SectionKind) -> Result<Vec<u8>> {
        let entry = self.required_entry(kind)?;
        let len = usize::try_from(entry.len)
            .map_err(|_| Error::corrupt(&self.path, "section does not fit in memory"))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(len)
            .map_err(|_| Error::corrupt(&self.path, "cannot allocate section buffer"))?;
        bytes.resize(len, 0);
        read_checked(&self.file, &self.path, &mut bytes, entry.offset)?;
        if crc32fast::hash(&bytes) != entry.crc32 {
            return Err(Error::corrupt(&self.path, format!("{kind:?} checksum mismatch")));
        }
        Ok(bytes)
    }

    /// Reads `buf.len()` bytes at `offset` within a section (no checksum). `Corrupt` if
    /// the range is outside the section.
    pub fn read_at(&self, kind: SectionKind, offset: u64, buf: &mut [u8]) -> Result<()> {
        let entry = self.required_entry(kind)?;
        if offset > entry.len || buf.len() as u64 > entry.len - offset {
            return Err(Error::corrupt(&self.path, "read outside section"));
        }
        read_checked(&self.file, &self.path, buf, entry.offset + offset)
    }

    /// Verifies every section's checksum.
    pub fn verify_all(&self) -> Result<()> {
        // Full verification must not require a full-precision vector array in RAM.
        let mut buffer = [0u8; 64 * 1024];
        for entry in &self.entries {
            let mut hasher = crc32fast::Hasher::new();
            let mut position = 0;
            while position < entry.len {
                let take = (entry.len - position).min(buffer.len() as u64) as usize;
                read_checked(
                    &self.file,
                    &self.path,
                    &mut buffer[..take],
                    entry.offset + position,
                )?;
                hasher.update(&buffer[..take]);
                position += take as u64;
            }
            if hasher.finalize() != entry.crc32 {
                return Err(Error::corrupt(
                    &self.path,
                    format!("{:?} checksum mismatch", entry.kind),
                ));
            }
        }
        Ok(())
    }

    fn required_entry(&self, kind: SectionKind) -> Result<&SectionEntry> {
        self.entry(kind)
            .ok_or_else(|| Error::corrupt(&self.path, format!("missing section {kind:?}")))
    }
}

/// Reads exactly `buf.len()` bytes at `offset` without moving a shared cursor
/// (`FileExt::read_exact_at` on Unix, a `seek_read` loop on Windows).
pub fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    if offset.checked_add(buf.len() as u64).is_none() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "read offset overflow"));
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut remaining = buf;
        let mut position = offset;
        while !remaining.is_empty() {
            match file.seek_read(remaining, position) {
                Ok(0) => return Err(io::Error::from(io::ErrorKind::UnexpectedEof)),
                Ok(n) => {
                    position += n as u64;
                    remaining = &mut remaining[n..];
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
    #[cfg(not(any(unix, windows)))]
    {
        Err(io::Error::new(io::ErrorKind::Unsupported, "positional reads are unavailable"))
    }
}

/// Writes `dir/name` atomically: a temporary file, fsync, rename, fsync of `dir`.
pub fn atomic_write(dir: &Path, name: &str, bytes: &[u8]) -> Result<()> {
    let mut components = Path::new(name).components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err(Error::InvalidArgument("expected a file name within the directory".into()));
    }
    let final_path = dir.join(name);
    let tmp_path = temporary_path(&final_path);
    let mut file = create_temporary(&tmp_path)?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|e| Error::io(&tmp_path, e))?;
    drop(file);
    fs::rename(&tmp_path, &final_path).map_err(|e| Error::io(&final_path, e))?;
    sync_dir(dir)
}

/// Fsyncs a directory (a no-op where directories cannot be opened, as on Windows).
pub fn sync_dir(dir: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        let dir = if dir.as_os_str().is_empty() { Path::new(".") } else { dir };
        File::open(dir)
            .and_then(|file| file.sync_all())
            .map_err(|e| Error::io(dir, e))
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
        Ok(())
    }
}

/// Little-endian encodings of slices, and their inverses (`None` on a length that is
/// not a multiple of the element size).
pub fn f32s_to_le(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

pub fn f32s_from_le(bytes: &[u8]) -> Option<Vec<f32>> {
    let (words, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        return None;
    }
    Some(words.iter().map(|&word| f32::from_le_bytes(word)).collect())
}

#[cfg(any(test, feature = "internals"))]
pub fn u32s_to_le(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

pub fn u32s_from_le(bytes: &[u8]) -> Option<Vec<u32>> {
    let (words, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        return None;
    }
    Some(words.iter().map(|&word| u32::from_le_bytes(word)).collect())
}

#[cfg(feature = "internals")]
pub fn u16s_to_le(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

pub fn u16s_from_le(bytes: &[u8]) -> Option<Vec<u16>> {
    let (words, remainder) = bytes.as_chunks::<2>();
    if !remainder.is_empty() {
        return None;
    }
    Some(words.iter().map(|&word| u16::from_le_bytes(word)).collect())
}

// Only called on fixed-size fields or chunks whose size has already been checked.
fn le_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn le_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

fn read_checked(file: &File, path: &Path, bytes: &mut [u8], offset: u64) -> Result<()> {
    read_exact_at(file, bytes, offset).map_err(|e| {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            Error::corrupt(path, "truncated file")
        } else {
            Error::io(path, e)
        }
    })
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".tmp");
    PathBuf::from(name)
}

fn parent_dir(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn require_absent(path: &Path) -> Result<()> {
    // symlink_metadata also catches a dangling symlink at the final name.
    match fs::symlink_metadata(path) {
        Ok(_) => Err(Error::AlreadyExists(path.to_owned())),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(Error::io(path, e)),
    }
}

fn create_temporary(path: &Path) -> Result<File> {
    // Remove rather than truncate: a stale symlink/hard link must not cause us to
    // modify a published file. Callers serialize writes to the same final name.
    match fs::remove_file(path) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(Error::io(path, e)),
    }
    File::options()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| Error::io(path, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            loop {
                let path = std::env::temp_dir().join(format!(
                    "mhrv-format-{}-{}",
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

    fn fix_header_crc(bytes: &mut [u8]) {
        let crc = crc32fast::hash(&bytes[..60]);
        bytes[60..64].copy_from_slice(&crc.to_le_bytes());
    }

    fn assert_corrupt(path: &Path, bytes: &[u8]) {
        fs::write(path, bytes).unwrap();
        for verify in [VerifyMode::Headers, VerifyMode::Full] {
            match SectionFile::open(path, SEGMENT_MAGIC, verify) {
                Err(Error::Corrupt { path: found, .. }) => assert_eq!(found, path),
                other => panic!("expected corruption for {verify:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn checksummed_malformed_metadata_is_rejected() {
        let dir = TestDir::new();
        let path = dir.0.join("metadata.mhrv");
        let mut writer = SectionWriter::create(&path, SEGMENT_MAGIC).unwrap();
        writer.write_section(SectionKind::Info, &[1; 129]).unwrap();
        writer.write_section(SectionKind::KeyBytes, b"keys").unwrap();
        writer.write_section(SectionKind::Vectors, &[2; 8]).unwrap();
        writer.finish().unwrap();
        let good = fs::read(&path).unwrap();
        let table_offset = le_u64(&good[16..24]) as usize;

        let table_changes = [
            (0, 99u32.to_le_bytes().to_vec()), // unknown kind
            (32, 1u32.to_le_bytes().to_vec()), // duplicate kind
            (4, 1u32.to_le_bytes().to_vec()),  // flags
            (28, 1u32.to_le_bytes().to_vec()), // reserved word
            (8, 0u64.to_le_bytes().to_vec()),  // overlaps header
            (8, 65u64.to_le_bytes().to_vec()), // misaligned
            (8, u64::MAX.to_le_bytes().to_vec()),
            (16, u64::MAX.to_le_bytes().to_vec()),
            (40, 64u64.to_le_bytes().to_vec()), // overlapping data
            (16, (table_offset as u64).to_le_bytes().to_vec()), // overlaps table
        ];
        for (offset, patch) in table_changes {
            let mut bad = good.clone();
            let at = table_offset + offset;
            bad[at..at + patch.len()].copy_from_slice(&patch);
            let crc = crc32fast::hash(&bad[table_offset..]);
            bad[32..36].copy_from_slice(&crc.to_le_bytes());
            fix_header_crc(&mut bad);
            assert_corrupt(&path, &bad);
        }
        let header_changes = [
            (12, u32::MAX.to_le_bytes().to_vec()),
            (16, u64::MAX.to_le_bytes().to_vec()),
            (24, u64::MAX.to_le_bytes().to_vec()),
            (36, vec![1]),
        ];
        for (at, patch) in header_changes {
            let mut bad = good.clone();
            bad[at..at + patch.len()].copy_from_slice(&patch);
            fix_header_crc(&mut bad);
            assert_corrupt(&path, &bad);
        }
        for at in (0..64).chain(table_offset..good.len()) {
            let mut bad = good.clone();
            bad[at] ^= 1;
            assert_corrupt(&path, &bad);
        }
        for end in [0, 1, 8, 32, 60, 63, 64, 65, 4096, table_offset, good.len() - 1] {
            assert_corrupt(&path, &good[..end]);
        }
        let mut trailing = good.clone();
        trailing.push(0);
        assert_corrupt(&path, &trailing);
    }

    #[test]
    fn empty_sections_and_streaming_state() {
        let dir = TestDir::new();
        let empty_path = dir.0.join("empty.mhrv");
        let writer = SectionWriter::create(&empty_path, SEGMENT_MAGIC).unwrap();
        assert_eq!(writer.finish().unwrap(), HEADER_LEN);
        let empty = SectionFile::open(&empty_path, SEGMENT_MAGIC, VerifyMode::Full).unwrap();
        assert!(empty.entries().is_empty());

        let path = dir.0.join("sections.mhrv");
        let mut writer = SectionWriter::create(&path, SEGMENT_MAGIC).unwrap();
        assert!(writer.write_chunk(&[]).is_err());
        assert!(writer.end_section().is_err());
        writer.begin_section(SectionKind::Vectors).unwrap();
        assert!(writer.begin_section(SectionKind::Info).is_err());
        assert!(writer.write_section(SectionKind::Info, b"refused").is_err());
        writer.write_chunk(&[]).unwrap();
        writer.end_section().unwrap();
        writer.write_section(SectionKind::Info, &[]).unwrap();
        writer.write_section(SectionKind::Codes, b"codes").unwrap();
        assert!(writer.begin_section(SectionKind::Vectors).is_err());
        assert!(!path.exists());
        writer.finish().unwrap();
        let file = SectionFile::open(&path, SEGMENT_MAGIC, VerifyMode::Full).unwrap();
        assert!(file.read_section(SectionKind::Vectors).unwrap().is_empty());
        file.read_at(SectionKind::Vectors, 0, &mut []).unwrap();
        file.read_at(SectionKind::Codes, 5, &mut []).unwrap();
        assert!(file.read_at(SectionKind::Codes, u64::MAX, &mut [0]).is_err());
        assert!(file.read_at(SectionKind::Codes, 6, &mut []).is_err());
        assert_eq!(file.read_section(SectionKind::Codes).unwrap(), b"codes");

        let unfinished = dir.0.join("unfinished.mhrv");
        let mut writer = SectionWriter::create(&unfinished, SEGMENT_MAGIC).unwrap();
        writer.begin_section(SectionKind::Info).unwrap();
        assert!(writer.finish().is_err());
        assert!(!unfinished.exists());
    }

    #[test]
    fn positional_reads_leave_cursor_unchanged_and_detect_truncation() {
        let dir = TestDir::new();
        let path = dir.0.join("positional.mhrv");
        let mut writer = SectionWriter::create(&path, SEGMENT_MAGIC).unwrap();
        writer.write_section(SectionKind::Info, b"abcdef").unwrap();
        writer.finish().unwrap();
        let file = SectionFile::open(&path, SEGMENT_MAGIC, VerifyMode::Headers).unwrap();
        let mut cursor = &file.file;
        cursor.seek(SeekFrom::Start(10)).unwrap();
        std::thread::scope(|scope| {
            for i in 0..6 {
                let file = &file;
                scope.spawn(move || {
                    for _ in 0..20 {
                        let mut byte = [0];
                        file.read_at(SectionKind::Info, i, &mut byte).unwrap();
                        assert_eq!(byte[0], b'a' + i as u8);
                    }
                });
            }
        });
        assert_eq!(cursor.stream_position().unwrap(), 10);
        assert!(read_exact_at(&file.file, &mut [0; 2], u64::MAX).is_err());
        File::options().write(true).open(&path).unwrap().set_len(HEADER_LEN).unwrap();
        assert!(matches!(file.read_section(SectionKind::Info), Err(Error::Corrupt { .. })));
        assert!(matches!(file.read_at(SectionKind::Info, 0, &mut [0]), Err(Error::Corrupt { .. })));
        assert!(matches!(file.verify_all(), Err(Error::Corrupt { .. })));
    }

    #[test]
    fn encoders_preserve_all_float_bits() {
        let bits = [0, 0x8000_0000, 1, 0x7f80_0000, 0xff80_0000, 0x7f80_0001, 0xffc0_1234];
        let floats: Vec<_> = bits.iter().map(|&bits| f32::from_bits(bits)).collect();
        let bytes = f32s_to_le(&floats);
        assert_eq!(bytes, u32s_to_le(&bits));
        let decoded = f32s_from_le(&bytes).unwrap();
        assert_eq!(decoded.iter().map(|value| value.to_bits()).collect::<Vec<_>>(), bits);
        assert_eq!(f32s_from_le(&[]), Some(Vec::new()));
        assert_eq!(u32s_from_le(&[]), Some(Vec::new()));
        assert_eq!(u16s_from_le(&[]), Some(Vec::new()));
    }

    #[test]
    fn stale_temporary_hard_links_do_not_modify_published_files() {
        let dir = TestDir::new();
        let published = dir.0.join("published");
        fs::write(&published, b"untouched").unwrap();
        let new_path = dir.0.join("new.mhrv");
        fs::hard_link(&published, temporary_path(&new_path)).unwrap();
        SectionWriter::create(&new_path, SEGMENT_MAGIC).unwrap().finish().unwrap();
        assert_eq!(fs::read(&published).unwrap(), b"untouched");
        fs::hard_link(&published, dir.0.join("current.tmp")).unwrap();
        atomic_write(&dir.0, "current", b"updated").unwrap();
        assert_eq!(fs::read(&published).unwrap(), b"untouched");
        assert_eq!(fs::read(dir.0.join("current")).unwrap(), b"updated");
    }
}
