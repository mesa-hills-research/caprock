//! Validate every resident table before exposing infallible ordinal accessors.
use std::path::Path;

use crate::config::{CodeKind, IndexConfig, VerifyMode};
use crate::error::{Error, Result};
use crate::format::{SEGMENT_MAGIC, SectionFile, SectionKind, u16s_from_le, u32s_from_le};
use crate::quant::AnyCodec;
use crate::vamana::Graph;

use super::{Info, Segment, string_at};

pub(super) fn open(
    dir: &Path,
    name: &str,
    config: &IndexConfig,
    verify: VerifyMode,
) -> Result<Segment> {
    let path = dir.join(name);
    let file = SectionFile::open(&path, SEGMENT_MAGIC, verify)?;
    let corrupt = |detail| Error::corrupt(&path, detail);
    bounded_len(&file, SectionKind::Info, 1, 4096)?;
    let info: Info = serde_json::from_slice(&file.read_section(SectionKind::Info)?)
        .map_err(|e| corrupt(format!("invalid segment Info: {e}")))?;
    if info.dim == 0 || info.dim > crate::MAX_DIM || info.len == 0 {
        return Err(corrupt("invalid segment dimension or length".into()));
    }
    if info.dim != config.dim || info.metric != config.metric || info.codes != config.codes {
        return Err(corrupt(
            "segment dimension, metric or codes differ from configuration".into(),
        ));
    }
    let n = u64::from(info.len);
    let dim = info.dim as usize;
    // Parameters are O(dim), not O(n). Bound allocation before decoding metadata.
    bounded_len(&file, SectionKind::CodecParams, 0, 64 + 16 * dim as u64)?;
    let codec = AnyCodec::from_bytes(
        info.codes,
        info.metric,
        dim,
        &file.read_section(SectionKind::CodecParams)?,
    )
    .map_err(|e| corrupt(format!("invalid codec: {e}")))?;
    exact_len(&file, SectionKind::Codes, n * codec.code_len() as u64)?;
    if info.codes == CodeKind::F32 {
        if file.entry(SectionKind::Vectors).is_some() {
            return Err(corrupt(
                "f32 segments must store vectors only in Codes".into(),
            ));
        }
    } else {
        exact_len(&file, SectionKind::Vectors, n * dim as u64 * 4)?;
    }
    if info.has_graph != file.entry(SectionKind::Graph).is_some() {
        return Err(corrupt("graph section and Info disagree".into()));
    }
    exact_len(&file, SectionKind::KeyOffsets, (n + 1) * 4)?;
    exact_len(&file, SectionKind::KeyIndex, n * 4)?;
    exact_len(&file, SectionKind::PathOfOrdinal, n * 4)?;
    exact_len(&file, SectionKind::LanguageOfOrdinal, n * 2)?;
    bounded_len(
        &file,
        SectionKind::KeyBytes,
        n,
        (n * crate::MAX_KEY_BYTES as u64).min(u32::MAX as u64),
    )?;
    let offsets_len = bounded_len(&file, SectionKind::PathOffsets, 8, (n + 1) * 4)?;
    if !offsets_len.is_multiple_of(4) {
        return Err(corrupt("incomplete path offset".into()));
    }
    let paths = offsets_len / 4 - 1;
    bounded_len(
        &file,
        SectionKind::PathBytes,
        0,
        (paths * crate::MAX_PATH_BYTES as u64).min(u32::MAX as u64),
    )?;
    // Each JSON string needs at most six bytes per source byte (a control escape),
    // two quotes and a comma. There are at most 65536 distinct u16 language ids.
    bounded_len(
        &file,
        SectionKind::Languages,
        4,
        2 + n.min(65536) * (6 * crate::MAX_LANGUAGE_BYTES as u64 + 3),
    )?;

    let key_bytes = file.read_section(SectionKind::KeyBytes)?;
    let key_offsets = read_u32s(&file, SectionKind::KeyOffsets)?;
    validate_strings(&path, &key_bytes, &key_offsets, 1, crate::MAX_KEY_BYTES)?;
    let key_index = file.read_section(SectionKind::KeyIndex)?;
    let mut previous_key = None;
    for &word in key_index.as_chunks::<4>().0 {
        let ordinal = u32::from_le_bytes(word);
        if ordinal >= info.len {
            return Err(corrupt("key index ordinal out of range".into()));
        }
        let key = string_at(&key_bytes, &key_offsets, ordinal as usize);
        if previous_key.is_some_and(|previous| previous >= key) {
            return Err(corrupt(
                "key index is not a strictly key-sorted permutation".into(),
            ));
        }
        // n distinct in-range ordinals imply a complete permutation. No extra
        // bitset or duplicate key allocation is necessary to prove coverage.
        previous_key = Some(key);
    }
    let path_bytes = file.read_section(SectionKind::PathBytes)?;
    let path_offsets = read_u32s(&file, SectionKind::PathOffsets)?;
    validate_strings(&path, &path_bytes, &path_offsets, 0, crate::MAX_PATH_BYTES)?;
    for i in 1..paths as usize {
        if string_at(&path_bytes, &path_offsets, i - 1) >= string_at(&path_bytes, &path_offsets, i)
        {
            return Err(corrupt("distinct paths are not strictly sorted".into()));
        }
    }
    let path_of_ordinal = read_u32s(&file, SectionKind::PathOfOrdinal)?;
    if path_of_ordinal.first() != Some(&0)
        || path_of_ordinal.last().copied().map(u64::from) != Some(paths - 1)
    {
        return Err(corrupt("path ordinals do not cover the path table".into()));
    }
    for (i, pair) in path_of_ordinal.windows(2).enumerate() {
        if u64::from(pair[1]) >= paths || pair[1] < pair[0] || pair[1] - pair[0] > 1 {
            return Err(corrupt(
                "path ordinals are not a contiguous sorted table".into(),
            ));
        }
        if pair[0] == pair[1]
            && string_at(&key_bytes, &key_offsets, i) >= string_at(&key_bytes, &key_offsets, i + 1)
        {
            return Err(corrupt("ordinals are not sorted by (path, key)".into()));
        }
    }
    let languages: Vec<String> =
        serde_json::from_slice(&file.read_section(SectionKind::Languages)?)
            .map_err(|e| corrupt(format!("invalid language table: {e}")))?;
    if languages.is_empty()
        || languages.len() > 65536
        || languages.len() as u64 > n
        || languages
            .iter()
            .any(|name| name.len() > crate::MAX_LANGUAGE_BYTES)
        || languages.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(corrupt(
            "language names are not a valid sorted table".into(),
        ));
    }
    let language_of_ordinal = u16s_from_le(&file.read_section(SectionKind::LanguageOfOrdinal)?)
        .ok_or_else(|| corrupt("invalid language ids".into()))?;
    let mut used_languages = vec![false; languages.len()];
    for &id in &language_of_ordinal {
        let used = used_languages
            .get_mut(usize::from(id))
            .ok_or_else(|| corrupt("language id out of range".into()))?;
        *used = true;
    }
    if used_languages.contains(&false) {
        return Err(corrupt("unused language table entry".into()));
    }
    let codes = file.read_section(SectionKind::Codes)?;
    validate_codes(&path, &codec, dim, &codes)?;
    let graph = if info.has_graph {
        bounded_len(&file, SectionKind::Graph, 16, 16 + n * 256 * 4)?;
        Some(
            Graph::from_bytes(&file.read_section(SectionKind::Graph)?, info.len as usize)
                .map_err(|e| corrupt(format!("invalid graph: {e}")))?,
        )
    } else {
        None
    };
    Ok(Segment {
        id: info.id,
        file_name: name.to_owned(),
        file,
        len: info.len,
        dim,
        codec,
        codes,
        graph,
        key_bytes,
        key_offsets,
        key_index,
        path_bytes,
        path_offsets,
        path_of_ordinal,
        languages,
        language_of_ordinal,
    })
}

fn bounded_len(file: &SectionFile, kind: SectionKind, min: u64, max: u64) -> Result<u64> {
    let entry = file
        .entry(kind)
        .ok_or_else(|| Error::corrupt(file.path(), format!("missing {kind:?}")))?;
    if entry.len < min || entry.len > max {
        return Err(Error::corrupt(
            file.path(),
            format!("invalid {kind:?} length"),
        ));
    }
    Ok(entry.len)
}

fn exact_len(file: &SectionFile, kind: SectionKind, len: u64) -> Result<()> {
    bounded_len(file, kind, len, len).map(|_| ())
}

fn read_u32s(file: &SectionFile, kind: SectionKind) -> Result<Vec<u32>> {
    u32s_from_le(&file.read_section(kind)?)
        .ok_or_else(|| Error::corrupt(file.path(), format!("incomplete u32 in {kind:?}")))
}

fn validate_strings(
    path: &Path,
    bytes: &[u8],
    offsets: &[u32],
    min: usize,
    max: usize,
) -> Result<()> {
    let corrupt = || Error::corrupt(path, "invalid string table offsets or UTF-8");
    let text = std::str::from_utf8(bytes).map_err(|_| corrupt())?;
    if offsets.first() != Some(&0)
        || offsets.last().copied().map(u64::from) != Some(bytes.len() as u64)
    {
        return Err(corrupt());
    }
    for pair in offsets.windows(2) {
        let (start, end) = (pair[0] as usize, pair[1] as usize);
        if start > end || !(min..=max).contains(&(end - start)) || !text.is_char_boundary(end) {
            return Err(corrupt());
        }
    }
    Ok(())
}

fn validate_codes(path: &Path, codec: &AnyCodec, dim: usize, codes: &[u8]) -> Result<()> {
    let corrupt = || Error::corrupt(path, "invalid resident code values");
    match codec.kind() {
        CodeKind::F32 => {
            for &word in codes.as_chunks::<4>().0 {
                let value = f32::from_le_bytes(word);
                if !value.is_finite() || value.abs() > crate::distance::MAX_ABS_COMPONENT {
                    return Err(corrupt());
                }
            }
        }
        CodeKind::Binary => {
            let bit_bytes = dim.div_ceil(64) * 8;
            for code in codes.chunks_exact(codec.code_len()) {
                let tail = code[bit_bytes..].as_chunks::<4>().0;
                let norm = f32::from_le_bytes(tail[0]);
                let centre_dot = f32::from_le_bytes(tail[1]);
                if !norm.is_finite() || norm < 0.0 || !centre_dot.is_finite() {
                    return Err(corrupt());
                }
                if !dim.is_multiple_of(64) {
                    let last = u64::from_le_bytes(
                        code[bit_bytes - 8..bit_bytes]
                            .try_into()
                            .expect("full sign word"),
                    );
                    if last >> (dim % 64) != 0 {
                        return Err(corrupt());
                    }
                }
            }
        }
        CodeKind::Int8 => {} // Every byte is a valid scalar code.
    }
    Ok(())
}
