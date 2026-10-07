//! Streaming, deterministic construction. Only graph builds retain encoded vectors.
use std::path::Path;

use crate::config::{AnnKind, CodeKind, IndexConfig};
use crate::error::{Error, Result};
use crate::format::{SEGMENT_MAGIC, SectionKind, SectionWriter};
use crate::quant::{AnyCodec, MAX_TRAINING_VECTORS, training_sample, with_codec};
use crate::util::mix64;

use super::{Info, RecordSource, SegmentBuild, SegmentInfo};

const BUFFER_BYTES: usize = 64 * 1024;

// Permutation and key index coexist with training, language sorting, or graph
// construction. String bytes are streamed, not copied into build RAM.
pub(super) fn memory_required(config: &IndexConfig, n: u64, graph: bool) -> u64 {
    let dim = u64::from(config.dim);
    let sample = n.min(MAX_TRAINING_VECTORS as u64);
    let training = if config.codes == CodeKind::F32 {
        0
    } else {
        // Sample matrix, row ids, and sampler's temporary ordered set.
        sample.saturating_mul(4 * dim + 88)
    };
    let languages = n.saturating_mul(std::mem::size_of::<&str>() as u64);
    let graph_bytes = if let (true, AnnKind::Vamana(params)) = (graph, config.ann.kind) {
        let code_len = match config.codes {
            CodeKind::F32 => 4 * dim,
            CodeKind::Int8 => dim,
            CodeKind::Binary => dim.div_ceil(64) * 8 + 8,
        };
        // Codes, adjacency, reverse-edge/pruning workspace and candidate lists.
        n.saturating_mul(
            code_len + 16 * u64::from(params.max_degree) + 32 * u64::from(params.build_beam) + 256,
        )
    } else {
        0
    };
    n.saturating_mul(8)
        .saturating_add(training.max(languages).max(graph_bytes))
        // Streaming buffers, codec workspace, small tables and allocator slack.
        .saturating_add(2 * BUFFER_BYTES as u64 + 128 * dim + 8192)
}

pub(super) fn write(
    dir: &Path,
    source: &dyn RecordSource,
    build: &SegmentBuild<'_>,
) -> Result<SegmentInfo> {
    let config = build.config;
    config.validate()?;
    let n = source.len();
    if n == 0 || n > u32::MAX as usize {
        return Err(Error::InvalidArgument(
            "segment length must be in 1..=u32::MAX".into(),
        ));
    }
    if build.threads == 0 {
        return Err(Error::InvalidArgument(
            "build threads must be positive".into(),
        ));
    }
    if build.graph && matches!(config.ann.kind, AnnKind::Flat) {
        return Err(Error::InvalidArgument(
            "a graph requires Vamana parameters".into(),
        ));
    }
    let required = memory_required(config, n as u64, build.graph);
    if required > build.memory_budget {
        return Err(Error::Budget {
            budget: build.memory_budget,
            detail: format!("segment build requires a workspace allowance of {required} bytes"),
        });
    }
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_unstable_by(|&a, &b| {
        (source.path(a as usize), source.key(a as usize))
            .cmp(&(source.path(b as usize), source.key(b as usize)))
    });
    // KeyIndex is a key-sorted permutation of ordinals, not input row numbers.
    let mut key_index: Vec<u32> = (0..n as u32).collect();
    key_index.sort_unstable_by(|&a, &b| {
        source
            .key(order[a as usize] as usize)
            .cmp(source.key(order[b as usize] as usize))
    });
    if key_index.windows(2).any(|pair| {
        source.key(order[pair[0] as usize] as usize) == source.key(order[pair[1] as usize] as usize)
    }) {
        return Err(Error::InvalidArgument(
            "segment keys must be distinct".into(),
        ));
    }
    validate_metadata(source, &order)?;
    let dim = config.dim as usize;
    let seed = mix64(config.seed ^ mix64(build.id));
    let codec = if config.codes == CodeKind::F32 {
        // F32 has no trained parameters: don't materialize a useless sample.
        AnyCodec::train(config.codes, config.metric, dim, &[], seed)?
    } else {
        let rows = training_sample(n, seed);
        let mut sample = vec![0.0; rows.len() * dim];
        for (&row, vector) in rows.iter().zip(sample.chunks_exact_mut(dim)) {
            source.read_vector(order[row] as usize, vector)?;
        }
        AnyCodec::train(config.codes, config.metric, dim, &sample, seed)?
    };
    let file_name = format!("seg-{:020}.mhrv", build.id);
    let mut writer = SectionWriter::create(&dir.join(&file_name), SEGMENT_MAGIC)?;
    let info = Info {
        id: build.id,
        len: n as u32,
        dim: config.dim,
        metric: config.metric,
        codes: config.codes,
        has_graph: build.graph,
    };
    writer.write_section(SectionKind::Info, &json(&info)?)?;
    writer.write_section(SectionKind::CodecParams, &codec.to_bytes())?;
    let mut vector = vec![0.0; dim];
    let mut encoded = vec![0; codec.code_len()];
    let mut vector_bytes = vec![0; dim * 4];
    let mut graph_codes = if build.graph {
        vec![
            0;
            n.checked_mul(codec.code_len())
                .ok_or_else(|| Error::Budget {
                    budget: build.memory_budget,
                    detail: "code array exceeds address space".into(),
                })?
        ]
    } else {
        Vec::new()
    };
    let vector_kind = if config.codes == CodeKind::F32 {
        SectionKind::Codes
    } else {
        SectionKind::Vectors
    };
    section(&mut writer, vector_kind, |out| {
        for (ordinal, &row) in order.iter().enumerate() {
            source.read_vector(row as usize, &mut vector)?;
            // Stored-form vectors must not be normalized a second time.
            crate::distance::validate(&vector, dim)?;
            for (&value, bytes) in vector.iter().zip(vector_bytes.as_chunks_mut::<4>().0) {
                *bytes = value.to_le_bytes();
            }
            out.push(&vector_bytes)?;
            if build.graph {
                let start = ordinal * codec.code_len();
                codec.encode(&vector, &mut graph_codes[start..start + codec.code_len()]);
            }
        }
        Ok(())
    })?;
    if config.codes != CodeKind::F32 {
        if build.graph {
            writer.write_section(SectionKind::Codes, &graph_codes)?;
        } else {
            section(&mut writer, SectionKind::Codes, |out| {
                for &row in &order {
                    source.read_vector(row as usize, &mut vector)?;
                    codec.encode(&vector, &mut encoded);
                    out.push(&encoded)?;
                }
                Ok(())
            })?;
        }
    }
    // Graph workspace and language-sort workspace do not coexist. Bound worker
    // scratch independently of the requested thread count, as the allowance above
    // covers at most eight simultaneous searches. This cannot change graph bytes.
    if build.graph {
        let AnnKind::Vamana(params) = config.ann.kind else {
            unreachable!()
        };
        let graph = with_codec!(&codec, c => crate::vamana::build_graph(
            c, &graph_codes, &params, seed, build.threads.min(8)
        ))?;
        // Stream Graph's documented representation, avoiding a second adjacency.
        section(&mut writer, SectionKind::Graph, |out| {
            out.push(&graph.degree.to_le_bytes())?;
            out.push(&graph.entry.to_le_bytes())?;
            out.push(&(n as u64).to_le_bytes())?;
            for neighbor in &graph.adjacency {
                out.push(&neighbor.to_le_bytes())?;
            }
            Ok(())
        })?;
    }
    drop(graph_codes);
    write_metadata(&mut writer, source, &order, &key_index)?;
    let bytes = writer.finish()?;
    Ok(SegmentInfo {
        id: build.id,
        file: file_name,
        vectors: n as u64,
        has_graph: build.graph,
        bytes,
    })
}

fn validate_metadata(source: &dyn RecordSource, order: &[u32]) -> Result<()> {
    let mut key_bytes = 0u64;
    let mut path_bytes = 0u64;
    let mut previous = None;
    for &row in order {
        let row = row as usize;
        let key = source.key(row);
        let path = source.path(row);
        if key.is_empty() || key.len() > crate::MAX_KEY_BYTES {
            return Err(Error::InvalidKey(
                "key length outside 1..=MAX_KEY_BYTES".into(),
            ));
        }
        if path.len() > crate::MAX_PATH_BYTES
            || source.language(row).len() > crate::MAX_LANGUAGE_BYTES
        {
            return Err(Error::InvalidRecord(
                "path or language exceeds length limit".into(),
            ));
        }
        key_bytes += key.len() as u64;
        if previous != Some(path) {
            path_bytes += path.len() as u64;
            previous = Some(path);
        }
    }
    if key_bytes > u64::from(u32::MAX) || path_bytes > u64::from(u32::MAX) {
        return Err(Error::InvalidArgument(
            "string table exceeds u32 offsets".into(),
        ));
    }
    Ok(())
}

fn write_metadata(
    writer: &mut SectionWriter,
    source: &dyn RecordSource,
    order: &[u32],
    key_index: &[u32],
) -> Result<()> {
    section(writer, SectionKind::KeyBytes, |out| {
        for &row in order {
            out.push(source.key(row as usize).as_bytes())?;
        }
        Ok(())
    })?;
    section(writer, SectionKind::KeyOffsets, |out| {
        let mut offset = 0u32;
        out.push(&offset.to_le_bytes())?;
        for &row in order {
            offset += source.key(row as usize).len() as u32;
            out.push(&offset.to_le_bytes())?;
        }
        Ok(())
    })?;
    section(writer, SectionKind::KeyIndex, |out| {
        for id in key_index {
            out.push(&id.to_le_bytes())?;
        }
        Ok(())
    })?;
    section(writer, SectionKind::PathBytes, |out| {
        let mut previous = None;
        for &row in order {
            let path = source.path(row as usize);
            if previous != Some(path) {
                out.push(path.as_bytes())?;
                previous = Some(path);
            }
        }
        Ok(())
    })?;
    section(writer, SectionKind::PathOffsets, |out| {
        let mut previous = None;
        let mut offset = 0u32;
        out.push(&offset.to_le_bytes())?;
        for &row in order {
            let path = source.path(row as usize);
            if previous != Some(path) {
                offset += path.len() as u32;
                out.push(&offset.to_le_bytes())?;
                previous = Some(path);
            }
        }
        Ok(())
    })?;
    section(writer, SectionKind::PathOfOrdinal, |out| {
        let mut previous = None;
        let mut id = 0u32;
        for &row in order {
            let path = source.path(row as usize);
            if previous.is_some_and(|previous| previous != path) {
                id += 1;
            }
            previous = Some(path);
            out.push(&id.to_le_bytes())?;
        }
        Ok(())
    })?;
    let mut languages: Vec<&str> = order
        .iter()
        .map(|&row| source.language(row as usize))
        .collect();
    languages.sort_unstable();
    languages.dedup();
    if languages.len() > usize::from(u16::MAX) + 1 {
        return Err(Error::InvalidArgument(
            "too many distinct languages in a segment".into(),
        ));
    }
    section(writer, SectionKind::Languages, |out| {
        out.push(b"[")?;
        for (i, language) in languages.iter().enumerate() {
            if i != 0 {
                out.push(b",")?;
            }
            out.push(&json(language)?)?;
        }
        out.push(b"]")
    })?;
    section(writer, SectionKind::LanguageOfOrdinal, |out| {
        for &row in order {
            let id = languages
                .binary_search(&source.language(row as usize))
                .expect("collected language");
            out.push(&(id as u16).to_le_bytes())?;
        }
        Ok(())
    })
}

fn json(value: &impl serde::Serialize) -> Result<Vec<u8>> {
    serde_json::to_vec(value)
        .map_err(|e| Error::InvalidArgument(format!("cannot serialize metadata: {e}")))
}

struct Buffer<'a> {
    writer: &'a mut SectionWriter,
    bytes: Vec<u8>,
}

impl Buffer<'_> {
    fn push(&mut self, bytes: &[u8]) -> Result<()> {
        if self.bytes.len() + bytes.len() > BUFFER_BYTES {
            self.flush()?;
        }
        if bytes.len() >= BUFFER_BYTES {
            self.writer.write_chunk(bytes)
        } else {
            self.bytes.extend_from_slice(bytes);
            Ok(())
        }
    }
    fn flush(&mut self) -> Result<()> {
        self.writer.write_chunk(&self.bytes)?;
        self.bytes.clear();
        Ok(())
    }
}

fn section(
    writer: &mut SectionWriter,
    kind: SectionKind,
    fill: impl FnOnce(&mut Buffer<'_>) -> Result<()>,
) -> Result<()> {
    writer.begin_section(kind)?;
    let mut buffer = Buffer {
        writer,
        bytes: Vec::with_capacity(BUFFER_BYTES),
    };
    fill(&mut buffer)?;
    buffer.flush()?;
    writer.end_section()
}
