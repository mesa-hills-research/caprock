//! Flat search and rescoring within one segment (unit U13; docs/requirements.md, "Search").

use crate::distance;
use crate::error::Result;
use crate::filter::SegmentFilter;
use crate::quant::{Codec, with_codec};
use crate::segment::Segment;
use crate::tombstones::Tombstones;
use crate::topk::TopK;
use crate::types::Hit;

/// One segment's part of a search.
#[derive(Debug, Clone, Copy)]
pub struct SegmentQuery<'a> {
    /// The query in stored form (normalized for cosine).
    pub query: &'a [f32],
    pub k: usize,
    /// Candidates to rescore at full precision (at least `k`).
    pub rescore: usize,
    /// `None`: every record.
    pub filter: Option<&'a SegmentFilter>,
    pub tombstones: &'a Tombstones,
}

/// Scans the codes of every live record the filter allows, keeps the `rescore` best
/// estimates (ties by key, through [`crate::topk::TopK`]), and returns [`rescore`]'s
/// top `k`. With f32 codes the estimates are canonical, so the result is exact.
pub fn flat_search(segment: &Segment, q: &SegmentQuery<'_>) -> Result<Vec<Hit>> {
    if nothing_to_search(segment, q) {
        return Ok(Vec::new());
    }
    // Dispatch once, not once per vector, so each scan is codec-specialized.
    let candidates = with_codec!(segment.codec(), codec => scan_codes(segment, q, codec));
    rescore(segment, q.query, &candidates, q.k)
}

/// The exact top `k` over full-precision vectors of every live record the filter
/// allows: canonical scores, result order. Equal to the oracle restricted to this
/// segment.
pub fn exact_search(segment: &Segment, q: &SegmentQuery<'_>) -> Result<Vec<Hit>> {
    if nothing_to_search(segment, q) {
        return Ok(Vec::new());
    }
    score_vectors(segment, q.query, matching_ordinals(segment, q), q.k)
}

/// The top `k` of `candidates` (estimate, ordinal) after rescoring with canonical scores
/// from full-precision vectors, in result order.
#[rustfmt::skip]
pub fn rescore(segment: &Segment, query: &[f32], candidates: &[(f32, u32)], k: usize) -> Result<Vec<Hit>> {
    if k == 0 || candidates.is_empty() {
        return Ok(Vec::new());
    }
    // Estimates no longer participate in ranking. Reading in ordinal order lets the
    // segment coalesce adjacent vectors; repeated candidates must not repeat hits.
    let mut ordinals: Vec<u32> = candidates.iter().map(|&(_, ordinal)| ordinal).collect();
    ordinals.sort_unstable();
    ordinals.dedup();
    score_vectors(segment, query, ordinals.into_iter(), k)
}

/// The [`Hit`] for an ordinal with a score.
pub fn hit(segment: &Segment, ordinal: u32, score: f32) -> Hit {
    Hit {
        key: segment.key(ordinal).to_owned(),
        score,
        path: segment.path(ordinal).to_owned(),
        language: segment.language(ordinal).to_owned(),
    }
}

fn nothing_to_search(segment: &Segment, q: &SegmentQuery<'_>) -> bool {
    q.k == 0
        || segment.is_empty()
        || q.tombstones.live() == 0
        || q.filter.is_some_and(SegmentFilter::is_empty)
}

fn matching_ordinals<'a>(
    segment: &'a Segment,
    q: &'a SegmentQuery<'_>,
) -> impl Iterator<Item = u32> + 'a {
    // A compiled filter has disjoint, sorted ranges. Skip unrelated paths entirely,
    // especially important when a selective filter makes a large segment scan flat.
    let all = q.filter.is_none().then_some(0..segment.len());
    all.into_iter()
        .chain(q.filter.into_iter().flat_map(|f| f.ranges.iter().cloned()))
        .flatten()
        .filter(move |&ordinal| {
            !q.tombstones.is_deleted(ordinal) && q.filter.is_none_or(|f| f.allows(ordinal, segment))
        })
}

fn scan_codes<C: Codec>(segment: &Segment, q: &SegmentQuery<'_>, codec: &C) -> Vec<(f32, u32)> {
    let prepared = codec.prepare(q.query);
    let wanted = q.rescore.max(q.k).min(q.tombstones.live() as usize);
    let mut top = TopK::new(wanted);
    let width = codec.code_len();
    let codes = segment.codes();
    for ordinal in matching_ordinals(segment, q) {
        let start = ordinal as usize * width;
        let score = codec.score(&prepared, &codes[start..start + width]);
        top.offer(score, ordinal, segment);
    }
    top.into_sorted(segment)
}

// Bound full-precision scratch independently of the segment or candidate count.
// Segment::read_vectors also bounds its temporary byte buffer and coalesces reads.
const VECTOR_BATCH_BYTES: usize = 64 * 1024;

fn score_vectors(
    segment: &Segment,
    query: &[f32],
    mut ordinals: impl Iterator<Item = u32>,
    k: usize,
) -> Result<Vec<Hit>> {
    let k = k.min(segment.len() as usize);
    if k == 0 {
        return Ok(Vec::new());
    }
    let dim = segment.dim();
    let metric = with_codec!(segment.codec(), codec => codec.metric());
    let batch_size = (VECTOR_BATCH_BYTES / (dim * std::mem::size_of::<f32>()))
        .max(1)
        .min(segment.len() as usize);
    let mut batch = Vec::with_capacity(batch_size);
    let mut vectors = Vec::new();
    let mut top = TopK::new(k);
    loop {
        batch.clear();
        batch.extend(ordinals.by_ref().take(batch_size));
        if batch.is_empty() {
            break;
        }
        vectors.resize(batch.len() * dim, 0.0);
        segment.read_vectors(&batch, &mut vectors)?;
        for (&ordinal, vector) in batch.iter().zip(vectors.chunks_exact(dim)) {
            // The query is already in stored form. In particular, never normalize a
            // cosine query again: that would change the canonical score's bits.
            top.offer(distance::score(metric, query, vector), ordinal, segment);
        }
    }
    Ok(top
        .into_sorted(segment)
        .into_iter()
        .map(|(score, ordinal)| hit(segment, ordinal, score))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CodeKind, IndexConfig, Metric, VerifyMode};
    use crate::error::Error;
    use crate::filter::Filter;
    use crate::segment::{MemoryRecords, SegmentBuild, write_segment};
    use crate::topk::hit_order;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Fixture {
        segment: Segment,
        dir: PathBuf,
        expected: Vec<Hit>,
        query: Vec<f32>,
    }

    impl Fixture {
        fn new(metric: Metric, codes: CodeKind) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let dir = std::env::temp_dir().join(format!(
                "mhrv-flat-unit-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&dir).unwrap();
            // Three full-precision batches, with a partial last batch and many ties.
            let dim = 257;
            let mut records = MemoryRecords {
                dim,
                ..MemoryRecords::default()
            };
            for i in 0..137 {
                let vector: Vec<f32> = (0..dim)
                    .map(|j| ((i % 11 + 1) * (j % 13 + 1)) as f32 / 31.0 - 0.5)
                    .collect();
                records
                    .vectors
                    .extend(distance::prepare(metric, &vector, dim).unwrap());
                records.keys.push(format!("key-{:03}", 137 - i));
                let root = if i % 3 == 0 { "src" } else { "other" };
                records.paths.push(format!("{root}/p{}", i % 7));
                records
                    .languages
                    .push(if i % 2 == 0 { "rust" } else { "go" }.into());
            }
            let query = distance::prepare(metric, &vec![0.3; dim], dim).unwrap();
            let mut expected: Vec<_> = records
                .vectors
                .chunks_exact(dim)
                .enumerate()
                .map(|(i, vector)| Hit {
                    key: records.keys[i].clone(),
                    score: distance::score(metric, &query, vector),
                    path: records.paths[i].clone(),
                    language: records.languages[i].clone(),
                })
                .collect();
            expected.sort_unstable_by(hit_order);
            let config = IndexConfig::new(dim as u32, metric).with_codes(codes);
            let info = write_segment(
                &dir,
                &records,
                &SegmentBuild {
                    id: 1,
                    config: &config,
                    graph: false,
                    threads: 1,
                    memory_budget: 64 << 20,
                },
            )
            .unwrap();
            let segment = Segment::open(&dir, &info.file, &config, VerifyMode::Full).unwrap();
            Self {
                segment,
                dir,
                expected,
                query,
            }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    fn assert_hits(actual: &[Hit], expected: &[Hit]) {
        assert_eq!(actual, expected);
        for (a, e) in actual.iter().zip(expected) {
            assert_eq!(a.score.to_bits(), e.score.to_bits());
        }
    }

    #[test]
    fn batched_reads_and_rescoring_are_canonical_for_every_codec() {
        for metric in [Metric::Cosine, Metric::Dot, Metric::L2] {
            for codes in [CodeKind::F32, CodeKind::Int8, CodeKind::Binary] {
                let fx = Fixture::new(metric, codes);
                let tombstones = Tombstones::none(fx.segment.len());
                // Estimates are irrelevant; reversed and duplicated ordinals must
                // produce the same unique hits and bits, independent of read order.
                let candidates: Vec<_> = (0..fx.segment.len())
                    .rev()
                    .flat_map(|ordinal| [(f32::NAN, ordinal), (f32::INFINITY, ordinal)])
                    .collect();
                for k in [1, 25, usize::MAX] {
                    let q = SegmentQuery {
                        query: &fx.query,
                        k,
                        rescore: usize::MAX,
                        filter: None,
                        tombstones: &tombstones,
                    };
                    let expected = &fx.expected[..k.min(fx.expected.len())];
                    assert_hits(&exact_search(&fx.segment, &q).unwrap(), expected);
                    assert_hits(&flat_search(&fx.segment, &q).unwrap(), expected);
                    assert_hits(
                        &rescore(&fx.segment, &fx.query, &candidates, k).unwrap(),
                        expected,
                    );
                }
            }
        }
    }

    #[test]
    fn filters_deletions_empty_results_and_rescore_lower_bound() {
        for codes in [CodeKind::F32, CodeKind::Int8, CodeKind::Binary] {
            let fx = Fixture::new(Metric::Dot, codes);
            let filter = Filter {
                path_prefixes: vec!["src/p0".into(), "src".into(), "src".into()],
                languages: vec!["rust".into()],
            };
            let compiled = SegmentFilter::compile(&filter, &fx.segment);
            let tombstones = Tombstones::none(fx.segment.len())
                .with_deleted(&(0..fx.segment.len()).step_by(4).collect::<Vec<_>>());
            let expected: Vec<_> = fx
                .expected
                .iter()
                .filter(|h| {
                    filter.matches(&h.path, &h.language)
                        && !tombstones.is_deleted(fx.segment.find(&h.key).unwrap())
                })
                .cloned()
                .collect();
            let mut q = SegmentQuery {
                query: &fx.query,
                k: usize::MAX,
                rescore: 0,
                filter: Some(&compiled),
                tombstones: &tombstones,
            };
            assert!(!expected.is_empty());
            assert_hits(&flat_search(&fx.segment, &q).unwrap(), &expected);
            assert_hits(&exact_search(&fx.segment, &q).unwrap(), &expected);
            q.k = 5;
            let hits = flat_search(&fx.segment, &q).unwrap();
            assert_eq!(hits.len(), 5);
            assert!(hits.iter().all(|h| expected.contains(h)));
            if codes == CodeKind::F32 {
                assert_hits(&hits, &expected[..5]);
            }
            q.k = 0;
            assert!(flat_search(&fx.segment, &q).unwrap().is_empty());
            assert!(exact_search(&fx.segment, &q).unwrap().is_empty());
            q.k = 5;
            let missing = SegmentFilter::compile(&Filter::language("absent"), &fx.segment);
            q.filter = Some(&missing);
            assert!(flat_search(&fx.segment, &q).unwrap().is_empty());
            assert!(exact_search(&fx.segment, &q).unwrap().is_empty());
            let deleted = tombstones.with_deleted(&(0..fx.segment.len()).collect::<Vec<_>>());
            q.filter = None;
            q.tombstones = &deleted;
            assert!(flat_search(&fx.segment, &q).unwrap().is_empty());
            assert!(exact_search(&fx.segment, &q).unwrap().is_empty());
        }
    }

    #[test]
    fn rescore_handles_empty_inputs_and_propagates_read_errors() {
        let fx = Fixture::new(Metric::L2, CodeKind::Int8);
        assert!(rescore(&fx.segment, &fx.query, &[], 10).unwrap().is_empty());
        assert!(
            rescore(&fx.segment, &fx.query, &[(1.0, 0)], 0)
                .unwrap()
                .is_empty()
        );
        assert!(matches!(
            rescore(&fx.segment, &fx.query, &[(0.0, fx.segment.len())], 1),
            Err(Error::InvalidArgument(_))
        ));
        fs::OpenOptions::new()
            .write(true)
            .open(fx.dir.join(fx.segment.file_name()))
            .unwrap()
            .set_len(0)
            .unwrap();
        assert!(rescore(&fx.segment, &fx.query, &[(1.0, 0)], 1).is_err());
        let tombstones = Tombstones::none(fx.segment.len());
        let q = SegmentQuery {
            query: &fx.query,
            k: 1,
            rescore: 4,
            filter: None,
            tombstones: &tombstones,
        };
        assert!(exact_search(&fx.segment, &q).is_err());
        assert!(flat_search(&fx.segment, &q).is_err());
    }
}
