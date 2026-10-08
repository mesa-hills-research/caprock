//! Searching a snapshot.
//!
//! For each segment: compile the filter; when the search is exact, or the segment has no
//! graph, or at most `filter_flat_below` live records match, search it flat (exact search
//! for `exact`); otherwise beam-search its graph (beam `max(params.beam or
//! search_beam, k, rescore)`), with separate routing and matching frontiers when
//! admission excludes nodes, and rescore. If a graph search finds fewer than
//! `min(k, matching live)` records, fall back to a flat search of that segment.
//! Search the delta exactly. Merge everything into the top `k`.

use super::Snapshot;
use crate::config::{AnnKind, SearchParams};
use crate::distance;
use crate::error::{Error, Result};
use crate::filter::{Filter, SegmentFilter};
use crate::flat::{self, SegmentQuery};
use crate::quant::{Codec, with_codec};
use crate::segment::Segment;
use crate::tombstones::Tombstones;
use crate::topk::merge_hits;
use crate::types::Hit;
use crate::vamana::search::filtered_beam_search;
use crate::vamana::{SearchScratch, beam_search};

impl Snapshot {
    /// The top `k` live records matching `filter` for a raw `query`, in result order
    /// (score descending, key ascending), with canonical scores. Exactly
    /// `min(k, live records matching)` hits. Errors: `InvalidVector` for a bad query,
    /// `InvalidArgument` for `k` of 0 or above [`crate::MAX_K`], `Io`/`Corrupt` from
    /// reading vectors.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
        params: &SearchParams,
    ) -> Result<Vec<Hit>> {
        if !(1..=crate::MAX_K).contains(&k) {
            return Err(Error::InvalidArgument(format!(
                "k must be in 1..={}",
                crate::MAX_K
            )));
        }
        let config = self.config();
        // Prepare even an empty snapshot's query, and normalize cosine only once.
        let query = distance::prepare(config.metric, query, config.dim as usize)?;
        let rescore = params
            .rescore
            .map_or_else(
                || {
                    k.saturating_mul(config.rescore_factor as usize)
                        .max(config.rescore_min as usize)
                },
                |n| n as usize,
            )
            .max(k);
        let default_beam = match config.ann.kind {
            AnnKind::Vamana(p) => p.search_beam as usize,
            AnnKind::Flat => 0,
        };
        let beam = params
            .beam
            .map_or(default_beam, |n| n as usize)
            .max(rescore);
        let filter = filter.filter(|f| !f.is_empty());
        let mut scratch = SearchScratch::new();
        let mut lists = Vec::with_capacity(self.inner.segments.len() + 1);
        for (segment, tombstones) in &self.inner.segments {
            if tombstones.live() == 0 {
                continue;
            }
            let compiled = filter.map(|f| SegmentFilter::compile(f, segment.as_ref()));
            if compiled.as_ref().is_some_and(SegmentFilter::is_empty) {
                continue;
            }
            let q = SegmentQuery {
                query: &query,
                k,
                rescore,
                filter: compiled.as_ref(),
                tombstones,
            };
            let hits = if params.exact {
                flat::exact_search(segment, &q)?
            } else if let Some(graph) = segment.graph() {
                let matching = matching_live(segment, tombstones, compiled.as_ref());
                if matching == 0 {
                    continue;
                }
                if matching <= config.ann.filter_flat_below {
                    flat::flat_search(segment, &q)?
                } else {
                    let admit = |ordinal| {
                        !tombstones.is_deleted(ordinal)
                            && compiled
                                .as_ref()
                                .is_none_or(|f| f.allows(ordinal, segment.as_ref()))
                    };
                    let candidates = with_codec!(segment.codec(), codec => {
                        let prepared = codec.prepare(&query);
                        // A full result list alone does not imply good filtered
                        // recall. Let matching nodes make their own progress even
                        // when better-scoring rejected nodes dominate navigation.
                        if matching < segment.len() {
                            filtered_beam_search(
                                graph, codec, segment.codes(), &prepared, beam,
                                rescore.min(matching as usize), &admit, &mut scratch,
                            )
                        } else {
                            beam_search(
                                graph, codec, segment.codes(), &prepared, beam,
                                rescore.min(matching as usize), &admit, &mut scratch,
                            )
                        }
                    });
                    // Rejected/deleted nodes still route the graph, but a selective
                    // filter can leave too few admitted candidates. Never return a
                    // short list when a flat search can supply the missing hits.
                    if candidates.len() < k.min(matching as usize) {
                        flat::flat_search(segment, &q)?
                    } else {
                        flat::rescore(segment, &query, &candidates, k)?
                    }
                }
            } else {
                flat::flat_search(segment, &q)?
            };
            lists.push(hits);
        }
        lists.push(self.inner.delta.search(&query, k, filter));
        // Writer publication and manifest loading preserve a single live owner for
        // each key, so the per-source lists have no keys in common.
        Ok(merge_hits(lists, k))
    }
}

fn matching_live(
    segment: &Segment,
    tombstones: &Tombstones,
    filter: Option<&SegmentFilter>,
) -> u32 {
    let Some(filter) = filter else {
        return tombstones.live();
    };
    if tombstones.deleted() == 0 {
        return filter.count(segment);
    }
    filter
        .ranges
        .iter()
        .flat_map(|r| r.clone())
        .filter(|&ordinal| !tombstones.is_deleted(ordinal) && filter.allows(ordinal, segment))
        .count() as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CodeKind, IndexConfig, Metric, StoreOptions, VamanaParams};
    use crate::store::Writer;
    use crate::topk::hit_order;
    use crate::types::Record;
    use std::cell::RefCell;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Fixture {
        writer: Writer,
        dir: PathBuf,
    }

    impl Fixture {
        fn new(metric: Metric, codes: CodeKind) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let dir = std::env::temp_dir().join(format!(
                "caprock-search-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let mut config = IndexConfig::new(5, metric).with_codes(codes);
            config.ann.graph_min_vectors = 2;
            config.ann.filter_flat_below = 1;
            config.ann.kind = AnnKind::Vamana(VamanaParams {
                max_degree: 4,
                build_beam: 8,
                ..VamanaParams::default()
            });
            let options = StoreOptions {
                threads: 1,
                ..StoreOptions::default()
            };
            let mut writer = Writer::create(&dir, config, options).unwrap();
            let mut state = 17u32;
            for i in 0..96 {
                let vector: Vec<_> = (0..5)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        (state % 2000) as f32 / 1000.0 - 1.0
                    })
                    .collect();
                writer
                    .upsert(Record {
                        key: &format!("key-{i:03}"),
                        vector: &vector,
                        path: &format!("row/{i:03}"),
                        language: if i % 2 == 0 { "rust" } else { "go" },
                    })
                    .unwrap();
            }
            writer.commit(None).unwrap();
            writer.compact().unwrap();
            assert!(writer.snapshot().inner.segments[0].0.graph().is_some());
            Self { writer, dir }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    #[test]
    fn exhaustive_overrides_and_small_beams_preserve_counts_and_canonical_scores() {
        for metric in [Metric::Cosine, Metric::Dot, Metric::L2] {
            for codes in [CodeKind::F32, CodeKind::Int8, CodeKind::Binary] {
                let mut fx = Fixture::new(metric, codes);
                for i in (0..96).step_by(3) {
                    fx.writer.delete(&format!("key-{i:03}")).unwrap();
                }
                fx.writer
                    .upsert(Record {
                        key: "key-001",
                        vector: &[0.9; 5],
                        path: "row/001",
                        language: "rust",
                    })
                    .unwrap();
                let snapshot = fx.writer.snapshot();
                let raw = [1.0, 0.5, -0.8, 0.2, 0.0];
                let query = distance::prepare(metric, &raw, 5).unwrap();
                for filter in [
                    Filter::all(),
                    Filter::language("rust"),
                    Filter::path("absent"),
                ] {
                    let mut expected: Vec<Hit> = snapshot
                        .keys()
                        .iter()
                        .filter_map(|key| {
                            let record = snapshot.get(key).unwrap().unwrap();
                            filter.matches(&record.path, &record.language).then(|| Hit {
                                score: distance::score(metric, &query, &record.vector),
                                key: record.key,
                                path: record.path,
                                language: record.language,
                            })
                        })
                        .collect();
                    expected.sort_unstable_by(hit_order);
                    for k in [1, 10, crate::MAX_K] {
                        for count in [0, u32::MAX] {
                            let params = SearchParams {
                                beam: Some(count),
                                rescore: Some(count),
                                exact: false,
                            };
                            let hits = snapshot.search(&raw, k, Some(&filter), &params).unwrap();
                            assert_eq!(hits.len(), k.min(expected.len()));
                            assert!(hits.windows(2).all(|h| hit_order(&h[0], &h[1]).is_lt()));
                            for hit in &hits {
                                let exact = expected.iter().find(|e| e.key == hit.key).unwrap();
                                assert_eq!(hit, exact);
                                assert_eq!(hit.score.to_bits(), exact.score.to_bits());
                            }
                            if count == u32::MAX {
                                assert_eq!(hits, expected[..k.min(expected.len())]);
                            }
                        }
                        assert_eq!(
                            snapshot
                                .search(&raw, k, Some(&filter), &SearchParams::exact())
                                .unwrap(),
                            expected[..k.min(expected.len())]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_graph_with_no_admitted_candidates_falls_back_to_flat() {
        let fx = Fixture::new(Metric::Dot, CodeKind::F32);
        let snapshot = fx.writer.snapshot();
        let segment = &snapshot.inner.segments[0].0;
        let query = [1.0, 0.5, -0.8, 0.2, 0.0];
        let visited = RefCell::new(vec![false; segment.len() as usize]);
        with_codec!(segment.codec(), codec => {
            let prepared = codec.prepare(&query);
            beam_search(segment.graph().unwrap(), codec, segment.codes(), &prepared,
                1, 1, &|i| {
                    visited.borrow_mut()[i as usize] = true;
                    false
                }, &mut SearchScratch::new())
        });
        // Admission never changes navigation. This filter rejects every node the
        // beam scored but admits many other live nodes, forcing the short-list path.
        let filter = Filter {
            path_prefixes: (0..segment.len())
                .filter(|&i| !visited.borrow()[i as usize])
                .map(|i| segment.path(i).to_owned())
                .collect(),
            languages: Vec::new(),
        };
        assert!(filter.path_prefixes.len() > 1);
        let params = SearchParams {
            beam: Some(1),
            rescore: Some(1),
            exact: false,
        };
        let hits = snapshot.search(&query, 1, Some(&filter), &params).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(
            hits,
            snapshot
                .search(&query, 1, Some(&filter), &SearchParams::exact())
                .unwrap()
        );
    }
}
