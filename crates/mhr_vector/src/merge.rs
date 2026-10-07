//! Merge planning and the memory math (unit U16; REQUIREMENTS.md, "Maintenance" and
//! "Memory").
//!
//! Planning is a pure function of segment summaries, so the same segments always give
//! the same plans (determinism of layouts). Plans never share inputs.

use std::collections::BTreeMap;

use crate::config::{AnnConfig, AnnKind, CodeKind, IndexConfig, MergePolicy};
use crate::error::{Error, Result};
use crate::segment::{RecordSource, Segment, build_memory_required};
use crate::tombstones::Tombstones;

/// What a planner knows about a segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentSummary {
    pub id: u64,
    pub vectors: u64,
    pub live: u64,
    pub has_graph: bool,
}

/// Why a merge was planned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeReason {
    /// Enough segments of one size tier.
    Tier,
    /// A segment's deleted share reached the policy's limit.
    Deletes,
    /// A flat segment is large enough for a graph.
    Graph,
    /// [`plan_compaction`].
    Compact,
}

/// One planned merge: rewrite the live records of `inputs` into one new segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergePlan {
    /// Input segment ids, oldest first.
    pub inputs: Vec<u64>,
    /// Whether the new segment gets a graph (its live count reaches
    /// `ann.graph_min_vectors` and the ANN kind is Vamana).
    pub graph: bool,
    pub reason: MergeReason,
}

/// The merges the policy wants now, in a deterministic order, with disjoint inputs, each
/// producing at most `max_segment_vectors` live records. Segments are given oldest first.
pub fn plan(
    segments: &[SegmentSummary],
    policy: &MergePolicy,
    ann: &AnnConfig,
    max_segment_vectors: u64,
) -> Vec<MergePlan> {
    // Power-of-two live-count tiers keep the inputs of a tier merge within a
    // factor of two. Use live sizes: tombstoned records will not reach the output.
    let mut tiers: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (i, segment) in segments.iter().enumerate() {
        if segment.vectors > 0 && segment.live > 0 && segment.live <= max_segment_vectors {
            tiers.entry(segment.live.ilog2()).or_default().push(i);
        }
    }

    let mut used = vec![false; segments.len()];
    let mut groups = Vec::new();
    let fan_in = policy.segments_per_tier.max(2) as usize;
    for indices in tiers.values_mut() {
        // Take the smallest first so that an oversized group cannot hide a later
        // group that would fit. Equal sizes are resolved by age, not segment id.
        indices.sort_unstable_by_key(|&i| (segments[i].live, i));
        for chunk in indices.chunks(fan_in) {
            if chunk.len() < fan_in {
                break;
            }
            let Some(live) = chunk.iter().try_fold(0u64, |sum, &i| {
                sum.checked_add(segments[i].live)
                    .filter(|&total| total <= max_segment_vectors)
            }) else {
                break;
            };
            for &i in chunk {
                used[i] = true;
            }
            groups.push(Group {
                indices: chunk.to_vec(),
                live,
                reason: MergeReason::Tier,
            });
        }
    }

    // A tier merge already purges deletions and builds the required graph. Only
    // schedule singleton rewrites for segments not covered by those merges.
    for (i, segment) in segments.iter().enumerate() {
        if used[i] || segment.vectors == 0 || segment.live > max_segment_vectors {
            continue;
        }
        let reason = if segment.live < segment.vectors
            && ((segment.vectors - segment.live) as f64 / segment.vectors as f64) as f32
                >= policy.max_deleted_fraction
        {
            Some(MergeReason::Deletes)
        } else if !segment.has_graph && wants_graph(ann, segment.live) {
            Some(MergeReason::Graph)
        } else {
            None
        };
        if let Some(reason) = reason {
            groups.push(Group {
                indices: vec![i],
                live: segment.live,
                reason,
            });
        }
    }
    finish_plans(groups, segments, ann)
}

/// Merges that bring the index to a compact state: every deletion purged, every output
/// at most `max_segment_vectors` live records, no two remaining segments that could
/// merge within that limit, and a graph on every segment large enough for one.
/// Deterministic; empty when already compact. Segments without records
/// (`vectors == 0`) are ignored; a plan whose inputs hold no live record means "drop
/// them" (the writer writes no empty segment).
pub fn plan_compaction(
    segments: &[SegmentSummary],
    ann: &AnnConfig,
    max_segment_vectors: u64,
) -> Vec<MergePlan> {
    let mut indices = Vec::new();
    let mut dead = Vec::new();
    for (i, segment) in segments.iter().enumerate() {
        if segment.vectors == 0 {
            continue;
        }
        if segment.live == 0 {
            dead.push(i);
        } else if segment.live <= max_segment_vectors {
            indices.push(i);
        }
    }
    indices.sort_unstable_by_key(|&i| (std::cmp::Reverse(segments[i].live), i));

    // First-fit decreasing. When a new bin is opened, its first segment does not
    // fit in any earlier bin. Bins only grow, so no pair of final bins can fit
    // together. Thus even one round reaches the required fixed point.
    let mut groups: Vec<Group> = Vec::new();
    for i in indices {
        let live = segments[i].live;
        if let Some(group) = groups
            .iter_mut()
            .find(|group| live <= max_segment_vectors - group.live)
        {
            group.indices.push(i);
            group.live += live;
        } else {
            groups.push(Group {
                indices: vec![i],
                live,
                reason: MergeReason::Compact,
            });
        }
    }
    groups.retain(|group| {
        if group.indices.len() > 1 {
            return true;
        }
        let segment = &segments[group.indices[0]];
        segment.live < segment.vectors || (!segment.has_graph && wants_graph(ann, segment.live))
    });
    if !dead.is_empty() {
        groups.push(Group {
            indices: dead,
            live: 0,
            reason: MergeReason::Compact,
        });
    }
    finish_plans(groups, segments, ann)
}

/// The most records one segment build may hold within `budget` bytes for this
/// configuration (REQUIREMENTS.md, "Memory"). `Budget` when fewer than 1,024 fit.
pub fn max_segment_vectors(config: &IndexConfig, budget: u64) -> Result<u64> {
    config.validate()?;
    let minimum = build_allowance(config, 1024);
    if minimum > budget {
        return Err(Error::Budget {
            budget,
            detail: format!("at least {minimum} bytes are needed to build 1,024 records"),
        });
    }

    // Segment ordinals are u32. Searching a bounded, monotone estimate also avoids
    // rounding errors around the training-sample cap and arithmetic overflow.
    let mut low = 1024;
    let mut high = u64::from(u32::MAX);
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if build_allowance(config, mid) <= budget {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    Ok(low)
}

fn wants_graph(ann: &AnnConfig, live: u64) -> bool {
    matches!(ann.kind, AnnKind::Vamana(_)) && live >= u64::from(ann.graph_min_vectors)
}

struct Group {
    indices: Vec<usize>,
    live: u64,
    reason: MergeReason,
}

fn finish_plans(
    mut groups: Vec<Group>,
    segments: &[SegmentSummary],
    ann: &AnnConfig,
) -> Vec<MergePlan> {
    for group in &mut groups {
        group.indices.sort_unstable();
    }
    groups.sort_unstable_by_key(|group| group.indices[0]);
    groups
        .into_iter()
        .map(|group| MergePlan {
            inputs: group.indices.into_iter().map(|i| segments[i].id).collect(),
            graph: wants_graph(ann, group.live),
            reason: group.reason,
        })
        .collect()
}

fn build_allowance(config: &IndexConfig, vectors: u64) -> u64 {
    // Use U11's actual workspace estimate rather than maintaining another graph
    // scratch formula here. Reserve graph workspace whenever ANN is enabled, even
    // below its threshold: the size cap is safe for either a seal or a graph build.
    let graph = matches!(config.ann.kind, AnnKind::Vamana(_));
    let workspace = build_memory_required(config, vectors, graph);
    let dim = u64::from(config.dim);
    let code_len = match config.codes {
        CodeKind::F32 => 4 * dim,
        CodeKind::Int8 => dim,
        CodeKind::Binary => dim.div_ceil(64) * 8 + 8,
    };
    // Flat writes stream their codes, but the resulting segment must be openable
    // too. LiveRecords' compact ordinal map coexists with the build workspace;
    // its capacity is exact and it never holds full-precision vectors or strings.
    workspace
        .max(vectors.saturating_mul(code_len))
        .saturating_add(vectors.saturating_mul(std::mem::size_of::<(u32, u32)>() as u64))
}

/// The live records of several segments, as one [`RecordSource`] (input order, then
/// ordinal order).
pub struct LiveRecords<'a> {
    inputs: Vec<(&'a Segment, &'a Tombstones)>,
    index: Vec<(u32, u32)>,
}

impl<'a> LiveRecords<'a> {
    /// Borrows the inputs and indexes their live ordinals without copying records.
    /// Panics if a tombstone set has a different length from its segment.
    pub fn new(inputs: Vec<(&'a Segment, &'a Tombstones)>) -> LiveRecords<'a> {
        let mut len = 0usize;
        for &(segment, tombstones) in &inputs {
            assert_eq!(
                segment.len(),
                tombstones.ordinals(),
                "tombstone size mismatch"
            );
            len = len
                .checked_add(tombstones.live() as usize)
                .expect("live record count exceeds address space");
        }
        let mut index = Vec::with_capacity(len);
        for (i, &(segment, tombstones)) in inputs.iter().enumerate() {
            let input = u32::try_from(i).expect("too many input segments");
            for ordinal in 0..segment.len() {
                if !tombstones.is_deleted(ordinal) {
                    index.push((input, ordinal));
                }
            }
        }
        LiveRecords { inputs, index }
    }
}

impl RecordSource for LiveRecords<'_> {
    fn len(&self) -> usize {
        self.index.len()
    }

    fn key(&self, i: usize) -> &str {
        let (input, ordinal) = self.index[i];
        self.inputs[input as usize].0.key(ordinal)
    }

    fn path(&self, i: usize) -> &str {
        let (input, ordinal) = self.index[i];
        self.inputs[input as usize].0.path(ordinal)
    }

    fn language(&self, i: usize) -> &str {
        let (input, ordinal) = self.index[i];
        self.inputs[input as usize].0.language(ordinal)
    }

    fn read_vector(&self, i: usize, out: &mut [f32]) -> Result<()> {
        let &(input, ordinal) = self
            .index
            .get(i)
            .ok_or_else(|| Error::InvalidArgument("live record index out of range".into()))?;
        self.inputs[input as usize].0.read_vector(ordinal, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Metric;

    fn summary(id: u64, vectors: u64, live: u64, has_graph: bool) -> SegmentSummary {
        SegmentSummary {
            id,
            vectors,
            live,
            has_graph,
        }
    }

    #[test]
    fn tier_merges_take_precedence_and_preserve_age_not_id_order() {
        let ann = AnnConfig {
            graph_min_vectors: 8,
            ..AnnConfig::default()
        };
        let policy = MergePolicy {
            segments_per_tier: 2,
            ..MergePolicy::default()
        };
        let segments = [
            summary(90, 8, 8, false),
            summary(3, 15, 15, true),
            summary(77, 12, 8, false),
            summary(44, 15, 15, true),
            summary(9, 16, 16, true),
            summary(11, 8, 6, false),
            summary(1, 0, 0, false),
        ];
        assert_eq!(
            plan(&segments, &policy, &ann, 20),
            vec![
                MergePlan {
                    inputs: vec![90, 77],
                    graph: true,
                    reason: MergeReason::Tier,
                },
                MergePlan {
                    inputs: vec![11],
                    graph: false,
                    reason: MergeReason::Deletes,
                },
            ]
        );
    }

    #[test]
    fn deletion_boundaries_and_drop_plans() {
        let policy = MergePolicy {
            max_deleted_fraction: 0.1,
            ..MergePolicy::default()
        };
        let ann = AnnConfig::default();
        let segments = [summary(5, 100, 90, true), summary(2, 100, 91, true)];
        let plans = plan(&segments, &policy, &ann, 100);
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].inputs, [5]);
        assert_eq!(plans[0].reason, MergeReason::Deletes);
        let dead = [summary(4, 0, 0, false), summary(1, 100, 0, true)];
        for plans in [
            plan(&dead, &policy, &ann, 0),
            plan_compaction(&dead, &ann, 0),
        ] {
            assert_eq!(plans.len(), 1);
            assert_eq!(plans[0].inputs, [1]);
            assert!(!plans[0].graph);
        }
        assert!(plan_compaction(&dead[..1], &ann, 0).is_empty());
    }

    #[test]
    fn size_limits_do_not_overflow() {
        let ann = AnnConfig {
            kind: AnnKind::Flat,
            ..AnnConfig::default()
        };
        let policy = MergePolicy {
            segments_per_tier: 2,
            ..MergePolicy::default()
        };
        let segments = [
            summary(10, u64::MAX - 1, u64::MAX - 1, false),
            summary(4, u64::MAX - 1, u64::MAX - 1, false),
            summary(7, 1, 1, false),
        ];
        assert!(plan(&segments, &policy, &ann, u64::MAX).is_empty());
        assert_eq!(
            plan_compaction(&segments, &ann, u64::MAX),
            vec![MergePlan {
                inputs: vec![10, 7],
                graph: false,
                reason: MergeReason::Compact,
            }]
        );
        // Inputs larger than a changed limit cannot be split by a MergePlan;
        // neither planner may emit an oversized output for them.
        assert!(plan(&segments, &policy, &ann, 1).is_empty());
        assert!(plan_compaction(&segments, &ann, 1).is_empty());
    }

    #[test]
    fn exhaustive_small_compactions_finish_in_one_round() {
        for kind in [AnnKind::Flat, AnnConfig::default().kind] {
            let ann = AnnConfig {
                kind,
                graph_min_vectors: 4,
                ..AnnConfig::default()
            };
            for mut layout in 0..7usize.pow(4) {
                let mut segments = Vec::new();
                for i in 0..4 {
                    let (vectors, live, graph) = match layout % 7 {
                        0 => (0, 0, false),
                        1 => (10, 0, true),
                        2 => (4, 1, false),
                        3 => (2, 2, false),
                        4 => (3, 3, true),
                        5 => (5, 5, false),
                        _ => (6, 6, true),
                    };
                    segments.push(summary(91 - i * 7, vectors, live, graph));
                    layout /= 7;
                }
                let plans = plan_compaction(&segments, &ann, 7);
                assert_eq!(plans, plan_compaction(&segments, &ann, 7));
                let mut used = [false; 4];
                let mut outputs = Vec::new();
                for (n, p) in plans.iter().enumerate() {
                    assert_eq!(p.reason, MergeReason::Compact);
                    let mut live = 0;
                    let mut previous = None;
                    for id in &p.inputs {
                        let i = segments.iter().position(|s| s.id == *id).unwrap();
                        assert!(!used[i]);
                        assert!(previous.is_none_or(|last| last < i));
                        used[i] = true;
                        previous = Some(i);
                        live += segments[i].live;
                    }
                    assert!(live <= 7);
                    assert_eq!(p.graph, wants_graph(&ann, live));
                    if live > 0 {
                        outputs.push(summary(100 + n as u64, live, live, p.graph));
                    }
                }
                outputs.extend(
                    segments
                        .into_iter()
                        .enumerate()
                        .filter_map(|(i, s)| (!used[i]).then_some(s)),
                );
                assert!(plan_compaction(&outputs, &ann, 7).is_empty());
                for (i, s) in outputs.iter().enumerate() {
                    assert_eq!(s.vectors, s.live);
                    assert!(!wants_graph(&ann, s.live) || s.has_graph);
                    for t in &outputs[i + 1..] {
                        assert!(s.live == 0 || t.live == 0 || s.live + t.live > 7);
                    }
                }
            }
        }
    }

    #[test]
    fn memory_limits_match_the_allowance_and_cover_segment_builds() {
        for codes in [CodeKind::F32, CodeKind::Int8, CodeKind::Binary] {
            for dim in [8, 1024, 4096] {
                for kind in [AnnKind::Flat, AnnConfig::default().kind] {
                    let mut config = IndexConfig::new(dim, Metric::Dot).with_codes(codes);
                    config.ann.kind = kind;
                    let minimum = build_allowance(&config, 1024);
                    assert!(matches!(
                        max_segment_vectors(&config, minimum - 1),
                        Err(Error::Budget { .. })
                    ));
                    assert_eq!(max_segment_vectors(&config, minimum).unwrap(), 1024);
                    for budget in [minimum, 1 << 30, u64::MAX] {
                        let max = max_segment_vectors(&config, budget).unwrap();
                        assert!(build_allowance(&config, max) <= budget);
                        assert!(max <= u64::from(u32::MAX));
                        if max < u64::from(u32::MAX) {
                            assert!(build_allowance(&config, max + 1) > budget);
                        }
                        assert!(
                            build_memory_required(&config, max, wants_graph(&config.ann, max))
                                <= budget
                        );
                    }
                }
            }
        }
        assert!(matches!(
            max_segment_vectors(&IndexConfig::new(0, Metric::Dot), u64::MAX),
            Err(Error::InvalidConfig(_))
        ));
    }

    #[test]
    fn empty_live_source_and_invalid_read() {
        let records = LiveRecords::new(Vec::new());
        assert!(records.is_empty());
        assert!(matches!(
            records.read_vector(0, &mut []),
            Err(Error::InvalidArgument(_))
        ));
    }
}
