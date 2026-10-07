//! Merge planning and the memory math (unit U16; REQUIREMENTS.md, "Maintenance" and
//! "Memory").
//!
//! Planning is a pure function of segment summaries, so the same segments always give
//! the same plans (determinism of layouts). Plans never share inputs.

use crate::config::{AnnConfig, IndexConfig, MergePolicy};
use crate::error::Result;
use crate::segment::{RecordSource, Segment};
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
    todo!("U16 merge: plan")
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
    todo!("U16 merge: plan_compaction")
}

/// The most records one segment build may hold within `budget` bytes for this
/// configuration (REQUIREMENTS.md, "Memory"). `Budget` when fewer than 1,024 fit.
pub fn max_segment_vectors(config: &IndexConfig, budget: u64) -> Result<u64> {
    todo!("U16 merge: max_segment_vectors")
}

/// The live records of several segments, as one [`RecordSource`] (input order, then
/// ordinal order).
pub struct LiveRecords<'a> {
    inputs: Vec<(&'a Segment, &'a Tombstones)>,
    index: Vec<(u32, u32)>,
}

impl<'a> LiveRecords<'a> {
    pub fn new(inputs: Vec<(&'a Segment, &'a Tombstones)>) -> LiveRecords<'a> {
        todo!("U16 merge: LiveRecords::new")
    }
}

impl RecordSource for LiveRecords<'_> {
    fn len(&self) -> usize {
        todo!("U16 merge: LiveRecords::len")
    }

    fn key(&self, i: usize) -> &str {
        todo!("U16 merge: LiveRecords::key")
    }

    fn path(&self, i: usize) -> &str {
        todo!("U16 merge: LiveRecords::path")
    }

    fn language(&self, i: usize) -> &str {
        todo!("U16 merge: LiveRecords::language")
    }

    fn read_vector(&self, i: usize, out: &mut [f32]) -> Result<()> {
        todo!("U16 merge: LiveRecords::read_vector")
    }
}
