//! Flat search and rescoring within one segment (unit U13; REQUIREMENTS.md, "Search").

use crate::error::Result;
use crate::filter::SegmentFilter;
use crate::segment::Segment;
use crate::tombstones::Tombstones;
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
    todo!("U13 flat: flat_search")
}

/// The exact top `k` over full-precision vectors of every live record the filter
/// allows: canonical scores, result order. Equal to the oracle restricted to this
/// segment.
pub fn exact_search(segment: &Segment, q: &SegmentQuery<'_>) -> Result<Vec<Hit>> {
    todo!("U13 flat: exact_search")
}

/// The top `k` of `candidates` (estimate, ordinal) after rescoring with canonical scores
/// from full-precision vectors, in result order.
pub fn rescore(segment: &Segment, query: &[f32], candidates: &[(f32, u32)], k: usize) -> Result<Vec<Hit>> {
    todo!("U13 flat: rescore")
}

/// The [`Hit`] for an ordinal with a score.
pub fn hit(segment: &Segment, ordinal: u32, score: f32) -> Hit {
    todo!("U13 flat: hit")
}
