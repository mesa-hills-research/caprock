//! Beam search over a Vamana graph (unit U15).
//!
//! Greedy best-first search from the graph's entry with a candidate list of `beam` nodes,
//! scoring nodes with the codec's estimates. Nodes that `admit` rejects (deleted, or
//! outside a filter) are still expanded, so they keep connecting the graph, but are never
//! returned. Deterministic: ties in estimates are broken by ordinal.

use crate::quant::Codec;
use crate::vamana::Graph;

/// Reusable per-thread buffers (visited set, candidate lists).
#[derive(Debug, Default)]
pub struct SearchScratch {
    visited: Vec<u64>,
    epoch: u32,
}

impl SearchScratch {
    pub fn new() -> SearchScratch {
        SearchScratch::default()
    }
}

/// Searches `graph` for `query` with a candidate list of `beam` (at least `wanted`) and
/// returns up to `wanted` admitted nodes with the best estimates found, best first
/// (estimate descending, ordinal ascending).
#[allow(clippy::too_many_arguments)]
pub fn beam_search<C: Codec>(
    graph: &Graph,
    codec: &C,
    codes: &[u8],
    query: &C::Query,
    beam: usize,
    wanted: usize,
    admit: &dyn Fn(u32) -> bool,
    scratch: &mut SearchScratch,
) -> Vec<(f32, u32)> {
    todo!("U15 vamana-search: beam_search")
}
