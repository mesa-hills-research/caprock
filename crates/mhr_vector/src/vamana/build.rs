//! Deterministic Vamana construction (unit U15).
//!
//! Suggested method (ParlayANN's deterministic batch insertion, Manohar et al., PPoPP
//! 2024): insert points in batches of doubling size in a seeded order; within a batch,
//! search the graph as it was before the batch (in parallel), compute each point's pruned
//! out-edges, then add the reverse edges in a fixed order and prune overfull rows.
//! Any method meeting the acceptance tests and the recall targets is fine, as long as the
//! result does not depend on `threads`.

use crate::config::VamanaParams;
use crate::error::Result;
use crate::quant::Codec;
use crate::vamana::Graph;

/// Builds a graph over the `codes.len() / codec.code_len()` codes, with
/// `codec.score_codes` as the similarity. Deterministic in `(codes, params, seed)`.
/// Every row holds at most `params.max_degree` distinct neighbours, never itself.
pub fn build_graph<C: Codec>(
    codec: &C,
    codes: &[u8],
    params: &VamanaParams,
    seed: u64,
    threads: usize,
) -> Result<Graph> {
    todo!("U15 vamana-build: build_graph")
}
