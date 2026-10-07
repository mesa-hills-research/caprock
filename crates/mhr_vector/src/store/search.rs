//! Searching a snapshot (unit U19; REQUIREMENTS.md, "Search").
//!
//! For each segment: compile the filter; when the search is exact, or the segment has no
//! graph, or at most `filter_flat_below` live records match, search it flat (exact search
//! for `exact`); otherwise beam-search its graph (beam `max(params.beam or
//! search_beam, k, rescore)`), admitting live matching ordinals, and rescore. If a graph
//! search finds fewer than `min(k, matching live)` records, fall back to a flat search
//! of that segment. Search the delta exactly. Merge everything into the top `k`.

use super::Snapshot;
use crate::config::SearchParams;
use crate::error::Result;
use crate::filter::Filter;
use crate::types::Hit;

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
        todo!("U19 search: Snapshot::search")
    }
}
