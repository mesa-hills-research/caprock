//! Beam search over a Vamana graph (unit U15).
//!
//! Greedy best-first search from the graph's entry with a candidate list of `beam` nodes,
//! scoring nodes with the codec's estimates. Nodes that `admit` rejects (deleted, or
//! outside a filter) are still expanded, so they keep connecting the graph, but are never
//! returned. Deterministic: ties in estimates are broken by ordinal.

use std::cmp::Ordering;

use crate::quant::Codec;
use crate::vamana::Graph;

// Less means better. Numeric comparison deliberately treats signed zeros as equal,
// without modifying the score bits returned by the codec.
fn estimate_order(a: &(f32, u32), b: &(f32, u32)) -> Ordering {
    b.0.partial_cmp(&a.0)
        .expect("graph estimates must not be NaN")
        .then_with(|| a.1.cmp(&b.1))
}

/// A bounded best-first frontier, shared by construction and querying.
///
/// Settled nodes belong to the progress pool, not the pending pool: letting them
/// keep pending slots can stop at a local cluster before expanding a worse-scoring
/// bridge. Conversely, simply popping settled nodes would scan the whole graph.
/// Stop after `beam` consecutive expansions fail to improve the best `beam`
/// estimates seen. Both pools stay bounded by the requested beam; admission never
/// affects progress or routing. The caller offers each ordinal at most once.
#[derive(Debug, Default)]
pub(super) struct Frontier {
    pending: Vec<(f32, u32)>,
    best: Vec<(f32, u32)>,
    beam: usize,
    stalled: usize,
}

impl Frontier {
    pub(super) fn reset(&mut self, beam: usize) {
        debug_assert!(beam > 0);
        self.pending.clear();
        self.best.clear();
        self.beam = beam;
        self.stalled = 0;
    }

    pub(super) fn offer(&mut self, estimate: (f32, u32)) {
        if offer_bounded(&mut self.best, estimate, self.beam) {
            self.stalled = 0;
        }
        offer_bounded(&mut self.pending, estimate, self.beam);
    }

    pub(super) fn pop(&mut self) -> Option<u32> {
        if self.pending.is_empty() || self.stalled >= self.beam {
            return None;
        }
        self.stalled += 1;
        Some(self.pending.remove(0).1)
    }
}

// Scores, including signed-zero ties, use the same order in all three pools.
fn offer_bounded(pool: &mut Vec<(f32, u32)>, estimate: (f32, u32), limit: usize) -> bool {
    if pool.len() == limit && estimate_order(&estimate, &pool[limit - 1]) != Ordering::Less {
        return false;
    }
    let position = pool.partition_point(|c| estimate_order(c, &estimate) == Ordering::Less);
    if pool.len() == limit {
        pool.pop();
    }
    pool.insert(position, estimate);
    true
}

/// Reusable per-thread buffers (visited set, candidate lists).
#[derive(Debug, Default)]
pub struct SearchScratch {
    visited: Vec<u32>,
    epoch: u32,
    frontier: Frontier,
    admitted: Vec<(f32, u32)>,
}

impl SearchScratch {
    pub fn new() -> SearchScratch {
        SearchScratch::default()
    }

    fn begin(&mut self, nodes: usize) {
        self.admitted.clear();
        // Keep the entire allocation when switching to a smaller graph. On wrap we
        // must clear the tail too, since a later query may use a larger graph again.
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.visited.fill(0);
            self.epoch = 1;
        }
        if self.visited.len() < nodes {
            self.visited.resize(nodes, 0);
        }
    }

    fn visit(&mut self, node: u32) -> bool {
        let stamp = &mut self.visited[node as usize];
        if *stamp == self.epoch {
            return false;
        }
        *stamp = self.epoch;
        true
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
    let nodes = graph.len();
    if wanted == 0 || nodes == 0 {
        return Vec::new();
    }
    let beam = beam.max(wanted).min(nodes);
    let wanted = wanted.min(nodes);
    scratch.begin(nodes);
    scratch.frontier.reset(beam);

    let code_len = codec.code_len();
    let estimate = |node: u32| {
        let start = node as usize * code_len;
        (codec.score(query, &codes[start..start + code_len]), node)
    };
    scratch.visit(graph.entry);
    let entry = estimate(graph.entry);
    scratch.frontier.offer(entry);
    if admit(graph.entry) {
        offer_bounded(&mut scratch.admitted, entry, wanted);
    }

    while let Some(node) = scratch.frontier.pop() {
        for &neighbor in graph.neighbors(node) {
            if !scratch.visit(neighbor) {
                continue;
            }
            let scored = estimate(neighbor);
            // Include all scored, admitted nodes, even outside the pending beam.
            if admit(neighbor) {
                offer_bounded(&mut scratch.admitted, scored, wanted);
            }
            scratch.frontier.offer(scored);
        }
    }
    scratch.admitted.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Metric;
    use crate::quant::F32Codec;
    use crate::vamana::NO_NEIGHBOR;

    fn codec_and_codes(scores: &[f32]) -> (F32Codec, Vec<u8>) {
        let codec = F32Codec {
            metric: Metric::Dot,
            dim: 1,
        };
        let codes = scores.iter().flat_map(|s| s.to_le_bytes()).collect();
        (codec, codes)
    }

    #[test]
    fn expansion_is_best_first_bounded_and_visits_each_node_once() {
        // A beam of one expands 2 before 1, then follows 2 to 3. It must not
        // exhaustively explore the pruned route through 1 to the best node, 4.
        // The edge back to 0 checks that an already visited node is not rescored.
        let graph = Graph {
            degree: 2,
            entry: 0,
            adjacency: [
                [1, 2],
                [4, NO_NEIGHBOR],
                [0, 3],
                [NO_NEIGHBOR; 2],
                [NO_NEIGHBOR; 2],
            ]
            .concat(),
        };
        let (codec, codes) = codec_and_codes(&[-10.0, 1.0, 2.0, 10.0, 20.0]);
        let visits = std::cell::RefCell::new(Vec::new());
        let found = beam_search(
            &graph,
            &codec,
            &codes,
            &vec![1.0],
            1,
            1,
            &|node| {
                visits.borrow_mut().push(node);
                true
            },
            &mut SearchScratch::new(),
        );
        assert_eq!(found, vec![(10.0, 3)]);
        assert_eq!(*visits.borrow(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn settled_nodes_do_not_crowd_a_bridge_out_of_the_frontier() {
        // After expanding 1 and 3 the two best scores are 11 and 10. Node 4 is
        // worse than both, but is the best pending route to the other lobe (5).
        let graph = Graph {
            degree: 2,
            entry: 0,
            adjacency: [
                [1, 2],
                [3, NO_NEIGHBOR],
                [NO_NEIGHBOR; 2],
                [4, NO_NEIGHBOR],
                [5, NO_NEIGHBOR],
                [NO_NEIGHBOR; 2],
            ]
            .concat(),
        };
        let (codec, codes) = codec_and_codes(&[0.0, 10.0, 8.0, 11.0, 9.0, 20.0]);
        let mut previous = None;
        for filtered in [false, true] {
            let visits = std::cell::RefCell::new(Vec::new());
            let found = beam_search(
                &graph,
                &codec,
                &codes,
                &vec![1.0],
                2,
                1,
                &|node| {
                    visits.borrow_mut().push(node);
                    !filtered || node == 5
                },
                &mut SearchScratch::new(),
            );
            assert_eq!(found, [(20.0, 5)]);
            // Filtering even the bridging node cannot change the traversal.
            if let Some(previous) = &previous {
                assert_eq!(&*visits.borrow(), previous);
            }
            previous = Some(visits.into_inner());
        }
    }

    #[test]
    fn a_non_improving_frontier_stops_instead_of_scanning_the_graph() {
        let nodes = 64;
        let graph = Graph {
            degree: 1,
            entry: 0,
            adjacency: (1..nodes).chain([NO_NEIGHBOR]).collect(),
        };
        let scores: Vec<_> = (0..nodes).map(|i| (nodes - i) as f32).collect();
        let (codec, codes) = codec_and_codes(&scores);
        for beam in [1, 2, 4, 8] {
            let visits = std::cell::RefCell::new(Vec::new());
            let found = beam_search(
                &graph,
                &codec,
                &codes,
                &vec![1.0],
                beam,
                1,
                &|node| {
                    visits.borrow_mut().push(node);
                    true
                },
                &mut SearchScratch::new(),
            );
            assert_eq!(found, [(64.0, 0)]);
            // Fill the progress pool, then allow only `beam` unsuccessful
            // expansions. Merely discovering a new node is not progress.
            assert_eq!(*visits.borrow(), (0..2 * beam as u32).collect::<Vec<_>>());
        }
    }

    #[test]
    fn frontier_pools_are_bounded_and_reset_independently_of_scores() {
        let mut frontier = Frontier::default();
        for beam in [1, 3, 8, 2] {
            frontier.reset(beam);
            assert!(frontier.pop().is_none());
            for node in 0..100 {
                frontier.offer((node as f32, node));
                assert!(frontier.pending.len() <= beam);
                assert!(frontier.best.len() <= beam);
            }
            assert_eq!(frontier.pop(), Some(99));
            assert_eq!(frontier.best[0], (99.0, 99));
            while frontier.pop().is_some() {}
            assert!(frontier.pop().is_none());
        }
    }

    #[test]
    fn rejected_nodes_still_route_and_results_include_nodes_outside_the_beam() {
        // Node 2 is only reachable through the rejected entry and node 1. Node 3
        // has a worse estimate than every routing node, but is an admitted result.
        let graph = Graph {
            degree: 2,
            entry: 0,
            adjacency: vec![
                1,
                NO_NEIGHBOR,
                2,
                3,
                NO_NEIGHBOR,
                NO_NEIGHBOR,
                NO_NEIGHBOR,
                NO_NEIGHBOR,
            ],
        };
        let (codec, codes) = codec_and_codes(&[0.0, 1.0, 2.0, -1.0]);
        assert_eq!(
            beam_search(
                &graph,
                &codec,
                &codes,
                &vec![1.0],
                1,
                2,
                &|node| node >= 2,
                &mut SearchScratch::new(),
            ),
            vec![(2.0, 2), (-1.0, 3)],
        );
    }

    #[test]
    fn tied_estimates_use_ordinals_not_discovery_order() {
        let graph = Graph {
            degree: 3,
            entry: 3,
            adjacency: vec![
                NO_NEIGHBOR,
                NO_NEIGHBOR,
                NO_NEIGHBOR,
                NO_NEIGHBOR,
                NO_NEIGHBOR,
                NO_NEIGHBOR,
                NO_NEIGHBOR,
                NO_NEIGHBOR,
                NO_NEIGHBOR,
                2,
                1,
                0,
            ],
        };
        let (codec, codes) = codec_and_codes(&[1.0; 4]);
        assert_eq!(
            beam_search(
                &graph,
                &codec,
                &codes,
                &vec![1.0],
                0,
                2,
                &|_| true,
                &mut SearchScratch::new(),
            ),
            vec![(1.0, 0), (1.0, 1)],
        );
        assert_eq!(estimate_order(&(-0.0, 0), &(0.0, 1)), Ordering::Less);
        assert_eq!(estimate_order(&(0.0, 0), &(-0.0, 0)), Ordering::Equal);
    }

    #[test]
    fn epochs_wrap_and_survive_switching_graph_sizes() {
        let mut scratch = SearchScratch::new();
        scratch.begin(8);
        assert!(scratch.visit(7));
        assert!(!scratch.visit(7));
        scratch.epoch = u32::MAX;
        scratch.begin(1);
        assert_eq!(scratch.epoch, 1);
        // Even the part beyond the smaller graph was cleared on rollover.
        assert!(scratch.visit(7));
        scratch.begin(16);
        assert!(scratch.visit(7));
        assert!(scratch.visit(15));
        scratch.begin(1);
        assert!(scratch.visit(0));
    }

    #[test]
    fn empty_searches_and_disconnected_graphs() {
        let (codec, codes) = codec_and_codes(&[1.0, 2.0]);
        let mut scratch = SearchScratch::new();
        let graph = Graph {
            degree: 1,
            entry: 1,
            adjacency: vec![NO_NEIGHBOR; 2],
        };
        for graph in [&Graph::default(), &graph] {
            assert!(
                beam_search(
                    graph,
                    &codec,
                    &codes,
                    &vec![1.0],
                    8,
                    0,
                    &|_| panic!("empty searches do not need admission"),
                    &mut scratch,
                )
                .is_empty()
            );
        }
        assert!(
            beam_search(
                &Graph::default(),
                &codec,
                &codes,
                &vec![1.0],
                8,
                1,
                &|_| true,
                &mut scratch,
            )
            .is_empty()
        );
        assert_eq!(
            beam_search(
                &graph,
                &codec,
                &codes,
                &vec![1.0],
                usize::MAX,
                usize::MAX,
                &|_| true,
                &mut scratch,
            ),
            vec![(2.0, 1)],
        );
    }
}
