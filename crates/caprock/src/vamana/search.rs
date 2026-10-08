//! Beam search over a Vamana graph.
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
/// estimates seen. Both pools stay bounded by the requested beam. The caller
/// chooses which nodes belong to this frontier and offers each ordinal at most once.
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

// Scores, including signed-zero ties, use the same order in every pool.
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
    matching_frontier: Frontier,
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
    search(
        graph, codec, codes, query, beam, wanted, admit, scratch, false,
    )
}

/// Searches with independent bounded frontiers for matching results and rejected
/// routing nodes. Unlike an unfiltered walk followed by admission, matching nodes
/// cannot be crowded out by better-scoring nodes outside the filter. Rejected
/// nodes still route, but progress toward their optimum cannot terminate progress
/// toward the filtered optimum (or vice versa). Each frontier has `beam` slots
/// (raised to at least `wanted`); results use the same ordering as [`beam_search`].
#[allow(clippy::too_many_arguments)]
pub fn filtered_beam_search<C: Codec>(
    graph: &Graph,
    codec: &C,
    codes: &[u8],
    query: &C::Query,
    beam: usize,
    wanted: usize,
    admit: &dyn Fn(u32) -> bool,
    scratch: &mut SearchScratch,
) -> Vec<(f32, u32)> {
    search(
        graph, codec, codes, query, beam, wanted, admit, scratch, true,
    )
}

#[allow(clippy::too_many_arguments)]
fn search<C: Codec>(
    graph: &Graph,
    codec: &C,
    codes: &[u8],
    query: &C::Query,
    beam: usize,
    wanted: usize,
    admit: &dyn Fn(u32) -> bool,
    scratch: &mut SearchScratch,
    filtered: bool,
) -> Vec<(f32, u32)> {
    let nodes = graph.len();
    if wanted == 0 || nodes == 0 {
        return Vec::new();
    }
    let beam = beam.max(wanted).min(nodes);
    let wanted = wanted.min(nodes);
    scratch.begin(nodes);
    scratch.frontier.reset(beam);
    scratch.matching_frontier.reset(beam);

    let code_len = codec.code_len();
    let estimate = |node: u32| {
        let start = node as usize * code_len;
        (codec.score(query, &codes[start..start + code_len]), node)
    };
    scratch.visit(graph.entry);
    let entry = estimate(graph.entry);
    let offer = |scratch: &mut SearchScratch, scored: (f32, u32)| {
        let allowed = admit(scored.1);
        if allowed {
            offer_bounded(&mut scratch.admitted, scored, wanted);
        }
        if filtered && allowed {
            scratch.matching_frontier.offer(scored);
        } else {
            scratch.frontier.offer(scored);
        }
    };
    offer(scratch, entry);

    // First find routes from the entry, then refine the matching neighborhood.
    // Each node goes into exactly one frontier and is scored/expanded at most
    // once. Either frontier can discover improvements that restart the other.
    while let Some(node) = scratch
        .frontier
        .pop()
        .or_else(|| scratch.matching_frontier.pop())
    {
        for &neighbor in graph.neighbors(node) {
            if !scratch.visit(neighbor) {
                continue;
            }
            let scored = estimate(neighbor);
            // Include all scored, admitted nodes, even outside the pending beam.
            offer(scratch, scored);
        }
    }
    scratch.admitted.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AnnKind, IndexConfig, Metric};
    use crate::quant::{BinaryCodec, F32Codec, Int8Codec};
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
    fn filtered_progress_is_not_terminated_by_a_better_unfiltered_cluster() {
        // The unfiltered optimum (0, 1) stalls the routing beam before it can
        // expand the matching lobe (2 -> 6 -> 8). There are already `wanted`
        // matches (2, 3), so the short-list fallback cannot fix this miss.
        // The matching lobe in turn discovers a new rejected bridge (9 -> 10).
        let graph = Graph {
            degree: 3,
            entry: 0,
            adjacency: [
                [1, 2, 3],
                [4, NO_NEIGHBOR, NO_NEIGHBOR],
                [6, NO_NEIGHBOR, NO_NEIGHBOR],
                [7, NO_NEIGHBOR, NO_NEIGHBOR],
                [5, NO_NEIGHBOR, NO_NEIGHBOR],
                [NO_NEIGHBOR; 3],
                [8, NO_NEIGHBOR, NO_NEIGHBOR],
                [NO_NEIGHBOR; 3],
                [9, NO_NEIGHBOR, NO_NEIGHBOR],
                [10, NO_NEIGHBOR, NO_NEIGHBOR],
                [0, 8, NO_NEIGHBOR],
            ]
            .concat(),
        };
        let (codec, codes) = codec_and_codes(&[
            100.0, 99.0, 10.0, 9.0, 98.0, 97.0, 20.0, 11.0, 30.0, 101.0, 40.0,
        ]);
        let query = codec.prepare(&[1.0]);
        let admit = |node| matches!(node, 2 | 3 | 6 | 7 | 8 | 10);
        let mut scratch = SearchScratch::new();
        for _ in 0..3 {
            assert_eq!(
                beam_search(&graph, &codec, &codes, &query, 2, 2, &admit, &mut scratch),
                [(10.0, 2), (9.0, 3)]
            );
            let visits = std::cell::RefCell::new(vec![0; graph.len()]);
            assert_eq!(
                filtered_beam_search(
                    &graph,
                    &codec,
                    &codes,
                    &query,
                    2,
                    2,
                    &|node| {
                        visits.borrow_mut()[node as usize] += 1;
                        admit(node)
                    },
                    &mut scratch,
                ),
                [(40.0, 10), (30.0, 8)]
            );
            // Cyclic routes and switching between frontier modes must not cause
            // rescoring, re-expansion, or leftover candidates from the last query.
            assert!(visits.borrow().iter().all(|&count| count == 1));
            for frontier in [&scratch.frontier, &scratch.matching_frontier] {
                assert!(frontier.pending.len() <= 2);
                assert!(frontier.best.len() <= 2);
            }
        }
    }

    #[test]
    fn default_int8_filtered_search_refines_an_already_full_shortlist() {
        let k = 10;
        let config = IndexConfig::new(2, Metric::Cosine);
        let AnnKind::Vamana(params) = config.ann.kind else {
            panic!("default int8 configuration must use Vamana");
        };
        let wanted = (k * config.rescore_factor as usize).max(config.rescore_min as usize);
        let beam = (params.search_beam as usize).max(wanted);
        let rejected = 4 * beam;
        let bridge = rejected + wanted;
        let best = bridge + 1..bridge + 1 + k;
        let matching = (config.ann.filter_flat_below as usize + 1).max(wanted + 1 + k + 4 * beam);
        let nodes = rejected + matching;

        // A globally better rejected cluster supplies a full rescore shortlist
        // of weak matches. Only expanding those matches reveals the bridge to
        // the filtered optimum. A full-shortlist fallback cannot detect this
        // failure, and neither the beam nor the rescore budget is overridden.
        let vectors: Vec<f32> = (0..nodes)
            .flat_map(|node| {
                let x: f32 = if node < rejected {
                    1.0
                } else if node < bridge {
                    0.4
                } else if node == bridge {
                    0.2
                } else if best.contains(&node) {
                    0.8
                } else {
                    0.0
                };
                [x, (1.0 - x * x).sqrt()]
            })
            .collect();
        let codec = Int8Codec::train(Metric::Cosine, 2, &vectors, 0).unwrap();
        let width = codec.code_len();
        let mut codes = vec![0; nodes * width];
        for (vector, code) in vectors
            .as_chunks::<2>()
            .0
            .iter()
            .zip(codes.chunks_exact_mut(width))
        {
            codec.encode(vector, code);
        }
        let query = codec.prepare(&[1.0, 0.0]);
        let mut rows = vec![[NO_NEIGHBOR; 2]; nodes];
        for (node, row) in rows.iter_mut().enumerate() {
            if node < rejected || (node >= bridge && node + 1 < nodes) {
                row[0] = node as u32 + 1;
            }
            if node < wanted {
                row[1] = (rejected + node) as u32;
            }
        }
        rows[rejected][0] = bridge as u32;
        let graph = Graph {
            degree: 2,
            entry: 0,
            adjacency: rows.concat(),
        };
        let admit = |node: u32| node as usize >= rejected;
        let mut flat: Vec<_> = codes
            .chunks_exact(width)
            .enumerate()
            .filter(|&(node, _)| admit(node as u32))
            .map(|(node, code)| (codec.score(&query, code), node as u32))
            .collect();
        flat.sort_unstable_by(estimate_order);
        assert_eq!(
            flat[..k]
                .iter()
                .map(|&(_, node)| node as usize)
                .collect::<Vec<_>>(),
            best.collect::<Vec<_>>()
        );

        let mut scratch = SearchScratch::new();
        for _ in 0..2 {
            let unfiltered = beam_search(
                &graph,
                &codec,
                &codes,
                &query,
                beam,
                wanted,
                &admit,
                &mut scratch,
            );
            assert_eq!(unfiltered.len(), wanted);
            assert!(unfiltered.iter().all(|&(_, node)| (node as usize) < bridge));
            let visits = std::cell::RefCell::new(vec![0; nodes]);
            let found = filtered_beam_search(
                &graph,
                &codec,
                &codes,
                &query,
                beam,
                wanted,
                &|node| {
                    visits.borrow_mut()[node as usize] += 1;
                    admit(node)
                },
                &mut scratch,
            );
            assert_eq!(found, flat[..wanted]);
            let visits = visits.borrow();
            assert!(visits.iter().all(|&count| count <= 1));
            // Separate progress must not degenerate into exhaustive search of
            // either the rejected routing cluster or the matching population.
            assert!(visits[..rejected].contains(&0));
            assert!(visits[rejected..].contains(&0));
            for frontier in [&scratch.frontier, &scratch.matching_frontier] {
                assert!(frontier.pending.len() <= beam);
                assert!(frontier.best.len() <= beam);
            }
        }
    }

    #[test]
    fn filtered_frontiers_remain_bounded_and_handle_uniform_admission() {
        let nodes = 128u32;
        let graph = Graph {
            degree: 1,
            entry: 0,
            adjacency: (1..nodes).chain([NO_NEIGHBOR]).collect(),
        };
        let scores: Vec<_> = (0..nodes).map(|i| (nodes - i) as f32).collect();
        let (codec, codes) = codec_and_codes(&scores);
        let query = codec.prepare(&[1.0]);
        let mut scratch = SearchScratch::new();
        for beam in [1, 2, 4, 8] {
            let visits = std::cell::RefCell::new(Vec::new());
            let found = filtered_beam_search(
                &graph,
                &codec,
                &codes,
                &query,
                beam,
                1,
                &|node| {
                    visits.borrow_mut().push(node);
                    node % 2 == 1
                },
                &mut scratch,
            );
            assert_eq!(found, [(127.0, 1)]);
            // Independent progress is not permission to exhaust the graph: both
            // frontiers must stop after their own non-improving expansions.
            assert!(visits.borrow().len() <= 4 * beam);
            for all in [true, false] {
                assert_eq!(
                    filtered_beam_search(
                        &graph,
                        &codec,
                        &codes,
                        &query,
                        beam,
                        1,
                        &|_| all,
                        &mut scratch,
                    ),
                    beam_search(
                        &graph,
                        &codec,
                        &codes,
                        &query,
                        beam,
                        1,
                        &|_| all,
                        &mut scratch,
                    )
                );
            }
        }
        assert!(
            filtered_beam_search(
                &graph,
                &codec,
                &codes,
                &query,
                1,
                0,
                &|_| panic!("zero wanted does not need admission"),
                &mut scratch,
            )
            .is_empty()
        );
        assert!(
            filtered_beam_search(
                &Graph::default(),
                &codec,
                &[],
                &query,
                usize::MAX,
                usize::MAX,
                &|_| panic!("empty graphs do not need admission"),
                &mut scratch,
            )
            .is_empty()
        );
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
    fn binary_plateau_does_not_hide_a_lower_scoring_bridge() {
        let dim = 64;
        let codec = BinaryCodec::train(Metric::Cosine, dim, &[], 0).unwrap();
        let width = codec.code_len();
        let mut codes = vec![0; 6 * width];
        for (node, negatives) in [32, 8, 12, 8, 10, 0].into_iter().enumerate() {
            let mut vector = vec![0.125; dim];
            vector[..negatives].fill(-0.125);
            // Distinct sign patterns can have identical query estimates.
            vector.rotate_left(node * 7);
            codec.encode(&vector, &mut codes[node * width..(node + 1) * width]);
        }
        let query = codec.prepare(&vec![0.125; dim]);
        let estimates: Vec<_> = codes
            .chunks_exact(width)
            .map(|code| codec.score(&query, code))
            .collect();
        assert_eq!(estimates[1], estimates[3]);
        assert!(estimates[4] < estimates[1]);
        assert!(estimates[4] > estimates[2]);
        assert!(estimates[5] > estimates[1]);

        // The tied estimates at 1 and 3 fill the best-beam pool. Keeping those
        // settled nodes in the pending pool used to discard the worse bridge 4,
        // making the best candidate 5 unreachable regardless of admission.
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
        let mut scratch = SearchScratch::new();
        for filtered in [false, true, false] {
            let visits = std::cell::RefCell::new(Vec::new());
            let found = beam_search(
                &graph,
                &codec,
                &codes,
                &query,
                2,
                2,
                &|node| {
                    visits.borrow_mut().push(node);
                    !filtered || node == 5
                },
                &mut scratch,
            );
            let mut expected = vec![(estimates[5], 5)];
            if !filtered {
                expected.push((estimates[1], 1));
            }
            assert_eq!(found, expected);
            assert_eq!(*visits.borrow(), [0, 1, 2, 3, 4, 5]);
        }
    }

    #[test]
    fn int8_k50_rescore_pool_does_not_block_routes_to_better_neighbors() {
        let k = 50;
        let config = IndexConfig::new(2, Metric::Cosine);
        let AnnKind::Vamana(params) = config.ann.kind else {
            panic!("default int8 configuration must use Vamana");
        };
        let wanted = (k * config.rescore_factor as usize).max(config.rescore_min as usize);
        assert_eq!(wanted, 200);
        let nodes = wanted + 1 + k + 3 * wanted;

        // Fill the entire rescore-sized progress pool with a local plateau.
        // Its last node leads through a worse bridge to all fifty exact nearest
        // neighbors. Retaining settled nodes in the pending pool rejects that
        // bridge even though the result list already has `wanted` candidates.
        let vectors: Vec<f32> = (0..nodes)
            .flat_map(|node| {
                let x: f32 = if node < wanted {
                    0.75
                } else if node == wanted {
                    0.5
                } else if node <= wanted + k {
                    1.0
                } else {
                    0.0
                };
                [x, (1.0 - x * x).sqrt()]
            })
            .collect();
        let codec = Int8Codec::train(Metric::Cosine, 2, &vectors, 0).unwrap();
        let width = codec.code_len();
        let mut codes = vec![0; nodes * width];
        for (vector, code) in vectors
            .as_chunks::<2>()
            .0
            .iter()
            .zip(codes.chunks_exact_mut(width))
        {
            codec.encode(vector, code);
        }
        let query = codec.prepare(&[1.0, 0.0]);
        let graph = Graph {
            degree: 2,
            entry: 0,
            adjacency: (0..nodes as u32)
                .flat_map(|node| {
                    let next = if node + 1 < nodes as u32 {
                        node + 1
                    } else {
                        NO_NEIGHBOR
                    };
                    [next, if node == 0 { NO_NEIGHBOR } else { 0 }]
                })
                .collect(),
        };
        let mut flat: Vec<_> = codes
            .chunks_exact(width)
            .enumerate()
            .map(|(node, code)| (codec.score(&query, code), node as u32))
            .collect();
        flat.sort_unstable_by(estimate_order);
        let exact_top: Vec<_> = (wanted as u32 + 1..=wanted as u32 + k as u32).collect();
        assert_eq!(
            flat[..k].iter().map(|&(_, node)| node).collect::<Vec<_>>(),
            exact_top
        );

        let mut scratch = SearchScratch::new();
        for beam in [params.search_beam as usize, 32, 64, 128, 256, 0] {
            let visits = std::cell::RefCell::new(vec![0; nodes]);
            let found = beam_search(
                &graph,
                &codec,
                &codes,
                &query,
                beam,
                wanted,
                &|node| {
                    visits.borrow_mut()[node as usize] += 1;
                    true
                },
                &mut scratch,
            );
            assert_eq!(found, flat[..wanted]);
            // Recover the entire rescore shortlist without turning this into a
            // flat scan. Cyclic edges must not cause repeated codec evaluations.
            let visits = visits.borrow();
            assert!(visits.iter().all(|&count| count <= 1));
            assert!(visits.contains(&0));
            assert!(exact_top.iter().all(|&node| visits[node as usize] == 1));
            assert!(scratch.frontier.pending.len() <= beam.max(wanted));
            assert!(scratch.frontier.best.len() <= beam.max(wanted));
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
