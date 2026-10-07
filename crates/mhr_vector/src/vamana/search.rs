//! Beam search over a Vamana graph (unit U15).
//!
//! Greedy best-first search from the graph's entry with a candidate list of `beam` nodes,
//! scoring nodes with the codec's estimates. Nodes that `admit` rejects (deleted, or
//! outside a filter) are still expanded, so they keep connecting the graph, but are never
//! returned. Deterministic: ties in estimates are broken by ordinal.

use std::cmp::Ordering;

use crate::quant::Codec;
use crate::vamana::Graph;

#[derive(Debug)]
struct Candidate {
    estimate: (f32, u32),
    expanded: bool,
}

// Less means better. Numeric comparison deliberately treats signed zeros as equal,
// without modifying the score bits returned by the codec.
fn estimate_order(a: &(f32, u32), b: &(f32, u32)) -> Ordering {
    b.0.partial_cmp(&a.0)
        .expect("graph estimates must not be NaN")
        .then_with(|| a.1.cmp(&b.1))
}

/// Reusable per-thread buffers (visited set, candidate lists).
#[derive(Debug, Default)]
pub struct SearchScratch {
    visited: Vec<u32>,
    epoch: u32,
    candidates: Vec<Candidate>,
    admitted: Vec<(f32, u32)>,
}

impl SearchScratch {
    pub fn new() -> SearchScratch {
        SearchScratch::default()
    }

    fn begin(&mut self, nodes: usize) {
        self.candidates.clear();
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

    // Keep a sorted, bounded navigation pool, including already expanded nodes.
    // Its worst estimate is the stopping threshold; expanding a node does not free
    // a slot, which would otherwise turn a bounded beam search into a full scan.
    fn offer_candidate(&mut self, estimate: (f32, u32), beam: usize) -> Option<usize> {
        if self.candidates.len() == beam
            && estimate_order(&estimate, &self.candidates[beam - 1].estimate) != Ordering::Less
        {
            return None;
        }
        let position = self
            .candidates
            .partition_point(|c| estimate_order(&c.estimate, &estimate) == Ordering::Less);
        if self.candidates.len() == beam {
            self.candidates.pop();
        }
        self.candidates.insert(
            position,
            Candidate {
                estimate,
                expanded: false,
            },
        );
        Some(position)
    }

    // Admission affects only this result pool, never the navigation threshold.
    // Include all scored nodes, even those outside the final navigation beam.
    fn offer_admitted(&mut self, estimate: (f32, u32), wanted: usize) {
        if self.admitted.len() == wanted
            && estimate_order(&estimate, &self.admitted[wanted - 1]) != Ordering::Less
        {
            return;
        }
        let position = self
            .admitted
            .partition_point(|c| estimate_order(c, &estimate) == Ordering::Less);
        if self.admitted.len() == wanted {
            self.admitted.pop();
        }
        self.admitted.insert(position, estimate);
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

    let code_len = codec.code_len();
    let estimate = |node: u32| {
        let start = node as usize * code_len;
        (codec.score(query, &codes[start..start + code_len]), node)
    };
    scratch.visit(graph.entry);
    let entry = estimate(graph.entry);
    scratch.offer_candidate(entry, beam);
    if admit(graph.entry) {
        scratch.offer_admitted(entry, wanted);
    }

    let mut cursor = 0;
    while cursor < scratch.candidates.len() {
        if scratch.candidates[cursor].expanded {
            cursor += 1;
            continue;
        }
        let node = scratch.candidates[cursor].estimate.1;
        scratch.candidates[cursor].expanded = true;
        let mut next = cursor + 1;
        for &neighbor in graph.neighbors(node) {
            if !scratch.visit(neighbor) {
                continue;
            }
            let scored = estimate(neighbor);
            if admit(neighbor) {
                scratch.offer_admitted(scored, wanted);
            }
            if let Some(position) = scratch.offer_candidate(scored, beam) {
                // A newly discovered node may precede the current cursor. Rewind
                // to it so the next expansion is always the best unexpanded node.
                next = next.min(position);
            }
        }
        cursor = next;
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
