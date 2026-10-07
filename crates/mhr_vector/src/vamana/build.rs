//! Deterministic Vamana construction (unit U14).
//!
//! Searches and robust pruning within a batch read the pre-batch graph. Out-edges
//! are then installed in insertion order, and reverse edges in ordinal order. Batch
//! boundaries depend only on the number of points, never on the worker count. This
//! is the deterministic batch-insertion approach used by ParlayANN.

use std::cmp::Ordering;

use crate::config::{Metric, VamanaParams};
use crate::error::{Error, Result};
use crate::quant::Codec;
use crate::util::{Bitset, SplitMix64};
use crate::vamana::{Graph, NO_NEIGHBOR};

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
    validate(codec, params, threads)?;
    let width = codec.code_len();
    if width == 0 || !codes.len().is_multiple_of(width) {
        return Err(Error::InvalidArgument("incomplete graph codes".into()));
    }
    let nodes = codes.len() / width;
    if nodes > NO_NEIGHBOR as usize {
        return Err(Error::InvalidArgument("too many graph nodes".into()));
    }
    let degree = params.max_degree as usize;
    let cells = nodes
        .checked_mul(degree)
        .filter(|&cells| cells <= isize::MAX as usize / size_of::<u32>())
        .ok_or_else(|| Error::InvalidArgument("graph size overflow".into()))?;
    let mut graph = Graph {
        degree: params.max_degree,
        entry: 0,
        adjacency: vec![NO_NEIGHBOR; cells],
    };
    if nodes <= 1 {
        return Ok(graph);
    }

    let mut space = CodeSpace {
        codec,
        codes,
        width,
        metric: codec.metric(),
        dot_shift: 0.0,
    };
    if space.metric == Metric::Dot {
        // Cauchy-Schwarz bounds every dot by the largest self-dot. The common
        // shift leaves rankings unchanged and makes robust-prune distances
        // nonnegative (using -dot, or -cosine, would invert the pruning rule).
        for node in 0..nodes as u32 {
            space.dot_shift = space.dot_shift.max(space.score(node, node));
        }
    }
    let mut rng = SplitMix64::new(seed);
    graph.entry = space.entry(nodes, &mut rng);

    // A seeded random graph gives the first pass long-range routes. Reserve a
    // cyclic edge initially so that even a very unlucky sample stays connected.
    for node in 0..nodes as u32 {
        let mut row: Vec<u32> = rng
            .sample((nodes - 1) as u64, degree)
            .into_iter()
            .map(|id| id as u32 + u32::from(id >= u64::from(node)))
            .collect();
        let next = (node + 1) % nodes as u32;
        if !row.contains(&next) {
            *row.last_mut().unwrap() = next;
        }
        set_row(&mut graph, node, &row);
    }

    let workers = threads.min(
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1),
    );
    let mut order: Vec<u32> = (0..nodes as u32).collect();
    // Bounded batches avoid retaining another full adjacency array. Starting
    // small lets useful local routes develop before the larger parallel batches.
    let max_batch = nodes.div_ceil(32).clamp(32, 1024);
    for pass in 0..params.passes {
        for i in (1..nodes).rev() {
            let j = rng.below((i + 1) as u64) as usize;
            order.swap(i, j);
        }
        let alpha = if pass == 0 { 1.0 } else { params.alpha };
        let mut start = 0;
        let mut batch_size = 1;
        while start < nodes {
            let end = (start + batch_size).min(nodes);
            let batch = &order[start..end];
            let rows = search_batch(
                &space,
                &graph,
                batch,
                params.build_beam as usize,
                alpha,
                workers,
            )?;
            install_batch(&space, &mut graph, &rows, alpha);
            start = end;
            batch_size = (batch_size * 2).min(max_batch);
        }
    }
    repair_reachability(&mut graph);
    Ok(graph)
}

fn validate<C: Codec>(codec: &C, params: &VamanaParams, threads: usize) -> Result<()> {
    if !(4..=256).contains(&params.max_degree)
        || params.build_beam < params.max_degree
        || params.build_beam > 4096
        || !(1.0..=2.0).contains(&params.alpha)
        || !(1..=2).contains(&params.passes)
        || !(1..=4096).contains(&params.search_beam)
    {
        return Err(Error::InvalidConfig("invalid Vamana parameters".into()));
    }
    if threads == 0 {
        return Err(Error::InvalidArgument("graph build needs a worker".into()));
    }
    if !(1..=crate::MAX_DIM as usize).contains(&codec.dim()) {
        return Err(Error::InvalidArgument(
            "invalid graph code dimension".into(),
        ));
    }
    Ok(())
}

struct CodeSpace<'a, C> {
    codec: &'a C,
    codes: &'a [u8],
    width: usize,
    metric: Metric,
    dot_shift: f32,
}

impl<C: Codec> CodeSpace<'_, C> {
    fn code(&self, node: u32) -> &[u8] {
        let start = node as usize * self.width;
        &self.codes[start..start + self.width]
    }

    fn score(&self, a: u32, b: u32) -> f32 {
        self.codec.score_codes(self.code(a), self.code(b))
    }

    fn distance(&self, a: u32, b: u32) -> f32 {
        self.distance_from_score(self.score(a, b))
    }

    fn distance_from_score(&self, score: f32) -> f32 {
        match self.metric {
            Metric::Cosine => (1.0 - score).max(0.0),
            Metric::L2 => (-score).max(0.0),
            Metric::Dot => (self.dot_shift - score).max(0.0),
        }
    }

    fn entry(&self, nodes: usize, rng: &mut SplitMix64) -> u32 {
        // A sample medoid: score every possible entry against the same fixed,
        // seeded sample. For L2/cosine this selects the point nearest the sample
        // centroid without requiring a codec-specific decoder. The small-graph
        // case uses every point. Accumulation order is identical for all workers.
        let sample = rng.sample(nodes as u64, 64);
        let mut best = f64::NEG_INFINITY;
        let mut entry = 0;
        for node in 0..nodes as u32 {
            let sum: f64 = sample
                .iter()
                .map(|&id| self.score(node, id as u32) as f64)
                .sum();
            if sum > best {
                best = sum;
                entry = node;
            }
        }
        entry
    }

    fn prune(&self, node: u32, mut candidates: Vec<u32>, degree: usize, alpha: f32) -> Vec<u32> {
        candidates.sort_unstable();
        candidates.dedup();
        let mut ranked: Vec<Candidate> = candidates
            .into_iter()
            .filter(|&id| id != node)
            .map(|id| Candidate {
                score: self.score(node, id),
                node: id,
                expanded: false,
            })
            .collect();
        ranked.sort_unstable_by(Candidate::compare);
        let mut selected = Vec::with_capacity(degree);
        for i in 0..ranked.len() {
            if ranked[i].expanded {
                continue;
            }
            let chosen = ranked[i].node;
            selected.push(chosen);
            if selected.len() == degree {
                break;
            }
            for candidate in &mut ranked[i + 1..] {
                if !candidate.expanded
                    && alpha * self.distance(chosen, candidate.node)
                        <= self.distance_from_score(candidate.score)
                {
                    candidate.expanded = true;
                }
            }
        }
        selected
    }
}

struct Candidate {
    score: f32,
    node: u32,
    expanded: bool,
}

impl Candidate {
    fn compare(&self, other: &Self) -> Ordering {
        // Navigation uses the original score, not a shifted distance: subtracting
        // a large dot shift can otherwise collapse distinct scores to the same f32.
        // Numeric equality also treats -0 and +0 as ties.
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or_else(|| other.score.total_cmp(&self.score))
            .then(self.node.cmp(&other.node))
    }
}

/// Scratch space is local to a worker. Only touched bits are reset between queries.
struct BuildScratch {
    seen: Bitset,
    touched: Vec<u32>,
    beam: Vec<Candidate>,
    expanded: Vec<u32>,
}

impl BuildScratch {
    fn new(nodes: usize) -> Self {
        Self {
            seen: Bitset::new(nodes),
            touched: Vec::new(),
            beam: Vec::new(),
            expanded: Vec::new(),
        }
    }

    fn search<C: Codec>(
        &mut self,
        space: &CodeSpace<'_, C>,
        graph: &Graph,
        node: u32,
        beam: usize,
    ) -> Vec<u32> {
        for id in self.touched.drain(..) {
            self.seen.clear(id as usize);
        }
        self.beam.clear();
        self.expanded.clear();
        self.seen.set(graph.entry as usize);
        self.touched.push(graph.entry);
        self.beam.push(Candidate {
            score: space.score(node, graph.entry),
            node: graph.entry,
            expanded: false,
        });
        while let Some(index) = self.beam.iter().position(|candidate| !candidate.expanded) {
            let current = self.beam[index].node;
            self.beam[index].expanded = true;
            self.expanded.push(current);
            for &neighbor in graph.neighbors(current) {
                if !self.seen.set(neighbor as usize) {
                    continue;
                }
                self.touched.push(neighbor);
                let candidate = Candidate {
                    score: space.score(node, neighbor),
                    node: neighbor,
                    expanded: false,
                };
                if self.beam.len() == beam
                    && candidate.compare(self.beam.last().unwrap()) != Ordering::Less
                {
                    continue;
                }
                let position = self
                    .beam
                    .binary_search_by(|other| other.compare(&candidate))
                    .unwrap_or_else(|position| position);
                self.beam.insert(position, candidate);
                self.beam.truncate(beam);
            }
        }
        let mut candidates = self.expanded.clone();
        candidates.extend_from_slice(graph.neighbors(node));
        candidates
    }
}

type Rows = Vec<(u32, Vec<u32>)>;

fn search_batch<C: Codec>(
    space: &CodeSpace<'_, C>,
    graph: &Graph,
    batch: &[u32],
    beam: usize,
    alpha: f32,
    workers: usize,
) -> Result<Rows> {
    let run = |chunk: &[u32]| {
        let mut scratch = BuildScratch::new(graph.len());
        chunk
            .iter()
            .map(|&node| {
                let candidates = scratch.search(space, graph, node, beam);
                let row = space.prune(node, candidates, graph.degree as usize, alpha);
                (node, row)
            })
            .collect::<Rows>()
    };
    // Parallelism changes only the division of this immutable batch's work. Small
    // batches avoid thread startup entirely, including in multi-thread builds.
    let workers = workers.min(batch.len().div_ceil(8));
    if workers <= 1 {
        return Ok(run(batch));
    }
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for chunk in batch.chunks(batch.len().div_ceil(workers)) {
            handles.push(
                std::thread::Builder::new()
                    .spawn_scoped(scope, || run(chunk))
                    .map_err(|error| Error::io("", error))?,
            );
        }
        let mut rows = Vec::with_capacity(batch.len());
        // Joining in chunk order, not completion order, fixes insertion order.
        for handle in handles {
            match handle.join() {
                Ok(chunk) => rows.extend(chunk),
                Err(panic) => std::panic::resume_unwind(panic),
            }
        }
        Ok(rows)
    })
}

fn set_row(graph: &mut Graph, node: u32, neighbors: &[u32]) {
    let width = graph.degree as usize;
    let start = node as usize * width;
    let row = &mut graph.adjacency[start..start + width];
    row[..neighbors.len()].copy_from_slice(neighbors);
    row[neighbors.len()..].fill(NO_NEIGHBOR);
}

fn install_batch<C: Codec>(space: &CodeSpace<'_, C>, graph: &mut Graph, rows: &Rows, alpha: f32) {
    let mut reverse = Vec::new();
    for (node, neighbors) in rows {
        set_row(graph, *node, neighbors);
        reverse.extend(neighbors.iter().map(|&neighbor| (neighbor, *node)));
    }
    reverse.sort_unstable();
    let mut start = 0;
    while start < reverse.len() {
        let node = reverse[start].0;
        let mut candidates = graph.neighbors(node).to_vec();
        let mut end = start;
        while end < reverse.len() && reverse[end].0 == node {
            let neighbor = reverse[end].1;
            if !candidates.contains(&neighbor) {
                candidates.push(neighbor);
            }
            end += 1;
        }
        if candidates.len() > graph.degree as usize {
            candidates = space.prune(node, candidates, graph.degree as usize, alpha);
        }
        set_row(graph, node, &candidates);
        start = end;
    }
}

/// Robust prune can isolate identical points (all their distances are zero). Join
/// unreachable components without exceeding R or deleting a reachability-tree edge.
fn repair_reachability(graph: &mut Graph) {
    let mut seen = Bitset::new(graph.len());
    let mut stack = vec![graph.entry];
    seen.set(graph.entry as usize);
    let mut leaf = graph.entry;
    let mut next = 0;
    loop {
        while let Some(node) = stack.pop() {
            leaf = node;
            for &neighbor in graph.neighbors(node) {
                if seen.set(neighbor as usize) {
                    stack.push(neighbor);
                }
            }
        }
        while next < graph.len() && seen.get(next) {
            next += 1;
        }
        if next == graph.len() {
            return;
        }
        // The last expanded vertex discovered no new vertices, so none of its
        // outgoing edges belong to the discovery tree. Even if its row is full,
        // replacing its final edge preserves every previously reached vertex.
        let width = graph.degree as usize;
        let used = graph.neighbors(leaf).len();
        graph.adjacency[leaf as usize * width + used.min(width - 1)] = next as u32;
        seen.set(next);
        stack.push(next as u32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::F32Codec;

    fn codec() -> F32Codec {
        F32Codec {
            metric: Metric::Cosine,
            dim: 2,
        }
    }

    #[test]
    fn empty_singleton_and_invalid_build_arguments() {
        let codec = codec();
        let params = VamanaParams::default();
        let empty = build_graph(&codec, &[], &params, 3, 1).unwrap();
        assert!(empty.is_empty());
        assert_eq!(Graph::from_bytes(&empty.to_bytes(), 0).unwrap(), empty);
        let mut code = vec![0; codec.code_len()];
        codec.encode(&[1.0, 0.0], &mut code);
        let single = build_graph(&codec, &code, &params, 3, 7).unwrap();
        assert_eq!(single.len(), 1);
        assert!(single.neighbors(0).is_empty());
        assert_eq!(Graph::from_bytes(&single.to_bytes(), 1).unwrap(), single);
        assert!(build_graph(&codec, &code[..7], &params, 3, 1).is_err());
        assert!(build_graph(&codec, &code, &params, 3, 0).is_err());
        for dim in [0, crate::MAX_DIM as usize + 1, usize::MAX] {
            assert!(build_graph(&F32Codec { dim, ..codec }, &[], &params, 3, 1).is_err());
        }
        for bad in [
            VamanaParams {
                max_degree: 0,
                ..params
            },
            VamanaParams {
                max_degree: 257,
                ..params
            },
            VamanaParams {
                build_beam: 63,
                ..params
            },
            VamanaParams {
                build_beam: 4097,
                ..params
            },
            VamanaParams {
                alpha: f32::NAN,
                ..params
            },
            VamanaParams {
                alpha: 0.99,
                ..params
            },
            VamanaParams {
                alpha: 2.01,
                ..params
            },
            VamanaParams {
                passes: 0,
                ..params
            },
            VamanaParams {
                passes: 3,
                ..params
            },
            VamanaParams {
                search_beam: 0,
                ..params
            },
        ] {
            assert!(build_graph(&codec, &code, &bad, 3, 1).is_err());
        }
    }

    fn reached(graph: &Graph) -> usize {
        let mut seen = Bitset::new(graph.len());
        let mut stack = vec![graph.entry];
        seen.set(graph.entry as usize);
        while let Some(node) = stack.pop() {
            for &neighbor in graph.neighbors(node) {
                if seen.set(neighbor as usize) {
                    stack.push(neighbor);
                }
            }
        }
        seen.count_ones()
    }

    #[test]
    fn duplicate_codes_are_fully_reachable_in_both_pass_modes() {
        let codec = codec();
        let mut code = vec![0; codec.code_len()];
        codec.encode(&[1.0, 0.0], &mut code);
        let codes = code.repeat(200);
        for passes in [1, 2] {
            let params = VamanaParams {
                max_degree: 8,
                build_beam: 16,
                passes,
                ..VamanaParams::default()
            };
            let graph = build_graph(&codec, &codes, &params, 9, 1).unwrap();
            assert_eq!(reached(&graph), 200);
            assert_eq!(Graph::from_bytes(&graph.to_bytes(), 200).unwrap(), graph);
            for threads in [2, 4, 7] {
                let other = build_graph(&codec, &codes, &params, 9, threads).unwrap();
                assert_eq!(graph.to_bytes(), other.to_bytes());
            }
        }
    }

    #[test]
    fn reachability_repair_preserves_full_rows_and_a_nonzero_entry() {
        let mut graph = Graph {
            degree: 2,
            entry: 2,
            adjacency: vec![1, 2, 0, 2, 0, 1, 4, 5, 3, 5, 3, 4, 7, 8, 6, 8, 6, 7],
        };
        assert_eq!(reached(&graph), 3);
        repair_reachability(&mut graph);
        assert_eq!(reached(&graph), 9);
        assert_eq!(Graph::from_bytes(&graph.to_bytes(), 9).unwrap(), graph);
    }

    #[test]
    fn score_order_does_not_collapse_dots_or_distinguish_signed_zero() {
        let candidate = |score, node| Candidate {
            score,
            node,
            expanded: false,
        };
        assert_eq!(
            candidate(-0.0, 0).compare(&candidate(0.0, 1)),
            Ordering::Less
        );
        assert_eq!(
            candidate(1.0, 0).compare(&candidate(2.0, 1)),
            Ordering::Greater
        );
        assert_eq!(
            candidate(2.0, 0).compare(&candidate(2.0, 1)),
            Ordering::Less
        );
    }
}
