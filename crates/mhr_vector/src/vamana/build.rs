//! Deterministic Vamana construction.
//!
//! Searches and robust pruning within a batch read the pre-batch graph. Out-edges
//! are then installed, and every reverse edge's target is re-pruned from its row
//! after that installation. No target's new row depends on another's, so targets
//! are grouped and pruned in parallel, and the result equals installing them one
//! by one in ordinal order. Batch boundaries depend only on the number of points,
//! never on the worker count. This is the deterministic batch-insertion approach
//! used by ParlayANN.
//!
//! The workers live for the whole build. Each batch has two phases separated by
//! barriers: the batch's searches (claimed one node at a time), then the
//! installation of its rows (claimed one range of target ordinals at a time).
//! Every random draw is taken from the seeded sequence in the order of the
//! sequential algorithm; workers that need a later part of it skip ahead.

use std::any::Any;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering::Relaxed};
use std::sync::{Barrier, Condvar, Mutex, MutexGuard};

use crate::config::{Metric, VamanaParams};
use crate::error::{Error, Result};
use crate::quant::Codec;
use crate::util::{Bitset, SplitMix64};
use crate::vamana::{Graph, NO_NEIGHBOR};

/// Nodes per work item when choosing the entry and drawing the initial rows.
const PREPARE_CHUNK: usize = 1024;
/// Target ranges per worker in the installation phase (load balance).
const RANGES_PER_WORKER: usize = 8;

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
    if nodes <= 1 {
        return Ok(Graph {
            degree: params.max_degree,
            entry: 0,
            adjacency: vec![NO_NEIGHBOR; cells],
        });
    }

    // The draws, in the sequential order: the entry's sample, `per_node` draws
    // for each node's random initial row (a seeded random graph gives the first
    // pass long-range routes), then one shuffle per pass.
    let mut rng = SplitMix64::new(seed);
    let sample = rng.sample(nodes as u64, 64);
    let initial = rng.clone();
    let per_node = if degree >= nodes - 1 { 0 } else { degree as u64 };
    rng.advance(per_node * nodes as u64);
    let mut order: Vec<u32> = (0..nodes as u32).collect();
    let mut orders = Vec::with_capacity(params.passes as usize);
    for _ in 0..params.passes {
        for i in (1..nodes).rev() {
            let j = rng.below((i + 1) as u64) as usize;
            order.swap(i, j);
        }
        orders.push(order.clone());
    }
    drop(order);

    let workers = threads.min(
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1),
    );
    let ranges = if workers == 1 {
        1
    } else {
        workers * RANGES_PER_WORKER
    };
    // Bounded batches avoid retaining another full adjacency array. Starting
    // small lets useful local routes develop before the larger parallel batches.
    let max_batch = nodes.div_ceil(32).clamp(32, 1024);
    let build = Build {
        codec,
        codes,
        width,
        params,
        nodes,
        degree,
        workers,
        ranges,
        max_batch,
        sample: &sample,
        initial,
        per_node,
        orders: &orders,
        rows: Rows::new(cells, degree),
        forward: (0..max_batch * degree)
            .map(|_| AtomicU32::new(NO_NEIGHBOR))
            .collect(),
        slots: (0..workers * ranges)
            .map(|_| Mutex::new(Slot::default()))
            .collect(),
        prepared: Mutex::new(vec![None; nodes.div_ceil(PREPARE_CHUNK)]),
        next_prepare: AtomicUsize::new(0),
        next_forward: AtomicUsize::new(0),
        next_reverse: AtomicUsize::new(0),
        barrier: Barrier::new(workers),
        failed: AtomicBool::new(false),
        panic: Mutex::new(None),
    };
    let entry = build.run()?;
    let mut graph = Graph {
        degree: params.max_degree,
        entry,
        adjacency: build.rows.into_vec(),
    };
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

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The adjacency under construction. Between two barriers each row is either only
/// read, or written by the one worker that owns it, so relaxed atomics suffice:
/// the barriers order the phases.
struct Rows {
    degree: usize,
    cells: Vec<AtomicU32>,
}

impl Rows {
    fn new(cells: usize, degree: usize) -> Rows {
        Rows {
            degree,
            cells: (0..cells).map(|_| AtomicU32::new(NO_NEIGHBOR)).collect(),
        }
    }

    #[cfg(test)]
    fn from_graph(graph: &Graph) -> Rows {
        Rows {
            degree: graph.degree as usize,
            cells: graph.adjacency.iter().map(|&n| AtomicU32::new(n)).collect(),
        }
    }

    fn row(&self, node: u32) -> &[AtomicU32] {
        let start = node as usize * self.degree;
        &self.cells[start..start + self.degree]
    }

    /// Appends `node`'s neighbours (its row up to the first padding) to `out`.
    fn read(&self, node: u32, out: &mut Vec<u32>) {
        read_row(self.row(node), out);
    }

    fn write(&self, node: u32, neighbors: &[u32]) {
        write_row(self.row(node), neighbors);
    }

    fn into_vec(self) -> Vec<u32> {
        self.cells.into_iter().map(AtomicU32::into_inner).collect()
    }
}

fn read_row(row: &[AtomicU32], out: &mut Vec<u32>) {
    for cell in row {
        let neighbor = cell.load(Relaxed);
        if neighbor == NO_NEIGHBOR {
            break;
        }
        out.push(neighbor);
    }
}

fn write_row(row: &[AtomicU32], neighbors: &[u32]) {
    for (cell, &neighbor) in row.iter().zip(neighbors) {
        cell.store(neighbor, Relaxed);
    }
    for cell in &row[neighbors.len()..] {
        cell.store(NO_NEIGHBOR, Relaxed);
    }
}

/// One worker's contribution to one target range: reverse edges `(target,
/// source)`, and the batch's own nodes in the range `(node, batch index)`.
#[derive(Default)]
struct Slot {
    edges: Vec<(u32, u32)>,
    nodes: Vec<(u32, u32)>,
}

/// A chunk's largest self-score (dot shift), and its best entry sum and node.
type Prepared = (f32, f64, u32);

/// State shared by the workers of one build.
struct Build<'a, C> {
    codec: &'a C,
    codes: &'a [u8],
    width: usize,
    params: &'a VamanaParams,
    nodes: usize,
    degree: usize,
    workers: usize,
    ranges: usize,
    max_batch: usize,
    sample: &'a [u64],
    /// The generator after the entry's sample: the initial rows' draws start here.
    initial: SplitMix64,
    per_node: u64,
    orders: &'a [Vec<u32>],
    rows: Rows,
    /// The current batch's new rows, by batch index.
    forward: Vec<AtomicU32>,
    /// `workers x ranges` handoffs from the search phase to the installation phase.
    slots: Vec<Mutex<Slot>>,
    prepared: Mutex<Vec<Option<Prepared>>>,
    next_prepare: AtomicUsize,
    next_forward: AtomicUsize,
    next_reverse: AtomicUsize,
    barrier: Barrier,
    failed: AtomicBool,
    panic: Mutex<Option<Box<dyn Any + Send>>>,
}

/// Holds spawned workers until every spawn has succeeded (or one has failed), so
/// that the barrier always has all of its parties.
#[derive(Default)]
struct Gate {
    open: Mutex<Option<bool>>,
    signal: Condvar,
}

impl Gate {
    fn wait(&self) -> bool {
        let mut open = lock(&self.open);
        loop {
            if let Some(go) = *open {
                return go;
            }
            open = self
                .signal
                .wait(open)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    fn set(&self, go: bool) {
        *lock(&self.open) = Some(go);
        self.signal.notify_all();
    }
}

/// A worker's reusable buffers.
struct Local {
    scratch: BuildScratch,
    prune: PruneScratch,
    candidates: Vec<(u32, f32)>,
    ids: Vec<u32>,
    row: Vec<u32>,
    edges: Vec<Vec<(u32, u32)>>,
    nodes: Vec<Vec<(u32, u32)>>,
    gathered: Vec<(u32, u32)>,
}

impl<C: Codec> Build<'_, C> {
    /// Runs the workers (this thread is worker 0) and returns the entry.
    fn run(&self) -> Result<u32> {
        let gate = Gate::default();
        let entry = std::thread::scope(|scope| {
            let mut spawned = Ok(());
            for id in 1..self.workers {
                let gate = &gate;
                let worker = std::thread::Builder::new().spawn_scoped(scope, move || {
                    if gate.wait() {
                        self.work(id);
                    }
                });
                if let Err(error) = worker {
                    spawned = Err(Error::io("", error));
                    break;
                }
            }
            gate.set(spawned.is_ok());
            spawned.map(|()| self.work(0))
        })?;
        if let Some(panic) = lock(&self.panic).take() {
            resume_unwind(panic);
        }
        Ok(entry)
    }

    /// Runs `work` unless a worker has failed. A panic is kept for the caller, and
    /// the worker carries on to the barriers (doing nothing) so no one waits forever.
    fn phase(&self, work: impl FnOnce()) {
        if self.failed.load(Relaxed) {
            return;
        }
        if let Err(panic) = catch_unwind(AssertUnwindSafe(work)) {
            self.failed.store(true, Relaxed);
            lock(&self.panic).get_or_insert(panic);
        }
    }

    fn space(&self, dot_shift: f32) -> CodeSpace<'_, C> {
        CodeSpace {
            codec: self.codec,
            codes: self.codes,
            width: self.width,
            metric: self.codec.metric(),
            dot_shift,
        }
    }

    fn work(&self, id: usize) -> u32 {
        self.phase(|| self.prepare());
        self.barrier.wait();
        let (entry, dot_shift) = self.entry_and_shift();
        let space = self.space(dot_shift);
        let mut local = Local {
            scratch: BuildScratch::new(self.nodes),
            prune: PruneScratch::default(),
            candidates: Vec::new(),
            ids: Vec::new(),
            row: Vec::with_capacity(self.degree),
            edges: vec![Vec::new(); self.ranges],
            nodes: vec![Vec::new(); self.ranges],
            gathered: Vec::new(),
        };
        for (pass, order) in self.orders.iter().enumerate() {
            let alpha = if pass == 0 { 1.0 } else { self.params.alpha };
            let mut start = 0;
            let mut batch_size = 1;
            while start < self.nodes {
                let end = (start + batch_size).min(self.nodes);
                let batch = &order[start..end];
                self.phase(|| self.search_batch(id, &space, entry, batch, alpha, &mut local));
                self.barrier.wait();
                self.phase(|| self.install_batch(id, &space, alpha, &mut local));
                self.barrier.wait();
                start = end;
                batch_size = (batch_size * 2).min(self.max_batch);
            }
        }
        entry
    }

    fn range_of(&self, node: u32) -> usize {
        (node as u64 * self.ranges as u64 / self.nodes as u64) as usize
    }

    /// Per chunk of nodes: the largest self-score (Dot only), the best entry, and the
    /// seeded random initial rows (with a cyclic edge, so that even a very unlucky
    /// sample stays connected).
    fn prepare(&self) {
        let space = self.space(0.0);
        let mut row = Vec::with_capacity(self.degree);
        loop {
            let chunk = self.next_prepare.fetch_add(1, Relaxed);
            let start = chunk * PREPARE_CHUNK;
            if start >= self.nodes || self.failed.load(Relaxed) {
                return;
            }
            let mut dot_shift = 0.0f32;
            let mut best = f64::NEG_INFINITY;
            let mut entry = 0;
            for node in start as u32..(start + PREPARE_CHUNK).min(self.nodes) as u32 {
                if space.metric == Metric::Dot {
                    // Cauchy-Schwarz bounds every dot by the largest self-dot. The
                    // common shift leaves rankings unchanged and makes robust-prune
                    // distances nonnegative (using -dot, or -cosine, would invert
                    // the pruning rule).
                    dot_shift = dot_shift.max(space.score(node, node));
                }
                // A sample medoid: score every possible entry against the same
                // fixed, seeded sample. For L2/cosine this selects the point
                // nearest the sample centroid without requiring a codec-specific
                // decoder. The small-graph case uses every point.
                let sum: f64 = self
                    .sample
                    .iter()
                    .map(|&id| space.score(node, id as u32) as f64)
                    .sum();
                if sum > best {
                    best = sum;
                    entry = node;
                }
                let mut rng = self.initial.clone();
                rng.advance(u64::from(node) * self.per_node);
                row.clear();
                row.extend(
                    rng.sample((self.nodes - 1) as u64, self.degree)
                        .into_iter()
                        .map(|id| id as u32 + u32::from(id >= u64::from(node))),
                );
                let next = (node + 1) % self.nodes as u32;
                if !row.contains(&next) {
                    *row.last_mut().unwrap() = next;
                }
                self.rows.write(node, &row);
            }
            lock(&self.prepared)[chunk] = Some((dot_shift, best, entry));
        }
    }

    /// Combines the chunks in node order, as one sequential scan would: the first
    /// node with the best sum, and the largest self-score. (The shift is only
    /// compared, so the sign of a zero shift cannot matter.)
    fn entry_and_shift(&self) -> (u32, f32) {
        let mut dot_shift = 0.0f32;
        let mut best = f64::NEG_INFINITY;
        let mut entry = 0;
        for chunk in lock(&self.prepared).iter() {
            // Missing only after a failure, when no further work is done.
            let Some((shift, sum, node)) = *chunk else {
                return (0, 0.0);
            };
            dot_shift = dot_shift.max(shift);
            if sum > best {
                best = sum;
                entry = node;
            }
        }
        (entry, dot_shift)
    }

    /// Searches and prunes the batch's nodes against the frozen graph, then hands the
    /// new rows' reverse edges to the installation phase, grouped by target range.
    fn search_batch(
        &self,
        id: usize,
        space: &CodeSpace<'_, C>,
        entry: u32,
        batch: &[u32],
        alpha: f32,
        local: &mut Local,
    ) {
        if id == 0 {
            // Nobody uses the installation counter until the next barrier.
            self.next_reverse.store(0, Relaxed);
        }
        loop {
            let index = self.next_forward.fetch_add(1, Relaxed);
            if index >= batch.len() || self.failed.load(Relaxed) {
                break;
            }
            let node = batch[index];
            local.scratch.search(
                space,
                &self.rows,
                entry,
                node,
                self.params.build_beam as usize,
                &mut local.candidates,
            );
            space.prune(
                node,
                &mut local.candidates,
                self.degree,
                alpha,
                &mut local.prune,
                &mut local.row,
            );
            write_row(
                &self.forward[index * self.degree..(index + 1) * self.degree],
                &local.row,
            );
            for &neighbor in &local.row {
                local.edges[self.range_of(neighbor)].push((neighbor, node));
            }
            local.nodes[self.range_of(node)].push((node, index as u32));
        }
        // Each slot was emptied by the last installation phase; swapping keeps
        // both buffers' allocations.
        for range in 0..self.ranges {
            let mut slot = lock(&self.slots[id * self.ranges + range]);
            std::mem::swap(&mut slot.edges, &mut local.edges[range]);
            std::mem::swap(&mut slot.nodes, &mut local.nodes[range]);
        }
    }

    /// Installs the batch's rows, one target range at a time: first the batch's own
    /// new rows, then each reverse-edge target's row with its new sources appended
    /// in ordinal order, re-pruned when it overflows. A target's new row depends
    /// only on its installed row and its sources, never on another target.
    fn install_batch(&self, id: usize, space: &CodeSpace<'_, C>, alpha: f32, local: &mut Local) {
        if id == 0 {
            // Nobody uses the search counter until the next barrier.
            self.next_forward.store(0, Relaxed);
        }
        loop {
            let range = self.next_reverse.fetch_add(1, Relaxed);
            if range >= self.ranges || self.failed.load(Relaxed) {
                break;
            }
            local.gathered.clear();
            for worker in 0..self.workers {
                let mut slot = lock(&self.slots[worker * self.ranges + range]);
                for &(node, index) in &slot.nodes {
                    let index = index as usize;
                    local.row.clear();
                    read_row(
                        &self.forward[index * self.degree..(index + 1) * self.degree],
                        &mut local.row,
                    );
                    self.rows.write(node, &local.row);
                }
                slot.nodes.clear();
                local.gathered.extend_from_slice(&slot.edges);
                slot.edges.clear();
            }
            local.gathered.sort_unstable();
            let mut start = 0;
            while start < local.gathered.len() {
                let node = local.gathered[start].0;
                local.ids.clear();
                self.rows.read(node, &mut local.ids);
                let mut end = start;
                while end < local.gathered.len() && local.gathered[end].0 == node {
                    let neighbor = local.gathered[end].1;
                    if !local.ids.contains(&neighbor) {
                        local.ids.push(neighbor);
                    }
                    end += 1;
                }
                if local.ids.len() > self.degree {
                    local.candidates.clear();
                    local.candidates.extend(local.ids.iter().map(|&id| (id, UNSCORED)));
                    space.prune(
                        node,
                        &mut local.candidates,
                        self.degree,
                        alpha,
                        &mut local.prune,
                        &mut local.row,
                    );
                    self.rows.write(node, &local.row);
                } else {
                    self.rows.write(node, &local.ids);
                }
                start = end;
            }
        }
    }
}

struct CodeSpace<'a, C> {
    codec: &'a C,
    codes: &'a [u8],
    width: usize,
    metric: Metric,
    dot_shift: f32,
}

/// Marks a candidate whose score is not known yet (scores are never NaN).
const UNSCORED: f32 = f32::NAN;

#[derive(Default)]
struct PruneScratch {
    ranked: Vec<Candidate>,
    ids: Vec<u32>,
    at: Vec<u32>,
    scores: Vec<f32>,
}

impl<C: Codec> CodeSpace<'_, C> {
    fn code(&self, node: u32) -> &[u8] {
        let start = node as usize * self.width;
        &self.codes[start..start + self.width]
    }

    fn score(&self, a: u32, b: u32) -> f32 {
        self.codec.score_codes(self.code(a), self.code(b))
    }

    /// `score(a, id)` for each id, into `scores`.
    fn score_many(&self, a: u32, ids: &[u32], scores: &mut Vec<f32>) {
        scores.clear();
        self.codec
            .score_codes_many(self.code(a), self.codes, ids, scores);
    }

    fn distance_from_score(&self, score: f32) -> f32 {
        match self.metric {
            Metric::Cosine => (1.0 - score).max(0.0),
            Metric::L2 => (-score).max(0.0),
            Metric::Dot => (self.dot_shift - score).max(0.0),
        }
    }

    /// Robust prune of `candidates` (any order; duplicates and `node` itself are
    /// ignored) into `selected`. Each candidate carries `score(node, id)` or
    /// [`UNSCORED`]. A function of the candidate set alone.
    fn prune(
        &self,
        node: u32,
        candidates: &mut Vec<(u32, f32)>,
        degree: usize,
        alpha: f32,
        scratch: &mut PruneScratch,
        selected: &mut Vec<u32>,
    ) {
        candidates.sort_unstable_by_key(|&(id, _)| id);
        candidates.dedup_by(|later, kept| {
            if later.0 != kept.0 {
                return false;
            }
            if kept.1.is_nan() {
                kept.1 = later.1;
            }
            true
        });
        scratch.ids.clear();
        scratch.ids.extend(
            candidates
                .iter()
                .filter(|&&(id, score)| id != node && score.is_nan())
                .map(|&(id, _)| id),
        );
        self.score_many(node, &scratch.ids, &mut scratch.scores);
        let mut computed = scratch.scores.iter();
        let ranked = &mut scratch.ranked;
        ranked.clear();
        for &(id, score) in candidates.iter().filter(|&&(id, _)| id != node) {
            let score = if score.is_nan() {
                *computed.next().unwrap()
            } else {
                score
            };
            ranked.push(Candidate {
                score,
                distance: self.distance_from_score(score),
                node: id,
                occluded: false,
            });
        }
        // Candidate::compare is a total order on distinct nodes, so the sorted
        // order does not depend on the input order.
        ranked.sort_unstable_by(Candidate::compare);
        selected.clear();
        for i in 0..ranked.len() {
            if ranked[i].occluded {
                continue;
            }
            let chosen = ranked[i].node;
            selected.push(chosen);
            if selected.len() == degree {
                break;
            }
            // Each remaining candidate's test depends only on its own distances.
            scratch.ids.clear();
            scratch.at.clear();
            for (j, candidate) in ranked.iter().enumerate().skip(i + 1) {
                if !candidate.occluded {
                    scratch.ids.push(candidate.node);
                    scratch.at.push(j as u32);
                }
            }
            self.score_many(chosen, &scratch.ids, &mut scratch.scores);
            for (&j, &score) in scratch.at.iter().zip(&scratch.scores) {
                let candidate = &mut ranked[j as usize];
                if alpha * self.distance_from_score(score) <= candidate.distance {
                    candidate.occluded = true;
                }
            }
        }
    }
}

struct Candidate {
    score: f32,
    /// `distance_from_score(score)`.
    distance: f32,
    node: u32,
    occluded: bool,
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

/// An estimate ordered worst first: numerically lower scores, then higher
/// ordinals (the query frontier's order, reversed; signed zeros are ties).
#[derive(Clone, Copy, PartialEq)]
struct Estimate(f32, u32);

impl Eq for Estimate {}

impl Ord for Estimate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Greater means worse. Scores are never NaN.
        other
            .0
            .partial_cmp(&self.0)
            .expect("graph estimates must not be NaN")
            .then(self.1.cmp(&other.1))
    }
}

impl PartialOrd for Estimate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The query frontier's rules (`search::Frontier`) with cheaper pools: the best
/// `beam` estimates seen (progress) as a heap whose root is the worst of them, and
/// the best `beam` unexpanded estimates as a deque sorted best first. Stops after
/// `beam` consecutive expansions fail to improve the progress pool. Each pool's
/// contents equal the sorted pools' contents, so every decision is the same.
#[derive(Default)]
struct BuildFrontier {
    pending: VecDeque<Estimate>,
    best: BinaryHeap<Estimate>,
    beam: usize,
    stalled: usize,
}

impl BuildFrontier {
    fn reset(&mut self, beam: usize) {
        self.pending.clear();
        self.best.clear();
        self.beam = beam;
        self.stalled = 0;
    }

    fn offer(&mut self, score: f32, node: u32) {
        let estimate = Estimate(score, node);
        if self.best.len() < self.beam {
            self.best.push(estimate);
            self.stalled = 0;
        } else if let Some(mut worst) = self.best.peek_mut()
            && estimate < *worst
        {
            *worst = estimate;
            self.stalled = 0;
        }
        if self.pending.len() == self.beam {
            if self.pending.back().is_some_and(|worst| estimate >= *worst) {
                return;
            }
            self.pending.pop_back();
        }
        let position = self.pending.partition_point(|other| *other < estimate);
        self.pending.insert(position, estimate);
    }

    fn pop(&mut self) -> Option<(f32, u32)> {
        if self.stalled >= self.beam {
            return None;
        }
        let Estimate(score, node) = self.pending.pop_front()?;
        self.stalled += 1;
        Some((score, node))
    }
}

/// Scratch space is local to a worker. Only touched bits are reset between queries.
struct BuildScratch {
    seen: Bitset,
    touched: Vec<u32>,
    frontier: BuildFrontier,
    fresh: Vec<u32>,
    scores: Vec<f32>,
}

impl BuildScratch {
    fn new(nodes: usize) -> Self {
        Self {
            seen: Bitset::new(nodes),
            touched: Vec::new(),
            frontier: BuildFrontier::default(),
            fresh: Vec::new(),
            scores: Vec::new(),
        }
    }

    /// Writes the search's expanded nodes with their scores, then `node`'s current
    /// neighbours ([`UNSCORED`]), to `candidates`.
    fn search<C: Codec>(
        &mut self,
        space: &CodeSpace<'_, C>,
        rows: &Rows,
        entry: u32,
        node: u32,
        beam: usize,
        candidates: &mut Vec<(u32, f32)>,
    ) {
        for id in self.touched.drain(..) {
            self.seen.clear(id as usize);
        }
        self.frontier.reset(beam);
        candidates.clear();
        self.seen.set(entry as usize);
        self.touched.push(entry);
        self.frontier.offer(space.score(node, entry), entry);
        while let Some((score, current)) = self.frontier.pop() {
            candidates.push((current, score));
            // Offers do not affect which neighbours are new, so scoring them
            // together keeps the order of offers.
            self.fresh.clear();
            for cell in rows.row(current) {
                let neighbor = cell.load(Relaxed);
                if neighbor == NO_NEIGHBOR {
                    break;
                }
                if self.seen.set(neighbor as usize) {
                    self.touched.push(neighbor);
                    self.fresh.push(neighbor);
                }
            }
            space.score_many(node, &self.fresh, &mut self.scores);
            for (&neighbor, &score) in self.fresh.iter().zip(&self.scores) {
                self.frontier.offer(score, neighbor);
            }
        }
        for cell in rows.row(node) {
            let neighbor = cell.load(Relaxed);
            if neighbor == NO_NEIGHBOR {
                break;
            }
            candidates.push((neighbor, UNSCORED));
        }
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
    use crate::quant::{BinaryCodec, F32Codec, Int8Codec};

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
    fn construction_search_also_expands_a_lower_scoring_bridge() {
        let codec = F32Codec {
            metric: Metric::Dot,
            dim: 1,
        };
        let codes: Vec<_> = [0.0f32, 10.0, 8.0, 11.0, 9.0, 20.0, 1.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let space = CodeSpace {
            codec: &codec,
            codes: &codes,
            width: codec.code_len(),
            metric: Metric::Dot,
            dot_shift: 400.0,
        };
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
                [NO_NEIGHBOR; 2],
            ]
            .concat(),
        };
        let rows = Rows::from_graph(&graph);
        let mut scratch = BuildScratch::new(graph.len());
        let mut candidates = Vec::new();
        let mut selected = Vec::new();
        for _ in 0..2 {
            scratch.search(&space, &rows, graph.entry, 6, 2, &mut candidates);
            assert!(candidates.contains(&(5, 20.0)));
            let mut prune = PruneScratch::default();
            space.prune(6, &mut candidates, 2, 1.2, &mut prune, &mut selected);
            assert_eq!(selected[0], 5);
        }
    }

    #[test]
    fn score_order_does_not_collapse_dots_or_distinguish_signed_zero() {
        let candidate = |score, node| Candidate {
            score,
            distance: 0.0,
            node,
            occluded: false,
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

    /// The sequential algorithm as it was before construction became parallel
    /// (839e575), with its own scalar search and prune: every build must equal it
    /// byte for byte.
    mod sequential {
        use super::super::*;
        use crate::vamana::search::Frontier;

        struct Space<'a, C> {
            codec: &'a C,
            codes: &'a [u8],
            width: usize,
            metric: Metric,
            dot_shift: f32,
        }

        impl<C: Codec> Space<'_, C> {
            fn score(&self, a: u32, b: u32) -> f32 {
                let code = |n: u32| &self.codes[n as usize * self.width..][..self.width];
                self.codec.score_codes(code(a), code(b))
            }

            fn distance_from_score(&self, score: f32) -> f32 {
                match self.metric {
                    Metric::Cosine => (1.0 - score).max(0.0),
                    Metric::L2 => (-score).max(0.0),
                    Metric::Dot => (self.dot_shift - score).max(0.0),
                }
            }

            fn prune(&self, node: u32, mut candidates: Vec<u32>, degree: usize, alpha: f32) -> Vec<u32> {
                candidates.sort_unstable();
                candidates.dedup();
                let mut ranked: Vec<Candidate> = candidates
                    .into_iter()
                    .filter(|&id| id != node)
                    .map(|id| Candidate {
                        score: self.score(node, id),
                        distance: 0.0,
                        node: id,
                        occluded: false,
                    })
                    .collect();
                ranked.sort_unstable_by(Candidate::compare);
                let mut selected = Vec::with_capacity(degree);
                for i in 0..ranked.len() {
                    if ranked[i].occluded {
                        continue;
                    }
                    let chosen = ranked[i].node;
                    selected.push(chosen);
                    if selected.len() == degree {
                        break;
                    }
                    for candidate in &mut ranked[i + 1..] {
                        if !candidate.occluded
                            && alpha * self.distance_from_score(self.score(chosen, candidate.node))
                                <= self.distance_from_score(candidate.score)
                        {
                            candidate.occluded = true;
                        }
                    }
                }
                selected
            }

            fn search(&self, graph: &Graph, node: u32, beam: usize) -> Vec<u32> {
                let mut seen = Bitset::new(graph.len());
                let mut frontier = Frontier::default();
                frontier.reset(beam);
                let mut expanded = Vec::new();
                seen.set(graph.entry as usize);
                frontier.offer((self.score(node, graph.entry), graph.entry));
                while let Some(current) = frontier.pop() {
                    expanded.push(current);
                    for &neighbor in graph.neighbors(current) {
                        if seen.set(neighbor as usize) {
                            frontier.offer((self.score(node, neighbor), neighbor));
                        }
                    }
                }
                expanded.extend_from_slice(graph.neighbors(node));
                expanded
            }
        }

        pub fn build<C: Codec>(codec: &C, codes: &[u8], params: &VamanaParams, seed: u64) -> Graph {
            let width = codec.code_len();
            let nodes = codes.len() / width;
            let degree = params.max_degree as usize;
            let mut graph = Graph {
                degree: params.max_degree,
                entry: 0,
                adjacency: vec![NO_NEIGHBOR; nodes * degree],
            };
            if nodes <= 1 {
                return graph;
            }
            let mut space = Space {
                codec,
                codes,
                width,
                metric: codec.metric(),
                dot_shift: 0.0,
            };
            if space.metric == Metric::Dot {
                for node in 0..nodes as u32 {
                    space.dot_shift = space.dot_shift.max(space.score(node, node));
                }
            }
            let mut rng = SplitMix64::new(seed);
            let sample = rng.sample(nodes as u64, 64);
            let mut best = f64::NEG_INFINITY;
            for node in 0..nodes as u32 {
                let sum: f64 = sample
                    .iter()
                    .map(|&id| space.score(node, id as u32) as f64)
                    .sum();
                if sum > best {
                    best = sum;
                    graph.entry = node;
                }
            }
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
            let mut order: Vec<u32> = (0..nodes as u32).collect();
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
                    let rows: Vec<(u32, Vec<u32>)> = order[start..end]
                        .iter()
                        .map(|&node| {
                            let candidates = space.search(&graph, node, params.build_beam as usize);
                            (node, space.prune(node, candidates, degree, alpha))
                        })
                        .collect();
                    let mut reverse = Vec::new();
                    for (node, neighbors) in &rows {
                        set_row(&mut graph, *node, neighbors);
                        reverse.extend(neighbors.iter().map(|&neighbor| (neighbor, *node)));
                    }
                    reverse.sort_unstable();
                    let mut i = 0;
                    while i < reverse.len() {
                        let node = reverse[i].0;
                        let mut candidates = graph.neighbors(node).to_vec();
                        while i < reverse.len() && reverse[i].0 == node {
                            if !candidates.contains(&reverse[i].1) {
                                candidates.push(reverse[i].1);
                            }
                            i += 1;
                        }
                        if candidates.len() > degree {
                            candidates = space.prune(node, candidates, degree, alpha);
                        }
                        set_row(&mut graph, node, &candidates);
                    }
                    start = end;
                    batch_size = (batch_size * 2).min(max_batch);
                }
            }
            repair_reachability(&mut graph);
            graph
        }

        fn set_row(graph: &mut Graph, node: u32, neighbors: &[u32]) {
            let width = graph.degree as usize;
            let row = &mut graph.adjacency[node as usize * width..(node as usize + 1) * width];
            row[..neighbors.len()].copy_from_slice(neighbors);
            row[neighbors.len()..].fill(NO_NEIGHBOR);
        }
    }

    fn vectors(n: usize, dim: usize, seed: u64, clusters: usize) -> Vec<f32> {
        let mut rng = SplitMix64::new(seed);
        let mut unit = || (rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
        let centres: Vec<f32> = (0..clusters * dim).map(|_| unit() * 4.0).collect();
        let mut out = Vec::with_capacity(n * dim);
        for i in 0..n {
            let centre = &centres[(i % clusters) * dim..(i % clusters + 1) * dim];
            out.extend(centre.iter().map(|&c| c + unit()));
        }
        out
    }

    fn encoded<C: Codec>(codec: &C, data: &[f32]) -> Vec<u8> {
        let mut codes = vec![0; data.len() / codec.dim() * codec.code_len()];
        for (vector, code) in data
            .chunks_exact(codec.dim())
            .zip(codes.chunks_exact_mut(codec.code_len()))
        {
            codec.encode(vector, code);
        }
        codes
    }

    fn assert_matches_sequential<C: Codec>(codec: &C, codes: &[u8], params: &VamanaParams) {
        let expected = sequential::build(codec, codes, params, 11).to_bytes();
        for threads in [1, 2, 3, 8] {
            let graph = build_graph(codec, codes, params, 11, threads).unwrap();
            assert!(graph.to_bytes() == expected, "{threads} threads");
        }
    }

    fn data(metric: Metric, dim: usize, n: usize) -> Vec<f32> {
        let mut data = vectors(n, dim, n as u64, 9);
        if metric == Metric::Cosine {
            for v in data.chunks_exact_mut(dim) {
                let norm = crate::distance::dot(v, v).sqrt();
                v.iter_mut().for_each(|x| *x /= norm);
            }
        }
        data
    }

    /// Graph hashes (`util::stable_hash` of the graph bytes) of builds by 839e575,
    /// the last commit with sequential construction.
    #[test]
    fn builds_equal_the_sequential_commits_bytes() {
        let hash = |graph: Graph| crate::util::stable_hash(&graph.to_bytes(), 0);
        let small = VamanaParams {
            max_degree: 12,
            build_beam: 24,
            ..VamanaParams::default()
        };
        let pinned = [
            (Metric::Cosine, 24, 1500, [0x9a98e51088057834, 0x06a97c953e256ff1, 0x4b69280040ff3640]),
            (Metric::L2, 17, 1100, [0xd2a8a2c7be1f5fe9, 0xfbff10e0e857ccca, 0x4f17a2c6061a74eb]),
            (Metric::Dot, 33, 900, [0x2ca0ff5161355edb, 0xc01550b1ea407604, 0xc658eb9eba9ceead]),
        ];
        for (metric, dim, n, [int8_hash, binary_hash, f32_hash]) in pinned {
            let data = data(metric, dim, n);
            for threads in [1, 3] {
                let int8 = Int8Codec::train(metric, dim, &data, 0).unwrap();
                let graph = build_graph(&int8, &encoded(&int8, &data), &small, 11, threads);
                assert_eq!(hash(graph.unwrap()), int8_hash, "int8 {metric:?}");
                let binary = BinaryCodec::train(metric, dim, &data, 0).unwrap();
                let graph = build_graph(&binary, &encoded(&binary, &data), &small, 11, threads);
                assert_eq!(hash(graph.unwrap()), binary_hash, "binary {metric:?}");
                let f32s = F32Codec { metric, dim };
                let codes = encoded(&f32s, &data[..300 * dim]);
                let graph = build_graph(&f32s, &codes, &small, 11, threads);
                assert_eq!(hash(graph.unwrap()), f32_hash, "f32 {metric:?}");
            }
        }
        for (metric, pinned) in [
            (Metric::Cosine, 0x3a1334ea2c10a066),
            (Metric::L2, 0xc47d1840b98ad747),
        ] {
            let data = data(metric, 40, 4000);
            let int8 = Int8Codec::train(metric, 40, &data, 0).unwrap();
            let codes = encoded(&int8, &data);
            let graph = build_graph(&int8, &codes, &VamanaParams::default(), 5, 4);
            assert_eq!(hash(graph.unwrap()), pinned, "default int8 {metric:?}");
        }
    }

    #[test]
    fn parallel_builds_equal_the_sequential_algorithm() {
        let params = VamanaParams {
            max_degree: 12,
            build_beam: 24,
            ..VamanaParams::default()
        };
        for (metric, dim, n) in [
            (Metric::Cosine, 24, 1500),
            (Metric::L2, 17, 1100),
            (Metric::Dot, 33, 900),
        ] {
            let data = data(metric, dim, n);
            let int8 = Int8Codec::train(metric, dim, &data, 0).unwrap();
            assert_matches_sequential(&int8, &encoded(&int8, &data), &params);
            let binary = BinaryCodec::train(metric, dim, &data, 0).unwrap();
            assert_matches_sequential(&binary, &encoded(&binary, &data), &params);
            let f32s = F32Codec { metric, dim };
            assert_matches_sequential(&f32s, &encoded(&f32s, &data[..300 * dim]), &params);
        }
        // A degree as large as the graph: no random draws for the initial rows.
        let tiny = VamanaParams {
            max_degree: 8,
            build_beam: 8,
            ..VamanaParams::default()
        };
        let data = vectors(9, 4, 5, 2);
        let int8 = Int8Codec::train(Metric::L2, 4, &data, 0).unwrap();
        assert_matches_sequential(&int8, &encoded(&int8, &data), &tiny);
    }

    #[test]
    fn a_panicking_codec_fails_the_build_without_hanging() {
        struct Fails(F32Codec);
        impl Codec for Fails {
            type Query = Vec<f32>;
            fn kind(&self) -> crate::config::CodeKind {
                self.0.kind()
            }
            fn metric(&self) -> Metric {
                self.0.metric()
            }
            fn dim(&self) -> usize {
                self.0.dim()
            }
            fn code_len(&self) -> usize {
                self.0.code_len()
            }
            fn encode(&self, vector: &[f32], out: &mut [u8]) {
                self.0.encode(vector, out)
            }
            fn prepare(&self, query: &[f32]) -> Vec<f32> {
                self.0.prepare(query)
            }
            fn score(&self, query: &Vec<f32>, code: &[u8]) -> f32 {
                self.0.score(query, code)
            }
            fn score_codes(&self, a: &[u8], b: &[u8]) -> f32 {
                let score = self.0.score_codes(a, b);
                assert!(score != -4.0, "codec failure");
                score
            }
            fn to_bytes(&self) -> Vec<u8> {
                self.0.to_bytes()
            }
        }
        let codec = Fails(F32Codec {
            metric: Metric::L2,
            dim: 1,
        });
        let codes: Vec<u8> = (0..400).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let params = VamanaParams {
            max_degree: 8,
            build_beam: 16,
            ..VamanaParams::default()
        };
        for threads in [1, 3] {
            let result =
                std::panic::catch_unwind(|| build_graph(&codec, &codes, &params, 1, threads));
            assert!(result.is_err(), "{threads} threads");
        }
    }
}
