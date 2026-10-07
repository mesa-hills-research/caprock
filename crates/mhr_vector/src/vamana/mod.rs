//! Vamana graphs (units U15 build, U16 search; REQUIREMENTS.md, "Approximate search").
//!
//! A segment's graph is a fixed-degree adjacency array over its ordinals: row `i` holds
//! node `i`'s out-neighbours followed by [`NO_NEIGHBOR`] padding. Construction is
//! DiskANN's Vamana (greedy search + robust prune with `alpha`), made deterministic
//! (REQUIREMENTS.md, "Determinism"): the same codes, parameters and seed give the same
//! graph for any thread count.

pub mod build;
pub mod search;

pub use build::build_graph;
pub use search::{SearchScratch, beam_search};

use crate::error::Result;

/// Padding in adjacency rows.
pub const NO_NEIGHBOR: u32 = u32::MAX;

/// A graph over `adjacency.len() / degree` nodes.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Graph {
    /// Row width (the out-degree bound `R`).
    pub degree: u32,
    /// The start node of every search (the medoid, or the nearest node to the centroid).
    pub entry: u32,
    /// `nodes * degree` neighbour ids, each row's real neighbours first.
    pub adjacency: Vec<u32>,
}

impl Graph {
    /// Number of nodes.
    pub fn len(&self) -> usize {
        if self.degree == 0 {
            0
        } else {
            self.adjacency.len() / self.degree as usize
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Node `node`'s out-neighbours (its row up to the first [`NO_NEIGHBOR`]).
    pub fn neighbors(&self, node: u32) -> &[u32] {
        let width = self.degree as usize;
        let row = &self.adjacency[node as usize * width..(node as usize + 1) * width];
        let real = row.iter().position(|&n| n == NO_NEIGHBOR).unwrap_or(width);
        &row[..real]
    }

    /// The graph as saved in a segment's `Graph` section: degree (u32), entry (u32),
    /// node count (u64), then the adjacency (u32 each), little-endian.
    pub fn to_bytes(&self) -> Vec<u8> {
        todo!("U15 vamana-build: Graph::to_bytes")
    }

    /// The inverse of [`Graph::to_bytes`] for a segment of `nodes` ordinals. `Corrupt`
    /// (empty path) when the length, the node count, the entry or a neighbour id is out
    /// of range, or a row repeats a neighbour or names its own node.
    pub fn from_bytes(bytes: &[u8], nodes: usize) -> Result<Graph> {
        todo!("U15 vamana-build: Graph::from_bytes")
    }
}
