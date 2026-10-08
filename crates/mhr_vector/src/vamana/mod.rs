//! Vamana graphs (units U14 build, U15 search; docs/requirements.md, "Approximate search").
//!
//! A segment's graph is a fixed-degree adjacency array over its ordinals: row `i` holds
//! node `i`'s out-neighbours followed by [`NO_NEIGHBOR`] padding. Construction is
//! DiskANN's Vamana (greedy search + robust prune with `alpha`), made deterministic
//! (docs/requirements.md, "Determinism"): the same codes, parameters and seed give the same
//! graph for any thread count.

pub mod build;
pub mod search;

pub use build::build_graph;
pub use search::{SearchScratch, beam_search};

use crate::error::{Error, Result};

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
        let mut bytes = Vec::with_capacity(16 + self.adjacency.len() * 4);
        bytes.extend_from_slice(&self.degree.to_le_bytes());
        bytes.extend_from_slice(&self.entry.to_le_bytes());
        bytes.extend_from_slice(&(self.len() as u64).to_le_bytes());
        for neighbor in &self.adjacency {
            bytes.extend_from_slice(&neighbor.to_le_bytes());
        }
        bytes
    }

    /// The inverse of [`Graph::to_bytes`] for a segment of `nodes` ordinals. `Corrupt`
    /// (empty path) when the length, the node count, the entry or a neighbour id is out
    /// of range, or a row repeats a neighbour or names its own node.
    pub fn from_bytes(bytes: &[u8], nodes: usize) -> Result<Graph> {
        let corrupt = |detail| Error::corrupt("", detail);
        if bytes.len() < 16 {
            return Err(corrupt("truncated graph header"));
        }
        let degree = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        let entry = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let saved_nodes = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        if saved_nodes != nodes as u64 || nodes > NO_NEIGHBOR as usize {
            return Err(corrupt("graph node count mismatch or out of range"));
        }
        if (nodes == 0 && entry != 0) || (nodes != 0 && (degree == 0 || entry as usize >= nodes)) {
            return Err(corrupt("graph degree or entry out of range"));
        }
        let cells = nodes
            .checked_mul(degree as usize)
            .ok_or_else(|| corrupt("graph size overflow"))?;
        let expected = cells
            .checked_mul(4)
            .and_then(|size| size.checked_add(16))
            .ok_or_else(|| corrupt("graph size overflow"))?;
        if bytes.len() != expected {
            return Err(corrupt("graph adjacency length mismatch"));
        }

        // Check shape before allocating, including for empty graphs with a huge width.
        let adjacency: Vec<u32> = bytes[16..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&word| u32::from_le_bytes(word))
            .collect();
        if nodes != 0 {
            let mut neighbors = Vec::new();
            for (node, row) in adjacency.chunks(degree as usize).enumerate() {
                neighbors.clear();
                let mut padding = false;
                for &neighbor in row {
                    if neighbor == NO_NEIGHBOR {
                        padding = true;
                    } else {
                        if padding || neighbor as usize >= nodes || neighbor as usize == node {
                            return Err(corrupt("invalid graph neighbor or non-terminal padding"));
                        }
                        neighbors.push(neighbor);
                    }
                }
                neighbors.sort_unstable();
                if neighbors.windows(2).any(|pair| pair[0] == pair[1]) {
                    return Err(corrupt("repeated graph neighbor"));
                }
            }
        }
        Ok(Graph {
            degree,
            entry,
            adjacency,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_corrupt(bytes: &[u8], nodes: usize) {
        assert!(matches!(
            Graph::from_bytes(bytes, nodes),
            Err(Error::Corrupt { path, .. }) if path.as_os_str().is_empty()
        ));
    }

    #[test]
    fn empty_graphs_round_trip_and_nonempty_graphs_need_a_degree() {
        for degree in [0, 8, u32::MAX] {
            let graph = Graph {
                degree,
                ..Graph::default()
            };
            assert_eq!(Graph::from_bytes(&graph.to_bytes(), 0).unwrap(), graph);
            let mut bytes = graph.to_bytes();
            bytes[4..8].copy_from_slice(&1u32.to_le_bytes());
            assert_corrupt(&bytes, 0);
        }
        let mut bytes = Graph::default().to_bytes();
        bytes[8..16].copy_from_slice(&1u64.to_le_bytes());
        assert_corrupt(&bytes, 1);
    }

    #[test]
    fn decoding_rejects_every_truncation_extra_bytes_and_bad_padding() {
        let graph = Graph {
            degree: 2,
            entry: 0,
            adjacency: vec![1, NO_NEIGHBOR, 0, NO_NEIGHBOR],
        };
        let mut bytes = graph.to_bytes();
        for len in 0..bytes.len() {
            assert_corrupt(&bytes[..len], 2);
        }
        bytes.push(0);
        assert_corrupt(&bytes, 2);
        let hole = Graph {
            adjacency: vec![NO_NEIGHBOR, 1, 0, NO_NEIGHBOR],
            ..graph
        };
        assert_corrupt(&hole.to_bytes(), 2);
    }

    #[test]
    fn decoding_checks_untrusted_sizes_before_allocating() {
        let mut bytes = [0u8; 16];
        bytes[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        for nodes in [1, u32::MAX as usize, usize::MAX] {
            bytes[8..16].copy_from_slice(&(nodes as u64).to_le_bytes());
            assert_corrupt(&bytes, nodes);
        }
    }
}
