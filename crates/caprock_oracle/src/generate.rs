//! Seeded generators: vectors of several shapes, keys, paths with known filter
//! selectivities, standard filters, and operation sequences.
//!
//! Vector `i` of a spec depends only on `(seed, i)` (and the shape's shared parameters,
//! such as cluster centres), so a set of `n` vectors is a prefix of a set of `m > n`.

use crate::filter::Filter;
use crate::rng::{Rng, mix};

/// The distribution of generated vectors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Shape {
    /// Independent standard normal components.
    Gaussian,
    /// A mixture: `clusters` centres (standard normal), points at centre + `spread` x
    /// standard normal.
    Clustered { clusters: usize, spread: f32 },
    /// Exact copies of `distinct` Gaussian prototypes: many exactly tied scores.
    Duplicates { distinct: usize },
    /// Integer components in `-levels..=levels`: many exact ties for dot and L2.
    Lattice { levels: u32 },
    /// A shared offset plus small noise: every cosine is high, as with some text
    /// embedding models; hard for quantizers.
    Anisotropic,
}

impl Shape {
    /// A short name for reports: `gaussian`, `clustered`, ...
    pub fn name(&self) -> &'static str {
        match self {
            Shape::Gaussian => "gaussian",
            Shape::Clustered { .. } => "clustered",
            Shape::Duplicates { .. } => "duplicates",
            Shape::Lattice { .. } => "lattice",
            Shape::Anisotropic => "anisotropic",
        }
    }
}

/// A family of vectors: dimension, shape and seed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VectorSpec {
    pub dim: usize,
    pub shape: Shape,
    pub seed: u64,
}

const STREAM_CENTRES: u64 = 1 << 40;
const STREAM_QUERIES: u64 = 1 << 41;
const STREAM_OFFSET: u64 = 1 << 42;

fn gaussian_vector(rng: &mut Rng, dim: usize, scale: f32) -> Vec<f32> {
    (0..dim).map(|_| rng.gaussian() as f32 * scale).collect()
}

impl VectorSpec {
    fn centre(&self, c: usize) -> Vec<f32> {
        let mut rng = Rng::derive(self.seed, STREAM_CENTRES + c as u64);
        gaussian_vector(&mut rng, self.dim, 1.0)
    }

    /// Vector `i` (a pure function of the spec and `i`).
    pub fn vector(&self, i: u64) -> Vec<f32> {
        let mut rng = Rng::derive(self.seed, i);
        match self.shape {
            Shape::Gaussian => gaussian_vector(&mut rng, self.dim, 1.0),
            Shape::Clustered { clusters, spread } => {
                let c = rng.below(clusters.max(1) as u64) as usize;
                let centre = self.centre(c);
                centre
                    .iter()
                    .map(|x| x + rng.gaussian() as f32 * spread)
                    .collect()
            }
            Shape::Duplicates { distinct } => {
                let p = rng.below(distinct.max(1) as u64);
                let mut proto = Rng::derive(self.seed, STREAM_CENTRES + p);
                gaussian_vector(&mut proto, self.dim, 1.0)
            }
            Shape::Lattice { levels } => {
                let span = 2 * levels as u64 + 1;
                let v: Vec<f32> = (0..self.dim)
                    .map(|_| rng.below(span) as f32 - levels as f32)
                    .collect();
                if v.iter().all(|x| *x == 0.0) {
                    let mut v = v;
                    v[0] = 1.0;
                    v
                } else {
                    v
                }
            }
            Shape::Anisotropic => {
                let mut off = Rng::derive(self.seed, STREAM_OFFSET);
                let offset = gaussian_vector(&mut off, self.dim, 1.0);
                offset
                    .iter()
                    .map(|x| x + rng.gaussian() as f32 * 0.25)
                    .collect()
            }
        }
    }

    /// Vectors `0..n`.
    pub fn vectors(&self, n: usize) -> Vec<Vec<f32>> {
        (0..n as u64).map(|i| self.vector(i)).collect()
    }

    /// `n` queries: mostly fresh draws from the same distribution; every fifth is a
    /// copy of a base vector (`0..base_n`) with a small perturbation, and every
    /// twenty-fifth an exact copy (exact ties with a stored vector).
    pub fn queries(&self, n: usize, base_n: usize) -> Vec<Vec<f32>> {
        (0..n as u64)
            .map(|j| {
                let mut rng = Rng::derive(self.seed, STREAM_QUERIES + j);
                if base_n > 0 && j % 25 == 0 {
                    self.vector(rng.below(base_n as u64))
                } else if base_n > 0 && j % 5 == 0 {
                    let mut v = self.vector(rng.below(base_n as u64));
                    for x in v.iter_mut() {
                        *x += rng.gaussian() as f32 * 0.05;
                    }
                    v
                } else {
                    self.vector(STREAM_QUERIES + j)
                }
            })
            .collect()
    }
}

/// The key of record `i`: `k` and 16 hex digits of a bijective mix of `i`, so key
/// order is unrelated to insertion order.
pub fn key(i: u64) -> String {
    format!("k{:016x}", mix(i))
}

/// The path classes and their shares of records: `(top directory, share)`.
pub const PATH_CLASSES: [(&str, f64); 5] = [
    ("p50", 0.5),
    ("p10", 0.1),
    ("p01", 0.01),
    ("p001", 0.001),
    ("rest", 0.389),
];

/// Languages, their shares and file extensions.
pub const LANGUAGES: [(&str, f64, &str); 4] = [
    ("rust", 0.4, "rs"),
    ("python", 0.3, "py"),
    ("typescript", 0.2, "ts"),
    ("go", 0.1, "go"),
];

fn pick<T: Copy>(u: f64, table: &[(T, f64)]) -> T {
    let mut acc = 0.0;
    for (item, share) in table {
        acc += share;
        if u < acc {
            return *item;
        }
    }
    table[table.len() - 1].0
}

/// The path and language of record `i`, independent of its vector: the top directory
/// follows [`PATH_CLASSES`], the language [`LANGUAGES`]. Paths look like
/// `p10/d17/f0421.py`.
pub fn path_for(i: u64, seed: u64) -> (String, String) {
    let mut rng = Rng::derive(seed ^ 0x5041_5448, i);
    let class = pick(rng.next_f64(), &PATH_CLASSES);
    let lang_table: Vec<(usize, f64)> = LANGUAGES
        .iter()
        .enumerate()
        .map(|(n, (_, s, _))| (n, *s))
        .collect();
    let lang = pick(rng.next_f64(), &lang_table);
    let (language, _, ext) = LANGUAGES[lang];
    let dir = rng.below(37);
    let file = rng.below(1013);
    (
        format!("{class}/d{dir}/f{file:04}.{ext}"),
        language.to_string(),
    )
}

/// The standard filters, by name, with their expected selectivity on [`path_for`]
/// records.
pub fn standard_filters() -> Vec<(&'static str, Filter, f64)> {
    let f = |prefixes: &[&str], languages: &[&str]| Filter {
        path_prefixes: prefixes.iter().map(|s| s.to_string()).collect(),
        languages: languages.iter().map(|s| s.to_string()).collect(),
    };
    vec![
        ("none", Filter::all(), 1.0),
        ("p50", f(&["p50"], &[]), 0.5),
        ("p10", f(&["p10"], &[]), 0.1),
        ("p01", f(&["p01"], &[]), 0.01),
        ("p001", f(&["p001"], &[]), 0.001),
        ("go", f(&[], &["go"]), 0.1),
        ("p10-python", f(&["p10"], &["python"]), 0.03),
        ("p01-or-p001", f(&["p01", "p001/"], &[]), 0.011),
        ("p50-d7", f(&["p50/d7"], &[]), 0.5 / 37.0),
        ("nothing", f(&["zzz"], &[]), 0.0),
    ]
}

/// One step of an operation sequence.
#[derive(Debug, Clone, PartialEq)]
pub enum Step {
    /// Upserts and deletes, applied as one batch, in order.
    Apply(Vec<Op>),
    /// Make everything applied so far durable.
    Commit,
    /// Drop the index and open it again (uncommitted operations are lost).
    Reopen,
    /// Run maintenance (merges) to completion.
    Maintain,
    /// Merge everything into as few segments as the memory budget allows.
    Compact,
    /// Run the query set and check the results.
    Check,
}

/// An upsert or a delete.
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Upsert {
        key: String,
        vector: Vec<f32>,
        path: String,
        language: String,
    },
    Delete {
        key: String,
    },
}

/// Parameters of a random operation sequence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpsSpec {
    pub vectors: VectorSpec,
    /// Distinct keys used (smaller: more replacements and deletes of live keys).
    pub key_space: u64,
    pub steps: usize,
    /// Largest batch.
    pub max_batch: usize,
    pub p_delete: f64,
    pub p_commit: f64,
    pub p_reopen: f64,
    pub p_maintain: f64,
    pub p_compact: f64,
    pub p_check: f64,
    pub seed: u64,
}

impl OpsSpec {
    /// A reasonable default mix for `dim` and `seed`.
    pub fn mixed(dim: usize, seed: u64) -> OpsSpec {
        OpsSpec {
            vectors: VectorSpec {
                dim,
                shape: Shape::Clustered {
                    clusters: 16,
                    spread: 0.3,
                },
                seed,
            },
            key_space: 3000,
            steps: 40,
            max_batch: 400,
            p_delete: 0.2,
            p_commit: 0.15,
            p_reopen: 0.05,
            p_maintain: 0.08,
            p_compact: 0.03,
            p_check: 0.2,
            seed,
        }
    }

    /// The sequence. Every upsert of key `j` uses a fresh vector draw, so a replaced
    /// record really changes. A final `Check` always ends the sequence.
    pub fn steps(&self) -> Vec<Step> {
        let mut rng = Rng::derive(self.seed, 0x0F5);
        let mut draws = 0u64;
        let mut out = Vec::new();
        for _ in 0..self.steps {
            let u = rng.next_f64();
            let mut t = self.p_commit;
            if u < t {
                out.push(Step::Commit);
                continue;
            }
            t += self.p_reopen;
            if u < t {
                out.push(Step::Reopen);
                continue;
            }
            t += self.p_maintain;
            if u < t {
                out.push(Step::Maintain);
                continue;
            }
            t += self.p_compact;
            if u < t {
                out.push(Step::Compact);
                continue;
            }
            t += self.p_check;
            if u < t {
                out.push(Step::Check);
                continue;
            }
            let size = 1 + rng.below(self.max_batch.max(1) as u64) as usize;
            let mut ops = Vec::with_capacity(size);
            for _ in 0..size {
                let j = rng.below(self.key_space.max(1));
                if rng.chance(self.p_delete) {
                    ops.push(Op::Delete { key: key(j) });
                } else {
                    let vector = self.vectors.vector(draws);
                    draws += 1;
                    let (path, language) = path_for(j, self.seed);
                    ops.push(Op::Upsert {
                        key: key(j),
                        vector,
                        path,
                        language,
                    });
                }
            }
            out.push(Step::Apply(ops));
        }
        out.push(Step::Check);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_property_and_determinism() {
        let spec = VectorSpec {
            dim: 8,
            shape: Shape::Clustered {
                clusters: 4,
                spread: 0.1,
            },
            seed: 7,
        };
        let a = spec.vectors(10);
        let b = spec.vectors(20);
        assert_eq!(a[..], b[..10]);
        assert_eq!(spec.queries(5, 10), spec.queries(5, 10));
    }

    #[test]
    fn path_shares_are_close_to_the_table() {
        let n = 200_000u64;
        let p10 = (0..n)
            .filter(|&i| path_for(i, 1).0.starts_with("p10/"))
            .count();
        let go = (0..n).filter(|&i| path_for(i, 1).1 == "go").count();
        let share = |c: usize| c as f64 / n as f64;
        assert!((share(p10) - 0.1).abs() < 0.005, "{}", share(p10));
        assert!((share(go) - 0.1).abs() < 0.005, "{}", share(go));
    }

    #[test]
    fn keys_are_distinct() {
        let mut keys: Vec<String> = (0..10_000).map(key).collect();
        keys.sort();
        keys.dedup();
        assert_eq!(keys.len(), 10_000);
    }
}
