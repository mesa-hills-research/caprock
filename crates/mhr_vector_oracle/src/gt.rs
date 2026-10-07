//! Batch ground truth: for each query, the exact top `k` base rows under the canonical
//! arithmetic and the (score descending, key ascending) order, optionally restricted to
//! rows a mask allows. The same arithmetic as [`crate::Oracle::search`], spread over
//! threads by query; a test checks they agree.

use std::io::{self, Read, Write};
use std::path::Path;

use crate::canonical::{self, Metric, VectorProblem};
use crate::oracle::hit_order;

/// Puts raw rows (flattened, `dim` each) in stored form in place: validated, and
/// normalized for cosine. Returns the first bad row.
pub fn prepare_rows(
    metric: Metric,
    dim: usize,
    rows: &mut [f32],
) -> Result<(), (usize, VectorProblem)> {
    for (i, row) in rows.chunks_exact_mut(dim).enumerate() {
        let prepared = canonical::prepare(metric, row, dim).map_err(|p| (i, p))?;
        row.copy_from_slice(&prepared);
    }
    Ok(())
}

/// One query's exact list: `(row, score)` pairs in result order.
pub type TruthList = Vec<(u32, f32)>;

/// The exact top `k` of each query over prepared `rows` (flattened). `keys[i]` breaks
/// ties; `mask[i] == false` excludes row `i`.
#[allow(clippy::too_many_arguments)]
pub fn ground_truth(
    metric: Metric,
    dim: usize,
    rows: &[f32],
    keys: &[String],
    mask: Option<&[bool]>,
    queries: &[Vec<f32>],
    k: usize,
    threads: usize,
) -> Result<Vec<TruthList>, VectorProblem> {
    assert!(k > 0, "k must be at least 1");
    let n = rows.len() / dim;
    assert_eq!(keys.len(), n, "one key per row");
    if let Some(m) = mask {
        assert_eq!(m.len(), n, "one mask entry per row");
    }
    let prepared: Vec<Vec<f32>> = queries
        .iter()
        .map(|q| canonical::prepare(metric, q, dim))
        .collect::<Result<_, _>>()?;
    let threads = threads.max(1).min(prepared.len().max(1));
    let per = prepared.len().div_ceil(threads).max(1);
    let mut out: Vec<TruthList> = vec![Vec::new(); prepared.len()];
    std::thread::scope(|scope| {
        for (chunk_out, chunk_q) in out.chunks_mut(per).zip(prepared.chunks(per)) {
            scope.spawn(move || {
                for (slot, q) in chunk_out.iter_mut().zip(chunk_q) {
                    *slot = top_k(metric, dim, rows, keys, mask, q, k);
                }
            });
        }
    });
    Ok(out)
}

fn top_k(
    metric: Metric,
    dim: usize,
    rows: &[f32],
    keys: &[String],
    mask: Option<&[bool]>,
    q: &[f32],
    k: usize,
) -> TruthList {
    // A sorted list of at most k entries; a new row enters when it beats the last.
    let mut best: TruthList = Vec::with_capacity(k + 1);
    for (i, row) in rows.chunks_exact(dim).enumerate() {
        if let Some(m) = mask
            && !m[i]
        {
            continue;
        }
        let s = canonical::score(metric, q, row);
        let key = keys[i].as_str();
        if best.len() == k {
            let (wi, ws) = best[k - 1];
            if hit_order(s, key, ws, &keys[wi as usize]) != std::cmp::Ordering::Less {
                continue;
            }
        }
        let pos = best.partition_point(|&(bi, bs)| {
            hit_order(bs, &keys[bi as usize], s, key) == std::cmp::Ordering::Less
        });
        best.insert(pos, (i as u32, s));
        best.truncate(k);
    }
    best
}

const MAGIC: &[u8; 8] = b"MHRVGT01";

/// Writes ground truth: magic, `k` (u32), query count (u32), base rows (u64), then
/// per query `k` pairs of (row u32, score f32), padded with `(u32::MAX, 0.0)`.
pub fn write_truth(path: &Path, k: usize, base_rows: u64, lists: &[TruthList]) -> io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("tmp");
    {
        let mut w = io::BufWriter::new(std::fs::File::create(&tmp)?);
        w.write_all(MAGIC)?;
        w.write_all(&(k as u32).to_le_bytes())?;
        w.write_all(&(lists.len() as u32).to_le_bytes())?;
        w.write_all(&base_rows.to_le_bytes())?;
        for list in lists {
            for j in 0..k {
                let (row, score) = list.get(j).copied().unwrap_or((u32::MAX, 0.0));
                w.write_all(&row.to_le_bytes())?;
                w.write_all(&score.to_le_bytes())?;
            }
        }
        w.flush()?;
    }
    std::fs::rename(tmp, path)
}

/// Reads [`write_truth`]'s file: `(k, base_rows, lists)`.
pub fn read_truth(path: &Path) -> io::Result<(usize, u64, Vec<TruthList>)> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut bytes)?;
    let bad = || {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: corrupt", path.display()),
        )
    };
    if bytes.len() < 24 || &bytes[..8] != MAGIC {
        return Err(bad());
    }
    let k = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let nq = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let base_rows = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    if bytes.len() != 24 + nq * k * 8 {
        return Err(bad());
    }
    let mut lists = Vec::with_capacity(nq);
    for q in 0..nq {
        let mut list = Vec::with_capacity(k);
        for j in 0..k {
            let at = 24 + (q * k + j) * 8;
            let row = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
            let score = f32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap());
            if row != u32::MAX {
                list.push((row, score));
            }
        }
        lists.push(list);
    }
    Ok((k, base_rows, lists))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::Filter;
    use crate::generate::{Shape, VectorSpec, key, path_for};
    use crate::oracle::Oracle;

    #[test]
    fn agrees_with_the_oracle() {
        for (metric, shape) in [
            (Metric::Cosine, Shape::Gaussian),
            (Metric::Dot, Shape::Lattice { levels: 2 }),
            (Metric::L2, Shape::Duplicates { distinct: 30 }),
        ] {
            let spec = VectorSpec {
                dim: 19,
                shape,
                seed: 3,
            };
            let n = 700;
            let base = spec.vectors(n);
            let queries = spec.queries(40, n);
            let keys: Vec<String> = (0..n as u64).map(key).collect();
            let mut oracle = Oracle::new(metric, spec.dim);
            let filter = Filter {
                path_prefixes: vec!["p50".into()],
                languages: vec![],
            };
            let mut mask = Vec::new();
            for (i, v) in base.iter().enumerate() {
                let (path, language) = path_for(i as u64, 9);
                mask.push(filter.matches(&path, &language));
                oracle.upsert(&keys[i], v, &path, &language).unwrap();
            }
            let mut rows: Vec<f32> = base.concat();
            prepare_rows(metric, spec.dim, &mut rows).unwrap();
            for (m, f) in [(None, Filter::all()), (Some(&mask[..]), filter.clone())] {
                let lists =
                    ground_truth(metric, spec.dim, &rows, &keys, m, &queries, 10, 3).unwrap();
                for (q, list) in queries.iter().zip(&lists) {
                    let want = oracle.search(q, 10, &f).unwrap();
                    let got: Vec<(String, f32)> = list
                        .iter()
                        .map(|&(i, s)| (keys[i as usize].clone(), s))
                        .collect();
                    let want: Vec<(String, f32)> =
                        want.into_iter().map(|h| (h.key, h.score)).collect();
                    assert_eq!(got, want, "{metric:?} {shape:?}");
                }
            }
        }
    }

    #[test]
    fn truth_files_round_trip() {
        let dir = std::env::temp_dir().join(format!("mhrv-gt-{}", std::process::id()));
        let path = dir.join("t.bin");
        let lists = vec![vec![(3u32, 0.5f32), (1, 0.25)], vec![(7, -1.0)]];
        write_truth(&path, 2, 10, &lists).unwrap();
        assert_eq!(read_truth(&path).unwrap(), (2, 10, lists));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
