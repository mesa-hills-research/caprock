//! The harness's commands. Each returns a JSON value (printed by `main`).

use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

use caprock::{Filter, StoreOptions};
use caprock_oracle::dataset::{DatasetInfo, FORMAT, f32s_to_le};
use caprock_oracle::generate::{Shape, VectorSpec};
use caprock_oracle::rng::Rng;
use serde_json::{Value, json};

use crate::args::Args;
use crate::configs::{BuildOptions, ConfigRef, ConfigSpec, Part, Workload, datasets_root};
use crate::data::{Judge, Slice, named_filter};
use crate::engine::{self, EngineKind, Rec, Searcher};
use crate::measure::{self, median, quantile, secs};

/// Ground-truth depth kept in the cache (recall@k for any k up to it).
pub const GT_K: usize = 100;

type R<T> = Result<T, String>;

fn engine_kind(a: &Args) -> R<EngineKind> {
    EngineKind::parse(a.get("engine").unwrap_or("caprock"))
}

fn config_arg(a: &Args) -> R<ConfigSpec> {
    let raw = a.get("config").unwrap_or("vamana-int8");
    if raw.trim_start().starts_with('{') {
        serde_json::from_str::<ConfigSpec>(raw).map_err(|e| format!("--config: {e}"))
    } else {
        ConfigRef::Name(raw.to_string()).resolve()
    }
}

fn build_options(a: &Args) -> R<BuildOptions> {
    let d = BuildOptions::default();
    Ok(BuildOptions {
        threads: a.num("threads", d.threads)?,
        budget: a.num("budget", d.budget)?,
        batch: a.num("batch", d.batch)?,
        commit_every: a.num("commit-every", d.commit_every)?,
        delta_max_vectors: a.num("delta", d.delta_max_vectors)?,
    })
}

/// `synth`: writes a synthetic dataset, streaming (any size).
pub fn synth(a: &Args) -> R<Value> {
    let out = PathBuf::from(a.required("out")?);
    let dim: usize = a.num("dim", 128)?;
    let n: usize = a.num("n", 100_000)?;
    let nq: usize = a.num("queries", 1000)?;
    let seed: u64 = a.num("seed", 1)?;
    let metric = a.get("metric").unwrap_or("cosine").to_string();
    let shape = match a.get("shape").unwrap_or("clustered") {
        "gaussian" => Shape::Gaussian,
        "clustered" => Shape::Clustered {
            clusters: a.num("clusters", 1000)?,
            spread: a.num("spread", 0.35)?,
        },
        "duplicates" => Shape::Duplicates {
            distinct: a.num("distinct", n / 4 + 1)?,
        },
        "anisotropic" => Shape::Anisotropic,
        s => return Err(format!("unknown shape {s:?}")),
    };
    let spec = VectorSpec { dim, shape, seed };
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let mut w = std::io::BufWriter::new(
        std::fs::File::create(out.join("base.f32")).map_err(|e| e.to_string())?,
    );
    for i in 0..n as u64 {
        w.write_all(&f32s_to_le(&spec.vector(i)))
            .map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())?;
    let mut w = std::io::BufWriter::new(
        std::fs::File::create(out.join("queries.f32")).map_err(|e| e.to_string())?,
    );
    for q in spec.queries(nq, n) {
        w.write_all(&f32s_to_le(&q)).map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())?;
    let info = DatasetInfo {
        format: FORMAT.into(),
        name: a.get("name").map(str::to_string).unwrap_or_else(|| {
            out.file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default()
        }),
        metric,
        dim,
        base_count: n,
        query_count: nq,
        base: "base.f32".into(),
        queries: "queries.f32".into(),
        meta: None,
        query_texts: None,
        neighbors: None,
        neighbors_k: 0,
        source: json!({"generator": "caprock_bench synth", "shape": shape.name(), "seed": seed}),
    };
    std::fs::write(
        out.join("dataset.json"),
        serde_json::to_string_pretty(&info).map_err(|e| e.to_string())? + "\n",
    )
    .map_err(|e| e.to_string())?;
    Ok(json!({"dataset": out, "base": n, "queries": nq, "dim": dim}))
}

/// `gt`: computes (and caches) ground truth; cross-checks the source's neighbours.
pub fn gt(a: &Args) -> R<Value> {
    let root = datasets_root(a.get("datasets"));
    let slice = Slice::open(
        &root,
        a.required("dataset")?,
        a.num("base", 0)?,
        a.num("queries", 2000)?,
    )?;
    let threads: usize = a.num("threads", 8)?;
    let mut filters = vec!["none".to_string()];
    if let Some(f) = a.get("filters") {
        filters.extend(f.split(',').filter(|s| !s.is_empty()).map(str::to_string));
    }
    let mut out = serde_json::Map::new();
    for name in &filters {
        let started = Instant::now();
        let truth = slice.ground_truth(
            name,
            &named_filter(name)?,
            slice.base,
            &HashSet::new(),
            GT_K,
            threads,
        )?;
        let mut entry = json!({"seconds": secs(started), "queries": truth.len()});
        if name == "none"
            && slice.base == slice.ds.info.base_count
            && let Some(source) = slice
                .ds
                .source_neighbors(slice.queries.len())
                .map_err(|e| e.to_string())?
        {
            // Agreement on the first 10 neighbours (ids; ties may legitimately differ).
            let k = 10.min(slice.ds.info.neighbors_k);
            let mut agree = 0.0;
            for (t, s) in truth.iter().zip(&source) {
                let ours: HashSet<u32> = t.iter().take(k).map(|x| x.0).collect();
                agree += s
                    .iter()
                    .take(k)
                    .filter(|&&x| ours.contains(&(x as u32)))
                    .count() as f64
                    / k as f64;
            }
            entry["source_agreement_at_10"] = json!(agree / truth.len() as f64);
        }
        out.insert(name.clone(), entry);
    }
    Ok(json!({"dataset": slice.name(), "base": slice.base, "ground_truth": out}))
}

/// Loads the slice's rows into an index in batches, committing as configured; then
/// compacts. Returns timings.
fn load(
    slice: &Slice,
    index: &mut dyn engine::EngineIndex,
    rows: std::ops::Range<usize>,
    opts: &BuildOptions,
) -> R<(f64, f64)> {
    let started = Instant::now();
    let dim = slice.dim();
    let mut since_commit = 0;
    let mut start = rows.start;
    while start < rows.end {
        let count = opts.batch.min(rows.end - start);
        let flat = slice.read_rows(start, count)?;
        let meta: Vec<(String, String, String)> =
            (start..start + count).map(|i| slice.record(i)).collect();
        let recs: Vec<Rec<'_>> = meta
            .iter()
            .enumerate()
            .map(|(j, (k, p, l))| Rec {
                key: k,
                vector: &flat[j * dim..(j + 1) * dim],
                path: p,
                language: l,
            })
            .collect();
        index.apply(&recs, &[])?;
        since_commit += count;
        if since_commit >= opts.commit_every {
            index.commit()?;
            since_commit = 0;
        }
        start += count;
    }
    index.commit()?;
    let load_seconds = secs(started);
    let started = Instant::now();
    index.compact()?;
    index.commit()?;
    Ok((load_seconds, secs(started)))
}

/// `build`: builds an index from a dataset slice (meant to run in its own process, so
/// its peak memory is the build's).
pub fn build(a: &Args) -> R<Value> {
    let root = datasets_root(a.get("datasets"));
    let slice = Slice::open(&root, a.required("dataset")?, a.num("base", 0)?, 1)?;
    let spec = config_arg(a)?;
    let opts = build_options(a)?;
    let dir = PathBuf::from(a.required("index")?);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).map_err(|e| e.to_string())?;
    }
    let config = spec.index_config(slice.dim() as u32, slice.store_metric())?;
    let baseline_rss = measure::current_rss();
    let mut index = engine::create(engine_kind(a)?, &dir, config, opts.store_options())?;
    let rows = a.num("rows", slice.base)?.min(slice.base);
    let (load_seconds, compact_seconds) = load(&slice, index.as_mut(), 0..rows, &opts)?;
    Ok(json!({
        "dataset": slice.name(),
        "config": spec.name,
        "rows": rows,
        "load_seconds": load_seconds,
        "compact_seconds": compact_seconds,
        "build_seconds": load_seconds + compact_seconds,
        "vectors_per_second": rows as f64 / (load_seconds + compact_seconds).max(1e-9),
        "baseline_rss": baseline_rss,
        "peak_rss": measure::peak_rss(),
        "disk_bytes": measure::dir_bytes(&dir),
        "stats": index.stats(),
    }))
}

/// Recall over all queries at one beam, and the rules any result broke.
#[derive(Debug, Clone)]
struct BeamResult {
    recall: f64,
    violations: usize,
    first_violation: Option<String>,
}

/// Recall of every query at `beam`, on `threads` threads (untimed).
fn recall_pass(
    searcher: &Arc<dyn Searcher>,
    judge: &Judge<'_>,
    k: usize,
    beam: Option<u32>,
    threads: usize,
) -> R<BeamResult> {
    let n = judge.slice.queries.len();
    let threads = threads.max(1).min(n.max(1));
    let per = n.div_ceil(threads).max(1);
    let filter = if judge.filter.is_empty() {
        None
    } else {
        Some(&judge.filter)
    };
    let results: Vec<R<Vec<crate::data::Verdict>>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let searcher = searcher.clone();
                scope.spawn(move || {
                    let mut out = Vec::new();
                    for q in t * per..((t + 1) * per).min(n) {
                        let hits =
                            searcher.search(&judge.slice.queries[q], k, filter, beam, false)?;
                        out.push(judge.judge(q, &hits, k));
                    }
                    Ok(out)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut recalls = Vec::new();
    let mut violations = 0;
    let mut first = None;
    for r in results {
        for v in r? {
            recalls.push(v.recall);
            if let Some(msg) = v.violation {
                violations += 1;
                first.get_or_insert(msg);
            }
        }
    }
    Ok(BeamResult {
        recall: recalls.iter().sum::<f64>() / recalls.len().max(1) as f64,
        violations,
        first_violation: first,
    })
}

/// Timed single-thread passes: (median-pass QPS, p50 ms, p99 ms of that pass).
fn timed_pass(
    searcher: &Arc<dyn Searcher>,
    queries: &[Vec<f32>],
    k: usize,
    filter: Option<&Filter>,
    beam: Option<u32>,
    repeats: usize,
) -> R<(f64, f64, f64)> {
    let mut passes: Vec<(f64, Vec<f64>)> = Vec::new();
    // One warm-up pass.
    for q in queries {
        searcher.search(q, k, filter, beam, false)?;
    }
    for _ in 0..repeats.max(1) {
        let mut lat = Vec::with_capacity(queries.len());
        let started = Instant::now();
        for q in queries {
            let t = Instant::now();
            std::hint::black_box(searcher.search(q, k, filter, beam, false)?);
            lat.push(secs(t) * 1000.0);
        }
        passes.push((queries.len() as f64 / secs(started), lat));
    }
    passes.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (qps, lat) = &passes[(passes.len() - 1) / 2];
    Ok((*qps, quantile(lat, 0.5), quantile(lat, 0.99)))
}

/// QPS with `threads` threads each running the whole query list once.
fn parallel_qps(
    searcher: &Arc<dyn Searcher>,
    queries: &[Vec<f32>],
    k: usize,
    beam: Option<u32>,
    threads: usize,
) -> R<f64> {
    let started = Instant::now();
    let results: Vec<R<()>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let searcher = searcher.clone();
                scope.spawn(move || {
                    for i in 0..queries.len() {
                        let q = &queries[(i + t * 97) % queries.len()];
                        std::hint::black_box(searcher.search(q, k, None, beam, false)?);
                    }
                    Ok(())
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for r in results {
        r?;
    }
    Ok((threads * queries.len()) as f64 / secs(started))
}

fn judge_for<'a>(slice: &'a Slice, filter_name: &str, threads: usize) -> R<Judge<'a>> {
    let filter = named_filter(filter_name)?;
    let truth = slice.ground_truth(
        filter_name,
        &filter,
        slice.base,
        &HashSet::new(),
        GT_K,
        threads,
    )?;
    Judge::new(slice, truth, filter, slice.base, HashSet::new())
}

/// Measures queries of one part on an existing index.
fn query_part(slice: &Slice, searcher: &Arc<dyn Searcher>, part: &Part, w: &Workload) -> R<Value> {
    let judge = judge_for(slice, "none", w.recall_threads)?;
    let mut beams: Vec<Option<u32>> = vec![None];
    beams.extend(w.beams.iter().map(|b| Some(*b)));
    let mut rows = Vec::new();
    for beam in beams {
        let r = recall_pass(searcher, &judge, part.k, beam, w.recall_threads)?;
        let (qps, p50, p99) = timed_pass(
            searcher,
            &slice.queries,
            part.k,
            None,
            beam,
            w.timed_repeats,
        )?;
        let mut multi = serde_json::Map::new();
        for &t in &w.qps_threads {
            if t > 1 {
                multi.insert(
                    t.to_string(),
                    json!(parallel_qps(searcher, &slice.queries, part.k, beam, t)?),
                );
            }
        }
        rows.push(json!({
            "beam": beam, "recall": r.recall, "violations": r.violations,
            "first_violation": r.first_violation, "qps_1": qps, "p50_ms": p50, "p99_ms": p99,
            "qps_threads": multi,
        }));
    }
    let mut filtered = Vec::new();
    for name in &part.filters {
        let judge = judge_for(slice, name, w.recall_threads)?;
        let r = recall_pass(searcher, &judge, part.k, None, w.recall_threads)?;
        let (qps, p50, p99) = timed_pass(
            searcher,
            &slice.queries,
            part.k,
            Some(&judge.filter),
            None,
            1,
        )?;
        filtered.push(json!({
            "filter": name, "matching": judge.matching, "recall": r.recall,
            "violations": r.violations, "first_violation": r.first_violation,
            "qps_1": qps, "p50_ms": p50, "p99_ms": p99,
        }));
    }
    Ok(json!({"beams": rows, "filters": filtered}))
}

/// `query`: measures queries on an index built with `build`.
pub fn query(a: &Args) -> R<Value> {
    let root = datasets_root(a.get("datasets"));
    let slice = Slice::open(
        &root,
        a.required("dataset")?,
        a.num("base", 0)?,
        a.num("queries", 2000)?,
    )?;
    let dir = PathBuf::from(a.required("index")?);
    let searcher = engine::open_reader(engine_kind(a)?, &dir, StoreOptions::default())?;
    let part = Part {
        dataset: slice.name().to_string(),
        base: slice.base,
        queries: slice.queries.len(),
        config: ConfigRef::Name("vamana-int8".into()),
        k: a.num("k", 10)?,
        filters: a
            .get("filters")
            .map(|f| f.split(',').map(str::to_string).collect())
            .unwrap_or_default(),
        updates: false,
        open: false,
    };
    let w = Workload {
        name: "query".into(),
        description: String::new(),
        parts: vec![],
        build: BuildOptions::default(),
        timed_repeats: a.num("repeats", 3)?,
        recall_threads: a.num("recall-threads", 4)?,
        qps_threads: a.list("qps-threads", &[1usize])?,
        beams: a.list("beams", &[])?,
    };
    query_part(&slice, &searcher, &part, &w)
}

/// `open`: opens an index in this (fresh) process and runs one query.
pub fn open(a: &Args) -> R<Value> {
    let root = datasets_root(a.get("datasets"));
    let slice = Slice::open(&root, a.required("dataset")?, a.num("base", 0)?, 1)?;
    let dir = PathBuf::from(a.required("index")?);
    let cold = a.flag("drop-caches") && measure::drop_caches();
    let started = Instant::now();
    let searcher = engine::open_reader(engine_kind(a)?, &dir, StoreOptions::default())?;
    let open_seconds = secs(started);
    let started = Instant::now();
    searcher.search(&slice.queries[0], 10, None, None, false)?;
    Ok(json!({
        "cold": cold,
        "open_seconds": open_seconds,
        "first_query_seconds": secs(started),
        "rss_after_open": measure::current_rss(),
        "live": searcher.live(),
    }))
}

/// `update`: on an index built over the first `base` rows (with `--rows` = base minus
/// the held-out share), inserts held-out rows in batches, deletes random rows, and
/// measures throughput, visibility latency and recall afterwards (meant to run in its
/// own process on a scratch copy).
pub fn update(a: &Args) -> R<Value> {
    let root = datasets_root(a.get("datasets"));
    let slice = Slice::open(
        &root,
        a.required("dataset")?,
        a.num("base", 0)?,
        a.num("queries", 500)?,
    )?;
    let dir = PathBuf::from(a.required("index")?);
    let built_rows: usize = a.num("rows", slice.base)?;
    let opts = build_options(a)?;
    let kind = engine_kind(a)?;
    let mut index = engine::open_writer(kind, &dir, opts.store_options())?;
    let dim = slice.dim();
    let mut rng = Rng::new(a.num("seed", 7)?);

    // Visibility: batches of 100 (a watcher's typical burst), apply latency.
    let inserts = slice.base - built_rows;
    let mut small_latencies = Vec::new();
    let mut inserted = 0;
    let started = Instant::now();
    let mut big_batches = 0.0;
    while inserted < inserts {
        let batch = if small_latencies.len() < 50 {
            100
        } else {
            1000
        };
        let count = batch.min(inserts - inserted);
        let start = built_rows + inserted;
        let flat = slice.read_rows(start, count)?;
        let meta: Vec<(String, String, String)> =
            (start..start + count).map(|i| slice.record(i)).collect();
        let recs: Vec<Rec<'_>> = meta
            .iter()
            .enumerate()
            .map(|(j, (k, p, l))| Rec {
                key: k,
                vector: &flat[j * dim..(j + 1) * dim],
                path: p,
                language: l,
            })
            .collect();
        let t = Instant::now();
        index.apply(&recs, &[])?;
        let took = secs(t);
        if batch == 100 {
            small_latencies.push(took * 1000.0);
            // Read-your-writes: the last key must be live at once.
            if !index.searcher().contains(&meta[count - 1].0) {
                return Err(format!(
                    "read-your-writes: {} not live right after apply",
                    meta[count - 1].0
                ));
            }
        } else {
            big_batches += took;
        }
        inserted += count;
    }
    let insert_seconds = secs(started);
    let t = Instant::now();
    index.commit()?;
    let commit_seconds = secs(t);

    // Deletes: a tenth of the live rows, batches of 1000.
    let mut deleted = HashSet::new();
    let target = slice.base / 10;
    while deleted.len() < target {
        deleted.insert(rng.below(slice.base as u64) as usize);
    }
    let mut del_rows: Vec<usize> = deleted.iter().copied().collect();
    del_rows.sort();
    let started = Instant::now();
    for chunk in del_rows.chunks(1000) {
        let keys: Vec<String> = chunk.iter().map(|&i| slice.record(i).0).collect();
        let refs: Vec<&str> = keys.iter().map(String::as_str).collect();
        index.apply(&[], &refs)?;
    }
    let delete_seconds = secs(started);
    index.commit()?;

    let truth = slice.ground_truth(
        &format!("upd{}-{}", built_rows, target),
        &Filter::all(),
        slice.base,
        &deleted,
        GT_K,
        8,
    )?;
    let judge = Judge::new(&slice, truth, Filter::all(), slice.base, deleted)?;
    let k = a.num("k", 10)?;
    let before = recall_pass(&index.searcher(), &judge, k, None, 4)?;
    let t = Instant::now();
    index.maintain()?;
    let maintain_seconds = secs(t);
    let after_maintain = recall_pass(&index.searcher(), &judge, k, None, 4)?;
    let t = Instant::now();
    index.compact()?;
    index.commit()?;
    let compact_seconds = secs(t);
    let after_compact = recall_pass(&index.searcher(), &judge, k, None, 4)?;
    Ok(json!({
        "inserted": inserts,
        "insert_vectors_per_second": inserts as f64 / insert_seconds.max(1e-9),
        "insert_seconds": insert_seconds,
        "large_batch_seconds": big_batches,
        "apply_100_p50_ms": median(&small_latencies),
        "apply_100_p99_ms": quantile(&small_latencies, 0.99),
        "commit_seconds": commit_seconds,
        "deleted": target,
        "deletes_per_second": target as f64 / delete_seconds.max(1e-9),
        "recall_after_updates": before.recall,
        "violations_after_updates": before.violations + after_maintain.violations + after_compact.violations,
        "first_violation": before.first_violation.or(after_maintain.first_violation).or(after_compact.first_violation),
        "maintain_seconds": maintain_seconds,
        "recall_after_maintain": after_maintain.recall,
        "compact_seconds": compact_seconds,
        "recall_after_compact": after_compact.recall,
        "stats": index.stats(),
    }))
}

/// Runs this binary as a child with `args`; returns its JSON output.
pub fn child(args: &[String]) -> R<Value> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let out = Command::new(exe)
        .args(args)
        .output()
        .map_err(|e| e.to_string())?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    if !out.status.success() {
        return Err(format!(
            "child {:?} failed ({}): {}{}",
            args.first(),
            out.status,
            String::from_utf8_lossy(&out.stderr)
                .lines()
                .rev()
                .take(15)
                .collect::<Vec<_>>()
                .join("\n"),
            stdout.lines().last().unwrap_or("")
        ));
    }
    let last = stdout
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with('{'))
        .ok_or("child printed no JSON")?;
    serde_json::from_str(last).map_err(|e| e.to_string())
}

fn common_child_args(a: &Args, part: &Part, w: &Workload) -> R<Vec<String>> {
    let spec = part.config.resolve()?;
    Ok(vec![
        "--datasets".into(),
        datasets_root(a.get("datasets")).display().to_string(),
        "--dataset".into(),
        part.dataset.clone(),
        "--base".into(),
        part.base.to_string(),
        "--engine".into(),
        a.get("engine").unwrap_or("caprock").into(),
        "--config".into(),
        serde_json::to_string(&spec).map_err(|e| e.to_string())?,
        "--threads".into(),
        w.build.threads.to_string(),
        "--budget".into(),
        w.build.budget.to_string(),
        "--batch".into(),
        w.build.batch.to_string(),
        "--commit-every".into(),
        w.build.commit_every.to_string(),
        "--delta".into(),
        w.build.delta_max_vectors.to_string(),
    ])
}

/// `run`: a suite: per part, ground truth, a build (child), an open (child), queries,
/// updates (child). Writes `results.json` and `results.md` into `--out`.
pub fn run(a: &Args) -> R<Value> {
    let path = PathBuf::from(a.required("suite")?);
    let w = Workload::load(&path)?;
    let out = PathBuf::from(a.required("out")?);
    let work = a
        .get("work")
        .map(PathBuf::from)
        .unwrap_or_else(|| out.join("indexes"));
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let root = datasets_root(a.get("datasets"));
    let mut results = Vec::new();
    for (i, part) in w.parts.iter().enumerate() {
        let spec = part.config.resolve()?;
        eprintln!(
            "[{}/{}] {} {}",
            i + 1,
            w.parts.len(),
            part.dataset,
            spec.name
        );
        let slice = Slice::open(&root, &part.dataset, part.base, part.queries)?;
        let mut entry =
            json!({"dataset": part.dataset, "base": slice.base, "config": spec.name, "k": part.k});
        let gt_started = Instant::now();
        for f in std::iter::once("none").chain(part.filters.iter().map(String::as_str)) {
            slice.ground_truth(
                f,
                &named_filter(f)?,
                slice.base,
                &HashSet::new(),
                GT_K,
                w.recall_threads.max(8),
            )?;
        }
        entry["ground_truth_seconds"] = json!(secs(gt_started));
        let dir = work.join(format!("part{i}-{}-{}", part.dataset, spec.name));
        let mut args = vec![
            "build".to_string(),
            "--index".into(),
            dir.display().to_string(),
        ];
        args.extend(common_child_args(a, part, &w)?);
        let held_out = if part.updates { slice.base / 10 } else { 0 };
        args.extend(["--rows".into(), (slice.base - held_out).to_string()]);
        let built = child(&args);
        match built {
            Err(e) => {
                entry["error"] = json!(e);
                results.push(entry);
                continue;
            }
            Ok(b) => entry["build"] = b,
        }
        if part.open {
            for cold in [false, true] {
                let mut args = vec![
                    "open".to_string(),
                    "--index".into(),
                    dir.display().to_string(),
                ];
                args.extend(common_child_args(a, part, &w)?);
                if cold {
                    args.push("--drop-caches".into());
                }
                entry[if cold { "open_cold" } else { "open_warm" }] =
                    child(&args).unwrap_or_else(|e| json!({"error": e}));
            }
        }
        if held_out == 0 {
            let searcher = engine::open_reader(engine_kind(a)?, &dir, StoreOptions::default())?;
            entry["queries"] =
                query_part(&slice, &searcher, part, &w).unwrap_or_else(|e| json!({"error": e}));
        } else {
            // An update part reports recall after updates (its own ground truth); define
            // a second part without updates for query speed.
            let mut args = vec![
                "update".to_string(),
                "--index".into(),
                dir.display().to_string(),
            ];
            args.extend(common_child_args(a, part, &w)?);
            args.extend([
                "--rows".into(),
                (slice.base - held_out).to_string(),
                "--queries".into(),
                part.queries.min(500).to_string(),
            ]);
            entry["updates"] = child(&args).unwrap_or_else(|e| json!({"error": e}));
        }
        entry["disk_bytes"] = json!(measure::dir_bytes(&dir));
        if !a.flag("keep") {
            let _ = std::fs::remove_dir_all(&dir);
        }
        results.push(entry);
        std::fs::write(
            out.join("results.json"),
            serde_json::to_string_pretty(&results).unwrap(),
        )
        .map_err(|e| e.to_string())?;
    }
    let report = json!({"suite": w.name, "parts": results});
    std::fs::write(
        out.join("results.json"),
        serde_json::to_string_pretty(&report).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    std::fs::write(out.join("results.md"), crate::report::markdown(&report))
        .map_err(|e| e.to_string())?;
    Ok(json!({"results": out.join("results.json"), "markdown": out.join("results.md")}))
}

/// `datasets`: what is under the datasets root.
pub fn datasets(a: &Args) -> R<Value> {
    let root = datasets_root(a.get("datasets"));
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&root) {
        let mut names: Vec<_> = entries.flatten().map(|e| e.path()).collect();
        names.sort();
        for p in names {
            if let Ok(ds) = caprock_oracle::dataset::Dataset::open(&p) {
                out.push(json!({
                    "name": ds.info.name, "dir": p, "metric": ds.info.metric, "dim": ds.info.dim,
                    "base": ds.info.base_count, "queries": ds.info.query_count,
                    "meta": ds.info.meta.is_some(), "bytes": measure::dir_bytes(&p),
                }));
            }
        }
    }
    Ok(json!({"root": root, "datasets": out}))
}
