//! The conformance checks: the documented behaviour (docs/reference.md), checked through
//! the public API against the reference implementation in `mhr_vector_oracle`.
//! `mhr_vector_conform` runs each in its own process.
//!
//! A failure has a stable `kind`, and the check id and the kind together form its
//! signature. Details carry the seed and step to replay (`mhr_vector_conform one <id>`).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use mhr_vector::{
    CodeKind, ContentHash, EmbeddingCache, EmbeddingModel, Error, Filter, IndexConfig, Metric, Op,
    Reader, Record, SearchParams, StoreOptions, Writer,
};
use mhr_vector_oracle::Oracle;
use mhr_vector_oracle::canonical;
use mhr_vector_oracle::check::{Expectation, check_query};
use mhr_vector_oracle::filter as ofilter;
use mhr_vector_oracle::generate::{
    self, OpsSpec, Shape, Step, VectorSpec, path_for, standard_filters,
};
use mhr_vector_oracle::rng::Rng;
use mhr_vector_oracle::scenario::{Scenario, run};

use crate::adapter::{Scratch, StoreUnderTest, to_filter, to_ops};
use crate::engine::oracle_metric;

/// A failed check: a stable kind and the details.
#[derive(Debug, Clone, PartialEq)]
pub struct Failure {
    pub kind: String,
    pub detail: String,
}

fn fail(kind: &str, detail: impl Into<String>) -> Failure {
    Failure {
        kind: kind.to_string(),
        detail: detail.into(),
    }
}

type Outcome = Result<String, Failure>;

/// One check.
pub struct Check {
    pub id: String,
    pub run: Box<dyn Fn() -> Outcome + Send + Sync>,
}

fn check(id: impl Into<String>, f: impl Fn() -> Outcome + Send + Sync + 'static) -> Check {
    Check {
        id: id.into(),
        run: Box::new(f),
    }
}

/// Seeds per scenario check (`MHRV_CONFORM_SEEDS`, default 2).
fn seeds() -> u64 {
    std::env::var("MHRV_CONFORM_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2)
}

fn options(threads: usize) -> StoreOptions {
    StoreOptions {
        delta_max_vectors: 300,
        threads,
        build_memory_budget: 256 << 20,
        ..StoreOptions::default()
    }
}

fn metric_name(m: Metric) -> &'static str {
    match m {
        Metric::Cosine => "cosine",
        Metric::Dot => "dot",
        Metric::L2 => "l2",
    }
}

const METRICS: [Metric; 3] = [Metric::Cosine, Metric::Dot, Metric::L2];

fn shapes() -> Vec<Shape> {
    vec![
        Shape::Gaussian,
        Shape::Clustered {
            clusters: 16,
            spread: 0.3,
        },
        Shape::Duplicates { distinct: 200 },
        Shape::Lattice { levels: 2 },
        Shape::Anisotropic,
    ]
}

fn graph_config(dim: u32, metric: Metric, codes: CodeKind) -> IndexConfig {
    let mut c = IndexConfig::new(dim, metric).with_codes(codes);
    c.ann.graph_min_vectors = 400;
    c
}

fn ops_spec(dim: usize, shape: Shape, seed: u64) -> OpsSpec {
    let mut o = OpsSpec::mixed(dim, seed);
    o.vectors.shape = shape;
    o.steps = 50;
    o.key_space = 5000;
    o.max_batch = 700;
    o
}

/// Runs a scenario for each seed on a fresh store.
fn scenario_check(
    config: IndexConfig,
    params: SearchParams,
    shape: Shape,
    expect: Expectation,
    min_mean_recall: Option<f64>,
    ks: Vec<usize>,
    tweak: fn(&mut OpsSpec),
) -> Outcome {
    let mut summary = Vec::new();
    for seed in 1..=seeds() {
        let mut ops = ops_spec(config.dim as usize, shape, seed);
        tweak(&mut ops);
        let scenario = Scenario {
            metric: oracle_metric(config.metric),
            ops,
            queries: 8,
            ks: ks.clone(),
            filters: true,
            expect,
            min_mean_recall,
        };
        let mut store = StoreUnderTest::new(config.clone(), options(2), params)
            .map_err(|e| fail("create-error", e))?;
        let report =
            run(&mut store, &scenario).map_err(|f| fail(&f.kind, format!("seed {seed}, {f}")))?;
        summary.push(format!(
            "seed {seed}: mean recall {:.4} over {} checks",
            report.recall.mean, report.checks
        ));
    }
    Ok(summary.join("; "))
}

/// Every check, in a fixed order.
pub fn all() -> Vec<Check> {
    let mut out = vec![
        check("kernels/canonical", kernels),
        check("filter/semantics", filter_semantics),
    ];
    for metric in METRICS {
        for shape in shapes() {
            out.push(check(
                format!("exact/{}/{}", metric_name(metric), shape.name()),
                move || {
                    scenario_check(
                        IndexConfig::exact(24, metric),
                        SearchParams::default(),
                        shape,
                        Expectation::exact(),
                        None,
                        vec![1, 10, 50],
                        |_| {},
                    )
                },
            ));
        }
        for codes in [CodeKind::Int8, CodeKind::Binary] {
            out.push(check(
                format!("exact-params/{}/{}", codes.name(), metric_name(metric)),
                move || {
                    scenario_check(
                        graph_config(24, metric, codes),
                        SearchParams::exact(),
                        Shape::Gaussian,
                        Expectation::exact(),
                        None,
                        vec![10],
                        |_| {},
                    )
                },
            ));
        }
        for (codes, min) in [
            (CodeKind::Int8, 0.95),
            (CodeKind::F32, 0.95),
            (CodeKind::Binary, 0.9),
        ] {
            for shape in [
                Shape::Clustered {
                    clusters: 16,
                    spread: 0.3,
                },
                Shape::Duplicates { distinct: 200 },
            ] {
                out.push(check(
                    format!(
                        "ann/{}/{}/{}",
                        codes.name(),
                        metric_name(metric),
                        shape.name()
                    ),
                    move || {
                        scenario_check(
                            graph_config(32, metric, codes),
                            SearchParams::default(),
                            shape,
                            Expectation::rescored(None),
                            Some(min),
                            vec![1, 10, 50],
                            |_| {},
                        )
                    },
                ));
            }
        }
        out.push(check(
            format!("durability/{}", metric_name(metric)),
            move || {
                scenario_check(
                    IndexConfig::exact(16, metric),
                    SearchParams::default(),
                    Shape::Gaussian,
                    Expectation::exact(),
                    None,
                    vec![10],
                    |o| {
                        o.p_commit = 0.25;
                        o.p_reopen = 0.2;
                        o.steps = 80;
                    },
                )
            },
        ));
    }
    out.push(check("ann/int8/cosine/anisotropic", || {
        scenario_check(
            graph_config(64, Metric::Cosine, CodeKind::Int8),
            SearchParams::default(),
            Shape::Anisotropic,
            Expectation::rescored(None),
            Some(0.95),
            vec![10],
            |_| {},
        )
    }));
    out.push(check("durability/ann-int8", || {
        scenario_check(
            graph_config(32, Metric::Cosine, CodeKind::Int8),
            SearchParams::default(),
            Shape::Clustered {
                clusters: 16,
                spread: 0.3,
            },
            Expectation::rescored(None),
            Some(0.95),
            vec![10],
            |o| {
                o.p_commit = 0.25;
                o.p_reopen = 0.2;
                o.p_compact = 0.1;
                o.steps = 80;
            },
        )
    }));
    out.push(check("determinism/threads", determinism_threads));
    out.push(check("determinism/repeat", determinism_repeat));
    out.push(check("persistence/garbage", persistence_garbage));
    out.push(check(
        "persistence/corrupt-current",
        persistence_corrupt_current,
    ));
    out.push(check(
        "persistence/corrupt-segment",
        persistence_corrupt_segment,
    ));
    out.push(check("concurrency/readers", concurrency_readers));
    out.push(check("reader/refresh", reader_refresh));
    out.push(check("errors/refusals", errors_refusals));
    out.push(check("model/mismatch", model_mismatch));
    out.push(check("cache/roundtrip", cache_roundtrip));
    out
}

fn kernels() -> Outcome {
    let mut rng = Rng::new(1);
    let mut n = 0;
    for dim in (0..=70).chain([
        100, 127, 128, 129, 255, 256, 257, 512, 1000, 1023, 1024, 1025, 1536, 2048, 4096,
    ]) {
        for scale in [1e-3f32, 1.0, 1e3] {
            for _ in 0..5 {
                let a: Vec<f32> = (0..dim).map(|_| rng.gaussian() as f32 * scale).collect();
                let b: Vec<f32> = (0..dim).map(|_| rng.gaussian() as f32 * scale).collect();
                if mhr_vector::distance::dot(&a, &b).to_bits() != canonical::dot(&a, &b).to_bits() {
                    return Err(fail("kernel-mismatch", format!("dot, dim {dim}")));
                }
                if mhr_vector::distance::l2_squared(&a, &b).to_bits()
                    != canonical::l2_squared(&a, &b).to_bits()
                {
                    return Err(fail("kernel-mismatch", format!("l2_squared, dim {dim}")));
                }
                if dim > 0 {
                    let ours = mhr_vector::distance::normalize(&a)
                        .map_err(|e| fail("kernel-mismatch", format!("normalize refused: {e}")))?;
                    let theirs = canonical::normalize(&a)
                        .map_err(|e| fail("kernel-mismatch", e.to_string()))?;
                    if ours
                        .iter()
                        .map(|x| x.to_bits())
                        .ne(theirs.iter().map(|x| x.to_bits()))
                    {
                        return Err(fail("kernel-mismatch", format!("normalize, dim {dim}")));
                    }
                }
                n += 1;
            }
        }
    }
    Ok(format!("{n} vector pairs bit-identical"))
}

fn filter_semantics() -> Outcome {
    let prefixes = [
        "", "/", "p10", "p10/", "p1", "p10/d1", "p50/d7", "rest", "zzz", "p001", "src/foo",
    ];
    let mut paths: Vec<String> = [
        "src/foo",
        "src/foo/bar.rs",
        "src/foobar.rs",
        "src/foo-bar.rs",
        "src",
        "",
        "a/b",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    paths.extend((0..500).map(|i| path_for(i, 3).0));
    for p in &paths {
        for prefix in prefixes {
            if mhr_vector::filter::path_has_prefix(p, prefix) != ofilter::path_has_prefix(p, prefix)
            {
                return Err(fail(
                    "filter-mismatch",
                    format!("path {p:?} prefix {prefix:?}"),
                ));
            }
        }
    }
    for i in 0..2000 {
        let (p, l) = path_for(i, 4);
        for (name, f, _) in standard_filters() {
            if to_filter(&f).matches(&p, &l) != f.matches(&p, &l) {
                return Err(fail("filter-mismatch", format!("filter {name} on {p} {l}")));
            }
        }
    }
    Ok("filters agree".into())
}

fn apply_steps(w: &mut Writer, steps: &[Step]) -> Result<(), Failure> {
    for step in steps {
        match step {
            Step::Apply(ops) => {
                w.apply(&to_ops(ops))
                    .map_err(|e| fail("apply-error", e.to_string()))?;
            }
            Step::Commit => {
                w.commit(None)
                    .map_err(|e| fail("commit-error", e.to_string()))?;
            }
            Step::Maintain => {
                w.maintain()
                    .map_err(|e| fail("maintain-error", e.to_string()))?;
            }
            Step::Compact => {
                w.compact()
                    .map_err(|e| fail("compact-error", e.to_string()))?;
            }
            Step::Reopen | Step::Check => {}
        }
    }
    Ok(())
}

fn dir_image(dir: &std::path::Path) -> Vec<(String, Vec<u8>)> {
    let mut files: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir)
        .map(|r| {
            r.flatten()
                .filter(|e| e.file_name() != "LOCK")
                .map(|e| {
                    (
                        e.file_name().to_string_lossy().to_string(),
                        std::fs::read(e.path()).unwrap_or_default(),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    files
}

fn determinism_threads() -> Outcome {
    let spec = ops_spec(
        32,
        Shape::Clustered {
            clusters: 16,
            spread: 0.3,
        },
        9,
    );
    let steps = spec.steps();
    let queries = spec.vectors.queries(20, 1000);
    let mut images = Vec::new();
    let mut results = Vec::new();
    for threads in [1, 4] {
        let dir = Scratch::new("det");
        let mut w = Writer::create(
            dir.path(),
            graph_config(32, Metric::Cosine, CodeKind::Int8),
            options(threads),
        )
        .map_err(|e| fail("create-error", e.to_string()))?;
        apply_steps(&mut w, &steps)?;
        w.compact()
            .map_err(|e| fail("compact-error", e.to_string()))?;
        w.commit(None)
            .map_err(|e| fail("commit-error", e.to_string()))?;
        let s = w.snapshot();
        let mut r = Vec::new();
        for q in &queries {
            r.push(
                s.search(q, 10, None, &SearchParams::default())
                    .map_err(|e| fail("search-error", e.to_string()))?,
            );
        }
        results.push(r);
        drop(w);
        images.push(dir_image(dir.path()));
    }
    let names = |i: &Vec<(String, Vec<u8>)>| i.iter().map(|f| f.0.clone()).collect::<Vec<_>>();
    if names(&images[0]) != names(&images[1]) {
        return Err(fail(
            "layout-differs",
            format!("{:?} vs {:?}", names(&images[0]), names(&images[1])),
        ));
    }
    for (a, b) in images[0].iter().zip(&images[1]) {
        if a.1 != b.1 {
            return Err(fail(
                "bytes-differ",
                format!("{} differs between 1 and 4 threads", a.0),
            ));
        }
    }
    if results[0] != results[1] {
        return Err(fail(
            "results-differ",
            "search results differ between 1 and 4 threads",
        ));
    }
    Ok(format!("{} files identical", images[0].len()))
}

fn determinism_repeat() -> Outcome {
    let spec = ops_spec(32, Shape::Duplicates { distinct: 100 }, 10);
    let dir = Scratch::new("repeat");
    let mut w = Writer::create(
        dir.path(),
        graph_config(32, Metric::Dot, CodeKind::Binary),
        options(2),
    )
    .map_err(|e| fail("create-error", e.to_string()))?;
    apply_steps(&mut w, &spec.steps())?;
    let s = w.snapshot();
    let queries = spec.vectors.queries(20, 1000);
    let first: Vec<_> = queries
        .iter()
        .map(|q| {
            s.search(q, 10, None, &SearchParams::default())
                .map_err(|e| e.to_string())
        })
        .collect();
    let again: Vec<_> = std::thread::scope(|scope| {
        let h = scope.spawn(|| {
            queries
                .iter()
                .map(|q| {
                    s.search(q, 10, None, &SearchParams::default())
                        .map_err(|e| e.to_string())
                })
                .collect::<Vec<_>>()
        });
        h.join().unwrap()
    });
    if first != again {
        return Err(fail(
            "results-differ",
            "the same search gave different results",
        ));
    }
    Ok("repeatable".into())
}

fn small_store(dir: &std::path::Path) -> Result<(Writer, Oracle, Vec<Vec<f32>>), Failure> {
    let spec = VectorSpec {
        dim: 16,
        shape: Shape::Gaussian,
        seed: 3,
    };
    let mut w = Writer::create(dir, IndexConfig::exact(16, Metric::Cosine), options(1))
        .map_err(|e| fail("create-error", e.to_string()))?;
    let mut oracle = Oracle::new(canonical::Metric::Cosine, 16);
    let vectors = spec.vectors(1500);
    let keys: Vec<String> = (0..1500).map(generate::key).collect();
    let ops: Vec<Op> = vectors
        .iter()
        .zip(&keys)
        .map(|(v, k)| {
            Op::Upsert(Record {
                key: k,
                vector: v,
                path: "a/b.rs",
                language: "rust",
            })
        })
        .collect();
    for (v, k) in vectors.iter().zip(&keys) {
        oracle
            .upsert(k, v, "a/b.rs", "rust")
            .map_err(|e| fail("oracle", e.to_string()))?;
    }
    w.apply(&ops)
        .map_err(|e| fail("apply-error", e.to_string()))?;
    w.commit(None)
        .map_err(|e| fail("commit-error", e.to_string()))?;
    Ok((w, oracle, spec.queries(10, 1500)))
}

fn same_as_oracle(w: &Writer, oracle: &Oracle, queries: &[Vec<f32>]) -> Result<(), Failure> {
    let s = w.snapshot();
    for (i, q) in queries.iter().enumerate() {
        let hits = s
            .search(q, 10, None, &SearchParams::default())
            .map_err(|e| fail("search-error", e.to_string()))?;
        let pairs: Vec<(String, f32)> = hits.into_iter().map(|h| (h.key, h.score)).collect();
        check_query(
            oracle,
            q,
            10,
            &ofilter::Filter::all(),
            &pairs,
            Expectation::exact(),
        )
        .map_err(|v| fail(v.kind(), format!("query {i}: {v}")))?;
    }
    Ok(())
}

fn persistence_garbage() -> Outcome {
    let dir = Scratch::new("garbage");
    let (w, oracle, queries) = small_store(dir.path())?;
    drop(w);
    let stray = [
        "seg-00000000000000000999.mhrv",
        "seg-00000000000000000998.mhrv.tmp",
        "gen-00000000000000000000.json",
    ];
    for f in stray {
        std::fs::write(dir.path().join(f), b"junk").map_err(|e| fail("io", e.to_string()))?;
    }
    std::fs::write(dir.path().join("notes.txt"), b"mine").map_err(|e| fail("io", e.to_string()))?;
    let w = Writer::open(dir.path(), options(1)).map_err(|e| fail("open-error", e.to_string()))?;
    for f in stray {
        if dir.path().join(f).exists() {
            return Err(fail("garbage-kept", f));
        }
    }
    if !dir.path().join("notes.txt").exists() {
        return Err(fail("foreign-file-removed", "notes.txt"));
    }
    same_as_oracle(&w, &oracle, &queries)?;
    Ok("garbage removed, state intact".into())
}

fn persistence_corrupt_current() -> Outcome {
    let dir = Scratch::new("current");
    let (w, _, _) = small_store(dir.path())?;
    drop(w);
    std::fs::write(
        dir.path().join("CURRENT"),
        "gen-00000000000000000777.json\n",
    )
    .map_err(|e| fail("io", e.to_string()))?;
    match Writer::open(dir.path(), options(1)) {
        Err(Error::Corrupt { .. }) => {}
        Err(e) => return Err(fail("wrong-error", format!("expected Corrupt, got {e}"))),
        Ok(_) => {
            return Err(fail(
                "accepted-corrupt",
                "opened an index whose CURRENT names a missing manifest",
            ));
        }
    }
    match Reader::open(dir.path(), StoreOptions::default()) {
        Err(Error::Corrupt { .. }) => Ok("refused with Corrupt".into()),
        Err(e) => Err(fail(
            "wrong-error",
            format!("reader: expected Corrupt, got {e}"),
        )),
        Ok(_) => Err(fail("accepted-corrupt", "reader opened it")),
    }
}

fn persistence_corrupt_segment() -> Outcome {
    let dir = Scratch::new("segment");
    let (w, _, _) = small_store(dir.path())?;
    drop(w);
    let seg = std::fs::read_dir(dir.path())
        .map_err(|e| fail("io", e.to_string()))?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|x| x == "mhrv"))
        .ok_or_else(|| {
            fail(
                "no-segment",
                "a commit of 1,500 records wrote no segment file",
            )
        })?;
    let mut bytes = std::fs::read(&seg).map_err(|e| fail("io", e.to_string()))?;
    let at = bytes.len() / 3;
    for b in &mut bytes[at..at + 64] {
        *b ^= 0xA5;
    }
    std::fs::write(&seg, &bytes).map_err(|e| fail("io", e.to_string()))?;
    let full = StoreOptions {
        verify: mhr_vector::VerifyMode::Full,
        ..options(1)
    };
    match Writer::open(dir.path(), full) {
        Err(Error::Corrupt { .. }) => Ok("refused with Corrupt".into()),
        Err(e) => Err(fail("wrong-error", format!("expected Corrupt, got {e}"))),
        Ok(_) => Err(fail(
            "accepted-corrupt",
            "a full verification accepted a damaged segment",
        )),
    }
}

fn concurrency_readers() -> Outcome {
    let dim = 16;
    let dir = Scratch::new("concurrent");
    let mut w = Writer::create(
        dir.path(),
        IndexConfig::exact(dim as u32, Metric::Cosine),
        options(2),
    )
    .map_err(|e| fail("create-error", e.to_string()))?;
    let states: Arc<Mutex<HashMap<u64, Oracle>>> = Arc::new(Mutex::new(HashMap::new()));
    let mut oracle = Oracle::new(canonical::Metric::Cosine, dim);
    states.lock().unwrap().insert(0, oracle.clone());
    let reader = w.reader();
    let done = Arc::new(AtomicBool::new(false));
    let spec = ops_spec(
        dim,
        Shape::Clustered {
            clusters: 16,
            spread: 0.3,
        },
        11,
    );
    let queries = spec.vectors.queries(4, 1000);
    let errors: Arc<Mutex<Vec<Failure>>> = Arc::new(Mutex::new(Vec::new()));
    let mut handles = Vec::new();
    for t in 0..4 {
        let (reader, states, done, queries, errors) = (
            reader.clone(),
            states.clone(),
            done.clone(),
            queries.clone(),
            errors.clone(),
        );
        handles.push(std::thread::spawn(move || {
            let filters = standard_filters();
            let mut checked = 0usize;
            while !done.load(Ordering::Relaxed) || checked == 0 {
                let s = reader.snapshot();
                let oracle = loop {
                    if let Some(o) = states.lock().unwrap().get(&s.version()) {
                        break o.clone();
                    }
                    std::thread::yield_now();
                };
                let (name, f, _) = &filters[(checked + t) % filters.len()];
                for q in &queries {
                    let ours = to_filter(f);
                    let r = s.search(
                        q,
                        10,
                        if f.is_empty() { None } else { Some(&ours) },
                        &SearchParams::default(),
                    );
                    let outcome = match r {
                        Err(e) => Err(fail("search-error", e.to_string())),
                        Ok(hits) => {
                            let pairs: Vec<(String, f32)> =
                                hits.into_iter().map(|h| (h.key, h.score)).collect();
                            check_query(&oracle, q, 10, f, &pairs, Expectation::exact())
                                .map(|_| ())
                                .map_err(|v| {
                                    fail(
                                        v.kind(),
                                        format!("version {}, filter {name}: {v}", s.version()),
                                    )
                                })
                        }
                    };
                    if let Err(e) = outcome {
                        errors.lock().unwrap().push(e);
                        return checked;
                    }
                }
                checked += 1;
            }
            checked
        }));
    }
    let mut writer_error = None;
    for step in spec.steps() {
        let r = match step {
            Step::Apply(batch) => {
                for op in &batch {
                    match op {
                        generate::Op::Upsert {
                            key,
                            vector,
                            path,
                            language,
                        } => {
                            let _ = oracle.upsert(key, vector, path, language);
                        }
                        generate::Op::Delete { key } => {
                            oracle.delete(key);
                        }
                    }
                }
                let version = w.snapshot().version() + batch.len() as u64;
                states.lock().unwrap().insert(version, oracle.clone());
                w.apply(&to_ops(&batch))
                    .map(|_| ())
                    .map_err(|e| fail("apply-error", e.to_string()))
            }
            Step::Commit => w
                .commit(None)
                .map(|_| ())
                .map_err(|e| fail("commit-error", e.to_string())),
            Step::Maintain | Step::Compact => w
                .compact()
                .map(|_| ())
                .map_err(|e| fail("compact-error", e.to_string())),
            _ => Ok(()),
        };
        if let Err(e) = r {
            writer_error = Some(e);
            break;
        }
    }
    done.store(true, Ordering::Relaxed);
    let mut total = 0;
    for h in handles {
        total += h
            .join()
            .map_err(|_| fail("reader-panic", "a reader thread panicked"))?;
    }
    if let Some(e) = writer_error {
        return Err(e);
    }
    if let Some(e) = errors.lock().unwrap().first() {
        return Err(e.clone());
    }
    Ok(format!("{total} snapshot checks"))
}

fn reader_refresh() -> Outcome {
    let dir = Scratch::new("refresh");
    let (mut w, _, queries) = small_store(dir.path())?;
    let other = Reader::open(dir.path(), StoreOptions::default())
        .map_err(|e| fail("open-error", e.to_string()))?;
    let before = other.snapshot().len();
    w.apply(&[Op::Delete(&generate::key(0))])
        .map_err(|e| fail("apply-error", e.to_string()))?;
    if other
        .refresh()
        .map_err(|e| fail("refresh-error", e.to_string()))?
        || other.snapshot().len() != before
    {
        return Err(fail(
            "uncommitted-visible",
            "another reader saw an uncommitted delete",
        ));
    }
    w.commit(None)
        .map_err(|e| fail("commit-error", e.to_string()))?;
    if !other
        .refresh()
        .map_err(|e| fail("refresh-error", e.to_string()))?
        || other.snapshot().len() != before - 1
    {
        return Err(fail(
            "commit-invisible",
            "a reader's refresh missed a commit",
        ));
    }
    let a = other
        .snapshot()
        .search(&queries[0], 10, None, &SearchParams::default())
        .map_err(|e| fail("search-error", e.to_string()))?;
    let b = w
        .snapshot()
        .search(&queries[0], 10, None, &SearchParams::default())
        .map_err(|e| fail("search-error", e.to_string()))?;
    if a != b {
        return Err(fail(
            "results-differ",
            "a refreshed reader and the writer disagree",
        ));
    }
    Ok("refresh follows commits".into())
}

fn errors_refusals() -> Outcome {
    let dir = Scratch::new("refusals");
    let mut w = Writer::create(dir.path(), IndexConfig::new(4, Metric::Cosine), options(1))
        .map_err(|e| fail("create-error", e.to_string()))?;
    let good = [1.0f32, 0.0, 0.0, 0.0];
    let long_key = "k".repeat(256);
    let bad_vectors: [&[f32]; 5] = [
        &[1.0, 0.0],
        &[f32::NAN, 0.0, 0.0, 0.0],
        &[f32::INFINITY, 0.0, 0.0, 0.0],
        &[0.0; 4],
        &[2e15, 0.0, 0.0, 0.0],
    ];
    for v in bad_vectors {
        if !matches!(
            w.apply(&[Op::Upsert(Record {
                key: "k",
                vector: v,
                path: "a",
                language: "x"
            })]),
            Err(Error::InvalidVector(_))
        ) {
            return Err(fail("not-refused", format!("vector {v:?}")));
        }
    }
    for k in ["", long_key.as_str()] {
        if !matches!(
            w.apply(&[Op::Upsert(Record {
                key: k,
                vector: &good,
                path: "a",
                language: "x"
            })]),
            Err(Error::InvalidKey(_))
        ) {
            return Err(fail("not-refused", format!("key of {} bytes", k.len())));
        }
    }
    let s = w.snapshot();
    if !s.is_empty() {
        return Err(fail("partial-apply", "a refused batch changed the index"));
    }
    for k in [0, mhr_vector::MAX_K + 1] {
        if !matches!(
            s.search(&good, k, None, &SearchParams::default()),
            Err(Error::InvalidArgument(_))
        ) {
            return Err(fail("not-refused", format!("k = {k}")));
        }
    }
    if !matches!(
        s.search(&[1.0], 5, None, &SearchParams::default()),
        Err(Error::InvalidVector(_))
    ) {
        return Err(fail("not-refused", "query of the wrong dimension"));
    }
    let mut bad = IndexConfig::new(4, Metric::Cosine);
    bad.dim = 0;
    let d2 = Scratch::new("refusals2");
    if !matches!(
        Writer::create(d2.path(), bad, options(1)),
        Err(Error::InvalidConfig(_))
    ) {
        return Err(fail("not-refused", "dim 0"));
    }
    Ok("every bad input refused".into())
}

fn model_mismatch() -> Outcome {
    let dir = Scratch::new("model");
    let model = EmbeddingModel::voyage_code_4(256);
    drop(
        Writer::create(
            dir.path(),
            IndexConfig::for_model(model.clone()),
            options(1),
        )
        .map_err(|e| fail("create-error", e.to_string()))?,
    );
    Writer::open_for_model(dir.path(), &model, options(1))
        .map_err(|e| fail("open-error", e.to_string()))?;
    match Writer::open_for_model(dir.path(), &EmbeddingModel::voyage_code_4(512), options(1)) {
        Err(Error::ModelMismatch { .. }) => Ok("mismatch refused".into()),
        Err(e) => Err(fail("wrong-error", e.to_string())),
        Ok(_) => Err(fail(
            "accepted-mismatch",
            "opened a 256-dimension index for a 512-dimension model",
        )),
    }
}

fn cache_roundtrip() -> Outcome {
    let root = Scratch::new("cache");
    let model = EmbeddingModel::voyage_code_4(256);
    let h = |i: u32| -> ContentHash {
        let mut x = [7u8; 32];
        x[..4].copy_from_slice(&i.to_le_bytes());
        x
    };
    let v = |i: u32| -> Vec<f32> { (0..256).map(|j| (i * 7 + j) as f32 * 1e-3).collect() };
    {
        let mut c = EmbeddingCache::open(root.path(), &model)
            .map_err(|e| fail("open-error", e.to_string()))?;
        for i in 0..200 {
            c.put(&[(h(i), &v(i)[..])])
                .map_err(|e| fail("put-error", e.to_string()))?;
        }
        c.flush().map_err(|e| fail("flush-error", e.to_string()))?;
    }
    let c =
        EmbeddingCache::open(root.path(), &model).map_err(|e| fail("open-error", e.to_string()))?;
    for i in 0..200 {
        if c.get(&h(i)).map_err(|e| fail("get-error", e.to_string()))? != Some(v(i)) {
            return Err(fail("cache-lost", format!("entry {i}")));
        }
    }
    if c.missing(&[h(1), h(999)]) != vec![1] {
        return Err(fail("cache-missing", "missing() is wrong"));
    }
    Ok("200 entries survive a reopen".into())
}

/// Filters by name, for reports.
pub fn filter_names() -> Vec<&'static str> {
    standard_filters().into_iter().map(|(n, _, _)| n).collect()
}

/// A guard against an unused import when the standard set changes.
pub fn _filter(f: &Filter) -> bool {
    f.is_empty()
}
