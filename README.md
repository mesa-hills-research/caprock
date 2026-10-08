# mhr_vector

A vector store in pure, safe Rust, written for the dense search of arko_index, a code
search index. It holds embeddings of code and documentation chunks, takes a file
watcher's updates as they come, and answers top-k searches filtered by path or language.

- **Flat and graph search.** Flat search scans compact codes of every vector and rescores
  the best candidates at full precision. Vamana graphs (the DiskANN family) serve high
  query rates on large indexes. Any index can also answer exactly.
- **Exact scores.** int8 or binary codes stay in memory to find candidates, and the
  full-precision vectors on disk rescore them, so every returned score is the exact one.
- **Deterministic.** Scores follow a fixed summation order, ties break by key, and builds
  write byte-identical files at any thread count.
- **Incremental and crash-safe.** Upserts and deletes take effect as soon as `apply`
  returns, and a commit switches to a new generation atomically. Readers in any number of
  threads or processes search snapshots while one writer works.
- **Pure, safe Rust.** Every crate has `#![forbid(unsafe_code)]`, and the runtime
  dependencies are `serde`, `serde_json`, `thiserror`, `crc32fast` and `pulp` (runtime
  SIMD dispatch).
- **Embedding-aware.** An index records the embedding model its vectors come from, and an
  embedding cache keeps vectors by content hash, so an unchanged chunk is embedded once.

It was made by an AI agent swarm from a written contract, an exact oracle and an
acceptance test per unit. [docs/development](docs/development/README.md) tells how.

## Flat or graph

arko_index uses flat search with int8 codes by default. Its recall@10 was 1.0 on both sets
below, a query costs one pass over the codes, and merging segments builds no graph. The
Vamana graph is there for high-QPS serving, at about 14 times the flat query rate on the
code set with recall@10 of 0.97. With the default configuration every segment of 20,000
vectors or more gets one.

## Speed

Single-thread search, k = 10, on 1024-dimensional Voyage embeddings of 199,999 code chunks
and 34,601 documentation chunks:

| Search | Set | Recall@10 | Queries/s | p50 |
|---|---|---:|---:|---:|
| Flat, int8 codes | code | 1.000 | 75 | 13.4 ms |
| Flat, int8 codes | docs | 1.000 | 437 | 2.3 ms |
| Flat, binary codes | code | 0.996 | 110 | 9.0 ms |
| Exact, f32 | code | 1.000 | 38 | 26.2 ms |
| Vamana graph, int8 codes | code | 0.966 | 1,018 | 0.95 ms |
| Vamana graph, int8 codes | docs | 0.999 | 2,223 | 0.41 ms |

Measured on a 6-vCPU virtual machine on an Intel Core i9-13980HX. Graph builds write the
same bytes at any thread count: on an AMD EPYC 9354P a graph over 50,000 code vectors took
236.6 s on one thread and 18.9 s on 16. Every measurement, both machines and the method
are in [docs/benchmarks.md](docs/benchmarks.md).

## Installation

The crate is on GitHub for now:

```toml
[dependencies]
mhr_vector = { git = "https://github.com/mesa-hills-research/mhr_vector" }
```

It uses the 2024 edition and needs Rust 1.89 or later.

## Usage

```rust
use mhr_vector::{
    AnnKind, Filter, IndexConfig, Metric, Op, Reader, Record, SearchParams, StoreOptions, Writer,
};

let dir = std::env::temp_dir().join("mhr_vector_quickstart");
let _ = std::fs::remove_dir_all(&dir);

// int8 codes find candidates, full-precision vectors rescore them.
let mut config = IndexConfig::new(4, Metric::Cosine);
config.ann.kind = AnnKind::Flat; // without this, large segments get a Vamana graph
let mut writer = Writer::create(&dir, config, StoreOptions::default())?;

writer.apply(&[
    Op::Upsert(Record {
        key: "lib-0",
        vector: &[0.9, 0.1, 0.0, 0.2],
        path: "src/lib.rs",
        language: "rust",
    }),
    Op::Upsert(Record {
        key: "intro-0",
        vector: &[0.8, 0.0, 0.2, 0.1],
        path: "docs/intro.md",
        language: "markdown",
    }),
])?;

// Searchable at once, before any commit.
let query = [1.0, 0.1, 0.0, 0.1];
let filter = Filter::path("src");
let hits = writer
    .snapshot()
    .search(&query, 10, Some(&filter), &SearchParams::default())?;
for hit in &hits {
    println!("{:.4}  {}  {}", hit.score, hit.key, hit.path);
}

// Durable, and visible to readers in other processes.
writer.commit(None)?;
let reader = Reader::open(&dir, StoreOptions::default())?;
assert_eq!(reader.snapshot().len(), 2);
```

`cargo run -p mhr_vector --example quickstart` runs it. With arko_index's embeddings,
`IndexConfig::for_model(EmbeddingModel::voyage_code_4(1024))` takes the place of
`IndexConfig::new` and records the model with the index.

## Documentation

- [Requirements](docs/requirements.md): the behaviour contract, from the score arithmetic
  and filters to persistence, concurrency, determinism and memory.
- [Benchmarks](docs/benchmarks.md): every measurement, the machines and the method.
- [Datasets](docs/datasets.md): the Voyage sets of code and documentation, and how they
  were built.
- [Testing](docs/testing.md): the oracle, the acceptance and property tests, and the
  conformance checks.
- [Development](docs/development/README.md): the plan the agent swarm worked from.

## License

MIT or Apache-2.0, at your option ([LICENSE-MIT](LICENSE-MIT),
[LICENSE-APACHE](LICENSE-APACHE)).
