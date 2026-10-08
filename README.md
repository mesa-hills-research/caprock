# mhr_vector

A vector store for Rust, written in pure, safe Rust. It keeps compact codes of your
vectors in memory and the full vectors on disk, answers top-k searches with exact
scores, filters by path prefix and language, and takes updates as they come with
crash-safe commits.

- **Pure, safe Rust.** Every crate has `#![forbid(unsafe_code)]`, and the runtime
  dependencies are `serde`, `serde_json`, `thiserror`, `crc32fast` and `pulp` (SIMD
  kernels picked at run time).
- **Exact scores.** int8 or binary codes find candidates, and the full-precision vectors
  rescore them, so every returned score is the exact one. Any index can also answer by a
  brute-force scan.
- **Flat and graph search.** Flat search scans the codes of every vector. Vamana graphs
  (the DiskANN family) serve high query rates on large indexes.
- **Filters.** Each record carries a path and a language. Searches filter by path prefix,
  by language or both, and return `k` hits whenever `k` records match.
- **Incremental and crash-safe.** Upserts and deletes are searchable as soon as `apply`
  returns, and a commit switches to a new generation atomically. Any number of threads
  and processes search snapshots while one writer works.
- **Deterministic.** Scores follow a fixed summation order, ties break by key, and builds
  write the same bytes at any thread count.
- **Embedding-aware.** An index can record the model its vectors come from, and an
  embedding cache keeps vectors by content hash, so unchanged text is embedded once.

## Speed

Single-thread search, k = 10, on 1024-dimensional embeddings of 199,999 code chunks
(voyage-code-4) and 34,601 documentation chunks (voyage-4):

| Search | Set | Recall@10 | Queries/s | p50 |
|---|---|---:|---:|---:|
| Flat, int8 codes | code | 1.000 | 75 | 13.4 ms |
| Flat, int8 codes | docs | 1.000 | 437 | 2.3 ms |
| Flat, binary codes | code | 0.996 | 110 | 9.0 ms |
| Exact, f32 | code | 1.000 | 38 | 26.2 ms |
| Vamana graph, int8 codes | code | 0.966 | 1,018 | 0.95 ms |
| Vamana graph, int8 codes | docs | 0.999 | 2,223 | 0.41 ms |

Measured on 6 vCPUs of an Intel Core i9-13980HX. Flat search time grows with the index,
about 65 ns per vector with int8 codes, while graph search stays fast as the index grows.
Graph builds run in parallel: a graph over 50,000 code vectors took 236.6 s on one thread
and 18.9 s on 16. Every measurement, the machines and the method are in
[docs/benchmarks.md](docs/benchmarks.md).

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

`cargo run -p mhr_vector --example quickstart` runs it.

## Documentation

- [Reference](docs/reference.md): records, scores, filters, search, codes, updates and
  maintenance, persistence, concurrency, memory and configuration.
- [Benchmarks](docs/benchmarks.md): every measurement, the machines and the method, and
  how to benchmark your own data.

## Testing

```sh
cargo test --workspace --all-features
cargo run --release -p mhr_vector_bench --bin mhr_vector_conform -- run
```

The tests compare results with a brute-force reference implementation
(`mhr_vector_oracle`), and most of them need the test-only `internals` feature. The second
command runs end-to-end checks of the store, each in its own process.

## License

MIT or Apache-2.0, at your option ([LICENSE-MIT](LICENSE-MIT),
[LICENSE-APACHE](LICENSE-APACHE)).
