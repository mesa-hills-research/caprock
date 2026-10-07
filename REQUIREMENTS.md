# mhr_vector: requirements

The contract for `mhr_vector`, a pure-Rust vector store for arko_index's dense search.
The crate skeleton (`crates/mhr_vector`) states the API in code; this document states the
behaviour, the reasons, and the targets. Ground truth is the oracle crate
(`crates/mhr_vector_oracle`): when this document and the oracle disagree, the oracle is
the specification of results and this document is fixed.

Status: groundwork for the swarm (2026-10-07). Nothing below is implemented yet; every
function body in `mhr_vector` is a `todo!("Unn ...")` stub (see UNITS.md).

## 1. What it is for

arko_index chunks code (tree-sitter) and searches it with BM25. Dense search over
embeddings is to become an option (owner, 2026-10-07):

- **Voyage is the default provider**: `voyage-code-4`, 1024 dimensions by default; 256, 512
  and 2048 are also offered. Local embedding models come later through the same
  interface.
- **The embedding model is chosen per index** and saved with it; changing it forces a
  re-embed. Embeddings are cached by (model, content hash).
- **Hybrid retrieval.** On the owner's retrieval evaluation, dense beat lexical on depth
  questions (9/9 against 4/9) and lexical beat dense on "who calls X" (5/7 against 2/7).
  Fusion happens outside the store, by chunk key, so the store returns arko_index's own
  chunk IDs.

### Scale

| Corpus | Chunks | f32 at 1024 dims | int8 codes | binary codes |
|---|---|---|---|---|
| Typical repository | 5k-200k | 20 MB-0.8 GB | 5-205 MB | 0.6-26 MB |
| Linux kernel | ~1.34M | 5.5 GB | 1.37 GB | 172 MB |
| Largest test folder | ~5.8M | 23.8 GB | 5.9 GB | 742 MB |

So quantization matters (int8, binary; product quantization later), with rescoring at full
precision.

### Workload

- **Incremental.** The watcher adds and removes chunks as files change. A change must be
  searchable within about 2 s (read-your-writes); the store's share of that budget is
  small (section 7: `apply` returns in milliseconds and its effect is visible at once).
- **Queries.** Top-k with k ≤ 50 (the store accepts up to 1,000), by cosine or dot
  product; L2 is supported for benchmark datasets. Optionally filtered by path prefix
  or language.
- **Persistence.** Per index, on disk, crash-safe through an atomic generation switch, as
  arko_index's `index2` does it. The format is mmap-friendly and versioned.
- **Determinism.** Ties break by key. Builds don't depend on thread counts or timing.
- **Bounded memory.** Builds stay within a memory budget.

## 2. Design choices, and why

### 2.1 Segments: a mutable delta and immutable sealed segments

The index is a set of **immutable sealed segments** plus an in-memory **delta**, as in an
LSM tree or Lucene:

- An upsert goes to the delta. A delete or a replacement marks the old record deleted
  (**tombstones**: a bitset per segment, copy-on-write).
- When the delta reaches `delta_max_vectors`, it is **sealed** into a segment file.
- **Maintenance** merges segments: by size tiers, to purge deletions, and to build graphs.

Why this design:

- **Updates never rebuild a graph inline.** A watcher's burst costs a delta append. Graph
  construction happens only in merges, which run off the writer's critical path (plan,
  run anywhere, install).
- **Crash safety by construction.** Segment and tombstone files are written once, fsynced,
  and only then referenced by a new manifest. `CURRENT` switches atomically. No
  write-ahead log is needed: the caller's checkpoint token, stored in the manifest,
  tells it what is durable.
- **Bounded build memory.** A segment never exceeds what the budget can build
  (section 11). Large corpora become a few large segments, and searches fan out over
  them.
- **Simple concurrency.** A snapshot is a list of `Arc`ed immutable segments with their
  tombstones, plus a frozen view of the delta.
- **Deterministic layouts.** Seal and merge decisions are pure functions of segment
  sizes, so the same calls produce the same files.

### 2.2 Exact search first, with a canonical arithmetic

The flat (exact) path comes first, and every index can answer exactly
(`SearchParams::exact`). Scores follow a **canonical arithmetic** (section 4). Its
summation order is fixed and SIMD-friendly: 16 lanes, the order a 16-wide vector loop
produces anyway. So exact search can equal the oracle **bit for bit**: identical keys,
identical scores, identical order. Approximate search is judged by recall against the same
oracle, and the scores it returns are still canonical (they come from rescoring).

### 2.3 The approximate index: Vamana (DiskANN family), per sealed segment

Each sealed segment of at least `graph_min_vectors` vectors (default 20,000) carries a
Vamana graph. Smaller segments and the delta are searched flat. The candidates:

| | HNSW | IVF-PQ | Vamana / DiskANN |
|---|---|---|---|
| Incremental inserts | native, per insert | assign to nearest list (centroids drift) | FreshDiskANN-style, or rebuild per segment |
| Deletes | tombstones, repair is awkward | trivial (skip in lists) | tombstones; nodes still route; purge by merge |
| Memory layout | multi-layer lists, variable size | contiguous lists | one fixed-degree array: `n x R` u32 |
| Disk / mmap | layers complicate it | good | designed for SSD: compact codes in RAM, full vectors on disk |
| Quality at fixed memory | best in RAM | lowest (needs high nprobe and rescoring) | close to HNSW in RAM; best with full-precision rescoring |
| Filters | degrades when selective | easy (skip in lists) | same as HNSW; Filtered-DiskANN exists |
| Deterministic parallel build | hard (insertion races) | easy (seeded k-means) | shown by ParlayANN (batch insertion) |
| Build cost | high | low | high |

The choice is Vamana, built per sealed segment:

- **Single layer, fixed degree.** The format is a plain `n x R` array: aligned, contiguous,
  mmap-friendly, trivially validated.
- **Designed for exactly our split.** Compact codes stay in RAM for navigation;
  full-precision vectors stay on disk for rescoring. That is the quantization-plus-rescoring
  the workload needs, at 5.8M vectors.
- **Deterministic parallel construction is published.** ParlayANN (Manohar et al.,
  PPoPP 2024) builds it with batch insertion, so builds can be byte-identical for any thread
  count (section 10).
- **Updates and deletes fit the segment design.** Segments are immutable, so no in-place
  graph surgery is needed. Deleted nodes keep routing until a merge drops them, which is
  FreshDiskANN's lesson.
- **Filters are handled by planning.** Sorted segments give exact match counts in
  O(log n) (section 5). A selective filter is answered by a flat scan of exactly the
  matching records. A broad one is answered by graph search that admits only matching
  nodes, with a flat fallback when too few are found.

IVF and product quantization remain options for later (section 17). They would mainly
help much larger corpora or faster builds, and the segment design admits a second index
kind per segment without changing the format's structure.

### 2.4 Quantization, always rescored

Each segment keeps **resident codes** to find candidates:

- `int8`, the default: one byte per dimension, about 4x smaller than f32.
- `binary`: one bit per dimension plus at most 8 bytes of corrections, about 32x smaller.
  voyage-code-4 and text-embedding-3 are trained to survive both.
- `f32`: the codes are the vectors.

The best `max(k x rescore_factor, rescore_min)` candidates (int8: 4 and 32; binary: 16
and 128) are **rescored with the full-precision vectors**, read from the segment file with
positional reads. Returned scores are therefore always canonical, and the oracle can check
them with `==`. Full-precision vectors are always kept: on disk for int8 and binary, as
the codes themselves for f32.

### 2.5 Safe Rust, no mmap (for now)

`#![forbid(unsafe_code)]` everywhere. Mapping a file is `unsafe` in Rust (the file can
change underneath), so the store reads with **positional reads**: `read_exact_at` on Unix,
`seek_read` on Windows. Resident parts (codes, graph, keys, paths) are read into memory at
open; full-precision vectors are read on demand for rescoring. The format is
**mmap-friendly** anyway: little-endian fixed-width arrays, 4096-byte alignment for large
sections, no pointers, no varints in hot data. So a later, reviewed `mmap` feature can map
it without a format change.

Any `unsafe` needs a measured, large gain, reviewed and approved by the owner (section 16).

### 2.6 Filters by sorted ordinals

A sealed segment orders its records by (path, key). With component-wise prefix semantics
(section 5), the records under a path prefix form **at most two contiguous ordinal
ranges**: the path equal to the prefix, and the paths under `prefix/`. Binary search finds
them. A language filter is a small id set per segment. The planner knows each segment's
exact match count before searching.

## 3. API summary

The rustdoc of `crates/mhr_vector` is the detailed contract; the shape:

```rust
let mut writer = Writer::create(dir, IndexConfig::for_model(EmbeddingModel::voyage_code_4(1024)), StoreOptions::default())?;
writer.apply(&[Op::Upsert(Record { key, vector, path, language }), Op::Delete(old_key)])?; // visible at once
let hits: Vec<Hit> = writer.snapshot().search(&query, 10, Some(&Filter::path("src")), &SearchParams::default())?;
writer.commit(Some(&watcher_sequence.to_le_bytes()))?;      // durable + visible to other processes
if let Some(job) = writer.plan_maintenance() { let built = job.run()?; writer.install(built)?; } // or writer.maintain()
let reader = Reader::open(dir, StoreOptions::default())?;   // another process; reader.refresh()
```

- **`Writer`**: one per directory (exclusive lock). `create`, `open`, `open_for_model`,
  `apply`/`upsert`/`delete`, `snapshot`, `reader`, `commit`, `checkpoint`,
  `plan_maintenance`/`plan_compaction`, `install`, `maintain`, `compact`, `stats`.
- **`Snapshot`**: immutable, `Clone + Send + Sync`. `search`, `get`, `contains`, `len`,
  `keys`, `version`, `generation`, `config`.
- **`Reader`**: hands out snapshots of a writer (same process) or of a directory's last
  commit. `open`, `refresh`, `snapshot`.
- **`MaintenanceJob`** (`run`) and **`BuiltSegment`**: merges that run off the writer.
- **`IndexConfig`** (persisted): `dim`, `metric`, `embedding`, `codes`, `rescore_*`,
  `ann` (`kind`, `graph_min_vectors`, `filter_flat_below`), `seed`. Constructors: `new`,
  `for_model`, `exact`, `with_codes`. **`StoreOptions`** (per handle):
  `build_memory_budget` (1 GiB), `threads`, `delta_max_vectors` (8,192), `merge`,
  `verify`. **`SearchParams`**: `beam`, `rescore`, `exact`.
- **Records**: a key (1..=255 bytes, unique; arko_index's chunk ID); a vector; a path
  (≤ 4,096 bytes, relative and `/`-separated); a language (≤ 64 bytes).
- **Limits**: `MAX_DIM` 4,096, `MAX_K` 1,000.

## 4. Scores (the canonical arithmetic)

All scores are "higher is better":

| Metric | Stored vector | Score |
|---|---|---|
| `Cosine` | `normalize(v)` | `dot(normalize(q), x)` |
| `Dot` | `v` | `dot(q, x)` |
| `L2` | `v` | `-l2_squared(q, x)` |

- **`dot(a, b)`**: 16 `f32` lane accumulators. Element `i` goes to lane `i % 16`, in
  increasing `i`, as `acc = acc + (a[i] * b[i])`, the product and the sum each rounded to
  `f32` (no fused multiply-add). Then the reduction: for `w` in 8, 4, 2, 1,
  `acc[l] = acc[l] + acc[l + w]` for `l < w`. The result is `acc[0]`.
- **`l2_squared(a, b)`**: the same with `d = a[i] - b[i]`, `d * d`.
- **`normalize(v)`**: `norm = sqrt(dot(v, v))` in `f32`, then `v[i] / norm`. A zero squared
  norm is `VectorError::Zero`; a non-finite norm or result is `NotNormalizable`.
- **Valid vectors**: exactly `dim` finite components, each at most `1e15` in magnitude.
  Then no score can overflow, and scores are never NaN.
- **Result order**: score descending, then key ascending (bytes). `-0.0` equals `0.0`.

Candidate generation may use any approximation (`distance::dot_fast`, codecs, FMA under a
target feature), but **every returned score is computed with these functions**.

## 5. Filters

- A `Filter` has `path_prefixes` and `languages`. A record matches when its path matches
  **any** prefix (or there are none) **and** its language equals **any** listed language
  (or there are none).
- **Prefixes match whole components.** `src/foo` matches `src/foo` and `src/foo/bar.rs`,
  never `src/foobar.rs` or `lib/src/foo`. A trailing `/` is ignored; `""` and `/` match
  everything. Comparison is on bytes.
- The oracle's `filter.rs` is the reference.

## 6. Search

`snapshot.search(query, k, filter, params)`:

- **Validation.** `k` in `1..=MAX_K` (`InvalidArgument`); the query is validated and
  prepared like a stored vector (`InvalidVector`).
- **Exactly `min(k, live records matching the filter)` hits**, distinct keys, all live, all
  matching, in result order. An approximate search that finds too few must fall back
  (a flat search of the segment). Returning fewer is a bug (`too-few`).
- **Canonical scores** (section 4) on every hit.
- **Exact.** `IndexConfig::exact` (f32 codes, no graphs), or any index with
  `SearchParams::exact`, returns exactly the oracle's list.
- **Approximate.** Recall@k against the oracle meets section 13. Recall is defined in
  `mhr_vector_oracle::check`: ties at the k-th score are interchangeable, so duplicate
  vectors don't count as misses.
- **Parameters.**
  - `beam` is the Vamana candidate list (default `search_beam`, raised to at least `k` and
    to the rescore count).
  - `rescore` is the number of candidates rescored (default
    `max(k x rescore_factor, rescore_min)`, at least `k`).
- **The plan for each sealed segment**, after compiling the filter:
  - flat search when the segment has no graph, or at most `filter_flat_below` (2,048) of its
    live records match;
  - otherwise beam search over the graph, admitting live matching ordinals;
  - then rescoring.

  The delta is searched exactly. The per-segment lists are merged into the top `k`.

## 7. Updates and visibility

- **`apply(ops)`.**
  - The whole batch is validated first: any bad key, vector, path or language fails the
    call with **nothing applied**.
  - Operations then apply in order; a later operation on a key wins.
  - An upsert replaces a live record with the same key; a delete of an absent key is a
    no-op, counted in `ApplyReport::missing`.
  - Before returning, a new snapshot is published: **read-your-writes in the process,
    immediately**.
  - `apply` may seal the delta into new, uncommitted segments.
- **`version`.** Every operation of every successful `apply` counts, no-op deletes
  included. Snapshots carry it; a commit persists it.
- **`commit(checkpoint)`.**
  - Seals the delta, writes changed tombstones, and publishes generation `n + 1` with the
    caller's checkpoint token (at most 4 KiB; arko_index stores its watcher sequence).
  - Then removes files referenced by neither this generation nor the previous one.
  - Other processes see the commit after `Reader::refresh`. Uncommitted work is lost on a
    crash or a drop. The token tells the caller where to resume.
- **Maintenance.**
  - `plan_maintenance` is fast and does no I/O. It returns a `MaintenanceJob` holding `Arc`s
    of its input segments and their tombstones.
  - `job.run()` builds the merged segment anywhere, for example on another thread, within
    the memory budget.
  - `writer.install(built)` swaps the inputs for the output and deletes from it every
    record deleted or replaced since the plan.
  - A job from another writer, or whose inputs are gone, is `StaleJob`.
  - `maintain()` and `compact()` run the whole cycle in place. After `compact()` no deletion
    is left, no two segments could merge within the size limit, and every large-enough
    segment has a graph.
- **Visibility budget.** `apply` of 100 records at 1024 dimensions must return within the
  section 13 latency target, and its effect is visible to the next snapshot. Embedding
  latency (Voyage) is outside the store.

## 8. Persistence and crash safety

An index directory:

```text
LOCK                          the writer lock (std File::try_lock)
CURRENT                       the current manifest's name
gen-<20 digits>.json          one manifest per committed generation (no timestamps)
seg-<20 digits>.mhrv          sealed segments: section files, immutable
del-<seg 20>-<gen 20>.bits    tombstones of a segment as of a generation, immutable
*.tmp                         unfinished writes (garbage)
```

- **Section files** (format version 1):
  - a 64-byte header: magic, version, section count, table offset and length, table CRC,
    header CRC;
  - aligned sections: 4096 bytes for vectors, codes and the graph; 64 otherwise;
  - a section table: kind, offset, length, CRC-32 each.

  The layout is in `format.rs`; the sections of a segment are listed in
  `format::SectionKind`.
- **Write protocol.** Write `<name>.tmp`, fsync, rename, fsync the directory. A manifest is
  written only after every file it names is durable; `CURRENT` only after the manifest.
  - A crash at any point (process or power) leaves the last published generation intact.
  - The next writer `open` removes garbage: unreferenced segments, tombstones and
    manifests, and `*.tmp` files. It never removes a file whose name it doesn't recognise.
- **Readers.** Readers open every file of a generation when they take it, so the writer's
  garbage collection never breaks a reader that holds them (on Unix; on Windows the
  writer's deletes fail and are retried later). `Reader::refresh` retries when a file
  vanished between reading `CURRENT` and opening it.
- **Verification.**
  - `VerifyMode::Headers`, the default, checks headers, tables and every section loaded
    into memory.
  - `VerifyMode::Full` also checks the checksums of the on-demand sections (full-precision
    vectors).
  - Damage is `Error::Corrupt` with the file's path, never a panic.
- **Versioning.** Files carry `FORMAT_VERSION` (1); a newer version is
  `UnsupportedFormat`. While the crate is unpublished there is no compatibility promise
  (as with arko_index's on-disk data). Once arko depends on it, saved indexes stay
  loadable (AGENTS.md): new versions read old ones.

## 9. Concurrency

- **One writer per directory**, enforced by an exclusive file lock (in this process or
  another): a second `Writer::create` or `open` gets `Locked`.
- **Many readers.**
  - `Snapshot` is `Send + Sync` and immutable; any number of threads search it at once.
  - `Reader` hands out the latest snapshot. It never blocks the writer for more than an
    `Arc` clone.
- **Cross-process.** Readers in other processes see commits (`refresh`). A daemon serving
  several agents holds one writer and hands out `writer.reader()`.
- **Searches use the calling thread.** QPS at N threads comes from N concurrent callers.
  Builds use `StoreOptions::threads` threads.

## 10. Determinism

- **Results** are a pure function of the snapshot's contents, the query, `k`, the filter
  and the parameters. They never depend on thread counts, timing, allocation addresses or
  hash-map iteration order.
- **Builds are byte-identical** for any `threads`: a segment's bytes depend only on its
  records, the configuration and its id. Graph construction must be deterministic in
  parallel, for example by ParlayANN-style batch insertion.
- **Layouts.** The same sequence of `apply`, `commit`, `maintain`/`compact`, `plan_*` and
  `install` calls produces the same files. A test compares directories built with 1 and 4
  threads byte for byte.
- **Fixed algorithms.**
  - Persisted hashes use `util::stable_hash` (FNV-1a then the SplitMix64 finalizer), never
    std's unspecified `DefaultHasher`.
  - Random choices use `util::SplitMix64`, seeded from `IndexConfig::seed` and the
    segment id.
  - No timestamps in any file.

## 11. Memory

- **Resident per vector** (open segment):
  - the code: int8 `dim`+≤8, binary `dim/8`+≤8, f32 `4 x dim`;
  - the graph: `4 x R` (256 B at R = 64);
  - the key, plus at most 32 bytes of bookkeeping: key index, path ordinal, language id.

  Distinct path strings come on top. Tombstones take `len/8` bytes per segment.
  Full-precision vectors stay on disk (except for f32 codes).

  | | per vector at 1024 dims | Linux (1.34M) | 5.8M |
  |---|---|---|---|
  | int8 codes, R = 64 | ~1.4 KB | ~1.9 GB | ~8 GB |
  | binary codes, R = 64 | ~0.5 KB | ~0.65 GB | ~2.8 GB |

  At 256 dimensions (a Matryoshka prefix) the int8 figure drops to ~0.6 KB. arko_index
  chooses codes and dimensions from the machine's limits (an owner decision: caps from
  machine limits).
- **Builds.** A seal or merge uses at most `build_memory_budget` (default 1 GiB) beyond the
  open index. `merge::max_segment_vectors(config, budget)` gives the largest segment a
  budget can build, and every plan respects it. Measured as peak RSS minus the resident
  index minus the process's baseline. Full-precision vectors are streamed (positional
  reads and writes), never held all at once.
- **Delta**: at most `delta_max_vectors x (4 x dim + ~100)` bytes.
- **Queries**: O(beam x R + rescore x dim) per search, plus at most `len/8` bytes per segment
  for a filter bitset. No allocation proportional to the index for unfiltered queries.

## 12. Codec accuracy (unit level)

On the acceptance tests' generated data (2,000 vectors, 50 queries, clustered unless
noted):

- **int8**: the true top 10 is among the codec's top 40 estimates in ≥ 98% of cases
  (cosine at 128 and 256 dims, dot, L2, anisotropic cosine at 256). The mean absolute
  estimate error is ≤ 1% of the mean absolute score plus 0.001. `score_codes` is
  symmetric; training is deterministic.
- **binary**: true top 10 within the top 100 estimates ≥ 95% (clustered cosine), ≥ 90%
  (clustered dot and L2). Within the top 200 it must reach ≥ 50% on anisotropic data
  (plain sign bits reach ~34%; centring ~63%).

## 13. Targets

Measured by `mhr_vector_bench` (README.md) on the build box: AMD EPYC 9354P, Ubuntu 24.04,
default release profile (no `target-cpu` flags: what users get), warm page cache, k = 10,
default parameters unless noted.

### 13.1 Recall (hard floors; never lowered)

Datasets are in section 14.

| Configuration | Datasets | Recall@10 |
|---|---|---|
| exact (`IndexConfig::exact`, or `SearchParams::exact`) | all | **equal to the oracle**: same keys, bit-identical scores, same order |
| `flat-int8` | dbp3l-\*, code-voyage4-1024, sift-128 | ≥ 0.99 |
| `vamana-int8` (the default configuration) | dbp3l-1024, dbp3l-1536, dbp3l-256, code-voyage4-1024, sift-128-euclidean, synth-1024-6m | ≥ 0.95 at the default beam; recall@50 ≥ 0.95 at k = 50 |
| `vamana-int8` | glove-100-angular, nytimes-256-angular (hard sets) | ≥ 0.95 at some beam ≤ 512; the default beam's recall may not drop by more than 0.01 against main |
| `vamana-int8`, filtered | dbp3l-1024: p50, p10, p01, p001, go, p10-python; code set: `path:` and `lang:` filters | ≥ 0.95 for every filter |
| `vamana-binary` | dbp3l-1024, code-voyage4-1024 | ≥ 0.90 at the default beam; ≥ 0.95 at some beam ≤ 512 |
| after updates (10% inserts, then 10% deletes); after `maintain`; after `compact` | dbp3l-1024 | ≥ 0.95 at each point |
| conformance scenarios (`mhr_vector_conform`) | generated | exact checks equal; mean recall ≥ 0.95 (int8, f32), ≥ 0.90 (binary) |

The code set (~11k vectors) is below `graph_min_vectors`, so it is searched flat. That is
realistic for a typical repository and makes it a quantization and rescoring check.

### 13.2 Latency and throughput (goals)

| Workload | Goal |
|---|---|
| dbp3l-1024, 500k, `vamana-int8`, 1 thread | p50 ≤ 1.0 ms, p99 ≤ 3 ms (≥ 1,000 QPS) |
| dbp3l-1024, 990k (Linux-scale proxy), 1 thread | p50 ≤ 1.5 ms, p99 ≤ 4 ms |
| synth-1024-6m (5.8M-scale, several segments), 1 thread | p50 ≤ 5 ms, p99 ≤ 15 ms |
| code-voyage4-1024 (11k, flat), 1 thread | p50 ≤ 0.5 ms |
| filter p001 (0.1%) on dbp3l-1024 990k | p50 ≤ 2 ms |
| 32 threads, dbp3l-1024 500k | QPS ≥ 20 x the single-thread QPS |
| `apply` of 100 upserts (1024 dims) | p99 ≤ 20 ms (read-your-writes) |
| inserts in batches of 1,000 (maintenance excluded) | ≥ 50,000 vectors/s |
| deletes | ≥ 200,000 keys/s |
| `commit` of a delta of ≤ 8,192 vectors | ≤ 500 ms |
| build dbp3l-1024 500k (load + compact), 16 threads, 2 GiB budget | ≤ 3 min |
| build synth-1024-6m, 32 threads, 4 GiB budget | ≤ 60 min |
| peak build memory | ≤ resident index + budget + 512 MiB |
| open, 990k, warm / cold (NVMe) | ≤ 1 s / ≤ 5 s |
| disk after compaction | ≤ 1.15 x (4·dim + code + 4·R + 100) bytes per live vector |

These goals are the first estimates, made before any implementation exists. After the
first complete baseline the owner may adjust them; recall floors are never lowered.
The speed gate optimizes one number (SWARM-PACK.md, "The score").

## 14. Datasets

| Name | Source | Vectors | Dims | Metric | Role |
|---|---|---|---|---|---|
| dbp3l-1024 | text-embedding-3-large on DBpedia (HF Qdrant/…-1536-1M, MIT), Matryoshka prefix | 990k + 10k queries | 1024 | cosine | **the score**; guards; voyage-code-4 stand-in |
| dbp3l-1536, dbp3l-256 | same, other prefixes | same | 1536, 256 | cosine | dimension coverage (2048-class, 256) |
| code-voyage4-1024 | arko_index chunks of 11 pinned repositories via voyage-code-4 (`build_code_set.py`) | ~11.5k + 1,000 doc-comment queries | 1024 | cosine | the real thing: quantization, filters with real paths |
| glove-100-angular, nytimes-256-angular | ann-benchmarks | 1.18M, 290k | 100, 256 | cosine | hard sets (guards) |
| sift-128-euclidean | ann-benchmarks | 1M | 128 | L2 | L2 coverage |
| synth-1024-6m | `mhr_vector_bench synth` (clustered) | 6M | 1024 | cosine | scale, segments, memory bounds |
| glove-25-angular-100k, synth-small | ann-benchmarks slice; synthetic | 100k; 20k | 25; 32 | cosine | local smoke runs |

Datasets without real paths get synthetic paths and languages
(`generate::path_for(i, PATH_SEED)`), so the standard filters have known selectivities:
p50 50%, p10 10%, p01 1%, p001 0.1%, go 10%, p10-python 3%. Ground truth is always
computed by the oracle's arithmetic (`mhr_vector_bench gt`). The sources' own neighbour
lists are only a cross-check: on random-xs-20-angular the agreement is 1.0. Zero vectors in
angular sets (nytimes has 239) are dropped at conversion.

## 15. Embedding models and the cache (the arko_index surface)

- **`EmbeddingModel`**: provider, model, dimensions, document and query input types,
  output type, `normalized`, `text_format` (how arko_index renders a chunk;
  `arko_index/chunk-v1` = path, newline, source), and an optional revision.
  `EmbeddingModel::voyage_code_4(dims)` is the default.
- **`requires_reembed`.** Any change to provider, model, dimensions, document input type,
  output type, text format or revision changes document vectors. A change to the query
  input type or `normalized` does not.
- **`document_fingerprint`** is a file-name-safe identity, equal exactly when no re-embed
  is required.
- **The model is saved with the index.** It is part of `IndexConfig`, so it lives in every
  manifest. `Writer::open_for_model` refuses a different model (`ModelMismatch`, which
  carries both models); the caller creates a new index and re-embeds.
- **`EmbeddingCache`**: `<root>/<fingerprint>/`, an append-only checksummed log keyed by
  `ContentHash`. The hash is SHA-256 of the exact rendered text; the caller computes it.
  - A torn tail is truncated at open.
  - `missing(hashes)` says what to embed.
  - `put` and `flush` add vectors; `retain` compacts.
  - One writer per cache (`Locked`).
  - Switching models and switching back reuses the old cache.
- **Integration sketch (arko_index).**
  - Indexing:
    1. render each chunk;
    2. hash it;
    3. ask `cache.missing`;
    4. embed the missing texts with arko_ai's Voyage client (`input_type: document`);
    5. `cache.put` and `flush`;
    6. `writer.apply` upserts (key = chunk ID, path, language) and deletes (removed IDs);
    7. `writer.commit(watcher_sequence)` on the debounce tick.
  - Querying: embed the query (`input_type: query`), `snapshot.search(.., filter)`, and fuse
    with BM25 by key; reciprocal-rank fusion with k = 60 is a sound default.
  - Dense search is opt-in until a local embedding model exists (owner decision of
    2026-10-06).
- This crate never calls a provider and depends on nothing in arko.

## 16. Dependencies, safety and portability

- **`#![forbid(unsafe_code)]`** in every crate (workspace lint).
  - An exception needs a measured, large gain on the score workload, a minimal audited
    block in a separate module, and the owner's approval.
  - Candidates: mmap of segment files; runtime SIMD dispatch.
- **Runtime dependencies** (pure Rust, safe APIs): `serde`, `serde_json`, `thiserror`,
  `crc32fast`. Dev and host tooling adds `proptest`.
  - Adding a dependency is an owner decision. The gate refuses any change to `Cargo.toml`
    dependencies or a new package in `Cargo.lock`.
  - Never: `-sys` crates, build scripts that compile C or C++, FFI.
  - **Owner decision pending**: whether to pre-approve a safe-API SIMD dispatch crate
    (`pulp`) for the optimization phase. Without one, default builds autovectorize to the
    baseline target (SSE2 on x86-64, NEON on aarch64), and `popcnt` and AVX2 are reached
    only with `-C target-cpu`.
- **Portable.** Linux, macOS, Windows (std file locking, positional reads); no
  platform-specific code outside `format.rs` and `manifest.rs`.
- **Rust 1.89** or later: `File::lock`/`try_lock`, let-chains, edition 2024.

## 17. Out of scope for v1 (possible later)

- Product quantization, RaBitQ-style binary estimators (the binary codec may adopt one
  within its unit), IVF partitions.
- Storing f16 or int8 vectors only, without f32.
- An `mmap` feature (needs an `unsafe` approval).
- Background maintenance threads inside the store (callers run jobs on their own threads).
- Local embedding models (arko side).
- Multi-vector records, sparse vectors.
- Deletes by path prefix (callers delete by key; arko_index knows its keys).
