# Reference

How the store behaves: records, scores, filters, search, updates, persistence,
concurrency, memory and configuration. The rustdoc of each type has the details of its
API.

## Records

A record has four parts:

| Field | Limits | Use |
|---|---|---|
| `key` | 1 to 255 bytes, unique in the index | identifies the record and breaks score ties |
| `vector` | exactly `dim` finite components, each at most 1e15 in magnitude, non-zero for cosine | what searches score |
| `path` | at most 4,096 bytes | filters select records by path prefix |
| `language` | at most 64 bytes, and 65,536 distinct values per segment | filters select records by language |

A path is any `/`-separated hierarchy, such as a file path or `tenant/collection/item`,
and a language is any short label. Records that need neither can use empty strings.

`Snapshot::get` returns a stored record, and search hits carry the key, the score, the
path and the language.

## Scores and metrics

Every score is "higher is better":

| Metric | Stored vector | Score |
|---|---|---|
| `Cosine` | the vector normalized to unit length | `dot(normalize(query), stored)` |
| `Dot` | the vector as given | `dot(query, stored)` |
| `L2` | the vector as given | `-squared_distance(query, stored)` |

Scores use a fixed arithmetic. A dot product keeps 16 `f32` lane accumulators, adds
element `i` to lane `i % 16` in increasing order without fused multiply-add, and reduces
the lanes in a fixed tree. Squared distances work the same way. The SIMD kernels keep
this order, so a record's score has the same bits on every machine and through every
search path. The `distance` module documents the arithmetic in full.

Results are ordered by score, highest first, and then by key in byte order.

## Filters

A `Filter` holds a list of path prefixes and a list of languages. A record matches when
its path matches any of the prefixes and its language equals any of the languages. An
empty list matches everything.

Prefixes match whole path components. `src/foo` matches `src/foo` and `src/foo/bar.rs`,
but `src/foobar.rs` and `lib/src/foo` fall outside it. A trailing `/` is ignored, and
`""` and `/` match every path. Comparisons are on bytes.

```rust
let filter = Filter {
    path_prefixes: vec!["src".into(), "tests".into()],
    languages: vec!["rust".into()],
};
```

`Filter::path` and `Filter::language` build the one-element cases.

## Search

`Snapshot::search(query, k, filter, params)` returns exactly `min(k, matching live
records)` hits, with distinct keys, in result order, for `k` from 1 to 1,000. Every hit's
score is exact: candidates are rescored with the full-precision vectors before they are
returned.

Each segment is searched by one of two plans:

- **Flat:** score the segment's compact codes for every matching record, then rescore
  the best candidates at full precision. Used for segments without a graph, and for
  filtered searches where at most `filter_flat_below` (default 2,048) of a segment's live
  records match.
- **Graph:** a beam search over the segment's Vamana graph that returns only live,
  matching records, then rescoring. A graph search that finds too few records falls
  back to a flat search of that segment.

The in-memory delta (recent upserts) is searched exactly, and the lists of all segments
are merged into the top `k`.

`SearchParams` overrides the defaults for one search:

| Field | Default | Effect |
|---|---|---|
| `beam` | `search_beam` (64) | the graph search's candidate list, raised to at least the rescore count |
| `rescore` | `max(k × rescore_factor, rescore_min)` | candidates rescored at full precision, at least `k` |
| `exact` | `false` | score every matching record at full precision: the results of a brute-force scan |

## Codes and rescoring

Each segment keeps compact codes in memory to find candidates, and the full-precision
vectors on disk to rescore them:

| Codes | Size per vector | Rescored by default |
|---|---|---|
| `Int8` (the default) | `dim` bytes | `max(4k, 32)` |
| `Binary` | `dim` bits in 8-byte words, plus 8 bytes | `max(16k, 128)` |
| `F32` | `4 × dim` bytes, the vectors themselves | `k` |

Codes are trained per segment on a sample of its vectors, with a fixed seed. Int8 keeps a
minimum and a step per dimension. Binary keeps one sign bit per dimension, relative to a trained
centre, plus two corrections per vector. With `F32` codes the vectors live in
memory and nothing is read from disk at search time.

## Flat or graph

`IndexConfig::new` gives Vamana graphs to segments of at least `graph_min_vectors`
(default 20,000) vectors. Setting `config.ann.kind = AnnKind::Flat` searches every
segment flat.

- **Flat search** reads every code on every query, so its time grows with the index:
  about 65 ns per vector with int8 codes at 1024 dimensions on the benchmark machine.
  Its recall is close to exact (1.0 at k = 10 on both benchmark sets), and merges build
  no graphs, so indexing is fast.
- **Graph search** visits a small part of the index. On the 200,000-vector benchmark set
  it answered queries about 14 times faster than flat search, at recall@10 0.97. Graphs
  take longer to build and add `4 × max_degree` bytes per vector in memory.

`IndexConfig::exact(dim, metric)` keeps f32 codes and no graphs, so every search is a
brute-force scan.

## Updates and visibility

- **`apply(ops)`** takes a batch of `Op::Upsert(record)` and `Op::Delete(key)`. The whole
  batch is validated first, and a bad key, vector, path or language fails the call with
  nothing applied. The operations then apply in order, so a later operation on a key
  wins. An upsert replaces any live record with the same key. A delete of an absent key
  is counted in `ApplyReport::missing`.
- **Read-your-writes.** `apply` publishes a new snapshot before it returns, so its
  changes are visible to the next `snapshot()` in the same process.
- **Sealing.** When the delta reaches `delta_max_vectors` (default 8,192), it is written
  to a new segment, during `apply` if need be.
- **`commit(checkpoint)`** seals the delta, writes the tombstones that changed, and
  publishes the next generation with an optional caller token of up to 4 KiB, such as a
  sequence number of the caller's update stream. `Writer::checkpoint` returns it after a
  reopen. Uncommitted changes are lost when the process ends, and the token tells the
  caller where to resume.
- **`version`.** Every operation of every successful `apply` counts. Snapshots report it
  and commits persist it.

### Maintenance

Deletes leave tombstones, and every seal adds a segment. Merges rewrite segments to
purge deletions, combine small segments and build graphs. They run when the caller asks:

- `maintain()` runs the merges the policy wants: eight segments of one size tier
  (`segments_per_tier`), a segment with at least a quarter of its records deleted
  (`max_deleted_fraction`), or a large segment without a graph.
- `compact()` seals the delta and merges until no deletion is left, no two segments could
  merge within the memory budget, and every large-enough segment has a graph.
- For long merges off the writer's thread: `plan_maintenance()` (fast, no I/O) returns a
  `MaintenanceJob`, `job.run()` builds the merged segment on any thread, and
  `writer.install(built)` swaps it in, applying the deletes made in the meantime.

## Persistence and crash safety

An index directory holds:

```text
LOCK                          the writer lock
CURRENT                       the name of the current manifest
gen-<20 digits>.json          one manifest per committed generation
seg-<20 digits>.mhrv          sealed segments, immutable
del-<seg 20>-<gen 20>.bits    a segment's tombstones as of a generation, immutable
*.tmp                         unfinished writes
```

Every file is written as `<name>.tmp`, fsynced, renamed and followed by an fsync of the
directory. A manifest is written only after every file it names is durable, and
`CURRENT` only after the manifest. A crash at any point, of the process or the machine,
leaves the last committed generation intact. The next `Writer::open` removes the
leftovers of an unfinished commit, and each commit removes the files that neither it nor
the previous generation uses.

Segments are section files: a 64-byte header, sections aligned to 4,096 bytes for vectors,
codes and graphs, and a table of the sections with a CRC-32 each. Opening checks headers,
tables and every section it loads into memory. `VerifyMode::Full` also checks the
full-precision vectors. Damage is reported as `Error::Corrupt` with the file's path.

Files carry a format version, and files of another version are refused with
`Error::UnsupportedFormat`.

## Concurrency

- **One writer per directory**, held by an exclusive file lock. A second `Writer::create`
  or `Writer::open`, from this process or another, gets `Error::Locked`.
- **Any number of readers.** A `Snapshot` is immutable, `Send + Sync` and cheap to clone,
  and later writes leave it unchanged. `writer.reader()` follows the writer's snapshots in
  the same process.
- **Other processes** open the last commit with `Reader::open` and pick up newer commits
  with `Reader::refresh`. A reader keeps the files of its generation open, so it keeps
  working while the writer commits and cleans up.
- **Threads.** A search runs on the calling thread, so N concurrent callers give N-way
  parallelism. Seals, merges and graph builds use `StoreOptions::threads`.

## Determinism

- Search results depend only on the snapshot's records, the query, `k`, the filter and
  the parameters.
- A segment's bytes depend only on its records, the configuration and its id, whatever
  the thread count.
- The same sequence of calls produces the same files. Random choices come from a PRNG
  seeded by `IndexConfig::seed` and the segment id, and no file holds a timestamp.

## Memory

- **Resident while open**, per vector: the code (see [Codes and rescoring](#codes-and-rescoring)),
  `4 × max_degree` bytes of graph (256 bytes at the default 64) when the segment has one,
  and the key with a few dozen bytes of bookkeeping. Each distinct path is stored once
  per segment. Full-precision vectors stay on disk, except with f32 codes. At 1024
  dimensions an index of 200,000 vectors with int8 codes held 213 MiB, and 262 MiB with
  graphs.
- **Builds.** A seal or merge uses at most `build_memory_budget` (default 1 GiB) beyond
  the open index. The budget also caps the size of a segment, so large indexes become
  several segments, and searches cover them all.
- **The delta** holds at most `delta_max_vectors` records at full precision.
- **On disk**, each vector takes its full-precision copy plus what it holds in memory.

## Embedding models and the cache

An index can record the model its vectors come from. `IndexConfig::for_model(model)`
stores an `EmbeddingModel` (provider, model name, dimensions, input types, output type,
and a name for how texts are rendered before embedding) in the index's configuration.
`Writer::open_for_model` refuses an index built for a model whose document vectors
differ, with `Error::ModelMismatch`, and the caller re-embeds into a new index. A change
of the query input type alone keeps the index usable.

`EmbeddingCache` keeps vectors by content hash, one directory per model under a root, so
an unchanged text is embedded once across re-indexing and across indexes:

- `missing(hashes)` lists what to embed, `put` adds vectors and `flush` makes them
  durable.
- The cache is an append-only log with a checksum per record. A torn tail is truncated
  at open, and `retain` compacts the log.
- One handle per cache may write (`Error::Locked` otherwise).

The caller computes the hashes (SHA-256 of the exact text sent to the model, for
example) and calls the embedding provider.

## Configuration

`IndexConfig` is saved with the index and fixed at creation:

| Field | Default | Meaning |
|---|---|---|
| `dim` | | vector dimension, 1 to 4,096 |
| `metric` | | `Cosine`, `Dot` or `L2` |
| `embedding` | `None` | the model the vectors come from |
| `codes` | `Int8` | the codes kept in memory |
| `rescore_factor`, `rescore_min` | 4, 32 (int8) | candidates rescored: `max(k × factor, min)` |
| `ann.kind` | `Vamana` | `Flat`, or `Vamana` with its parameters below |
| `ann.graph_min_vectors` | 20,000 | smallest segment that gets a graph |
| `ann.filter_flat_below` | 2,048 | matching records below which a filtered search scans flat |
| `seed` | 0 | seeds code training and graph construction |

`VamanaParams`: `max_degree` 64, `build_beam` 128, `alpha` 1.2, `passes` 2,
`search_beam` 64.

`StoreOptions` apply to one open handle:

| Field | Default | Meaning |
|---|---|---|
| `build_memory_budget` | 1 GiB | memory a seal, merge or graph build may use |
| `threads` | available cores, at most 8 | threads for builds |
| `delta_max_vectors` | 8,192 | records held in memory before a seal |
| `merge.segments_per_tier` | 8 | segments of one size tier that trigger a merge |
| `merge.max_deleted_fraction` | 0.25 | deleted share that triggers a rewrite |
| `verify` | `Headers` | how much of each file `open` checks |

## Platforms and dependencies

The store runs on Linux, macOS and Windows and needs Rust 1.89 or later. It reads files
with positional reads, and every crate has `#![forbid(unsafe_code)]`. The runtime
dependencies are `serde`, `serde_json`, `thiserror`, `crc32fast` and `pulp`, which picks
SIMD kernels at run time.
