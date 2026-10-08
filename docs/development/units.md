# mhr_vector: the unit plan

19 units build `crates/mhr_vector` from its skeleton. Each unit owns files that no other
unit touches, implements the `todo!("Unn ...")` stubs in them, and passes its host-owned
acceptance test `crates/mhr_vector/tests/uNN_*.rs`. `docs/development/units.json` is the same plan in
machine-readable form (id, scope, `depends_on`, acceptance command, stub prefix, wave).

## Rules for every unit

- **Scope.** A unit's branch changes only its files (below). Anything else is refused by
  the gate before any check runs:
  - another unit's files;
  - the host-owned paths in docs/development/host-owned.md: tests, the oracle, the bench crate, `lib.rs`,
    `config.rs`, `types.rs`, `error.rs`, `distance.rs`'s signatures, `Cargo.toml`s,
    `Cargo.lock`, docs, workloads.
- **The skeleton's signatures are the contract.** `docs/development/signatures.txt` freezes every
  `pub` item: fn signatures, trait methods, pub fields. A unit may add private items, and
  new `pub` items in its own files. It may not change or remove frozen ones (the gate runs
  `scripts/signatures.py --check`). Two exceptions: the fields of `Int8Codec`,
  `BinaryCodec` and their query types, and the private fields of every struct, belong to
  the unit that owns the file.
- **When the contract is wrong** (a signature cannot work, or an acceptance test
  contradicts docs/requirements.md), the agent records it with `record_lesson` and stops with
  that report. The host fixes the skeleton or the test and resets the unit; agents never
  edit tests.
- **No `unsafe`, no new dependencies, `cargo fmt` clean, clippy clean** with
  `--all-targets --all-features -D warnings`.
- **Done** when the acceptance test passes, no `todo!("Unn` is left in the unit's files,
  and nothing that passed on `main` fails on the branch: other units' acceptance tests,
  `mhr_vector_conform` checks, oracle and bench tests.
- **Stay inside the dependency order.** A unit's code may call only units it depends on,
  directly or transitively. Calling anything else panics until that unit merges (its
  stubs are `todo!`). U07, U08 and U09 depend on U05 (they use its little-endian `f32s`
  helpers, `format::f32s_*_le`); keep their kernels otherwise self-contained.

Run a unit's tests from the repository root:

```sh
cargo test -p mhr_vector --features internals --test u07_quant
```

## Order

```text
wave 1  U01 distance  U02 util  U03 topk  U04 filter  U05 format  U08 int8  U09 binary
wave 2  U06 manifest (U05)  U07 quant-core (U01 U02)  U10 tombstones (U02 U05)
        U12 delta (U01 U03 U04)  U19 embedding (U05)
wave 3  U11 segment (U01 U02 U04 U05 U07 U08 U09)  U14 vamana-build (U01 U02 U07)
        U15 vamana-search (U07)
wave 4  U13 flat (U03 U04 U10 U11)  U16 merge (U10 U11)
wave 5  U17 store-writer (U06 U10 U11 U12 U14 U16 U19)
wave 6  U18 store-search (U13 U15 U17)
```

A unit becomes available when every unit it depends on has merged. With 16 workers, wave
1 runs 7 agents at once. Idle workers take parallel attempts (`parallel_attempts` 2) on
the largest open units (L: U11, U14, U17, U18).

## The units

Sizes: S ≈ 100-250 lines, M ≈ 250-600, L ≈ 600-1,500 (estimates).

### U01 distance (S): `src/distance.rs`
- The canonical `dot` and `l2_squared` (16 lanes, element `i` to lane `i % 16`, no FMA,
  the fixed tree reduction), `normalize`, `validate`, `prepare`, `score`. Also the fast
  kernels `dot_fast`, `l2_squared_fast` and `score_fast`, in any order within 1e-5
  relative.
- Write the canonical kernels so LLVM vectorizes them: an `[f32; 16]` accumulator over
  `chunks_exact(16)` gives identical bits.
- **Accepts** (`u01_distance`):
  - bit-identical to the oracle on 24 dimensions × 3 scales;
  - normalize and prepare identical;
  - the same refusals as the oracle;
  - fast kernels within tolerance;
  - a length mismatch panics.

### U02 util (S): `src/util.rs`
- `stable_hash` (FNV-1a 64 + SplitMix64 finalizer), `mix64`.
- `SplitMix64` with `next_u64`, `below` and `sample` (distinct, sorted, deterministic).
- `Bitset` (fixed length, `to_bytes`/`from_bytes` as little-endian words).
- **Accepts** (`u02_util`):
  - the hash equals a reference written from the spec;
  - the PRNG equals the oracle's SplitMix64;
  - sample properties;
  - bitset operations and their refusals.

### U03 topk (S): `src/topk.rs`
- `order`/`hit_order` (score descending, key ascending, `-0.0 == 0.0`).
- `TopK`: `offer` with key tie-breaks through `KeyLookup`, `threshold`, `into_sorted`.
- `merge_hits`.
- **Accepts** (`u03_topk`): equals a full sort over 200 random trials with heavy ties;
  `threshold` semantics; `offer`'s return value; merge equals top-k of the union.

### U04 filter (S): `src/filter.rs`
- `Filter::matches` and `path_has_prefix` (component-wise).
- `SegmentFilter::compile`/`allows`/`count`/`is_empty` against a `PathTable` (at most two
  ranges per prefix, plus language ids).
- **Accepts** (`u04_filter`):
  - equal to the oracle on generated and edge-case paths;
  - compiled filters allow exactly the matching ordinals of a brute-force table;
  - exact counts;
  - empty filters.

### U05 format (M): `src/format.rs`
- Section files per the module doc (header, aligned sections, CRC table), streaming
  sections, and `finish` (fsync, rename, dir fsync).
- `SectionFile::open` with `Headers`/`Full` verification, `read_section`, positional
  `read_at`.
- `read_exact_at` (Unix `FileExt`, Windows `seek_read`), `atomic_write`, `sync_dir`, and
  the LE encoders.
- **Accepts** (`u05_format`):
  - the byte layout is checked field by field;
  - round trips;
  - flipped bytes, wrong magic, truncation, newer versions: `Corrupt` or
    `UnsupportedFormat`, never a panic;
  - misuse refused; helpers.

### U06 manifest (S): `src/manifest.rs` (needs U05)
- File names; hex; `read_current`; `publish` (manifest then `CURRENT`, both atomic).
- `garbage`: exactly the unreferenced known files.
- `IndexLock` (`File::try_lock`).
- **Accepts** (`u06_manifest`): exact names; publish and read back; a missing or newer
  manifest is reported; the garbage list exactly; the lock is exclusive within one process.

### U07 quant-core (S): `src/quant/mod.rs` (needs U01, U02)
- `F32Codec` (codes are the little-endian vectors; scores canonical).
- `AnyCodec` dispatch, `train` (`F32` / `Int8::train` / `Binary::train`), `from_bytes`,
  `to_bytes`.
- `training_sample`.
- The `with_codec!` macro is given.
- **Accepts** (`u07_quant`): f32 codes and canonical scores; dispatch round trip; training
  samples equal `SplitMix64::sample`.

### U08 int8 (M): `src/quant/int8.rs`
- Train (for example per-dimension ranges from the sample, maybe quantile clipping),
  `encode`, `prepare` (an asymmetric f32×i8 or a symmetric i8×i8 estimate), `score`,
  `score_codes` (symmetric), `to_bytes`/`from_bytes`.
- Codes are `dim` bytes plus ≤ 8.
- Self-contained: no calls into other units.
- **Accepts** (`u08_int8`): true top-10 in the top-40 estimates ≥ 0.98 on clustered and
  anisotropic data for every metric; mean error ≤ 1% of mean |score| + 0.001;
  deterministic; symmetric; round trip; degenerate inputs.

### U09 binary (M): `src/quant/binary.rs`
- Sign bits after a trained centring (needed: the anisotropic test fails without it),
  packed in `u64` words, plus ≤ 8 bytes of corrections.
- Hamming or asymmetric estimates scaled to the metric. RaBitQ-style estimators are
  welcome within the unit.
- **Accepts** (`u09_binary`):
  - top-10 in the top-100: ≥ 0.95 for clustered cosine, ≥ 0.90 for clustered dot and L2;
  - ≥ 0.5 in the top-200 on anisotropic data;
  - layout bounds; deterministic; symmetric; round trip.

### U10 tombstones (S): `src/tombstones.rs` (needs U02, U05)
- Copy-on-write deleted-ordinal sets.
- The file format: magic, counts, bitset, CRC; written atomically.
- **Accepts** (`u10_tombstones`): `with_deleted` leaves the original alone; the file layout
  and round trip; damage, a wrong length and lying counts are `Corrupt`.

### U11 segment (L): `src/segment/` (needs U01, U02, U04, U05, U07, U08, U09)
- `MemoryRecords`.
- `write_segment`:
  1. sort a permutation by (path, key);
  2. train the codec on `training_sample` rows of the sorted order (seed mixed with the
     segment id);
  3. stream full-precision vectors in sorted order (f32 codes: the codes section is the
     vectors);
  4. write the codes, keys and key index, path table, language table, an `Info` JSON, and
     the graph when `build.graph` (call `vamana::build_graph`);
  5. respect `memory_budget` (`Budget` error); refuse an empty source (`InvalidArgument`).
- `Segment::open`, verifying the configuration (`Corrupt` on a dimension, metric or codes
  mismatch).
- Accessors; `find` through the key index; `read_vector(s)` by positional reads;
  `PathTable` (≤ 2 ranges by binary search); `KeyLookup`.
- **Accepts** (`u11_segment`):
  - every record round-trips;
  - sorted by (path, key);
  - codes equal the codec's encoding;
  - exact prefix ranges, including a path equal to a prefix and its siblings;
  - **byte-identical files** for shuffled input and 1 or 7 threads, with f32, int8 and
    binary codes;
  - refusals and damage.

### U12 delta (M): `src/delta.rs` (needs U01, U03, U04)
- The writer's in-memory records: upsert/replace, delete, `contains`.
- `view` must be cheap: shared append-only chunks, copy-on-write deletion marks.
- `take`.
- `DeltaView`: exact filtered search (canonical scores, key ties), `get`; a `RecordSource`
  in key order.
- **Accepts** (`u12_delta`):
  - identical to the oracle through 3,000 random upserts and deletes, every metric and
    standard filter, k of 1, 10 and 50;
  - views isolated; `take` empties;
  - 2,000 views of a 25 MB delta in < 2 s.

### U13 flat (M): `src/flat.rs` (needs U03, U04, U10, U11)
- `flat_search`: scan the codes of live matching ordinals, keep the best `rescore` (TopK
  with key ties), rescore.
- `exact_search`: every live matching vector at full precision.
- `rescore` (canonical, result order); `hit`.
- **Accepts** (`u13_flat`):
  - `exact_search` equals the oracle (Gaussian, duplicates, lattice ties; every metric;
    every filter; with deletions);
  - `flat_search` with f32 codes is exact;
  - int8 with 4× rescoring reaches mean recall ≥ 0.99 and returns canonical scores;
  - rescore ordering.

### U14 vamana-build (L): `src/vamana/build.rs`, `Graph::to_bytes`/`from_bytes` in `src/vamana/mod.rs` (needs U01, U02, U07)
- Deterministic Vamana: greedy search plus robust prune; the first pass with α = 1, the
  second with `alpha`; medoid or centroid-nearest entry; seeded order.
- Parallel only in a way that cannot change the result: ParlayANN batch insertion, where
  the batch's searches read the pre-batch graph and edges are applied in a fixed order.
- Graph serialization with full validation.
- **Accepts** (`u14_vamana_build`):
  - valid rows: ≤ R distinct neighbours, no self loops, padding at the end;
  - **identical graphs for 1, 2, 4 and 7 threads**;
  - ≥ 99.5% of nodes reachable from the entry;
  - a plain greedy search (written in the test) reaches recall@10 ≥ 0.95 at beam 64 on
    3,000 clustered points, cosine and L2;
  - tiny and degenerate inputs;
  - serialization round trip and refusals.

### U15 vamana-search (M): `src/vamana/search.rs` (needs U07)
- `beam_search`: best-first from the entry. Non-admitted nodes route but are never
  returned. The beam is raised to `wanted`. Ties by ordinal. `SearchScratch` reuse (a
  visited set by epoch) must not change results.
- **Accepts** (`u15_vamana_search`), on k-NN plus random-edge graphs built in the test:
  - recall@10 ≥ 0.95 at beam 64 for every metric;
  - results best first, with exact f32 estimates;
  - admission;
  - beam raised;
  - scratch reuse equals a fresh scratch;
  - one-node graphs.

### U16 merge (M): `src/merge.rs` (needs U10, U11)
- `plan`: tiers of `segments_per_tier` similar sizes; segments at
  `max_deleted_fraction`; flat segments that reached `graph_min_vectors`.
- `plan_compaction`: purge every deletion, pack segments so no two could still merge,
  give large segments graphs, ignore empty ones.
- `max_segment_vectors`: the memory math; `Budget` under 1,024.
- `LiveRecords`.
- Plans are deterministic, inputs disjoint and oldest first, outputs within the limit.
- **Accepts** (`u16_merge`):
  - tier, delete and graph plans;
  - compaction reaches a fixed point in < 20 rounds on 50 random layouts, with no
    deletions left, graphs where due, and no two segments mergeable;
  - the memory math's monotonicity and range (1024-dim int8 at 1 GiB: 100k-1.05M
    vectors);
  - live records skip deletions.

### U17 store-writer (L): `src/store/writer.rs`, `src/store/snapshot.rs`, `src/store/mod.rs` (needs U06, U10, U11, U12, U14, U16, U19)
- `create` (publishes generation 1) and `open` (garbage collection, segments, tombstones,
  checkpoint, version); `open_for_model` (`requires_reembed`).
- `apply`: validate everything first; key lookups in the delta and newest segments
  first; tombstones; seal at `delta_max_vectors`; publish a snapshot.
- `commit`: seal, tombstone files, manifest with checkpoint and version, garbage.
- Maintenance:
  - `plan_maintenance`/`plan_compaction` to a `MaintenanceJob`;
  - `run` uses `LiveRecords` and `write_segment`, with graph per plan;
  - `install` deletes from the output every key deleted or replaced in the inputs since
    the plan, and checks the epoch and the writer (`StaleJob`);
  - `maintain`, `compact`, `stats`.
- Snapshot accessors.
- May add fields to the structs in `store/mod.rs`.
- **Accepts** (`u17_writer`, public API):
  - create, commit and reopen; uncommitted work lost;
  - every refusal, with no partial batch;
  - model checks;
  - isolated snapshots;
  - sealing, maintenance and compaction (no deletions, graphs);
  - jobs run on another thread while records change, and their changes survive install;
  - `StaleJob`;
  - garbage removed, foreign files kept;
  - **byte-identical directories for 1 and 4 threads**.

### U18 store-search (L): `src/store/search.rs`, `src/store/reader.rs` (needs U13, U15, U17; may extend `src/store/mod.rs`)
- `Snapshot::search`:
  - validation (`k`, query);
  - per segment: compile the filter; flat when there is no graph, the search is exact, or
    the matches are at most `filter_flat_below`; else beam search plus rescore, with a
    flat fallback when it finds fewer than `min(k, matching live)`;
  - the delta exactly; merge.
- `Reader::open`/`refresh` (retry on files vanishing) and `snapshot`, for both sources.
- **Accepts** (`u18_search`):
  - exact configurations equal the oracle in random operation scenarios, every metric and
    filter, k of 1, 10 and 50;
  - `SearchParams::exact` equals the oracle on graph indexes;
  - graph indexes reach mean recall ≥ 0.95 (int8, f32) and ≥ 0.90 (binary) with every
    universal rule;
  - argument errors;
  - readers see commits only, refresh, held snapshots survive compaction;
  - four reader threads checked against the oracle at each snapshot's version while the
    writer applies, compacts and commits.

### U19 embedding (M): `src/embedding/` (needs U05)
- `requires_reembed`, `document_fingerprint`, `validate`.
- `EmbeddingCache`: per-fingerprint directory with `model.json` (`ModelMismatch`), a lock
  (`Locked`), an append-only log with a CRC per record, a torn tail truncated at open,
  later records win, `missing`, `put` (all-or-nothing validation), `flush`, `retain`
  (rewrite and switch atomically), `stats`.
- **Accepts** (`u19_embedding`): fingerprint text and safety; re-embed rules per field;
  round trip across reopen; `Locked`; refusals; torn tail; `retain`; another model's
  directory refused.

## After the units: tidy, milestones, optimization

These phases are in docs/development/swarm-pack.md. Their work items are **buckets**, not units:

- conformance failures by signature;
- property-test failures;
- clippy findings;
- stubs left by set-aside units;
- recall-guard failures;
- profile hotspots.

### End of the build phase (host)
When every unit has merged:

1. remove the skeleton allow line in `lib.rs` (`#![allow(unused_variables, dead_code, ...)]`);
2. turn every resulting warning into a tidy bucket;
3. run the full conformance suite, the property tests and the guard suite;
4. measure the first complete baseline (`workloads/full.json`), which the owner uses to
   confirm the goals of docs/requirements.md section 13.2.

### Stretch (not planned; needs the host's skeleton work first)
- **U20 pq**: product quantization. `CodeKind::Pq { subspaces }` in `config.rs` (host), a
  `quant/pq.rs` codec (k-means codebooks, ADC tables), dispatch in `AnyCodec`.
- **U21 rabitq**: if the binary codec's estimator is the limit.
- **U22 mmap**: only after an approved `unsafe` exception (docs/requirements.md section 16).
