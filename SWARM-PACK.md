# The mhr_vector task pack for agent_swarm

How this repository maps onto the `agent_swarm` engine. The engine is
`arko_workflows::swarm`, generalized from ts_port; its interfaces are in
`arko_workflows/examples/agent_swarm/DESIGN.md`, and its closest existing pack is
`packs/crate_fix`. The pack is called **`vector`** and runs in **merge mode**:

1. **prep**;
2. **build**: the 19 units;
3. **tidy**: buckets from the oracle;
4. **milestone**: the guard and full suites;
5. **optimize**: the speed score;
6. **optimize-build**, opt-in: the build score;
7. **done**.

There is no architect phase. The skeleton, the contract (`contract/signatures.txt`) and
the acceptance tests in this repository are the architect's output, written by the host.

## 1. What exists, and what the pack must add

**In this repository**:

| Piece | Path |
|---|---|
| the crate skeleton | `crates/mhr_vector` |
| per-unit acceptance tests | `crates/mhr_vector/tests/uNN_*.rs` |
| the unit plan | `units.json`, `UNITS.md` |
| the frozen interface | `contract/signatures.txt`, checked with `scripts/signatures.py` |
| the oracle | `crates/mhr_vector_oracle` |
| the conformance runner (the swarm's oracle) | `mhr_vector_conform` |
| the harness, the score and the build score | `mhr_vector_bench` |
| property tests | `crates/mhr_vector_bench/tests/properties.rs` |
| workloads | `workloads/` |
| dataset scripts | `scripts/` |
| host-owned paths | `HOST_OWNED.md` |

**The pack** (`arko_workflows/examples/agent_swarm/packs/vector/`, about crate_fix's size
plus ts_port's benchmark logic):

- settings;
- prep;
- `next_work` with unit dependencies;
- `check` (the gate);
- the bucket makers: conformance, tests, clippy, guards, profile;
- the agents' tools;
- prompts and status.

**Engine additions** (small):

1. **Side by side on a value, not on wall time.** `swarm::measure::side_by_side` times
   commands. The score is a "higher is better" number read from JSON, because the
   operating point (the beam) differs between main and a branch, so wall times are not
   comparable. The helper needed is
   `side_by_side_values(a, b, runs, parse: fn(&str) -> Result<f64, String>)`. It runs the
   sides alternately, collects `primary`, and offers `b_better(min_gain)` and `b_worse()`
   with the same noise rule. (ts_port kept its own benchmark code; this pack could too,
   but the helper is generic.)
2. Nothing else. Unit dependencies live in the pack's `next_work`. Scopes, owned paths,
   the standard gate, parallel attempts, tag-ins, notifications and stopping are the
   engine's.

## 2. Settings (`settings.json`, the pack's fields)

| Setting | Default | Meaning |
|---|---|---|
| `source` | none | This repository (a folder or a `git bundle`), copied once as the run's `main`. |
| `datasets` | `/work/mhr-vector/datasets` | `MHRV_DATASETS` for every harness and conformance command. |
| `score_workload` | `workloads/score.json` | The speed gate's workload (frozen: its fingerprint is recorded in prep). |
| `build_score_workload` | `workloads/build-score.json` | The build phase's workload. |
| `guards_workload` | `workloads/guards.json` | The recall guards (milestones, optimization gates). |
| `full_workload` | `workloads/full.json` | The milestone suite. |
| `bench_cpus`, `build_cpus` | `28-31`, `0-27,32-59` | Box CPUs: the score runs on 4 reserved physical cores (their SMT siblings 60-63 stay idle); everything else is pinned to the rest. |
| `bench_runs`, `min_gain` | 5, 0.02 | Side-by-side runs per side; the smallest gain beyond the noise. |
| `conform_jobs`, `conform_timeout_secs`, `conform_seeds` | 16, 900, 2 | Conformance processes, the per-check limit, and the seeds per scenario check (`MHRV_CONFORM_SEEDS`). |
| `test_timeout_secs` | 1800 | `cargo test` in a gate. |
| `optimize`, `optimize_build` | true, false | The phases after the milestone. |
| `target_score` | none | Stop the optimization phase at this score (none: the plateau rule decides). |
| `plateau_attempts` | 12 | Optimization ends after this many gated attempts in a row with no merge. |

Engine settings for the box: `workers` 16, `parallel_attempts` 2, `max_repairs` 3,
`max_attempts` 3, `service_tier` fast, `jobs_per_worker` 4, Astra 6 xhigh for every role.
`tag_ins` 8 (Opus 5.5 xhigh) for units or buckets that Astra failed twice; mainly U11,
U14, U17 and U18, and optimization.

## 3. Prep (host, no models)

1. Copy `source` into `work/main` (git). Check that the skeleton commit is there and that
   `python3 scripts/signatures.py --check contract/signatures.txt` passes.
2. Check the toolchain (rustc ≥ 1.89), `perf` (with `perf_event_paranoid` ≤ 1), `taskset`
   and Python 3.
3. Check the datasets with `mhr_vector_bench datasets`. Missing ones are reported, with
   `scripts/box_setup.sh` as the fix. The code set (`code-voyage4-1024`) is optional: until
   it exists, its guard parts are skipped with a notice in the status.
4. Build the host tools from `main`: `mhr_vector_bench`, `mhr_vector_conform`. Record
   main's **passing sets**: every test name passing in
   `cargo test --workspace --all-features --no-fail-fast` (libtest lines, crate_fix's
   `parse_tests`), and every passing conformance check. Before any unit both are nearly
   empty: only the oracle's and the bench crate's own tests pass.
5. Record the fingerprints of the workload files (`Workload::fingerprint`). A changed
   workload during a run is refused, as champion mode refuses a changed pack.
6. Plan the units from `units.json`.

## 4. The build phase (units)

**`next_work`**: the first pending unit, in id order, whose `depends_on` units are all
done. With no such unit and idle workers, a parallel attempt on an active L unit
(`parallel_attempts`). A set-aside unit blocks its dependants. Its stubs become a tidy
bucket after the phase, and the status names the blocked units.

**The prompt**:

- the unit's section of UNITS.md;
- "Rules for every unit";
- the REQUIREMENTS.md sections it implements;
- its acceptance command;
- its scope.

The essentials, verbatim: do not edit tests or host-owned files; do not change frozen
signatures; no `unsafe`, no dependencies; if the contract or a test is wrong, record a
lesson and finish with that report.

**The gate (`check`)**, in order, rejecting at the first failure with a precise reason
(names, the first lines of output):

1. **Scope**:
   - `outside_scope(changed, unit.scope)` is empty;
   - `owned_in(changed, HOST_OWNED)` is empty.
2. `cargo fmt --all -- --check`.
3. `cargo build --workspace --all-targets --all-features`.
4. **Safety and dependencies**:
   - no `unsafe` token in `crates/mhr_vector/src` other than the `forbid` attribute and
     comments (the compiler enforces `forbid` too);
   - `Cargo.lock` byte-identical to main's.
5. `cargo clippy --workspace --all-targets --all-features -- -D warnings`.
6. `python3 scripts/signatures.py --check contract/signatures.txt`.
7. **Stubs**: no `todo!("<unit id> ` left in the unit's scope.
8. **The unit's acceptance test** passes: `cargo test -p mhr_vector --features internals --test uNN_*`.
9. **No test regressions**: `cargo test --workspace --all-features --no-fail-fast`, and
   every test in main's passing set still passes.
10. **No conformance regressions**: `mhr_vector_conform run --jobs N`, and every check
    passing on main still passes.

On a pass, `main` fast-forwards and the passing sets are recomputed. On a rejection, the
reason goes back to the agent (`max_repairs`). Then fresh agents get the earlier attempts'
summaries (`max_attempts`). Then the unit is set aside.

## 5. The tidy phase (buckets from the oracle)

When no unit is pending or active, the host:

- removes the skeleton allow line from `lib.rs`;
- commits that on `main`;
- rebuilds the passing sets;
- makes the buckets.

Every bucket has a `kind`, a `signature`, `examples` (smallest first, capped), a
`count`, and a `scope`:

| Source | Kind | Signature | Examples | Scope |
|---|---|---|---|---|
| `mhr_vector_conform run` fail | `conform` | `<check id>: <failure kind>` (`recall-below`, `exact-mismatch`, `too-few`, `live-count`, `bytes-differ`, ...) | the detail (seed, step, query) and the replay command `MHRV_CONFORM_SEEDS=<n> mhr_vector_conform one <check id>` | from the check-family table below |
| conform panic or crash | `panic` | `panic: <file>: <message stem>` or `todo: Unn` | the check ids that hit it | the panicking file, plus its unit's files |
| conform timeout | `timeout` | `timeout: <check id>` | the check | the check family's scope |
| a failing acceptance or property test | `test` | `test: <binary>::<name>` | the assertion and, for proptest, the shrunk minimal input | the owning unit's scope (property tests: the store scope) |
| clippy, after the allow line is gone | `clippy` | `clippy: <lint>: <file>` | the warnings | that file |
| stubs of set-aside units | `stub` | `todo: Unn` | the stub list | the unit's scope |

**Scopes by check family** (so concurrent buckets don't collide; the ts_port lesson):

| Check family | Scope |
|---|---|
| `kernels/` | `distance.rs` |
| `filter/` | `filter.rs` |
| `exact/`, `exact-params/` | `store/search.rs`, `flat.rs`, `delta.rs`, `topk.rs` |
| `ann/` | `store/search.rs`, `vamana/`, `quant/`, `flat.rs` |
| `durability/`, `persistence/` | `store/writer.rs`, `store/snapshot.rs`, `store/mod.rs`, `manifest.rs`, `tombstones.rs`, `format.rs`, `segment/` |
| `determinism/` | `vamana/build.rs`, `segment/`, `merge.rs`, `store/writer.rs` |
| `concurrency/`, `reader/`, `errors/` | `store/` |
| `model/`, `cache/` | `embedding/`, `store/writer.rs` |

Two buckets with overlapping scopes are never active at once (`parallel_ok`).

**The tidy gate** is the build gate without steps 7 and 8. In their place: at least one
of the bucket's examples now passes. Buckets are recomputed after every merge. The phase
ends when no bucket is open (or all are set aside).

## 6. The milestone (host, no models)

On main:

- `mhr_vector_bench run --suite workloads/guards.json`;
- `mhr_vector_bench run --suite workloads/full.json`.

The reports (results.md) go to `reports/`; the owner gets a notification with the summary
table.

A recall floor broken (REQUIREMENTS.md section 13.1) becomes a `guard` bucket:

- signature `guard: <dataset>/<config>/<k or filter>`;
- example: the measured and required recall;
- scope: `vamana/`, `quant/`, `store/search.rs`, `merge.rs`.

The milestone repeats until the guards pass. Then the baseline score of main is recorded:
`prepare`, then `score` 5 times. The owner may adjust the goals of REQUIREMENTS.md section
13.2 then.

## 7. The optimization phase

### 7.1 The score

```sh
mhr_vector_bench prepare --workload workloads/score.json --index-root $PREP --datasets $MHRV_DATASETS
taskset -c $BENCH_CPUS mhr_vector_bench score --workload workloads/score.json --index-root $PREP --datasets $MHRV_DATASETS
# -> {"primary": 1234.5, "components": {"p0.qps": ..., "p0.beam": 40, "p0.recall": 0.9512, "p0.recall_default": 0.97, ...}, "detail": "..."}
```

**The score is the single-thread queries per second of the default configuration
(`vamana-int8`) at recall@10 ≥ 0.95, on the first 500,000 vectors of dbp3l-1024 (OpenAI
text-embedding-3-large, Matryoshka-truncated to voyage-code-4's 1024 dimensions), with its
first 2,000 held-out queries.**

- **Building the index.** `prepare` builds it through the public API:
  - batches of 10,000;
  - commits every 200,000;
  - `compact`;
  - 16 build threads, a 2 GiB budget.

  Main's prepared index is cached per main commit.
- **The operating point.** `score` searches the beam ladder (10, 12, ..., 512) upward with
  untimed recall passes (4 threads, oracle ground truth, tie-aware recall). It takes the
  smallest beam whose mean recall@10 reaches 0.95, warms up, and times 3 single-thread
  passes over the 2,000 queries. The median pass's QPS is `primary`.
- **`primary` is 0** when any result breaks a rule (a wrong count, a non-canonical score,
  a filter violation, ...) or no beam reaches 0.95. A better graph, faster kernels and
  cheaper rescoring all raise it. Trading recall away cannot, because the threshold is
  fixed.
- **The build.** The workspace release profile (opt-level 3, line tables for `perf`), no
  `RUSTFLAGS` and no `target-cpu`: what a user's `cargo build --release` produces. If the
  owner pre-approves a safe-API SIMD dispatch crate (REQUIREMENTS.md section 16), runtime
  dispatch is how faster instructions are reached.
- **Pinned.** `bench_cpus` are 4 physical cores kept free of everything else. The timed
  pass is single-threaded; the recall passes use the other 3 cores.

### 7.2 The gate for an optimization branch

All of these:

1. Build gate steps 1-6, 9 and 10 (scope: the bucket's; no stub step).
2. **Faster beyond the noise.**
   - Prepare the branch's index on `build_cpus`.
   - Run `score` for main and the branch alternately (A B, B A, ...), `bench_runs` (5)
     each, on `bench_cpus`.
   - Accept when `median(branch) ≥ median(main) × (1 + max(noise, min_gain))`. The noise
     is the larger side's largest relative distance from its median (`measure::noise`).
   - Reject as "slower" when the branch is below main beyond the noise.
3. **Recall floors.** Every recall floor of `workloads/guards.json`.
   - **The hard sets** (glove-100, nytimes-256) must still reach 0.95 on the ladder, and
     their default-beam recall may drop by at most 0.01 against main's (recorded at the
     milestone and updated at each merge).
   - The score's own `recall_default` must stay ≥ 0.95.
4. **Build and memory.**
   - `build-score` side by side (3 runs each): the branch builds no slower than main beyond
     `max(noise, 10%)`;
   - peak build RSS ≤ resident index + budget + 512 MiB;
   - prepared-index disk size ≤ main's × 1.02.
5. **Determinism** stays: the `determinism/*` conformance checks are part of step 10.
6. **Exact paths stay exact.** The canonical kernels' bits never change; step 10's exact
   checks enforce it.

### 7.3 Profiles to buckets (one workload)

**The invariant: the profiler, the bucket maker and the gate measure the same workload.**
Here they share one command, one workload file and one prepared-index recipe.

- **The profiler.** On main's prepared index:

  ```sh
  perf record -F 999 -g -o $RUN/perf.data -- \
    taskset -c $BENCH_CPUS mhr_vector_bench score --workload workloads/score.json --index-root $PREP_MAIN
  perf report -i $RUN/perf.data --no-children --sort symbol --demangle --stdio
  ```

  This is the exact command the gate times. `--demangle` was ts_port's lesson (c9920c5):
  without it, buckets point at mangled noise.
- **The bucket maker.** It groups self time by the module of each `mhr_vector::` symbol
  path:

  | Module | Files |
  |---|---|
  | `distance` | `distance.rs` |
  | `quant::int8` | `quant/int8.rs` |
  | `vamana::search` | `vamana/search.rs` |
  | `flat` | `flat.rs` |
  | `store::search` | `store/search.rs` |
  | `segment` | `segment/` |
  | `topk` | `topk.rs` |
  | `filter` | `filter.rs` |
  | `format` | `format.rs` |
  | `delta` | `delta.rs` |

  Plus two pseudo-modules:
  - `alloc` (malloc, free, realloc, memcpy, memmove);
  - `io` (the kernel: pread and page faults).

  Each module holding ≥ 3% of samples is a `perf` bucket:
  - examples: the top symbols with their percentages, the score's components (beam,
    p50, p99) and main's score;
  - scope: the module's files (`alloc` and `io`: the top two callers' modules, from the
    call graph).
- **The agents' tools.** Agents get a `profile` tool (the same command on their worktree's
  build, on `build_cpus`, informational) and a `score` tool (prepare plus score of their
  build, informational). The gate's measurement is the only one that counts.
- **After every merge**, main is re-prepared and re-profiled and the buckets recomputed. A
  `perf` bucket stays open after a merge (`reopen_after_merge`) until agents fail it
  `max_attempts` times.
- **The prompt** names the hard rules (identical exact results, recall floors, no `unsafe`,
  no dependencies, results independent of thread counts) and the usual levers:
  - code loops the autovectorizer handles (`chunks_exact`, fixed-width accumulators);
  - no allocation per query (scratch reuse);
  - batched and sorted positional reads for rescoring;
  - cache-friendly code layouts (code and neighbours adjacent);
  - early exits in top-k;
  - better graphs, which reach 0.95 at a smaller beam: degree, alpha, entry points, two
    passes;
  - cheaper filters.

### 7.4 Stopping rules

The optimization phase ends at the first of:

- `target_score` reached, when set (none by default);
- `plateau_attempts` (12) gated attempts in a row without a merge;
- every `perf` bucket set aside;
- the engine's `deadline`.

The run then runs the full suite on main and writes `report.md`:

- per-phase merges;
- the score's history;
- the final full-suite table;
- recall floors;
- `unsafe` uses: 0;
- clippy: 0.

### 7.5 The build optimization phase (opt-in: `optimize_build`)

The same structure with `workloads/build-score.json`:

- the score is `mhr_vector_bench build-score` (vectors per second, `primary` 0 when the
  built index's default recall falls below 0.95);
- the profile is `perf record -- mhr_vector_bench build-score ...`;
- buckets come from `vamana/build.rs`, `segment/`, `quant/` training, `merge.rs` and
  `format.rs`;
- the gate additionally keeps the query score not lower beyond the noise.

## 8. Tools for agents

Beside the engine's shell and `record_lesson`:

| Tool | Runs | Phases |
|---|---|---|
| `run_tests` | `cargo test -p mhr_vector --features internals --test <name>` (or every test), as the gate does | all |
| `conform` | `mhr_vector_conform run --filter <prefix>` or `one <id>` | all (meaningful once U18 merged) |
| `score` | `prepare` + `score` of the worktree's build (`build_cpus`) | optimize |
| `build_score` | `build-score` of the worktree's build | optimize-build |
| `profile` | `perf` as in section 7.3, on the worktree's build | optimize phases |

**Each agent's shell environment**:

- its own `CARGO_TARGET_DIR`;
- `CARGO_BUILD_JOBS=jobs_per_worker`;
- `MHRV_DATASETS`;
- `MHRV_SCRATCH` on `/work`;
- pinned to `build_cpus`.

## 9. The box

| What | Where |
|---|---|
| this repository | `/work/mhr-vector/repo`: from a bundle (`git bundle create vector.bundle --all` here, copied over; nothing is pushed) |
| datasets, Python tools, the harness build | `/work/mhr-vector/{datasets,venv,target-host}` (`scripts/box_setup.sh`, about 45 GB, mostly downloads) |
| the code set | `/work/mhr-vector/datasets/code-voyage4-1024` (the owner runs `scripts/build_code_set.py` with the Voyage key) |
| the run | `/work/mhr-vector/run` (settings, state, `work/main`, worktrees, reports) |

**CPUs**: EPYC 9354P, 32 cores and 64 threads. `bench_cpus` 28-31 with siblings 60-63
idle; `build_cpus` 0-27 and 32-59. 755 GB of RAM leave 16 workers with plenty of room: a
gate's builds and indexes take a few GB each.

**Launch**, once the pack exists:

```sh
cargo build --release -p arko_workflows --example agent_swarm
BIN=$PWD/target/release/examples/agent_swarm
$BIN vector init /work/mhr-vector/run --source /work/mhr-vector/repo --datasets /work/mhr-vector/datasets
$BIN vector check /work/mhr-vector/run       # prep: tools, datasets, passing sets; no model calls
$BIN vector selftest /work/mhr-vector/run    # the gate on scripted branches (below)
nohup $BIN vector run /work/mhr-vector/run > /work/mhr-vector/run/run.log 2>&1 &
```

**The selftest.** Scripted branches are made from the throwaway reference implementation
kept outside the repository (`~/Desktop/mhr-work/vector-validation/reference.patch`, never
in the run's repository or bundles). Each unit's files from the patch must pass that
unit's gate; a mutated copy must be rejected:

- a flipped comparison in `topk`, which conformance and the acceptance tests catch;
- a file outside the scope;
- an `unsafe` block;
- a changed signature;
- a new dependency.

## 10. Expected effort

- **The build phase** is about the size of ts_port's translation: 19 units against 21. ts_port took about 1.5
  hours of Astra 6 xhigh fast mode and 8% of the $200 plan's weekly window, though its
  units were translations and these are designs against tests. The critical path is six
  waves deep: U01/U02 → U07 → U11 → U16 → U17 → U18.
- **Tidy** depends on how much the conformance suite finds.
- **Optimization** runs until the plateau rule.
- **Gate time**:
  - build phase: 3-10 minutes, dominated by `cargo test --workspace` and the conformance
    run;
  - optimization: add a prepare (about 1-2 minutes at 500k), 10 score runs (about 10
    seconds each), the guard suite (5-10 minutes) and a 3+3 build-score comparison.
