# Moving the paused run to score-v2

The steps for the box, in order. The run at `/work/mhr-vector/run` is paused with all 19
units implemented and the milestone next. Its `work/main` holds score-v1's workloads.
[docs/datasets.md](../docs/datasets.md) describes the new sets.

## What the vector pack keeps that score-v1 shaped

From the pack at `arko_workflows/examples/agent_swarm/packs/vector` (41fc7a5: `phases.rs`,
`gate.rs`, `host.rs`, `state.rs`, `tools.rs`). The pack's fields sit at the top level of
`state.json` (the engine flattens them).

| What | Where | Needs |
|---|---|---|
| The workloads' fingerprints (FNV-1a of each file), recorded at prep | `state.json`: `prep.workloads` | Removing. Every `check` and `run` calls prep, which refuses a changed workload ("the score workload changed since this run started ... start a new run"). Prep records the new ones. |
| Guard parts skipped for missing datasets, and after a milestone the hard sets' default-beam recall | `state.json`: `guards` | Removing. Prep rewrites `skipped` only when something is missing, so old entries would linger in reports. The milestone measures `hard_default` again. |
| Main's score history (the baseline from the milestone, one point per merge) | `state.json`: `scores`, `build_scores` | Removing if present (they should be empty before the milestone). The target-score rule, the perf buckets' examples and the agents' tools read the last point. |
| Main's prepared score index | `bench/main-score/` (and `bench/main-build/`) | Deleting, to free disk. The host caches it in memory per main commit, and the host commit below changes main's commit, so the milestone prepares score-v2's indexes anyway. |
| The agents' prepared indexes | `targets/<slot>/tool-score/`, `tool-build-score/`, `tool-profile-index/` | Deleting if any exist. The `profile` tool reuses `tool-score/` when its `prepared.json` is there, and `score` refuses an index prepared for another workload. |
| The run's copies of the workloads | `bench/workloads/*.json` | Nothing: prep rewrites them from main. |
| Ground truth | `<datasets>/<set>/gt/` | Computing for the new sets (step 6). Prep would compute it too, with fewer threads. |
| The prompt text that describes the score to agents | the pack's `prompts.rs` | Updating before the optimize phase (step 9). It names dbp3l and 500,000 vectors. |

The milestone counter (`milestones`) only numbers the report folders and can stay.

## Steps

**1. Copy the groundwork and the Cloudflare docs to the box.** On this machine (`$BOX`
is the box's ssh address, as in `bin/vm3`):

```sh
git -C ~/Desktop/mhr-work/vector bundle create /tmp/vector-score-v2.bundle main
scp /tmp/vector-score-v2.bundle $BOX:/work/mhr-vector/
ssh $BOX mkdir -p /work/mhr-vector/cloudflare-docs
scp -r ~/Desktop/cloudflare-docs/docs ~/Desktop/cloudflare-docs/evaluation \
  ~/Desktop/cloudflare-docs/source-manifest.json $BOX:/work/mhr-vector/cloudflare-docs/
```

On the box, bring the groundwork copy up to date:

```sh
cd /work/mhr-vector/repo
git fetch /work/mhr-vector/vector-score-v2.bundle main && git merge --ff-only FETCH_HEAD
```

**2. Stop the run if anything is running**, and look at its state:

```sh
pgrep -af 'agent_swarm vector' || echo stopped
cd /work/mhr-vector/run
python3 -c 'import json; s = json.load(open("state.json")); print(s.get("phase"), "milestones", s.get("milestones", 0), "scores", len(s.get("scores", [])), "active", [u["plan"]["id"] for u in s.get("units", []) if u.get("status") == "active"], [b for b, v in s.get("buckets", {}).items() if v.get("status") == "active"])'
```

Expect phase `units` or `tidy` (or `milestone`), no scores and nothing active.

**3. Dry run** (no Voyage calls). It chunks every source, prints the selection, the
estimates and whether the caps cover them, and writes the plans to
`/work/mhr-vector/voyage-work/plans/`:

```sh
cd /work/mhr-vector/repo
scripts/voyage_sets.sh --dry-run | tee /work/mhr-vector/voyage-dry-run.json
```

Expected: about 200,000 code chunks with each of the 8 languages near 25,000 (fewer
where a language runs out), `docs-voyage4-200k` at about 57,000 chunks (11,626 of them
Cloudflare), 2,000 queries each, and `"fits": true` for both.

**4. Check that every guard filter matches chunks**, before paying. A filter that matches
nothing passes its guard with recall 1.0 and tests nothing:

```sh
python3 - <<'EOF'
import json
plans = "/work/mhr-vector/voyage-work/plans"
for part in json.load(open("/work/mhr-vector/repo/workloads/guards.json"))["parts"]:
    if part["dataset"] not in ("code-voyagecode4-200k", "docs-voyage4-200k") or not part.get("filters"):
        continue
    rows = [json.loads(line) for line in open(f"{plans}/{part['dataset']}-base.jsonl")]
    for f in part["filters"]:
        kind, value = f.split(":", 1)
        if kind == "lang":
            n = sum(r["language"] == value for r in rows)
        else:
            value = value.rstrip("/")
            n = sum((p := f"{r['source']}/{r['path']}") == value or p.startswith(value + "/") for r in rows)
        print(f"{part['dataset']:24} {f:40} {n:7} ({100 * n / len(rows):.2f}%)")
EOF
```

Expected roughly: `lang:go` and `lang:python` 10-13%, `path:redis` 2-4%,
`path:tokio/tokio/src/sync` about 0.5%, `path:tokio/tokio/src/signal` about 0.07%,
`path:mdn/files/en-us/web/javascript` 5-10%, `path:cloudflare/workers` 971 chunks
(1.7%), `path:mdn/files/en-us/glossary` about 1%, `path:cloudflare/kv` 57 chunks (0.1%).
If one is 0 or far off, pick another directory from the plan and change it in
`workloads/guards.json`, `workloads/full.json` and `scripts/box_setup.sh` before step 7.

**5. Embed.** The key comes from the environment and is never printed. A rerun after an
interruption resumes from the cache without billing twice:

```sh
cd /work/mhr-vector/repo
VOYAGE_API_KEY=... scripts/voyage_sets.sh 2>&1 | tee /work/mhr-vector/voyage-build.log
```

Expected bill: about 36M tokens for the code set (~$4.35 at $0.12 per million, before
free tokens) and about 45M for the docs set (~$2.75 at $0.06). The caps stop the code set
at 55M tokens and the docs set at 70M. `dataset.json` records the tokens billed.

**6. Ground truth** for every part of the new workloads (filters included). The bench
code is unchanged, so the existing harness build works:

```sh
B=/work/mhr-vector/target-host/release/mhr_vector_bench
export MHRV_DATASETS=/work/mhr-vector/datasets
T=60
$B gt --dataset code-voyagecode4-200k --queries 2000 --threads $T
$B gt --dataset code-voyagecode4-200k --queries 1000 --threads $T \
  --filters lang:go,lang:python,path:redis,path:tokio/tokio/src/sync,path:tokio/tokio/src/signal
$B gt --dataset code-voyagecode4-200k --base 100000 --queries 500 --threads $T
$B gt --dataset docs-voyage4-200k --queries 2000 --threads $T
$B gt --dataset docs-voyage4-200k --queries 1000 --threads $T \
  --filters path:mdn/files/en-us/web/javascript,path:cloudflare/workers,path:mdn/files/en-us/glossary,path:cloudflare/kv
$B datasets | python3 -m json.tool | grep -A1 -E '"name": "(code|docs)-voyage'
```

The first two cover the score (and build score). The others cover the guards: their
filtered parts, their unfiltered parts at 1,000 queries, and the update part. If step 4
changed a filter, use the new list here. `scripts/box_setup.sh --skip-synth` runs the same
commands.

**7. Clear the state score-v1 shaped** (with the run stopped):

```sh
cd /work/mhr-vector/run
cp state.json state.json.before-score-v2
python3 - <<'EOF'
import json
s = json.load(open("state.json"))
for key in ("prep", "guards", "scores", "build_scores"):
    print(key, "removed" if s.pop(key, None) is not None else "absent")
json.dump(s, open("state.json", "w"), indent=2)
EOF
rm -rf bench/main-score bench/main-build targets/*/tool-score targets/*/tool-build-score targets/*/tool-profile-index
```

**8. Commit the new groundwork into main as a host commit.** The run's main carries the
groundwork at c8ccae9 plus the units' work. The score-v2 change touches only host-owned
files, which no unit changes, so the diff applies as it is:

```sh
cd /work/mhr-vector/run/work/main
git status --short                        # expect nothing
git fetch -q /work/mhr-vector/repo main
git diff c8ccae9 FETCH_HEAD --stat
git diff c8ccae9 FETCH_HEAD | git apply --index
git commit -q -m "host: score-v2, Voyage-only datasets (code-voyagecode4-200k, docs-voyage4-200k)"
git log --oneline -1
```

If `git apply` refuses, nothing is changed. `git apply --index --3way` merges instead.

**9. Update the pack's description of the score** (before the optimize phase). In
`arko_workflows/examples/agent_swarm/packs/vector/prompts.rs` on the box's arko checkout,
replace the two descriptions:

- build score: "vectors per second to load and compact the first 500,000 vectors of
  dbp3l-1024 with 16 threads and a 2 GiB budget" becomes "the geometric mean, over
  code-voyagecode4-200k and docs-voyage4-200k, of the vectors per second to load and
  compact each with 16 threads and a 2 GiB budget".
- score: "on the first 500,000 vectors of dbp3l-1024 (OpenAI text-embedding-3-large on
  DBpedia, cut to 1024 dimensions)" becomes "on each of two sets, code-voyagecode4-200k
  (about 200,000 code chunks embedded with voyage-code-4) and docs-voyage4-200k (the
  Cloudflare and MDN docs embedded with voyage-4), all their vectors. The score is the
  geometric mean of the two parts' QPS". The sentence after it becomes "It is 0 when a
  result breaks a rule or either part reaches 0.95 at no beam."

The pack's README (its "The score" paragraph and the disk notes) says the same. Then
rebuild `agent_swarm` (`cargo build --release -p arko_workflows --example agent_swarm`).
The run's prompts are read from the binary, so the change takes effect at the restart.

**10. Restart.** Clear `stop_after` in `settings.json` if it was set to pause the run.
HOST_OWNED.md now lists `docs/`. Adding `"docs/"` to the settings' `owned` list makes the
gate enforce it too (bucket scopes already keep agents out of it). Then:

```sh
BIN=<the arko checkout>/target/release/examples/agent_swarm
$BIN vector check /work/mhr-vector/run
nohup $BIN vector run /work/mhr-vector/run > /work/mhr-vector/run/run.log 2>&1 &
```

`check` runs prep again. It records the new fingerprints and main's commit, warms the
ground truth (already cached), takes main's test and conformance results at the new
commit (a few minutes), and should print "the score's dataset: ready". Then the run goes
on to the milestone: guards-v2 on main, full-v2 in the background, the score-v2 baseline
(5 runs), and the optimize phase.

**11. Afterwards**, the OpenAI sets (`/work/mhr-vector/datasets/dbp3l-*`, about 11 GB)
are no longer used by any workload and can be deleted.
