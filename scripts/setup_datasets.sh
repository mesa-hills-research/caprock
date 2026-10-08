#!/usr/bin/env bash
# Downloads and converts the benchmark datasets and computes their ground truth.
#
#   scripts/setup_datasets.sh [--skip-synth]
#
# Variables:
#   MHRV_DATASETS  where the datasets go (default: datasets/ in this repository)
#   MHRV_VENV      the Python tools' virtual environment (default: .venv in this repository)
#   THREADS        threads for ground truth (default: every CPU)
# Disk: about 30 GB (downloads are deleted after conversion): ann sets ~3 GB, the 6M
# synthetic set ~25 GB, ground truth < 1 GB. --skip-synth leaves out the 6M set.
# The Voyage sets (scripts/voyage_sets.sh, build_code_set.py) need a Voyage key and are
# built separately. Once they exist, a rerun of this script adds their ground truth.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
DATASETS=${MHRV_DATASETS:-$REPO/datasets}
VENV=${MHRV_VENV:-$REPO/.venv}
THREADS=${THREADS:-$(getconf _NPROCESSORS_ONLN)}
SKIP_SYNTH=0
for arg in "$@"; do
  case $arg in
    --skip-synth) SKIP_SYNTH=1 ;;
    *) echo "unknown option $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$DATASETS"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --quiet -r "$REPO/scripts/requirements.txt"
fi
PY=$VENV/bin/python

(cd "$REPO" && cargo build --release -p mhr_vector_bench)
TARGET=${CARGO_TARGET_DIR:-$REPO/target}
case $TARGET in /*) ;; *) TARGET=$REPO/$TARGET ;; esac
BENCH=$TARGET/release/mhr_vector_bench

have() { [ -f "$DATASETS/$1/dataset.json" ]; }

# Standard ann-benchmarks sets (downloads ~1.3 GB).
for name in glove-100-angular nytimes-256-angular sift-128-euclidean; do
  have "$name" || "$PY" "$REPO/scripts/fetch_ann.py" "$name" --out "$DATASETS"
done
have glove-25-angular-100k || "$PY" "$REPO/scripts/fetch_ann.py" glove-25-angular --out "$DATASETS" --max-base 100000 --max-queries 1000

# Synthetic sets: a small one for smoke runs, a 6M x 1024 one for scale and memory.
have synth-small || "$BENCH" synth --out "$DATASETS/synth-small" --dim 32 --n 20000 --queries 200 --clusters 50
if [ "$SKIP_SYNTH" = 0 ]; then
  have synth-1024-6m || "$BENCH" synth --out "$DATASETS/synth-1024-6m" --dim 1024 --n 6000000 --queries 1000 --clusters 20000
fi

# Ground truth for the gate's workloads (the harness would compute it lazily anyway).
export MHRV_DATASETS=$DATASETS
# The Voyage sets: the score's slices, the guards' filters, the guards' update part.
if have code-voyagecode4-200k; then
  "$BENCH" gt --dataset code-voyagecode4-200k --queries 2000 --threads "$THREADS"
  "$BENCH" gt --dataset code-voyagecode4-200k --queries 1000 --threads "$THREADS" \
    --filters lang:go,lang:python,path:redis,path:tokio/tokio/src/sync,path:tokio/tokio/src/signal
  "$BENCH" gt --dataset code-voyagecode4-200k --base 100000 --queries 500 --threads "$THREADS"
fi
if have docs-voyage4-200k; then
  "$BENCH" gt --dataset docs-voyage4-200k --queries 2000 --threads "$THREADS"
  "$BENCH" gt --dataset docs-voyage4-200k --queries 1000 --threads "$THREADS" \
    --filters path:mdn/files/en-us/web/javascript,path:cloudflare/workers,path:mdn/files/en-us/glossary,path:cloudflare/kv
fi
if have code-voyage4-1024; then
  "$BENCH" gt --dataset code-voyage4-1024 --queries 1000 --threads "$THREADS" --filters path:ripgrep,path:flask/src,lang:go,lang:c
fi
"$BENCH" gt --dataset sift-128-euclidean --base 300000 --queries 1000 --threads "$THREADS"
"$BENCH" gt --dataset glove-100-angular --base 300000 --queries 1000 --threads "$THREADS"
"$BENCH" gt --dataset nytimes-256-angular --queries 1000 --threads "$THREADS"
# Full-size cross-checks against the sources' own neighbour lists.
"$BENCH" gt --dataset sift-128-euclidean --queries 2000 --threads "$THREADS"
"$BENCH" gt --dataset glove-100-angular --queries 2000 --threads "$THREADS"
"$BENCH" datasets
