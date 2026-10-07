#!/usr/bin/env bash
# Prepares the mhr_vector swarm's datasets and ground truth on the build box.
#
#   scripts/box_setup.sh [--skip-openai3] [--skip-synth]
#
# Layout (override the root with MHRV_ROOT):
#   $MHRV_ROOT/venv        Python tools (h5py, numpy, pyarrow)
#   $MHRV_ROOT/datasets    converted datasets (MHRV_DATASETS for the harness)
#   $MHRV_ROOT/target-host the harness build
# Disk: about 45 GB (downloads are deleted after conversion): ann sets ~3 GB, the
# OpenAI sets ~11 GB, the 6M synthetic set ~25 GB, ground truth < 1 GB.
# The code set (build_code_set.py) needs a Voyage key and is run separately.
set -euo pipefail

ROOT=${MHRV_ROOT:-/work/mhr-vector}
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATASETS=$ROOT/datasets
THREADS=${THREADS:-32}
SKIP_OPENAI3=0
SKIP_SYNTH=0
for arg in "$@"; do
  case $arg in
    --skip-openai3) SKIP_OPENAI3=1 ;;
    --skip-synth) SKIP_SYNTH=1 ;;
    *) echo "unknown option $arg" >&2; exit 2 ;;
  esac
done

mkdir -p "$DATASETS"
if [ ! -x "$ROOT/venv/bin/python" ]; then
  python3 -m venv "$ROOT/venv"
  "$ROOT/venv/bin/pip" install --quiet -r "$REPO/scripts/requirements.txt"
fi
PY=$ROOT/venv/bin/python

(cd "$REPO" && CARGO_TARGET_DIR=$ROOT/target-host cargo build --release -p mhr_vector_bench)
BENCH=$ROOT/target-host/release/mhr_vector_bench

have() { [ -f "$DATASETS/$1/dataset.json" ]; }

# Standard ann-benchmarks sets (downloads ~1.3 GB).
for name in glove-100-angular nytimes-256-angular sift-128-euclidean; do
  have "$name" || "$PY" "$REPO/scripts/fetch_ann.py" "$name" --out "$DATASETS"
done
have glove-25-angular-100k || "$PY" "$REPO/scripts/fetch_ann.py" glove-25-angular --out "$DATASETS" --max-base 100000 --max-queries 1000

# OpenAI text-embedding-3-large, 1M vectors, at 1536, 1024 and 256 dimensions.
if [ "$SKIP_OPENAI3" = 0 ] && ! have dbp3l-1024; then
  "$PY" "$REPO/scripts/fetch_openai3.py" --out "$DATASETS" --dims 1536,1024,256
fi

# Synthetic sets: a small one for smoke runs, a 6M x 1024 one for scale and memory.
have synth-small || "$BENCH" synth --out "$DATASETS/synth-small" --dim 32 --n 20000 --queries 200 --clusters 50
if [ "$SKIP_SYNTH" = 0 ]; then
  have synth-1024-6m || "$BENCH" synth --out "$DATASETS/synth-1024-6m" --dim 1024 --n 6000000 --queries 1000 --clusters 20000
fi

# Ground truth for the gate's workloads (the harness would compute it lazily anyway).
export MHRV_DATASETS=$DATASETS
if have dbp3l-1024; then
  "$BENCH" gt --dataset dbp3l-1024 --base 500000 --queries 2000 --threads "$THREADS"
  "$BENCH" gt --dataset dbp3l-1024 --base 200000 --queries 1000 --threads "$THREADS" --filters p10,p01,p001,go,p10-python
  "$BENCH" gt --dataset dbp3l-1536 --base 200000 --queries 1000 --threads "$THREADS"
  "$BENCH" gt --dataset dbp3l-256 --base 300000 --queries 1000 --threads "$THREADS"
fi
"$BENCH" gt --dataset sift-128-euclidean --base 300000 --queries 1000 --threads "$THREADS"
"$BENCH" gt --dataset glove-100-angular --base 300000 --queries 1000 --threads "$THREADS"
"$BENCH" gt --dataset nytimes-256-angular --queries 1000 --threads "$THREADS"
# Full-size cross-checks against the sources' own neighbour lists.
"$BENCH" gt --dataset sift-128-euclidean --queries 2000 --threads "$THREADS"
"$BENCH" gt --dataset glove-100-angular --queries 2000 --threads "$THREADS"
"$BENCH" datasets
