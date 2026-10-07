#!/usr/bin/env bash
# Builds score-v2's two datasets with Voyage (docs/datasets.md has the rules and costs):
#   code-voyagecode4-200k  ~200,000 code chunks in 8 languages, voyage-code-4
#   docs-voyage4-200k      every Cloudflare docs chunk plus MDN, up to 200,000, voyage-4
#
#   scripts/voyage_sets.sh --dry-run             # selection and estimates, no API calls
#   VOYAGE_API_KEY=... scripts/voyage_sets.sh    # embed (the key is never printed)
#
# Extra arguments go to both builds. Variables (defaults for the build box):
#   MHRV_ROOT  /work/mhr-vector (datasets in $MHRV_ROOT/datasets, work in $MHRV_ROOT/voyage-work)
#   REPOS      /work/safety/repos (the code checkouts and mdn__content)
#   CF_DOCS    $MHRV_ROOT/cloudflare-docs (a copy of the cloudflare-docs folder: docs/, evaluation/)
#   ARKO_CLI   /work/arko-main/target/release/examples/arko_index_cli
#   PYTHON     python3
# --max-tokens caps what each set may bill: 1.1 to 1.3 times the dry run's estimate at 3
# bytes per token, which itself reads above the expected bill (code ~36M tokens, docs ~45M).
set -euo pipefail

ROOT=${MHRV_ROOT:-/work/mhr-vector}
REPO=$(cd "$(dirname "$0")/.." && pwd)
export REPOS=${REPOS:-/work/safety/repos}
export CF_DOCS=${CF_DOCS:-$ROOT/cloudflare-docs}
ARKO_CLI=${ARKO_CLI:-/work/arko-main/target/release/examples/arko_index_cli}
PY=${PYTHON:-python3}

COMMON=(
  --work "$ROOT/voyage-work" --out "$ROOT/datasets" --arko-cli "$ARKO_CLI"
  --dim 1024 --seed 1 --query-count 2000
  --exclude 'node_modules/' --exclude 'vendor/' --exclude 'third_party/' --exclude '3rdparty/' --exclude '*.min.js'
)

"$PY" "$REPO/scripts/build_voyage_set.py" --name code-voyagecode4-200k --model voyage-code-4 \
  --sources "$REPO/scripts/sources/code.json" --target-chunks 200000 \
  --queries code-doc-comments \
  --max-tokens 55000000 --allow-over "${COMMON[@]}" "$@"

"$PY" "$REPO/scripts/build_voyage_set.py" --name docs-voyage4-200k --model voyage-4 \
  --sources "$REPO/scripts/sources/docs.json" --target-chunks 200000 \
  --queries "docs-questions:$CF_DOCS/evaluation/queries.json,docs-headings" \
  --max-tokens 70000000 --allow-over "${COMMON[@]}" "$@"
