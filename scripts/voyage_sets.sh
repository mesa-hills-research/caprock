#!/usr/bin/env bash
# Builds score-v2's two datasets with Voyage (docs/datasets.md has the rules and costs):
#   code-voyagecode4-200k  ~200,000 code chunks in 8 languages, voyage-code-4
#   docs-voyage4-200k      every Cloudflare docs chunk plus MDN, up to 200,000, voyage-4
#
#   scripts/voyage_sets.sh --dry-run             # selection and estimates, no API calls
#   VOYAGE_API_KEY=... scripts/voyage_sets.sh    # embed (the key is never printed)
#
# Extra arguments go to both builds. Variables:
#   REPOS      the folder holding the code checkouts of scripts/sources/code.json and
#              mdn__content (required)
#   CF_DOCS    a copy of the cloudflare-docs folder: docs/, evaluation/ (required)
#   ARKO_CLI   arko_index_cli, arko_index's chunker (required)
#   MHRV_ROOT  where the sets go (default: this repository): datasets in
#              $MHRV_ROOT/datasets, work files in $MHRV_ROOT/voyage-work
#   PYTHON     python3
# --max-tokens caps what each set may bill: 1.1 to 1.3 times the dry run's estimate at 3
# bytes per token, which itself reads above the expected bill (code ~36M tokens, docs ~45M).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
ROOT=${MHRV_ROOT:-$REPO}
export REPOS=${REPOS:?set REPOS to the folder holding the code checkouts and mdn__content}
export CF_DOCS=${CF_DOCS:?set CF_DOCS to a copy of the cloudflare-docs folder}
ARKO_CLI=${ARKO_CLI:?set ARKO_CLI to arko_index_cli}
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
