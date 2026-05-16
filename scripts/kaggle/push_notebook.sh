#!/usr/bin/env bash
# Kaggle Notebook (kernel) を push (PLAN.md L230, Phase 3+ 用)。
# kernel-metadata.json は kaggle_notebooks/<slug>/ に置く。
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

SLUG="${1:?Usage: $0 <slug>}"
DIR="kaggle_notebooks/$SLUG"

if [ ! -d "$DIR" ]; then
  echo "[push_notebook] $DIR not found" >&2
  exit 2
fi

if [ ! -f "$DIR/kernel-metadata.json" ]; then
  echo "[push_notebook] $DIR/kernel-metadata.json not found — generate first" >&2
  exit 3
fi

kaggle kernels push -p "$DIR" 2>&1
echo "[push_notebook] OK: $SLUG"
