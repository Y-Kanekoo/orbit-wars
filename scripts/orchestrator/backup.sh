#!/usr/bin/env bash
# 日次 backup (PLAN.md A-7)。
# - コード: git push (現在の branch + main)
# - state/: git status で uncommitted を確認、あれば warning
# - NN weights: Kaggle Dataset として版上げ (Phase 3+ 用、Phase 0 では skip)
# - 重要 replays: Phase 1+ で実装

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "[backup] === start $TS ==="

# 1. main を push (already-pushed なら no-op)
CURRENT=$(git symbolic-ref --short HEAD 2>/dev/null || echo "")
if [ -n "$CURRENT" ] && [ "$CURRENT" != "HEAD" ]; then
  if git ls-remote --exit-code --heads origin "$CURRENT" >/dev/null 2>&1; then
    git push origin "$CURRENT" 2>&1 | tail -3 || echo "[backup] push failed (continue)"
  fi
fi

# 2. state/ uncommitted 確認
if ! git diff --quiet -- state/ experiments/ledger.jsonl 2>/dev/null; then
  echo "[backup] WARN: state/ or ledger に uncommitted 変更あり"
  git status --short -- state/ experiments/ledger.jsonl | head -10
fi

# 3. NN weights backup (Phase 3+)
if [ -d data/checkpoints ] && command -v kaggle >/dev/null 2>&1; then
  echo "[backup] Phase 3+: NN checkpoints を Kaggle Dataset に push (未実装)"
  # kaggle datasets version -p data/checkpoints -m "daily backup $TS"
fi

echo "[backup] === end ==="
exit 0
