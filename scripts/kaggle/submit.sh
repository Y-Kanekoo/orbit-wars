#!/usr/bin/env bash
# 提出ラッパー。PLAN.md L584:
#   bash scripts/kaggle/submit.sh main.py "phase-0 baseline" [--dry-run]
#
# 流れ:
#   1. verify_submission.sh で動作確認 (失敗で abort)
#   2. tar に pack (src/ を一緒に含める)
#   3. quota-guard hook が settings.json 経由で自動 block
#   4. --dry-run なら kaggle 実行はせず終了 (verify のみ)
#   5. kaggle competitions submit
#   6. record-submission.sh hook が ledger.jsonl + quota.json を update

set -euo pipefail

AGENT="${1:?Usage: $0 <main.py> <message> [--dry-run]}"
MESSAGE="${2:?Usage: $0 <main.py> <message> [--dry-run]}"
DRY=""
if [ "${3:-}" = "--dry-run" ]; then
  DRY="--dry-run"
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

echo "[submit] step 1: verify_submission"
bash scripts/kaggle/verify_submission.sh "$AGENT"

echo "[submit] step 2: pack submission.tar.gz"
PACK=$(mktemp -t orbit-wars-pack-XXXX.tar.gz)
trap 'rm -f "$PACK"' EXIT

# main.py が src/ を import するか判定
if grep -qE 'from[[:space:]]+src\.' "$AGENT" || grep -qE 'import[[:space:]]+src\.' "$AGENT"; then
  tar -czf "$PACK" "$AGENT" src/
else
  tar -czf "$PACK" "$AGENT"
fi

echo "[submit] step 3: re-verify packed tar"
bash scripts/kaggle/verify_submission.sh "$PACK"

if [ -n "$DRY" ]; then
  echo "[submit] DRY-RUN: skipping kaggle competitions submit"
  echo "[submit] would submit: $PACK with message '$MESSAGE'"
  exit 0
fi

echo "[submit] step 4: kaggle competitions submit"
kaggle competitions submit orbit-wars -f "$PACK" -m "$MESSAGE"

echo "[submit] OK"
exit 0
