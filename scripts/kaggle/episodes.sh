#!/usr/bin/env bash
# Submission ID の episode 一覧取得。
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

SUB_ID="${1:?Usage: $0 <SUBMISSION_ID>}"

# CSV 形式で保存可能、stdout には table
kaggle competitions episodes "$SUB_ID" 2>&1 | tee "experiments/episodes_${SUB_ID}.txt"
echo "[episodes] saved to experiments/episodes_${SUB_ID}.txt"
