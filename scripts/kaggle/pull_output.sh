#!/usr/bin/env bash
# Kaggle Notebook の output を pull (PLAN.md A-7 backup 連動)。
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

KERNEL="${1:?Usage: $0 <username/kernel-slug> [dest_dir]}"
DEST="${2:-data/checkpoints}"

mkdir -p "$DEST"
kaggle kernels output "$KERNEL" -p "$DEST" 2>&1
echo "[pull_output] OK: $KERNEL → $DEST"
