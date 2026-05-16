#!/usr/bin/env bash
# supervisor 用の git worktree を作成する。
# 目的: autonomous tmux claude が ~/Projects/orbit-wars/ を占有する間、
# human supervisor (or 別 claude session) は ~/Projects/orbit-wars-watch/
# で working tree を分離する。
#
# iter 1 (2026-05-16) で同一 working tree 共有による branch 切替事故が
# 発生したため恒久対策として導入。
#
# 使い方:
#   bash scripts/orchestrator/setup_supervisor_worktree.sh
#   cd ~/Projects/orbit-wars-watch    # 以降は supervisor 専用
#
# 既に存在する場合は noop。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SUPERVISOR_PATH="${SUPERVISOR_WORKTREE_PATH:-$HOME/Projects/orbit-wars-watch}"
SUPERVISOR_BRANCH="${SUPERVISOR_BRANCH:-supervisor/observe}"

cd "$REPO_ROOT"

if [ -d "$SUPERVISOR_PATH" ]; then
  echo "[setup_supervisor_worktree] 既に存在: $SUPERVISOR_PATH"
  echo "  cd: cd $SUPERVISOR_PATH"
  exit 0
fi

if git show-ref --verify --quiet "refs/heads/$SUPERVISOR_BRANCH"; then
  # branch 既存 → そこに worktree を当てる
  git worktree add "$SUPERVISOR_PATH" "$SUPERVISOR_BRANCH"
else
  # 新規 branch を main 起点で作って worktree add
  git worktree add "$SUPERVISOR_PATH" -b "$SUPERVISOR_BRANCH"
fi

echo "[setup_supervisor_worktree] OK: $SUPERVISOR_PATH (branch: $SUPERVISOR_BRANCH)"
echo "  cd: cd $SUPERVISOR_PATH"
echo "  以降の supervisor 作業は必ずこの worktree で行うこと。"
echo "  autonomous tmux session が動く $REPO_ROOT は触らないこと。"
