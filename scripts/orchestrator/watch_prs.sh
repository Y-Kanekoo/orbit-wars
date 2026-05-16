#!/usr/bin/env bash
# CI 通知 watcher。notify_on_ci.sh を 60 秒毎にポーリング常駐。
# tmux session "orbit-wars-watch" 内で動かす想定。
#
# 使い方:
#   bash scripts/orchestrator/watch_prs.sh                     # foreground (Ctrl-C 停止)
#   nohup bash scripts/orchestrator/watch_prs.sh > /dev/null 2>&1 &  # background
#   tmux new -d -s orbit-wars-ci 'bash scripts/orchestrator/watch_prs.sh'

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INTERVAL="${WATCH_INTERVAL_SEC:-60}"

echo "watch_prs.sh 起動 (interval=${INTERVAL}s, repo=$REPO_ROOT)"
trap 'echo "stopping watch_prs.sh"; exit 0' INT TERM

while true; do
  bash "$REPO_ROOT/scripts/orchestrator/notify_on_ci.sh" || \
    echo "[$(date '+%H:%M:%S')] notify_on_ci.sh 失敗 (継続)"
  sleep "$INTERVAL"
done
