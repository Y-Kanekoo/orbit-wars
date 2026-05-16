#!/usr/bin/env bash
# Leaderboard 取得。state/lb_history.jsonl に append (A-15 risk 検知用)。
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
mkdir -p state

OUT=$(kaggle competitions leaderboard orbit-wars -s 2>&1 || true)
echo "$OUT" | head -20

# 上位 20 行を JSONL に save (簡易: 名前と score のみ)
echo "$OUT" | awk 'NR>2 && NF>=4 {print $1, $(NF-1), $NF}' | head -20 | while read team date score; do
  echo "{\"polled_at\":\"$TS\",\"team\":\"$team\",\"score\":\"$score\"}" >> state/lb_history.jsonl
done

echo "[leaderboard] state/lb_history.jsonl に追記"
