#!/usr/bin/env bash
# クラッシュ復旧 (PLAN.md L227)。
# - git status で uncommitted 確認
# - experiments/ledger.jsonl 末尾の ended_at: null 行を decision: "interrupted" で閉じる
# - tmux session 不在なら tmux_launcher 起動を案内

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

echo "[resume] === orbit-wars crash recovery ==="

# 1. uncommitted 確認
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "[resume] uncommitted 変更あり:"
  git status --short | head -10
  echo "[resume] 確認後、適切な branch で commit して下さい"
fi

# 2. ledger の interrupted 行を閉じる
LEDGER="experiments/ledger.jsonl"
if [ -f "$LEDGER" ] && command -v jq >/dev/null 2>&1; then
  LAST=$(tail -1 "$LEDGER" 2>/dev/null || echo "")
  if [ -n "$LAST" ]; then
    ENDED=$(echo "$LAST" | jq -r '.ended_at // empty' 2>/dev/null)
    DECISION=$(echo "$LAST" | jq -r '.decision // empty' 2>/dev/null)
    if [ -z "$ENDED" ] || [ "$ENDED" = "null" ] || [ -z "$DECISION" ]; then
      TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
      UPDATED=$(echo "$LAST" | jq --arg ts "$TS" '. + {ended_at: $ts, decision: "interrupted"}')
      # 末尾 1 行を置き換え (sed で in-place)
      LINES=$(wc -l < "$LEDGER")
      head -n $((LINES - 1)) "$LEDGER" > "$LEDGER.tmp"
      echo "$UPDATED" >> "$LEDGER.tmp"
      mv "$LEDGER.tmp" "$LEDGER"
      echo "[resume] ledger 末尾を interrupted で閉じました"
    fi
  fi
fi

# 3. tmux session 状況
if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t orbit-wars 2>/dev/null; then
    echo "[resume] tmux session 'orbit-wars' 生存中"
  else
    echo "[resume] tmux session なし。再起動: bash scripts/orchestrator/tmux_launcher.sh"
  fi
fi

echo "[resume] OK"
exit 0
