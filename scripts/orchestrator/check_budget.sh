#!/usr/bin/env bash
# Budget ceiling 監視。state/budget_ceiling.json を読み、tmux pane から
# 現在のセッション消費量 (claude 表示の $XX.XX) を抽出して soft cap 判定。
#
# 使い方:
#   bash scripts/orchestrator/check_budget.sh           # 1 回チェック
#   watch -n 600 bash scripts/orchestrator/check_budget.sh  # 10 分毎
#
# exit code:
#   0  OK (cap 内)
#   1  warning threshold 超過 (alert only)
#   2  cap 超過 (action 推奨)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

CEIL_FILE="state/budget_ceiling.json"
SESSION="orbit-wars"

if [ ! -f "$CEIL_FILE" ]; then
  echo "[check_budget] $CEIL_FILE 不在 — 監視 skip"
  exit 0
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "[check_budget] jq 未 install — 監視 skip"
  exit 0
fi

if ! command -v tmux >/dev/null 2>&1 || ! tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[check_budget] tmux session '$SESSION' 不在 — 監視 skip"
  exit 0
fi

CAP=$(jq -r '.daily_usd_estimated_cap' "$CEIL_FILE")
WARN=$(jq -r '.daily_usd_warning_threshold' "$CEIL_FILE")
ITER_HR_MAX=$(jq -r '.iteration_max_hours' "$CEIL_FILE")

# tmux pane status bar から $XX.XX を抽出 (claude が表示している cumulative spend)
PANE=$(tmux capture-pane -p -t "$SESSION" 2>/dev/null)
SPEND=$(echo "$PANE" | grep -oE '\$[0-9]+\.[0-9]+' | tail -1 | tr -d '$')

if [ -z "$SPEND" ]; then
  echo "[check_budget] 消費量を pane から抽出できず — 監視 skip (claude 起動直後 ?)"
  exit 0
fi

# 経過時間 (tmux pane 末尾の "Xh Ym" or "Y m")
ELAPSED_STR=$(echo "$PANE" | grep -oE '[0-9]+h[0-9]+m' | tail -1)

echo "[check_budget] current spend: \$${SPEND}  elapsed: ${ELAPSED_STR:-unknown}  cap: \$${CAP}  warn: \$${WARN}"

# 比較 (awk で float)
EXCEED_CAP=$(awk -v s="$SPEND" -v c="$CAP" 'BEGIN{print (s>=c)?1:0}')
EXCEED_WARN=$(awk -v s="$SPEND" -v w="$WARN" 'BEGIN{print (s>=w)?1:0}')

if [ "$EXCEED_CAP" = "1" ]; then
  echo "[check_budget] CAP 超過: \$${SPEND} >= \$${CAP}"
  echo "  推奨アクション: tmux kill-session -t $SESSION"
  echo "  対処: orchestrator prompt の簡素化 or scope 絞り込みを検討"
  exit 2
elif [ "$EXCEED_WARN" = "1" ]; then
  echo "[check_budget] WARN threshold 超過: \$${SPEND} >= \$${WARN}"
  echo "  daily_report に alert を記載"
  exit 1
fi

exit 0
