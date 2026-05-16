#!/usr/bin/env bash
# 日次レポート: autonomous loop の進捗を 1 ファイルに集約。
# supervisor が朝一で読むため `reports/YYYY-MM-DD.daily.md` を生成。
#
# 使い方:
#   bash scripts/orchestrator/daily_report.sh           # 当日分を生成
#   cron で 09:00 JST に定期実行推奨。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

DATE=$(date +%Y-%m-%d)
mkdir -p reports
OUT="reports/${DATE}.daily.md"

{
  echo "# orbit-wars daily report ${DATE}"
  echo ""
  echo "generated at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo ""

  echo "## current best"
  echo '```json'
  cat state/best_score.json 2>/dev/null || echo "(なし)"
  echo '```'
  echo ""

  echo "## quota"
  echo '```json'
  cat state/quota.json 2>/dev/null || echo "(なし)"
  echo '```'
  echo ""

  echo "## LB (top 10)"
  echo '```'
  kaggle competitions leaderboard orbit-wars -s 2>/dev/null | head -12 || echo "(LB 取得失敗)"
  echo '```'
  echo ""

  echo "## submissions today (kaggle 観点)"
  echo '```'
  kaggle competitions submissions orbit-wars 2>/dev/null | head -5 || echo "(submission 取得失敗)"
  echo '```'
  echo ""

  echo "## active hypotheses (top 5)"
  echo '```'
  grep -E '^\| H[0-9]+' state/hypotheses.md 2>/dev/null \
    | grep -E '\| active \|' \
    | sort -t'|' -k5 -r \
    | head -5 || echo "(なし)"
  echo '```'
  echo ""

  echo "## recent ledger (last 10)"
  echo '```jsonl'
  tail -10 experiments/ledger.jsonl 2>/dev/null || echo "(空)"
  echo '```'
  echo ""

  echo "## learned rules"
  echo '```'
  grep -E '^- AVOID' state/learned_rules.md 2>/dev/null || echo "(なし)"
  echo '```'
  echo ""

  echo "## open PRs"
  echo '```'
  gh pr list --state open --limit 10 2>/dev/null || echo "(取得失敗)"
  echo '```'
  echo ""

  echo "## recently merged PRs (7 days)"
  echo '```'
  gh pr list --state merged --limit 15 --json number,title,mergedAt 2>/dev/null \
    | jq -r '.[] | "#\(.number) \(.title) (\(.mergedAt))"' 2>/dev/null \
    | head -15 \
    || echo "(取得失敗)"
  echo '```'
  echo ""

  echo "## autonomous tmux session"
  if command -v tmux >/dev/null 2>&1 && tmux has-session -t orbit-wars 2>/dev/null; then
    echo '```'
    tmux capture-pane -p -t orbit-wars | tail -20
    echo '```'
  else
    echo "(tmux session 'orbit-wars' 不在)"
  fi
  echo ""

  echo "## disk usage"
  echo '```'
  du -sh experiments/ state/ .claude/worktrees/ 2>/dev/null || true
  echo '```'
} > "$OUT"

echo "[daily_report] OK: $OUT"

# 直前日との diff も表示 (LB delta が分かる)
YESTERDAY=$(date -v-1d +%Y-%m-%d 2>/dev/null || date -d "yesterday" +%Y-%m-%d 2>/dev/null || echo "")
if [ -n "$YESTERDAY" ] && [ -f "reports/${YESTERDAY}.daily.md" ]; then
  echo "[daily_report] 前日比 diff (best_score 抜粋):"
  diff <(grep -A1 'lb_score' "reports/${YESTERDAY}.daily.md" | head -5) \
       <(grep -A1 'lb_score' "$OUT" | head -5) || true
fi
