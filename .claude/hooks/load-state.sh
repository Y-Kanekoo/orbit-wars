#!/usr/bin/env bash
# SessionStart hook
# state/best_score.json + experiments/ledger.jsonl 末尾 5 行 +
# state/hypotheses.md 上位 3 件 + state/learned_rules.md を stdout に出力。
# Claude Code が SessionStart hook の stdout を context に注入する。
#
# 仕様: stdin 不使用。stdout は明示的に "## " ヘッダ付き Markdown 形式。

set -uo pipefail

cd "$(dirname "$0")/../.."

echo "## orbit-wars セッション状態 ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
echo

# best_score.json
if [ -f state/best_score.json ]; then
  echo "### state/best_score.json"
  echo '```json'
  cat state/best_score.json
  echo '```'
  echo
fi

# quota.json
if [ -f state/quota.json ]; then
  echo "### state/quota.json"
  echo '```json'
  cat state/quota.json
  echo '```'
  echo
fi

# ledger 末尾 5
if [ -f experiments/ledger.jsonl ]; then
  echo "### experiments/ledger.jsonl 末尾 5 行"
  echo '```jsonl'
  tail -n 5 experiments/ledger.jsonl 2>/dev/null || echo "(空)"
  echo '```'
  echo
fi

# hypotheses 上位 3 (active のみ、priority 高い順)
if [ -f state/hypotheses.md ]; then
  echo "### state/hypotheses.md (active priority 上位 3)"
  echo '```'
  # markdown table を雑に parse、active 行のみ抽出
  grep -E '^\| H[0-9]+' state/hypotheses.md 2>/dev/null | grep -E '\| active \|' | sort -t'|' -k5 -r | head -3 || echo "(空)"
  echo '```'
  echo
fi

# learned_rules 全文
if [ -f state/learned_rules.md ]; then
  echo "### state/learned_rules.md"
  echo '```'
  grep -E '^- AVOID' state/learned_rules.md 2>/dev/null || echo "(ルール無し)"
  echo '```'
  echo
fi

# 現 phase 推定 (best_score.lb_score から)
if [ -f state/best_score.json ] && command -v jq >/dev/null 2>&1; then
  LB=$(jq -r '.lb_score // 0' state/best_score.json)
  PHASE="0"
  awk -v lb="$LB" 'BEGIN{
    if (lb < 700) print "1";
    else if (lb < 950) print "2";
    else if (lb < 1200) print "3";
    else if (lb < 1400) print "4";
    else print "5";
  }' > /tmp/load-state-phase
  PHASE=$(cat /tmp/load-state-phase)
  echo "### 推定 Phase: ${PHASE} (LB ${LB})"
  echo
fi

exit 0
