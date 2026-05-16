#!/usr/bin/env bash
# Stop hook
# CLAUDE.md Intent と実施作業の乖離をログ → 同一パターン 3 回検出で
# learned_rules.md に昇格 (PLAN.md 自己改善ループ-1)。
# orbit-wars 固有として「LB target に対する進捗 delta」「quota 残」も毎回出力。
#
# 仕様: stdin から session の transcript_path 等を読む。出力は stdout に診断、
# 同時に state/error_counts.json を update。

set -uo pipefail

INPUT="$(cat)"
COUNTS="state/error_counts.json"
RULES="state/learned_rules.md"
BEST="state/best_score.json"
QUOTA="state/quota.json"

# state files が無い場合は何もしない (Phase 0 途中)
if [ ! -f "$BEST" ] || [ ! -f "$QUOTA" ]; then
  exit 0
fi

# orbit-wars 固有メトリクス出力
if command -v jq >/dev/null 2>&1; then
  CUR_LB=$(jq -r '.lb_score // 0' "$BEST")
  TARGET=1437.2  # top-9 ボーダー (2026-05-16 時点)
  DELTA=$(awk -v c="$CUR_LB" -v t="$TARGET" 'BEGIN{print t - c}')
  SUBS=$(jq -r '.submissions_today // 0' "$QUOTA")
  LIM=$(jq -r '.submissions_daily_limit // 5' "$QUOTA")
  REM=$((LIM - SUBS))
  echo "[analyze-deviations] LB ${CUR_LB} / target ${TARGET} (delta ${DELTA}), submissions today ${SUBS}/${LIM} (remaining ${REM})" >&2
fi

# transcript から最新ターン群を抽出して "繰り返しエラー" を集計するのは Phase 1+
# Phase 0 はメトリクス出力のみ。error_counts.json の昇格ロジックは score-analyzer 側で実装。

# error_counts.json の中身を見て、3 到達した signature を learned_rules.md に昇格
if [ -f "$COUNTS" ] && command -v jq >/dev/null 2>&1; then
  PROMOTED=0
  for sig in $(jq -r 'to_entries[] | select(.value >= 3) | .key' "$COUNTS" 2>/dev/null); do
    if [ -f "$RULES" ] && ! grep -qF "AVOID: $sig" "$RULES"; then
      echo "" >> "$RULES"
      echo "- AVOID: $sig — auto-promoted at $(date -u +%Y-%m-%dT%H:%M:%SZ) (count $(jq -r --arg s "$sig" '.[$s]' "$COUNTS"))" >> "$RULES"
      echo "[analyze-deviations] PROMOTED rule: AVOID $sig" >&2
      PROMOTED=$((PROMOTED + 1))
    fi
  done
fi

exit 0
