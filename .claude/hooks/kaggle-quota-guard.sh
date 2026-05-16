#!/usr/bin/env bash
# PreToolUse hook (matcher: Bash)
# kaggle submit / kaggle kernels push 系コマンドの quota guard。
# state/quota.json を読み、daily limit (5) や weekly Notebook GPU 30h を超えそうなら block。
#
# 仕様: stdin から tool_input JSON、command を抽出。
# kaggle 関連でなければ exit 0 (素通り)。

set -uo pipefail

INPUT="$(cat)"
QUOTA_FILE="state/quota.json"

if command -v jq >/dev/null 2>&1; then
  CMD=$(echo "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null)
else
  CMD=$(echo "$INPUT" | grep -oE '"command"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed -E 's/.*"command"[[:space:]]*:[[:space:]]*"([^"]*)".*/\1/')
fi

[ -z "$CMD" ] && exit 0

# kaggle 関連でなければ素通り
if ! echo "$CMD" | grep -qE '\bkaggle\b'; then
  exit 0
fi

# quota.json が無い場合は素通り (Phase 0 未完成時)
if [ ! -f "$QUOTA_FILE" ]; then
  exit 0
fi

block() {
  echo "[kaggle-quota-guard] BLOCK: $1" >&2
  echo "  command: $CMD" >&2
  echo "  quota: $(cat $QUOTA_FILE)" >&2
  exit 2
}

# submission 判定
if echo "$CMD" | grep -qE 'kaggle[[:space:]]+competitions[[:space:]]+submit\b'; then
  if command -v jq >/dev/null 2>&1; then
    TODAY=$(jq -r '.date' "$QUOTA_FILE")
    COUNT=$(jq -r '.submissions_today' "$QUOTA_FILE")
    LIMIT=$(jq -r '.submissions_daily_limit // 5' "$QUOTA_FILE")
    CURRENT_DATE=$(date -u +%Y-%m-%d)

    # 日付が変わっていればカウントリセット相当 (実際の reset は record-submission.sh で)
    if [ "$TODAY" != "$CURRENT_DATE" ]; then
      exit 0
    fi

    if [ "$COUNT" -ge "$LIMIT" ]; then
      block "daily submission quota 到達 ($COUNT/$LIMIT)"
    fi

    # 残 1 枠時は警告 (block しない)
    if [ "$COUNT" -ge $((LIMIT - 1)) ]; then
      echo "[kaggle-quota-guard] WARN: submission 残 $((LIMIT - COUNT)) 枠" >&2
    fi
  fi
fi

# Notebook (kernel) push 判定
if echo "$CMD" | grep -qE 'kaggle[[:space:]]+kernels[[:space:]]+push\b'; then
  if command -v jq >/dev/null 2>&1; then
    HOURS=$(jq -r '.kaggle_kernel_runs_this_week_hours // 0' "$QUOTA_FILE")
    LIMIT=$(jq -r '.kaggle_kernel_quota_hours_per_week // 30' "$QUOTA_FILE")
    # bash で float 比較は awk 委譲
    OVER=$(awk -v h="$HOURS" -v l="$LIMIT" 'BEGIN{print (h>=l)?1:0}')
    if [ "$OVER" = "1" ]; then
      block "weekly kernel GPU quota 到達 (${HOURS}h/${LIMIT}h)"
    fi
  fi
fi

exit 0
