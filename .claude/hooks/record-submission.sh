#!/usr/bin/env bash
# PostToolUse hook (matcher: Bash)
# tool_input.command が `kaggle ... competitions submit` を含むなら、
# experiments/ledger.jsonl 末尾行に submission_id を upsert し、
# state/quota.json.submissions_today を +1 する。
#
# 仕様: stdin から tool_input.command + tool_response.stdout を読む。
# stdout から submission_id を parse (kaggle CLI 出力に含まれる)。

set -uo pipefail

INPUT="$(cat)"
LEDGER="experiments/ledger.jsonl"
QUOTA="state/quota.json"

if command -v jq >/dev/null 2>&1; then
  CMD=$(echo "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null)
  STDOUT=$(echo "$INPUT" | jq -r '.tool_response.stdout // empty' 2>/dev/null)
else
  CMD=$(echo "$INPUT" | grep -oE '"command"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed -E 's/.*"command"[[:space:]]*:[[:space:]]*"([^"]*)".*/\1/')
  STDOUT=""
fi

[ -z "$CMD" ] && exit 0

# kaggle submit でなければ素通り
if ! echo "$CMD" | grep -qE 'kaggle[[:space:]]+competitions[[:space:]]+submit\b'; then
  exit 0
fi

# 失敗 (exit code != 0) なら quota 加算しない
if command -v jq >/dev/null 2>&1; then
  RC=$(echo "$INPUT" | jq -r '.tool_response.exit_code // 0' 2>/dev/null)
  if [ "$RC" != "0" ] && [ -n "$RC" ]; then
    exit 0
  fi
fi

# submission_id 抽出 (kaggle CLI v1.6 出力例: "submission 52123456")
SUB_ID=$(echo "$STDOUT" | grep -oE 'submission[[:space:]]+[0-9]+' | head -1 | awk '{print $2}')
[ -z "$SUB_ID" ] && SUB_ID="unknown"

TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
TODAY=$(date -u +%Y-%m-%d)

# ledger に append (簡易: 1 行 JSON、後で score-analyzer が enrich)
mkdir -p experiments
echo "{\"submission_id\":\"$SUB_ID\",\"submitted_at\":\"$TS\",\"status\":\"pending\",\"recorded_by\":\"record-submission.sh\"}" >> "$LEDGER"

# quota 加算
if [ -f "$QUOTA" ] && command -v jq >/dev/null 2>&1; then
  TMP=$(mktemp)
  jq --arg today "$TODAY" --arg ts "$TS" '
    if .date != $today then
      .submissions_today = 1 | .date = $today | .last_updated = $ts
    else
      .submissions_today += 1 | .last_updated = $ts
    end
  ' "$QUOTA" > "$TMP" && mv "$TMP" "$QUOTA"
fi

echo "[record-submission] recorded submission $SUB_ID" >&2
exit 0
