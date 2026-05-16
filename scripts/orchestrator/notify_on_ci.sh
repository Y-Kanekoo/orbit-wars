#!/usr/bin/env bash
# CI 完了通知。OPEN な PR 全てを 1 回 polling し、前回と比較して
# checks の状態遷移 (in_progress|queued → completed) を検知したら
# macOS notification + log に追記する。
#
# 使い方:
#   bash scripts/orchestrator/notify_on_ci.sh           # 1 回チェック
#   bash scripts/orchestrator/watch_prs.sh              # 60 秒毎の常駐
#
# state file: $XDG_STATE_HOME/orbit-wars/ci_state.json
#   (default: ~/.local/state/orbit-wars/ci_state.json)
# log file:   logs/ci_notifications.log (repo 内)
#
# exit code:
#   0  正常 (通知有無問わず)
#   1  gh CLI 未認証 / 取得失敗

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/orbit-wars"
STATE_FILE="$STATE_DIR/ci_state.json"
LOG_FILE="$REPO_ROOT/logs/ci_notifications.log"

mkdir -p "$STATE_DIR" "$(dirname "$LOG_FILE")"

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI が見つかりません" >&2
  exit 1
fi
if ! gh auth status >/dev/null 2>&1; then
  echo "gh auth 未認証" >&2
  exit 1
fi

# 現在の OPEN PR 全件の checks rollup を取得。
# checks: [{name, status, conclusion}] を pr 単位で集約し、
# 全 check 完了済 (status=COMPLETED) かどうかと conclusion 集計で
# 単一 state を計算する。
CURRENT_JSON="$(
  gh pr list --state open --json number,title,headRefName,statusCheckRollup \
    --jq '[.[] | {
      number,
      title,
      branch: .headRefName,
      state: (
        if (.statusCheckRollup // [] | length) == 0 then "no_checks"
        elif any(.statusCheckRollup[]; .status != null and .status != "COMPLETED") then "in_progress"
        elif any(.statusCheckRollup[]; .conclusion == "FAILURE" or .conclusion == "CANCELLED" or .conclusion == "TIMED_OUT") then "failure"
        else "success"
        end
      )
    }]'
)" || {
  echo "gh pr list 取得失敗" >&2
  exit 1
}

# 通知関数 (macOS osascript)。失敗しても続行。
notify() {
  local title="$1"
  local message="$2"
  osascript -e "display notification \"${message//\"/\\\"}\" with title \"${title//\"/\\\"}\" sound name \"Glass\"" 2>/dev/null || true
}

# 過去 state 読み込み (無ければ空)。
PREV_JSON="[]"
if [[ -f "$STATE_FILE" ]]; then
  PREV_JSON="$(cat "$STATE_FILE")"
fi

# diff を計算: 現在 state ごとに前回と比較し、
# (1) 前回 in_progress / 不在 → 今回 success/failure に遷移したら通知
echo "$CURRENT_JSON" | jq -c '.[]' | while read -r pr; do
  number="$(echo "$pr" | jq -r '.number')"
  title="$(echo "$pr" | jq -r '.title')"
  state="$(echo "$pr" | jq -r '.state')"

  prev_state="$(echo "$PREV_JSON" | jq -r --argjson n "$number" '.[] | select(.number == $n) | .state' 2>/dev/null || echo "")"

  # 通知対象: in_progress / no_checks / 不在 → success/failure に遷移
  if [[ "$state" == "success" || "$state" == "failure" ]]; then
    if [[ "$prev_state" != "$state" ]]; then
      ts="$(date '+%Y-%m-%d %H:%M:%S')"
      icon="✅"
      [[ "$state" == "failure" ]] && icon="❌"
      msg="$icon PR #$number $state: $title"
      echo "[$ts] $msg" >> "$LOG_FILE"
      notify "Orbit Wars CI" "PR #$number $state — $title"
    fi
  fi
done

# state を保存。
echo "$CURRENT_JSON" > "$STATE_FILE"
