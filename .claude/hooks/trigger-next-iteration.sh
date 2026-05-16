#!/usr/bin/env bash
# Stop hook
# tmux session "orbit-wars" が生存していれば、cooldown 60s を確認して
# 次イテレーションを kick する。loop continuation の唯一の正規ルート。
#
# 仕様: stdin 不使用 (Stop hook)。
# state/iteration_cooldown が touch されてから 60s 経過しないなら trigger しない。

set -uo pipefail

cd "$(dirname "$0")/../.."

COOLDOWN_FILE="state/iteration_cooldown"
COOLDOWN_SEC=60

mkdir -p state

# cooldown check
if [ -f "$COOLDOWN_FILE" ]; then
  LAST=$(stat -f %m "$COOLDOWN_FILE" 2>/dev/null || stat -c %Y "$COOLDOWN_FILE" 2>/dev/null || echo 0)
  NOW=$(date +%s)
  DIFF=$((NOW - LAST))
  if [ "$DIFF" -lt "$COOLDOWN_SEC" ]; then
    echo "[trigger-next-iteration] cooldown ${DIFF}s/${COOLDOWN_SEC}s — skip" >&2
    exit 0
  fi
fi

# tmux session 存在確認
if ! command -v tmux >/dev/null 2>&1; then
  echo "[trigger-next-iteration] tmux 未 install — skip" >&2
  exit 0
fi

if ! tmux has-session -t orbit-wars 2>/dev/null; then
  echo "[trigger-next-iteration] tmux session 'orbit-wars' 不在 — skip" >&2
  exit 0
fi

# claude -p 用に次の prompt をパイプ送信 (-l Enter で送信)
PROMPT='次イテレーション開始: state/best_score.json と state/hypotheses.md を再読込し、優先度最高の active 仮説で run_iteration.sh を起動してください。'

tmux send-keys -t orbit-wars "$PROMPT" Enter 2>&1 | tail -3 >&2

touch "$COOLDOWN_FILE"
echo "[trigger-next-iteration] tmux session に次イテレーション prompt を送信" >&2
exit 0
