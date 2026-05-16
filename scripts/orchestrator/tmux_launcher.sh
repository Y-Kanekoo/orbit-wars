#!/usr/bin/env bash
# tmux で claude -p --worktree を常駐起動する (PLAN.md L226)。
# 既に session があれば warning だけ、無ければ新規 detach 起動。
#
# 使い方:
#   bash scripts/orchestrator/tmux_launcher.sh           # 起動
#   tmux attach -t orbit-wars                            # 進行確認
#   tmux kill-session -t orbit-wars                      # 停止

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="orbit-wars"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[tmux_launcher] tmux 未 install。brew install tmux などで導入してください。" >&2
  exit 2
fi

if ! command -v claude >/dev/null 2>&1; then
  echo "[tmux_launcher] claude CLI 未 install。" >&2
  exit 3
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[tmux_launcher] session '$SESSION' は既に起動中。"
  echo "  attach: tmux attach -t $SESSION"
  echo "  kill:   tmux kill-session -t $SESSION"
  exit 0
fi

PROMPT='あなたは Orbit Wars 自律エージェントの orchestrator。

開始時:
1. docs/PLAN.md を読み、現在の Phase を把握 (state/best_score.json の lb_score から判定)
2. state/hypotheses.md の active priority 上位を確認
3. state/learned_rules.md の AVOID を遵守

各イテレーション:
- bash scripts/orchestrator/run_iteration.sh を実行 (まだ無ければ実装する)
- 1 iteration 終了で Stop hook trigger-next-iteration.sh が次を kick する
- quota 枯渇 (kaggle-quota-guard 発火) 時は state/quota.json の date が変わるまで wait

安全境界:
- main 直 push 禁止、必ず exp/<NNN>-<slug> ブランチ → PR → CodeRabbit + CI gate → squash merge
- テストファイル改変禁止
- --no-verify 禁止
- 同一エラー 3 回 → learned_rules 昇格、6 回 → TODO(autonomous) で skip'

cd "$REPO_ROOT"

tmux new-session -d -s "$SESSION" -n main "claude -p '$PROMPT' --worktree 2>&1 | tee logs/tmux_$(date -u +%Y%m%d_%H%M%S).log"

mkdir -p logs

sleep 2
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[tmux_launcher] OK: session '$SESSION' 起動済"
  echo "  attach: tmux attach -t $SESSION"
  echo "  status: tmux capture-pane -p -t $SESSION | tail -20"
  echo "  kill:   tmux kill-session -t $SESSION"
else
  echo "[tmux_launcher] FAIL: session が立ち上がりませんでした" >&2
  exit 4
fi
