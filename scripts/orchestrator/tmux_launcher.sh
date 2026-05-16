#!/usr/bin/env bash
# tmux で claude をインタラクティブ起動する。
# 設計理由: claude -p (--print) は print-and-exit のため trigger-next-iteration
# hook の send-keys が機能しない。インタラクティブ session に対し send-keys
# で次イテレーション prompt を送る方式に変更。
#
# 使い方:
#   bash scripts/orchestrator/tmux_launcher.sh           # 起動 (初回 prompt 送信付き)
#   tmux attach -t orbit-wars                            # 進行確認
#   tmux kill-session -t orbit-wars                      # 停止

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SESSION="orbit-wars"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[tmux_launcher] tmux 未 install。brew install tmux などで導入してください。" >&2
  exit 2
fi

# Claude Code CLI を優先 (~/.local/bin/claude)。/Applications/cmux.app 等の
# GUI ラッパーを誤って起動しないよう明示的に解決する。
CLAUDE_BIN=""
for cand in "$HOME/.local/bin/claude" "/opt/homebrew/bin/claude" "/usr/local/bin/claude"; do
  if [ -x "$cand" ]; then
    CLAUDE_BIN="$cand"
    break
  fi
done
if [ -z "$CLAUDE_BIN" ]; then
  echo "[tmux_launcher] claude CLI 未 install (~/.local/bin/claude にネイティブインストーラ推奨)。" >&2
  exit 3
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[tmux_launcher] session '$SESSION' は既に起動中。"
  echo "  attach: tmux attach -t $SESSION"
  echo "  kill:   tmux kill-session -t $SESSION"
  exit 0
fi

# 初回 orchestrator prompt。1 iteration を完走させる指示。
# Stop hook (trigger-next-iteration) が次イテレーションを send-keys する。
INIT_PROMPT='あなたは Orbit Wars 自律エージェントの orchestrator です。以下を順に実行してください。

ステップ 1: 状態把握
- docs/PLAN.md を読み、Phase 0-5 の構成を把握
- state/best_score.json で現 LB (388.6) と Phase を確認
- state/hypotheses.md で active 仮説の priority 上位 3 件を確認
- state/learned_rules.md の AVOID を確認

ステップ 2: 1 イテレーション実行
- priority 最高の active hypothesis を 1 件選択
- exp/<NNN>-<slug> ブランチを切る
- PLAN.md の該当 Phase 仕様に沿って実装 (src/ 配下を編集)
- pytest -q tests/ で既存 22 件が pass し続けることを確認
- python -m pytest で新 hypothesis 用 test も追加可能
- ローカル self-play (python scripts/selfplay/tournament.py --agent1 main.py --agent2 docs/competition/legacy-388/main.py --n 30) で勝率測定
- winrate >= 55% なら kaggle submit (bash scripts/kaggle/submit.sh main.py "exp NNN <hypothesis>")
- winrate < 55% なら ledger 記録のみで discard

ステップ 3: PR 作成と merge 判定
- gh pr create で PR (CI/CodeRabbit gate)
- PR pass + LB gain >= +20 なら main に squash merge し state/best_score.json 更新
- discard なら branch を残置 (Phase 4 ensemble pool 候補として)

ステップ 4: ledger 更新と終了
- experiments/ledger.jsonl に 1 行 append
- 状態を簡潔に報告して停止 (Stop hook が次イテレーションを kick する)

安全境界 (絶対):
- main 直 push 禁止 (settings.json deny rule に頼らず自主規制)
- テストファイル改変禁止
- --no-verify 禁止
- 同一エラー 3 回 → 異 approach、6 回 → state/hypotheses.md で status を stuck に変更
- Kaggle quota は state/quota.json を必ず参照、5/日上限を遵守
- 1 秒/turn 制約: src/utils/timing.py の deadline_iter を必ず使う

開始してください。'

cd "$REPO_ROOT"
mkdir -p logs

LOGFILE="logs/tmux_$(date -u +%Y%m%d_%H%M%S).log"

# インタラクティブ claude を tmux で起動。stdout/stderr は tmux pane に出る (capture-pane で確認可)。
# pipe-pane で外部ログ取得も別途可能だが、初期は pane capture のみ。
tmux new-session -d -s "$SESSION" -n main "cd '$REPO_ROOT' && '$CLAUDE_BIN'"

echo "[tmux_launcher] claude bin: $CLAUDE_BIN"
echo "[tmux_launcher] log path planned: $LOGFILE (pane の出力は tmux capture-pane で取得)"

# claude が起動して prompt 入力受付になるのを待つ (10-15s)
sleep 12

# 初回 prompt を送信
tmux send-keys -t "$SESSION" "$INIT_PROMPT" Enter

sleep 2
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[tmux_launcher] OK: session '$SESSION' 起動済 + 初回 prompt 送信済"
  echo "  attach: tmux attach -t $SESSION"
  echo "  status: tmux capture-pane -p -t $SESSION | tail -50"
  echo "  kill:   tmux kill-session -t $SESSION"
else
  echo "[tmux_launcher] FAIL: session が立ち上がりませんでした" >&2
  exit 4
fi
