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

# 初回 orchestrator prompt (iter 2 で 1iter≈1hr/$7 を観測したため簡素化、< 30min 目標)。
# Stop hook (trigger-next-iteration) が次イテレーションを send-keys する。
#
# 設計指針:
# - 思考時間 cap: advisor 1 回まで、deep cogitation は禁止
# - 出力期待: 各 step の完了時に短い stdout 報告のみ、過度の中間説明禁止
# - 1 仮説 1 iter、scope creep 禁止
INIT_PROMPT='Orbit Wars 自律 orchestrator。1 iteration を < 30min で完走させてください。

# 制約 (絶対)
- main 直 push 禁止、--no-verify 禁止、tests/ 改変禁止
- 1 iter で実装するのは hypothesis 1 件のみ (scope creep 禁止)
- advisor 呼出は 1 iter につき最大 1 回まで
- pytest -q tests/ が 22 件 pass し続けること
- Kaggle quota: state/quota.json を必ず参照、5/日超で abort
- 1 秒/turn 制約: src/utils/timing.py の deadline_iter を必ず使う
- 同一エラー 3 回 → state/learned_rules.md に AVOID 昇格 + 異 approach、6 回 → status を stuck に

# 実行 (各 step 完了で短く報告)

1. **load_state**: state/best_score.json + state/hypotheses.md + state/learned_rules.md を読む。priority 最高の active hypothesis を 1 件選択 (id, hypothesis, phase, validation_method, refs を出力)。

2. **branch**: `git checkout main && git pull --ff-only && git checkout -b exp/<NNN>-<slug>` (NNN=hypothesis id 末尾 3 桁、slug=hypothesis 要約)。state/hypotheses.md で当該行を `in_progress` に変更し commit。

3. **implement**: PLAN.md の該当 Phase 仕様に沿って src/ 配下を編集。1 ファイル 1 commit、メッセージ `[exp NNN] <subject>` で連打。

4. **test**: `pytest -q tests/` (既存 22 件 pass)。新 hypothesis の test は任意、書く場合 `tests/test_exp_NNN.py` に。

5. **self-play**: `python scripts/selfplay/tournament.py --agent1 main.py --agent2 docs/competition/legacy-388/main.py --n 30 2>&1 | tail -5` で winrate を見る。

6. **decision**:
   - winrate ≥ 0.55: `bash scripts/kaggle/submit.sh main.py "exp NNN <hypothesis>"` で提出。
   - winrate < 0.55: ledger に discard 記録のみ。提出しない。

7. **PR + merge** (提出した場合のみ): `gh pr create` で CI/CodeRabbit gate、pass + LB gain ≥ +20 で `gh pr merge --squash`、main pull、state/best_score.json 更新。

8. **ledger 更新**: experiments/ledger.jsonl に 1 行 append。state/hypotheses.md で status を done/discarded/stuck に変更。完了報告 (LB 結果、所要時間、次イテレーション入口) を 5 行以内で出力し停止。

Stop hook trigger-next-iteration が次を kick します。開始してください。'

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
