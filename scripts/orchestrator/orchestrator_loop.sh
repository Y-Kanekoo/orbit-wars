#!/usr/bin/env bash
# Headless orchestrator loop。
#
# 設計理由 (context 枯渇対策):
#   旧 tmux_launcher.sh は単一インタラクティブ session で全 iter を回すため、
#   3-4 iter で context が 100% に達して claude が停止していた。
#   本スクリプトは各 iteration を `claude -p` (print-and-exit) の独立 session
#   として起動するため、毎 iter が fresh context (0%) で始まり context 枯渇が
#   構造的に発生しない。loop continuation は本スクリプトが担う
#   (Stop hook trigger-next-iteration は不要 = headless では tmux session
#    "orbit-wars" を作らないので自然に no-op)。
#
# 安全:
#   - guard-dangerous-bash.sh (PreToolUse hook) は exit 2 で block するため
#     --permission-mode bypassPermissions でも deny path は有効。
#   - 多層停止: 最大 iter 数 / wall-clock 時間 / per-iter budget / 連続失敗。
#
# 使い方:
#   bash scripts/orchestrator/orchestrator_loop.sh           # 起動
#   bash scripts/orchestrator/orchestrator_loop.sh --dry-run # claude を呼ばず flow 確認
#   ORBIT_MAX_ITERS=5 bash scripts/orchestrator/orchestrator_loop.sh
#
# 停止: Ctrl-C、または下記いずれかの上限到達で自動停止。

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# supervisor worktree (orbit-wars-watch) での誤起動を防ぐ。
# autonomous loop は autonomous worktree (~/Projects/orbit-wars) 専用。
# supervisor worktree で回すと supervisor の作業 branch と競合し、AVOID rule
# supervisor_shared_working_tree に抵触する。
if [[ "$REPO_ROOT" == *orbit-wars-watch* ]]; then
  echo "[loop] ERROR: supervisor worktree ($REPO_ROOT) では起動禁止。" >&2
  echo "[loop] autonomous worktree で起動してください:" >&2
  echo "[loop]   cd ~/Projects/orbit-wars && bash scripts/orchestrator/orchestrator_loop.sh" >&2
  exit 5
fi

# 旧 tmux session との二重起動を防ぐ。
if command -v tmux >/dev/null 2>&1 && tmux has-session -t orbit-wars 2>/dev/null; then
  echo "[loop] ERROR: tmux session 'orbit-wars' が走行中。二重起動を防ぐため停止します。" >&2
  echo "[loop] 旧 loop を止めてから起動してください: tmux kill-session -t orbit-wars" >&2
  exit 6
fi

# ---- 設定 (環境変数で上書き可) ----
MAX_ITERS="${ORBIT_MAX_ITERS:-30}"              # ループ回数上限
MAX_HOURS="${ORBIT_MAX_HOURS:-12}"              # wall-clock 上限 (時間)
MAX_BUDGET_PER_ITER="${ORBIT_MAX_BUDGET_USD:-8}" # 1 iter のコスト上限 (USD)
MAX_CONSECUTIVE_FAILURES="${ORBIT_MAX_FAILS:-3}" # 連続失敗で停止
COOLDOWN_SEC="${ORBIT_COOLDOWN_SEC:-30}"        # iter 間 cooldown
MODEL="${ORBIT_MODEL:-opus}"
EFFORT="${ORBIT_EFFORT:-high}"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

# ---- claude CLI 解決 (cmux.app GUI ラッパー回避) ----
CLAUDE_BIN=""
for cand in "$HOME/.local/bin/claude" "/opt/homebrew/bin/claude" "/usr/local/bin/claude"; do
  if [ -x "$cand" ]; then CLAUDE_BIN="$cand"; break; fi
done
if [ -z "$CLAUDE_BIN" ]; then
  echo "[loop] claude CLI 未 install (~/.local/bin/claude 推奨)。" >&2
  exit 3
fi

mkdir -p logs state
LOGFILE="logs/orchestrator_loop_$(date -u +%Y%m%d_%H%M%S).log"

# ---- iteration prompt (各 iter で fresh context に渡す) ----
read -r -d '' ITER_PROMPT <<'PROMPT'
Orbit Wars 自律 orchestrator (headless / fresh context)。1 iteration を実行し、完了報告して停止してください。

# 前提
- これは新規 context です。同一 session で複数 iter は回りません (loop スクリプトが次を起動)。
- まず state を読んで現状把握すること。past の文脈は state ファイルにのみ存在します。

# 制約 (絶対)
- main 直 push 禁止、--no-verify 禁止、tests/ 改変禁止
- 1 iter で実装する hypothesis は 1 件のみ (scope creep 禁止)
- advisor 呼出は最大 1 回
- pytest -q tests/ が pass し続けること
- Kaggle quota: state/quota.json を参照、submissions_today >= 5 なら submit しない
- 1 秒/turn: src/utils/timing.py の deadline_iter を使用
- 同一エラー 3 回 -> state/learned_rules.md に AVOID 昇格 + 異 approach、6 回 -> status を stuck に

# 実行手順 (各 step 完了で短く報告)
1. load_state: state/best_score.json + state/hypotheses.md + state/learned_rules.md を読む。priority 最高の active hypothesis を 1 件選択 (id/hypothesis/phase/validation を出力)。active が無ければ researcher 的に新仮説を 1 件 hypotheses.md に追加してから進む。
2. branch: git checkout main && git pull --ff-only && git checkout -b exp/<NNN>-<slug>。hypotheses.md の当該行を in_progress に変更し commit。
3. implement: src/ 配下を編集。1 ファイル 1 commit、メッセージ [exp NNN] <subject>。
4. test: pytest -q tests/ が pass することを確認。
5. mix-eval: python scripts/selfplay/tournament.py --opponents random,nearest_sniper,prev_best --n 30 を実行 -> state/last_mix_eval.json 生成。
6. gate: python scripts/kaggle/check_mix_gate.py state/last_mix_eval.json で no-regression gate を判定。
7. decision (2 分岐 — 厳守):
   (A) gate FAIL: submit しない。ledger に discard 記録、hypotheses.md を discarded に。PR を作成し CI green 後に **即 merge** してよい (state/ledger 更新のみで LB 待ち不要)。
   (B) gate PASS: bash scripts/kaggle/submit.sh main.py "exp NNN <hypothesis>" で提出。PR を作成し本文に "STATUS: awaiting_lb" を明記。**この PR は merge しない**。ledger に submitted (lb_score=null) 記録。**LB 反映を待たずに iter を終了する** (supervisor が 24-48h 後の安定 LB で merge 判定)。
8. ledger 更新 + 完了報告を 5 行以内で出力して停止。次 iter は loop スクリプトが fresh context で起動します。
PROMPT

echo "[loop] === orchestrator loop start $(date -u +%FT%TZ) ==="
echo "[loop] claude=$CLAUDE_BIN model=$MODEL effort=$EFFORT"
echo "[loop] limits: max_iters=$MAX_ITERS max_hours=$MAX_HOURS budget/iter=\$$MAX_BUDGET_PER_ITER max_fails=$MAX_CONSECUTIVE_FAILURES dry=$DRY"
echo "[loop] log: $LOGFILE"

START_EPOCH=$(date +%s)
failures=0
iter=0

while [ "$iter" -lt "$MAX_ITERS" ]; do
  iter=$((iter + 1))

  # wall-clock 上限
  ELAPSED_H=$(( ($(date +%s) - START_EPOCH) / 3600 ))
  if [ "$ELAPSED_H" -ge "$MAX_HOURS" ]; then
    echo "[loop] wall-clock ${ELAPSED_H}h >= ${MAX_HOURS}h 上限 — 停止" | tee -a "$LOGFILE"
    break
  fi

  # active hypothesis 有無 (無ければ ITER_PROMPT 内で researcher 的に追加するが、
  # backlog 完全枯渇かつ追加もできない状況での空回りは avoid したいので警告のみ)
  if ! grep -qE '^\| H[0-9]+ \| active \|' state/hypotheses.md 2>/dev/null; then
    echo "[loop] warn: active hypothesis が hypotheses.md に見当たらない (iter 内で追加を試みる)" | tee -a "$LOGFILE"
  fi

  echo "[loop] === iter $iter/$MAX_ITERS start $(date -u +%FT%TZ) (elapsed ${ELAPSED_H}h) ===" | tee -a "$LOGFILE"

  if [ "$DRY" = "1" ]; then
    echo "[loop] DRY-RUN: claude -p を呼ばず prompt のみ表示" | tee -a "$LOGFILE"
    echo "----- ITER_PROMPT -----"
    echo "$ITER_PROMPT"
    echo "-----------------------"
    break
  fi

  # 1 iteration を fresh context で実行。
  if "$CLAUDE_BIN" -p "$ITER_PROMPT" \
       --permission-mode bypassPermissions \
       --max-budget-usd "$MAX_BUDGET_PER_ITER" \
       --model "$MODEL" \
       --effort "$EFFORT" 2>&1 | tee -a "$LOGFILE"; then
    failures=0
    echo "[loop] iter $iter 完了 $(date -u +%FT%TZ)" | tee -a "$LOGFILE"
  else
    failures=$((failures + 1))
    echo "[loop] iter $iter 失敗 (${failures}/${MAX_CONSECUTIVE_FAILURES}) $(date -u +%FT%TZ)" | tee -a "$LOGFILE"
    if [ "$failures" -ge "$MAX_CONSECUTIVE_FAILURES" ]; then
      echo "[loop] 連続失敗 ${MAX_CONSECUTIVE_FAILURES} 回 — 停止 (supervisor 介入が必要)" | tee -a "$LOGFILE"
      break
    fi
  fi

  sleep "$COOLDOWN_SEC"
done

echo "[loop] === orchestrator loop end $(date -u +%FT%TZ) (iters=$iter) ===" | tee -a "$LOGFILE"
