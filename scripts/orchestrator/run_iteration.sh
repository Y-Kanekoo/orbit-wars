#!/usr/bin/env bash
# 1 イテレーション本体 (PLAN.md L225)。
# Phase 0 段階は skeleton — Phase 1+ で score-analyzer / pr-author 連携を追加。
#
# 流れ:
#   1. state を read (load-state.sh と同じ内容を stdout 表示)
#   2. state/hypotheses.md から priority 最高の active 1 件 pick
#   3. exp/<NNN>-<slug> branch create
#   4. (Phase 1+) implementer subagent を呼んで実装
#   5. (Phase 1+) tournament.py で local self-play 評価
#   6. (Phase 1+) winrate >= 55% なら submit.sh、未満なら discard
#   7. ledger.jsonl に 1 行 append
#
# 引数:
#   --dry-run           # 実装/提出せず flow の通り抜けを確認
#   --hypothesis-id ID  # 特定 ID を強制選択

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

DRY=0
FORCE_ID=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --hypothesis-id) FORCE_ID="$2"; shift 2 ;;
    *) echo "[run_iteration] unknown arg: $1" >&2; exit 2 ;;
  esac
done

TS_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)

echo "[run_iteration] === start $TS_START ==="

# 1. state 表示
bash .claude/hooks/load-state.sh

# 2. hypothesis pick
if [ ! -f state/hypotheses.md ]; then
  echo "[run_iteration] state/hypotheses.md 不在 — researcher を起動して populate 必要" >&2
  exit 3
fi

# priority 列を見て active 上位 1 件を取得 (Phase 0 簡易 parse)
PICK=$(grep -E '^\| H[0-9]+' state/hypotheses.md 2>/dev/null \
  | grep -E '\| active \|' \
  | sort -t'|' -k5 -r \
  | head -1)

if [ -n "$FORCE_ID" ]; then
  PICK=$(grep -E "^\| $FORCE_ID " state/hypotheses.md || true)
fi

if [ -z "$PICK" ]; then
  echo "[run_iteration] active hypothesis 無し — researcher 起動が必要" >&2
  exit 4
fi

HID=$(echo "$PICK" | awk -F'|' '{gsub(/ /, "", $2); print $2}')
echo "[run_iteration] picked: $HID"
echo "$PICK"

# 3. branch
SLUG=$(echo "$PICK" | awk -F'|' '{print $8}' | head -c 40 | tr -c 'a-zA-Z0-9' '-' | sed 's/--*/-/g; s/^-//; s/-$//')
[ -z "$SLUG" ] && SLUG="exp"
NNN=$(echo "$HID" | sed 's/^H0*//')
NNN=$(printf "%03d" "$NNN")
BRANCH="exp/$NNN-$SLUG"
echo "[run_iteration] branch: $BRANCH"

if [ "$DRY" = "1" ]; then
  echo "[run_iteration] DRY-RUN: skip implementation / submit"
  exit 0
fi

git checkout main
git pull --ff-only
git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"

# 4-6. Phase 0 段階では実装ロジック未配線。Phase 1+ で implementer subagent
# を Task ツールで呼び、その後 tournament.py 評価 → submit.sh の連携を追加する。
echo "[run_iteration] Phase 0 stub: implementer/tournament/submit は Phase 1+ で配線"

# 7. ledger append (skeleton)
TS_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
mkdir -p experiments
echo "{\"exp_id\":\"$NNN\",\"branch\":\"$BRANCH\",\"hypothesis_id\":\"$HID\",\"started_at\":\"$TS_START\",\"ended_at\":\"$TS_END\",\"decision\":\"phase0_skeleton\"}" >> experiments/ledger.jsonl

echo "[run_iteration] === end $TS_END ==="
exit 0
