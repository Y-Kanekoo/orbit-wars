#!/usr/bin/env bash
# H021 submit gate: mix-eval (random/nearest_sniper/prev_best 各 N=30) を実行し、
# gate 条件を満たさなければ exit 1 (submit を block)。
#
# 使い方:
#   bash scripts/kaggle/mix_eval_gate.sh main.py [N_PER_OPPONENT]
#
# 結果は state/last_mix_eval.json に保存され、kaggle-quota-guard hook が二重チェックする。

set -uo pipefail

AGENT="${1:-main.py}"
N_PER_OPP="${2:-30}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PY="$REPO_ROOT/.venv/bin/python"
[ -x "$PY" ] || PY=python3

MIX_OUT="state/last_mix_eval.json"

echo "[mix_eval_gate] mix-eval 実行 (3 相手 x N=$N_PER_OPP)..." >&2
"$PY" scripts/selfplay/tournament.py \
  --agent "$AGENT" \
  --opponents random,nearest_sniper,prev_best \
  --n-per-opponent "$N_PER_OPP" \
  --out "$MIX_OUT" >&2 || true

if [ ! -f "$MIX_OUT" ]; then
  echo "[mix_eval_gate] FAIL: $MIX_OUT が生成されなかった" >&2
  exit 1
fi

echo "[mix_eval_gate] gate 判定..." >&2
"$PY" scripts/kaggle/check_mix_gate.py "$MIX_OUT"
