#!/usr/bin/env bash
# Phase 0 smoke test: 自律ループ起動前の最小診断スクリプト
# 失敗時は exit 1、各 check の結果を ✓/✗ で表示する。

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PASS=0
FAIL=0

check() {
  local name="$1"
  local cmd="$2"
  if eval "$cmd" >/dev/null 2>&1; then
    echo "✓ $name"
    PASS=$((PASS + 1))
  else
    echo "✗ $name  (cmd: $cmd)"
    FAIL=$((FAIL + 1))
  fi
}

echo "=== orbit-wars Phase 0 smoke test ==="
echo "repo: $REPO_ROOT"
echo

echo "--- 認証 ---"
check "kaggle CLI 存在" "command -v kaggle"
check "kaggle.json 存在 (600)" "test -f ~/.kaggle/kaggle.json && test \"\$(stat -f '%Lp' ~/.kaggle/kaggle.json)\" = '600'"
check "kaggle competitions list (auth 検証)" "kaggle competitions list --search orbit"
check "gh CLI 存在" "command -v gh"
check "gh auth status" "gh auth status"
check "git remote origin 設定" "git remote get-url origin"
echo

echo "--- ランタイム ---"
check "python3 >=3.11" "python3 -c 'import sys; assert sys.version_info >= (3,11)'"
check "kaggle-environments importable" "python3 -c 'import kaggle_environments'"
check "orbit_wars env 登録済" "python3 -c 'from kaggle_environments import make; make(\"orbit_wars\")'"
echo

echo "--- repo state ---"
check "state/best_score.json 存在" "test -f state/best_score.json"
check "state/quota.json 存在" "test -f state/quota.json"
check "experiments/ledger.jsonl 存在" "test -f experiments/ledger.jsonl"
check "docs/competition/legacy-388/main.py 存在" "test -f docs/competition/legacy-388/main.py"
echo

echo "--- legacy-388 動作確認 (optional, skip on failure) ---"
if python3 -c 'import kaggle_environments' >/dev/null 2>&1; then
  if python3 -c "
from kaggle_environments import make
env = make('orbit_wars', configuration={'seed': 42}, debug=False)
result = env.run(['docs/competition/legacy-388/main.py', 'random'])
final = env.steps[-1]
print('episode steps:', len(env.steps))
print('final rewards:', [s.reward for s in final])
" 2>/dev/null; then
    echo "✓ legacy-388 vs random 1 試合完走"
    PASS=$((PASS + 1))
  else
    echo "✗ legacy-388 vs random 失敗 (kaggle-environments 未 install ? legacy-388 module path ?)"
    FAIL=$((FAIL + 1))
  fi
else
  echo "- skip (kaggle-environments 未 install)"
fi
echo

echo "=== 結果: PASS=$PASS, FAIL=$FAIL ==="
if [ "$FAIL" -eq 0 ]; then
  echo "Phase 0 smoke OK ✓"
  exit 0
else
  echo "Phase 0 smoke 失敗 — 上記 ✗ を解消すること"
  exit 1
fi
