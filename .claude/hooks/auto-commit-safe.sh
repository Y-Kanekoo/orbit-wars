#!/usr/bin/env bash
# Stop hook
# 元案 auto-commit-on-stop を厳格化 (PLAN.md L153)。
# 未 commit 変更がある場合、critical tests を実行して pass のときのみ commit する。
# Phase 0 段階では tests/ が無い場合は smoke test の代替 (kaggle-environments
# import) のみ実施。
#
# 仕様: stdin 不使用 (Stop hook はセッション終了時に呼ばれる)。

set -uo pipefail

cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"

# 変更が無ければ素通り
if git diff --quiet && git diff --cached --quiet && [ -z "$(git ls-files --others --exclude-standard 2>/dev/null)" ]; then
  exit 0
fi

# main branch 直接編集中なら commit しない (PR 経由原則を守る)
CURRENT_BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo "")
if [ "$CURRENT_BRANCH" = "main" ] || [ "$CURRENT_BRANCH" = "master" ]; then
  echo "[auto-commit-safe] main 上での auto-commit はスキップ (PR 経由で commit してください)" >&2
  exit 0
fi

# critical tests 実行 (tests/ が存在する場合のみ)
TEST_RESULT=0
if [ -d tests ] && command -v pytest >/dev/null 2>&1; then
  PYTEST_TARGETS=""
  [ -f tests/test_timing.py ] && PYTEST_TARGETS="$PYTEST_TARGETS tests/test_timing.py"
  [ -f tests/test_action_legal.py ] && PYTEST_TARGETS="$PYTEST_TARGETS tests/test_action_legal.py"

  if [ -n "$PYTEST_TARGETS" ]; then
    if ! pytest -q $PYTEST_TARGETS 2>&1 | tail -20 >&2; then
      echo "[auto-commit-safe] critical tests 失敗 — 未 commit のまま残します" >&2
      TEST_RESULT=1
    fi
  fi
fi

# Phase 0 代替: kaggle-environments import 確認
if [ "$TEST_RESULT" = "0" ] && [ ! -d tests ]; then
  if command -v python3 >/dev/null 2>&1; then
    if ! python3 -c "import sys; sys.exit(0)" 2>/dev/null; then
      echo "[auto-commit-safe] python3 が動かない — auto-commit skip" >&2
      TEST_RESULT=1
    fi
  fi
fi

[ "$TEST_RESULT" != "0" ] && exit 0

# auto-commit (細かい変更を session 終了時に WIP として残す)
git add -A
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
git commit -m "[wip] auto-commit on session end at $TS

Stop hook auto-commit-safe.sh による自動コミット。
critical tests pass 確認済。後続セッションで内容を整理し、
適切な [type] commit に splitt/squash すること。
" --quiet 2>&1 | tail -3 >&2 || true

echo "[auto-commit-safe] WIP commit を作成しました (branch: $CURRENT_BRANCH)" >&2
exit 0
