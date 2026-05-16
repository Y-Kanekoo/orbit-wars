#!/usr/bin/env bash
# PostToolUse hook (matcher: Edit|Write)
# 編集された path が *.py なら ruff check --fix + black -q を実行。
# 失敗 (依然 lint error) なら exit 2 で feedback、それ以外 exit 0。
#
# 仕様: stdin から tool_input の .file_path を抽出。

set -uo pipefail

INPUT="$(cat)"

if command -v jq >/dev/null 2>&1; then
  PATH_TARGET=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty' 2>/dev/null)
else
  PATH_TARGET=$(echo "$INPUT" | grep -oE '"file_path"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed -E 's/.*"file_path"[[:space:]]*:[[:space:]]*"([^"]*)".*/\1/')
fi

[ -z "$PATH_TARGET" ] && exit 0

# .py のみ対象
case "$PATH_TARGET" in
  *.py) ;;
  *) exit 0 ;;
esac

[ ! -f "$PATH_TARGET" ] && exit 0

# ruff (autofix)
if command -v ruff >/dev/null 2>&1; then
  ruff check --fix --quiet "$PATH_TARGET" || true
fi

# black
if command -v black >/dev/null 2>&1; then
  black -q "$PATH_TARGET" || true
fi

# 最終チェック (autofix できなかった lint が残るか)
if command -v ruff >/dev/null 2>&1; then
  if ! ruff check --quiet "$PATH_TARGET" >/tmp/ruff-remain.log 2>&1; then
    echo "[auto-lint-fmt] ruff: 未修正 lint 残存" >&2
    cat /tmp/ruff-remain.log >&2
    exit 2
  fi
fi

exit 0
