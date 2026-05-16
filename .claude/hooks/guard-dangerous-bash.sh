#!/usr/bin/env bash
# PreToolUse hook (matcher: Bash)
# 危険コマンドを検知し block する。stdin から Claude Code が tool_input を JSON で渡す。
# block 条件:
#   - rm -rf /
#   - git push --force (main / master 宛て)
#   - --no-verify
#   - eval | sh | curl ... | sh などの pipe-to-shell
#   - secrets 含む git add (kaggle.json, .env)
#
# 仕様: exit 2 + stderr に reason で block、exit 0 で許可。
set -uo pipefail

# stdin から tool_input を読む
INPUT="$(cat)"

# command フィールドを抽出 (jq があれば jq、なければ grep フォールバック)
if command -v jq >/dev/null 2>&1; then
  CMD=$(echo "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null)
else
  CMD=$(echo "$INPUT" | grep -oE '"command"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed -E 's/.*"command"[[:space:]]*:[[:space:]]*"([^"]*)".*/\1/')
fi

if [ -z "$CMD" ]; then
  exit 0
fi

block() {
  echo "[guard-dangerous-bash] BLOCK: $1" >&2
  echo "  command: $CMD" >&2
  exit 2
}

# 1. rm -rf / 系
if echo "$CMD" | grep -qE '\brm[[:space:]]+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r)[[:space:]]+(/|/\*|~|~/)'; then
  block "rm -rf 危険ターゲット"
fi

# 2. git push --force / -f を main/master に
if echo "$CMD" | grep -qE 'git[[:space:]]+push.*(--force|-f)[[:space:]]+.*\b(main|master)\b'; then
  block "main/master への force push 禁止"
fi
if echo "$CMD" | grep -qE 'git[[:space:]]+push.*(--force|-f)\b' && echo "$CMD" | grep -qE '\borigin[[:space:]]+main\b'; then
  block "main への force push 禁止"
fi

# 3. --no-verify (commit / push)
if echo "$CMD" | grep -qE '\-\-no-verify\b'; then
  block "--no-verify (hook bypass) 禁止"
fi

# 4. pipe-to-shell (curl | sh, wget | bash, eval)
if echo "$CMD" | grep -qE '(curl|wget)[^|]*\|[[:space:]]*(sh|bash|zsh|dash)\b'; then
  block "curl/wget pipe to shell (任意コード実行)"
fi
if echo "$CMD" | grep -qE '\beval[[:space:]]+'; then
  block "eval (任意コード実行)"
fi

# 5. secrets を git add (kaggle.json, .env, *.pem)
if echo "$CMD" | grep -qE 'git[[:space:]]+add\b' && echo "$CMD" | grep -qE '(\.env\b|kaggle\.json|\.pem\b|id_rsa)'; then
  block "secrets を git add しようとしている (kaggle.json/.env/.pem/id_rsa)"
fi

# 6. /etc 配下への書き込み
if echo "$CMD" | grep -qE '(>|>>|tee)[[:space:]]+/etc/'; then
  block "/etc 配下への書き込み禁止"
fi

# 7. find -exec rm
if echo "$CMD" | grep -qE 'find[[:space:]]+.*-exec[[:space:]]+(rm|sh|bash)'; then
  block "find -exec rm/sh (一括破壊リスク)"
fi

exit 0
