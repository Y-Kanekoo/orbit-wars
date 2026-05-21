#!/usr/bin/env bash
# PLAN.md A-5: 完全に新しい python 環境で submission.tar.gz を解凍し動作確認。
# import 漏れ / 絶対 path 依存 / src/ 参照漏れ を早期検知。
#
# 使い方:
#   bash scripts/kaggle/verify_submission.sh submission.tar.gz
#   bash scripts/kaggle/verify_submission.sh main.py   (生 main.py のまま pack して検証)

set -euo pipefail

INPUT="${1:?Usage: $0 <submission.tar.gz | main.py>}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

TMP=$(mktemp -d -t orbit-wars-verify-XXXX)
trap 'rm -rf "$TMP"' EXIT

cd "$REPO_ROOT"

case "$INPUT" in
  *.tar.gz)
    tar -xzf "$INPUT" -C "$TMP"
    ;;
  *.py)
    # 単一ファイル: そのまま TMP にコピー
    cp "$INPUT" "$TMP/main.py"
    # main.py が src/ を import するなら src/ も一緒に運ぶ
    if grep -qE 'from[[:space:]]+src\.' "$INPUT" || grep -qE 'import[[:space:]]+src\.' "$INPUT"; then
      if [ -d src ]; then
        cp -r src "$TMP/"
      fi
    fi
    ;;
  *)
    echo "[verify_submission] 拡張子 unknown: $INPUT (.tar.gz / .py のみ)" >&2
    exit 2
    ;;
esac

if [ ! -f "$TMP/main.py" ]; then
  echo "[verify_submission] FAIL: main.py が tar 内にない (root に置くこと)" >&2
  exit 3
fi

cd "$TMP"

# venv python を優先 (kaggle env は numpy 等込みなので .venv が同等の deps を持つ前提)
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
[ -x "$PYTHON_BIN" ] || PYTHON_BIN="python3"

# 純粋 python (リポジトリ依存無し) で agent 起動を試す
"$PYTHON_BIN" - <<'PYEOF'
import sys
sys.path.insert(0, ".")
try:
    import main
except Exception as e:
    print(f"[verify_submission] FAIL: import main: {e}", file=sys.stderr)
    sys.exit(4)

if not hasattr(main, "agent") or not callable(main.agent):
    print("[verify_submission] FAIL: main.agent が無い or callable でない", file=sys.stderr)
    sys.exit(5)

# 最小 obs で agent を 1 回呼ぶ (kaggle_environments が無くても通る)
mock_obs = {
    "player": 0,
    "step": 0,
    "planets": [
        [0, 0, 10.0, 10.0, 1.0, 10, 1],
        [1, -1, 50.0, 50.0, 1.0, 3, 1],
    ],
    "fleets": [],
    "angular_velocity": 0.03,
    "remainingOverageTime": 60.0,
}

try:
    result = main.agent(mock_obs)
except Exception as e:
    print(f"[verify_submission] FAIL: main.agent raised: {e}", file=sys.stderr)
    sys.exit(6)

if not isinstance(result, list):
    print(f"[verify_submission] FAIL: agent return is not list: {type(result)}", file=sys.stderr)
    sys.exit(7)

for mv in result:
    if not isinstance(mv, list) or len(mv) != 3:
        print(f"[verify_submission] FAIL: bad move format: {mv}", file=sys.stderr)
        sys.exit(8)

print(f"[verify_submission] OK: agent returned {len(result)} move(s)")
PYEOF

echo "[verify_submission] all checks passed"
exit 0
