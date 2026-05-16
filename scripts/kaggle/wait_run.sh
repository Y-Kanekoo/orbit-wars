#!/usr/bin/env bash
# Kaggle Notebook の status を 60s polling、complete で抜ける (PLAN.md L231)。
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

KERNEL="${1:?Usage: $0 <username/kernel-slug>}"
MAX_WAIT_MIN="${2:-180}"   # 最大 3h
INTERVAL=60

ELAPSED=0
while [ "$ELAPSED" -lt $((MAX_WAIT_MIN * 60)) ]; do
  STATUS=$(kaggle kernels status "$KERNEL" 2>&1 | head -3 | tail -1 | awk '{print $NF}')
  echo "[wait_run] $(date -u +%H:%M:%S) status=$STATUS elapsed=${ELAPSED}s"

  case "$STATUS" in
    complete|completed|COMPLETE)
      echo "[wait_run] OK: complete"
      exit 0
      ;;
    error|cancelled|failed|ERROR|FAILED)
      echo "[wait_run] FAIL: $STATUS" >&2
      exit 2
      ;;
  esac

  sleep "$INTERVAL"
  ELAPSED=$((ELAPSED + INTERVAL))
done

echo "[wait_run] TIMEOUT after ${MAX_WAIT_MIN} min" >&2
exit 3
