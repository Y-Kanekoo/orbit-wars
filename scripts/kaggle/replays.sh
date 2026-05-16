#!/usr/bin/env bash
# Episode の replay JSON を pull。
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

EP_ID="${1:?Usage: $0 <EPISODE_ID>}"

mkdir -p experiments/replays
kaggle competitions replay "$EP_ID" -p experiments/replays/ 2>&1
echo "[replays] experiments/replays/${EP_ID}.json"
