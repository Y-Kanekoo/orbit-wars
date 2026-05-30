#!/usr/bin/env bash
# launchd LaunchAgent インストーラ。
# - orchestrator_loop.sh の KeepAlive 自動再起動
# - awaiting_lb_recommender.sh の 12h 定期実行
#
# 使い方:
#   bash scripts/orchestrator/launchd/install.sh           # 両方 install + load
#   bash scripts/orchestrator/launchd/install.sh --uninstall  # unload + 削除
#   bash scripts/orchestrator/launchd/install.sh --status     # 状態確認のみ

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/scripts/orchestrator/launchd"
TARGET_DIR="$HOME/Library/LaunchAgents"
DOMAIN="gui/$(id -u)"

PLISTS=(
  "com.orbit-wars.loop"
  "com.orbit-wars.lb-recommender"
)

action="${1:-install}"

ensure_target_dir() { mkdir -p "$TARGET_DIR"; }

install_plist() {
  local label="$1"
  local src="$SCRIPT_DIR/${label}.plist.template"
  local dst="$TARGET_DIR/${label}.plist"
  if [ ! -f "$src" ]; then
    echo "[install] ERROR: template not found: $src" >&2
    return 1
  fi
  # __HOME__ を実際の $HOME に置換して配置
  sed "s|__HOME__|$HOME|g" "$src" > "$dst"
  echo "[install] wrote $dst"
  # 既に bootstrap 済みなら一旦 bootout してから再 bootstrap
  launchctl bootout "$DOMAIN/${label}" 2>/dev/null || true
  launchctl bootstrap "$DOMAIN" "$dst"
  echo "[install] bootstrapped $label"
}

uninstall_plist() {
  local label="$1"
  local dst="$TARGET_DIR/${label}.plist"
  launchctl bootout "$DOMAIN/${label}" 2>/dev/null && echo "[uninstall] booted out $label" || echo "[uninstall] $label was not loaded"
  if [ -f "$dst" ]; then
    rm -f "$dst"
    echo "[uninstall] removed $dst"
  fi
}

status_plist() {
  local label="$1"
  echo "--- $label ---"
  launchctl print "$DOMAIN/${label}" 2>/dev/null | head -25 || echo "(not loaded)"
}

case "$action" in
  install)
    ensure_target_dir
    for p in "${PLISTS[@]}"; do install_plist "$p"; done
    echo ""
    echo "[install] 完了。ログ確認:"
    echo "  tail -f /tmp/orbit_loop_launchd.out"
    echo "  tail -f /tmp/orbit_lb_recommender_launchd.out"
    ;;
  --uninstall|uninstall)
    for p in "${PLISTS[@]}"; do uninstall_plist "$p"; done
    ;;
  --status|status)
    for p in "${PLISTS[@]}"; do status_plist "$p"; done
    ;;
  *)
    echo "Usage: $0 [install|--uninstall|--status]" >&2
    exit 1
    ;;
esac
