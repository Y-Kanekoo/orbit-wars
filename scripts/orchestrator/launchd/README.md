# orbit-wars launchd 自動運用

`orchestrator_loop.sh` と `awaiting_lb_recommender.sh` を macOS launchd で
常駐させ、人手による再起動・LB 監視を自動化する。

## 自動化される範囲
- ✅ **loop の自動再起動**: `MAX_HOURS=12` / `MAX_ITERS=30` / 連続失敗 3 / Mac 再起動 後も `KeepAlive=true` で復活。`ThrottleInterval=1800` (30 min) で crash loop 防止
- ✅ **awaiting_lb PR の LB 監視**: 12h ごとに `gh pr list` + Kaggle CLI で live LB を取得し、PR にコメント (🟢 merge推奨 / 🔴 revert推奨 / ⏳ 様子見) を冪等投稿
- ❌ **merge/revert 自体**: 自動化しない (推奨コメントを見て supervisor or 人間が手動判定)

## install / uninstall / status
```
bash scripts/orchestrator/launchd/install.sh             # install + load (両 LaunchAgent)
bash scripts/orchestrator/launchd/install.sh --uninstall # 停止 + 削除
bash scripts/orchestrator/launchd/install.sh --status    # 状態確認
```

## 動作確認
```
launchctl list | grep orbit-wars
tail -f /tmp/orbit_loop_launchd.out
tail -f /tmp/orbit_lb_recommender_launchd.out
```

## 二重起動防止
`orchestrator_loop.sh` 冒頭の pgrep guard (exit 7) が「launchd 管理 + 手動 bash 実行」の競合を防ぐ。launchd は ThrottleInterval (30 min) を尊重しつつ、先行プロセス終了後に自然に新インスタンスを上げる。

## awaiting_lb recommender の閾値調整
LaunchAgent の `EnvironmentVariables` または手動実行時の env で:
- `ORBIT_BASELINE_LB=414.2` — baseline LB (既定: state/best_score.json から自動取得)
- `ORBIT_LB_MARGIN=5` — merge/revert 推奨の baseline ±margin
- `ORBIT_LB_MIN_AGE_HOURS=24` — 推奨に必要な最低提出経過時間
- `ORBIT_LB_DELTA_MIN=2` — 再投稿の最小 LB 変動 (これ未満 + 同推奨で skip)
- `ORBIT_DRY_RUN=1` — 投稿せずログ出力のみ (動作確認用)
