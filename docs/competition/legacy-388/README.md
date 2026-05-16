# Legacy 388.6 Baseline (リファレンス専用)

## 出自
- `~/Projects/kaggle/orbit-wars/submissions/submission.tar.gz` (2026-04-25 01:26 作成、2026-04-25 13:19 提出)
- Kaggle submission ID **52032932**: "phase1+2 baseline (beam depth=2 width=16)" → **publicScore 388.6**

## 構成
```
main.py                       # 提出エントリポイント (agent function)
search/beam.py                # beam search depth=2 width=16
policy/inference.py           # heuristic policy
strategy/geometry.py          # geometry utilities
strategy/targeting.py         # target selection
strategy/threat.py            # threat evaluation
```

## このディレクトリの位置づけ
**リファレンス専用。Phase 1 以降の実装で評価関数・heuristic ロジックの基準として参照する。
PLAN.md に基づく新規実装はゼロベースで進め、必要時に上書き or 部分移植する。**

## 親プロジェクト (`~/Projects/kaggle/orbit-wars/`) の有用資産
新規実装で参考にしうるもの (採否は実装時に PLAN.md と整合性チェック後に判断):
- `agent/search/mcts.py` — MCTS スケルトン
- `agent/main.py` — MCTS 分岐 + search 経路 counter
- `training/{train_ppo,selfplay,arena,curriculum}.py` — PPO + self-play arena
- `tests/test_{mcts,beam_search,arena,auto_cycle}.py` — 既存テストパターン
- `scripts/{auto_cycle.sh,arena_gate.sh,promote_champion.sh,parse_kaggle_submissions.py}` — 既存ハーネス挙動 (※ Codex 委譲前提のため新規ハーネスでは設計再考必要)
- `notes/{strategy_roadmap.md,design-log.md}` — 過去の戦略検討記録 (合計 118KB)
- `config/harness.yaml` — 既存ハーネス設定 (参考)

これらは **新規実装の意思決定後に取り込み判断**。先取り import はしない。
