# orbit-wars

Kaggle コンペ [Orbit Wars](https://www.kaggle.com/competitions/orbit-wars) 用の自律エージェント実装＋ハーネス。

## ゴール
- Private LB **top-9 (一桁順位)** 達成
- 締切: 2026-06-23 23:59 UTC

## 状態
- Phase: 0 (環境構築)
- 詳細プラン: `docs/PLAN.md` (作成予定 — `/ultraplan` で生成)
- ハーネス仕様: `docs/HARNESS.md` (作成予定)

## クイックリンク
- コンペ仕様: `docs/competition/competition-README.md`
- スタータコード: `docs/competition/competition-starter-main.py`
- 仮説バックログ: `docs/HYPOTHESES.md` (作成予定)
- 実験 ledger: `experiments/ledger.jsonl`
- 現状 best: `state/best_score.json`

## 開発フロー
- main 保護、`exp/<NNN>-<slug>` ブランチで実験
- 最小コミット連打 → PR → merge
- 自律実行: `bash scripts/orchestrator/tmux_launcher.sh` (Phase 0 で実装)
