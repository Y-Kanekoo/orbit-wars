# Learned Rules

`analyze-deviations.sh` hook + `score-analyzer` subagent が同一エラー 3 回検出時にここへ昇格する (PLAN.md 自己改善ループ-1)。

`load-state.sh` hook が毎 session 開始時にこのファイルを context に注入し、`implementer` の system prompt に "次の rules を遵守: ..." として埋め込む。

## スキーマ
- `AVOID: <signature> — <root_cause>` 形式で 1 行 1 ルール
- `signature` は score-analyzer の `recurring_error_signature` 出力に対応

## ルール

- AVOID: `phase1_baseline_port_parity_skip` — src/ に移植した beam (legacy-388 と同 eval) が legacy-388 と N=30 mirror で winrate ≈ 50% に届かない状態で Phase 1 hypothesis (territory / projection / threat 等) を載せても、効果が parity gap に埋もれて誤判定される。Phase 1 着手前に必ず parity check (territory_weight=0 で legacy-388 vs new main.py が N=30 で 0.40-0.60) を通す。
  - 検出契機: exp/001-territory-eval で territory_weight=0/0.3 共に winrate 16.7%、seat swap でも 16.7%。
  - 根本原因 (未特定、要 H000 的調査): main.py wrapper の per-turn `_safe_act` pre-call、`sanitize` の二重実行、numpy lazy import overhead、または budget 計算式の差。
- AVOID: `mirror_match_same_path` — kaggle_environments の `env.run([a, a])` で同一 file path を渡すと 2 step で異常終了 (rewards=[0,0])。tournament.py が winner=0 をデフォルト集計して誤った 100% winrate を返す。mirror seat asymmetry を測る場合は別名 (e.g. `main_b.py`) のコピーを agent2 に指定する。
