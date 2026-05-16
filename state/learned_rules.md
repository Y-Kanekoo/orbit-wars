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
- AVOID: `supervisor_shared_working_tree` — human supervisor (or 別 claude session) が autonomous tmux session と同じ `~/Projects/orbit-wars/` working tree で git checkout/branch 操作すると autonomous claude も同じ branch を見てしまい状態混乱が起きる。supervisor は必ず `bash scripts/orchestrator/setup_supervisor_worktree.sh` で作成した `~/Projects/orbit-wars-watch/` 等の別 worktree で作業する。
  - 検出契機: 2026-05-16 iter 1 で supervisor が fix/guard-hook-executable ブランチを切った際、autonomous claude の git status が同 branch に切り替わり「ユーザーが切り替えた様子」を検知して state リセット動作を強いられた。
  - 防止策: CLAUDE.md「Intent: supervisor と autonomous loop の working tree 分離」+ scripts/orchestrator/setup_supervisor_worktree.sh で恒久化。
- AVOID: `hook_executable_bit_drift` — `.claude/hooks/*.sh` 新規追加時に `chmod +x` が抜けると git mode 100644 で commit され、autonomous loop で `Permission denied` の non-blocking hook error が継続発生する (deny rule のみが守りに残り、guard hook ロジックは機能不全)。
  - 検出契機: PR #8 (guard-dangerous-bash.sh) で chmod 抜け、iter 1 autonomous loop 走行中に Permission denied error が連発、後追い PR #24 で git update-index --chmod=+x 修正。
  - 防止策: smoke.sh に hooks の disk exec bit + git mode 100755 検証 check を追加 (今後の新規 hook で同じ抜けが起きたら Phase 0 smoke で fail する)。
