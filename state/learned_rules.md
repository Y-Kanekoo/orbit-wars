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
- AVOID: `mirror_match_shared_src_import` — file 名を分けても (`main_bare.py` と `main_bare_b.py`)、両方が `from src.* import` で同じ module を共有すると同様に 2 step で異常終了する (rewards=[0,0])。`mirror_match_same_path` を file 名分けで回避するだけでは不十分。回避策: relative + try/except fallback の import (legacy-388 方式) で別 module key として load する、または subprocess 完全分離。検出契機: exp/000-parity-fix で確認。
- INFO: `port_beam_search_works_in_isolation_but_not_in_env_run` — 直接呼び出し (`agent(obs)`) では移植 beam (`main_bare`) と legacy-388 が **完全に同じ moves** を返す (確認: planets=[[0,0,30,30,3,100,2],[1,1,70,70,3,10,2]] obs で両者 `[[0, 0.5759..., 11]]`)。しかし `env.run([main_bare, legacy])` 経由では main_bare winrate 16.7% (N=30、seat swap でも同値)。env 内で 1) `from src.* import` を共有して agent が壊れる、2) legacy が `..search.beam` (relative) → `search.beam` (absolute fallback) で別 module としてロード → kaggle env 内の何かで legacy が有利な module 解決をする、のいずれか。次イテレーションは agent2 の import を legacy 方式 (relative+fallback) に揃えて再測定するのが H000 解決の入口。
- INFO: `main_bare_agent_not_invoked_in_env_run` — exp/000b 検証 (2026-05-17): main_bare の `agent()` 関数に `/tmp/main_bare_dump.log` への file write を仕込んで `env.run([main_bare, legacy], seed=1)` を実行 → 145 秒の試合は走るが log file が一度も作成されない。一方 `python -c 'from main_bare import agent; agent(obs)'` の直接呼び出しでは file が即時作成される。**kaggle_environments の orbit_wars 環境が main_bare の `agent` 関数を呼んでいない可能性** が浮上。H000 の真犯人候補は前回仮説の「import 共有」より更に上流の「agent loader の関数解決」。次イテレーション入口は kaggle_environments.envs.orbit_wars の source 確認 (agent 関数名は `agent` で正しいか、 callable / module path / file path のどれを期待するか、 file path の場合の load 方式)。
