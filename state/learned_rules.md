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
- AVOID: `agent_uses_dunder_file_at_module_top` — `__file__` を module top-level で参照する agent (`_ROOT = Path(__file__).resolve().parent` 等) は kaggle_environments の `get_last_callable` 経由で **NameError で crash**する。 agent.py L47 が `compile(raw, path_str, "exec")` + `exec(code_object, env)` を `env={}` で呼ぶため、`__file__` が env に注入されない。 crash 時は `get_last_callable` が `InvalidArgument` を raise し、agent が一切呼ばれない (fallback / no-op agent に置き換わるなど) ため対戦は形だけ進み、相手が確実に勝つ。
  - 検出契機: 2026-05-17 supervisor 調査で kaggle_environments/agent.py 読解 + main.py / main_bare.py を実 exec して `NameError: name '__file__' is not defined` を再現、修正後 main.py vs legacy-388 N=5 で winrate 0.167 → 0.40 (parity 期待値レンジ)。
  - 防止策: agent module top-level で `try: _ROOT = Path(__file__).resolve().parent; except NameError: pass` の guard を必ず入れる。kaggle agent loader は exec_dir を sys.path に append 済 (agent.py L53) なので fallback で何もしないで OK。
  - 関連: legacy-388 は `try: from .search.beam ...; except (ImportError, KeyError): ...` の構造で同じ問題 (`__name__` not in globals での relative import 失敗) を回避していた。今回の 修正で main.py + main_bare.py を同等以上に robust 化。
- INFO: `single_opponent_winrate_not_lb_predictive` — 単一相手 (legacy-388 のみ) のローカル N=30 winrate は LB を予測しない。exp 002 (H001 territory `TERRITORY_WEIGHT=0.3`) が両 seat 平均 63.3% (p1=73.3%, swap=53.3%) でも LB publicScore は baseline 比 **-17.3** (410.7→393.4) で discard。LB は random / nearest_sniper / 他参加者 submission の **mix 分布** に対する性能を測るため、単一相手では盲点が埋もれる。
  - 検出契機: 2026-05-21 supervisor が exp 002 の Kaggle 提出 (id 52716957) の publicScore を確認、ローカル高 winrate と LB regression の乖離を発見。
  - 防止策: submit gate を mix-eval (H021) に格上げ。**no-regression 方式**: `winrate_random≥baseline_random AND winrate_sniper≥baseline_sniper AND winrate_prev_best≥baseline_prev_best+0.05 AND winrate_min≥0.50 AND errors_total=0` の全条件を満たさないと submit しない。baseline は best_score.json mix_eval から動的読込 (hardcode 回避)。特に `winrate_min≥0.50` で「どの相手にも負け越さない」を保証。
  - 補足 (exp/021b, supervisor 通知2): 当初の絶対値 gate (random≥0.90, sniper≥0.60) は現 baseline (random 0.8667) すら通らないため no-regression に変更。「baseline から悪化しない + prev_best のみ +0.05 改善」で、submit ごとに単調改善を保証しつつ baseline を不当に弾かない。
  - 関連: LB score は時間変動する (episode 蓄積で再計算: 52716957 は 360.9→393.4、52032932 は 383.5→410.7)。go/no-go 判定は提出直後でなく **24-48h 後の安定 score** で行う。
