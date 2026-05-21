# Hypotheses Backlog

仮説バックログ。`researcher` subagent が補充し、`implementer` が優先度順に消化する。
B-7 (Phase 0 完了基準) として ≥ 10 件の populate が必要 (PLAN.md L933)。

## スキーマ
| 列 | 説明 |
|---|---|
| id | 連番 (`H001`, `H002`, ...) |
| status | `active` / `in_progress` / `done` / `discarded` / `stuck` |
| phase | 該当 Phase 番号 (`1`, `2`, `3`, `4`, `5`) |
| priority | 1-5 (5 が最高) |
| effort_hr | 想定所要時間 (時間) |
| expected_gain_lb | 期待 LB 上昇値 |
| hypothesis | 仮説本文 (1-2 行) |
| validation_method | 検証方法 (例: `vs Phase 1 self-play N=100, winrate≥55%`) |
| refs | 参考リンク・lessons |

## Active

| id | status | phase | priority | effort_hr | expected_gain_lb | hypothesis | validation_method | refs |
|---|---|---|---|---|---|---|---|---|
| H000 | in_progress | 1 | 5 | 4 | +0 (基盤) | 移植 parity 修復: exp/000b で main_bare.agent への DUMP 仕込み → env.run 経由で agent が呼ばれていない疑い (file 作成されない、直接呼びでは作成)。仮説は「import 共有」より更に上流「agent loader の関数解決」へシフト。次イテレーション入口: kaggle_environments.envs.orbit_wars の source 確認 + main_bare の global を消して毎ターン write 検証 | parity check pass (winrate 0.40-0.60 N=30) | learned_rules.md AVOID phase1_baseline_port_parity_skip / mirror_match_shared_src_import / INFO main_bare_agent_not_invoked_in_env_run; exp/000-parity-fix branch |
| H021 | done | 1 | 5 | 4 | +0 (基盤) | mix-eval infra: tournament.py に `--opponents random,nearest_sniper,prev_best` 追加 (N=30/相手, seat split)。best_score.json schema v2 で mix_eval section。kaggle-quota-guard hook + submit.sh に submit gate enforce。**全 hypothesis の prerequisite** | submit gate (no-regression) = winrate_random≥baseline_random AND winrate_sniper≥baseline_sniper AND winrate_prev_best≥baseline_prev_best+0.05 AND winrate_min≥0.50 AND errors_total=0。baseline は best_score.json mix_eval から動的読込 (hardcode 回避) | exp/021 + exp/021b 完了。baseline mix_eval: random 0.8667 / sniper 0.5667 / prev_best 0.5667。gate を絶対値 (random≥0.90) から no-regression に変更 (絶対値だと baseline 自身が通らない問題、supervisor 通知2) |
| H001 | discarded | 1 | 5 | 3 | +80 | territory control map を eval に加算 (係数 0.3) | (再評価は H002+H003 land 後の combined base 上で mix-eval winrate_min≥0.50 必須、単体 sweep 禁止) | exp/001 + exp 002 で discard。exp 002 は LB regression -17.3 (ローカル 63.3% でも LB 悪化)。learned_rules.md INFO single_opponent_winrate_not_lb_predictive |
| H002 | discarded | 1 | 5 | 4 | +100 | 30 turn 先までの ship 在庫 projection (production + incoming fleet 減算) を eval に加算 (係数 0.3-0.8 sweep) | exp/002-projection で discard。projection_total はスカラー一律加算で係数 sweep 無効 (0.3/0.1 で prev_best 完全同一 0.3667)、production 二重カウントで強相手 regression。learned_rules.md AVOID eval_term_redundant_with_production | PLAN.md L268; projection.py 実装は残置 (default off)、再定式化は H022 |
| H003 | active | 1 | 5 | 3 | +120 | 敵 fleet ETA matrix から計算した threat を eval から減算 (係数 0.7)。**既存 production*8 と方向が異なる守備項** (projection の二重カウント問題を回避) | H021 mix-eval gate (no-regression: random≥0.8667, sniper≥0.5667, prev_best≥0.6167, winrate_min≥0.50, errors=0) | PLAN.md L270; H002 discard 教訓で方向性確認済 |
| H022 | active | 1 | 4 | 5 | +60 | projection 再定式化: projection_total でなく『最弱 own planet の projection』(失陥リスク防衛優先) or 『incoming threat 込み projection 差分』を eval に加算 (production 二重カウント回避) | H021 mix-eval gate (no-regression) | exp/002-projection 教訓 (AVOID eval_term_redundant_with_production)。projection.py 再利用 |
| H004 | active | 1 | 4 | 6 | +60 | per-turn beam depth=2 width=16 (compound action 1 整数圧縮)、time-budget gating で 200ms | vs heuristic N=50 winrate≥55%、max_act_ms<300 | PLAN.md L259 |
| H005 | active | 1 | 4 | 4 | +50 | sun blocking aware path planning: fleet 経路が sun を切るか事前判定して候補から除外 | vs H004 N=30 winrate≥55% | PLAN.md L271, B-12 |
| H006 | active | 1 | 4 | 5 | +50 | comet pre-positioning: step 40/140/240/340/440 で comet spawn 予測 zone に事前派兵 | vs H003 N=30 winrate≥55% | PLAN.md L270 |
| H007 | active | 1 | 3 | 6 | +50 | eval 係数の grid search (territory 0.1-0.5 / projection 0.3-0.8 / threat 0.5-1.0) | best 組合せ vs default N=30 winrate≥55% | PLAN.md L272 |
| H008 | active | 1 | 4 | 3 | +30 | beam の action_gen で空 action set `[]` も必ず候補に含める ("do nothing" 評価) | vs H004 N=30 winrate>50% (= 棄損しない) | PLAN.md B-8 |
| H009 | active | 1 | 4 | 5 | +60 | multi-launch per planet: 1 turn の compound action を最大 K=3-5 launch list で生成 | vs H004 N=30 winrate≥55% | PLAN.md B-3 |
| H010 | active | 1 | 3 | 4 | +40 | depth-1 defensive minimax (自分 1 手 → 敵最悪応手) を eval に組込 | vs H003 N=30 winrate≥55% | PLAN.md B-12 |
| H011 | active | 2 | 5 | 8 | +150 | PUCT-MCTS、rollout 200-400 (1秒予算)、rollout policy = Phase 1 heuristic | vs Phase 1 best N=50 winrate≥65%、LB ≥950 | PLAN.md L294, 303 |
| H012 | active | 2 | 4 | 4 | +50 | MCTS progressive widening (k=0.3-1.0 sweep) | best k vs default k N=30 winrate≥55% | PLAN.md L304 |
| H013 | active | 2 | 3 | 6 | +70 | offline opponent classifier (expansion/aggression/comet_priority 3-class) + online dispatch | vs MCTS w/o dispatch N=30 winrate≥55% | PLAN.md L308 |
| H014 | active | 2 | 3 | 6 | +30 | MCTS transposition table (Zobrist hash) で同一状態再訪コスト排除 | rollout/sec が 1.5-2x | PLAN.md A-16 |
| H015 | active | 2 | 3 | 5 | +50 | RAVE / AMAF rollout 共有 | vs base MCTS N=30 winrate≥55% | PLAN.md L309 |
| H016 | active | 3 | 4 | 16 | +100 | Transformer encoder (planets+fleets を token、4-layer/d=64) + NN value head のみ (policy は MCTS visit dist) | vs Phase 2 best N=30 winrate≥55%、LB ≥1100 | PLAN.md L328-334, L349 |
| H017 | active | 3 | 4 | 8 | +80 | NN policy prior 追加 (AlphaZero 風 (value, policy_logits) 出力) | vs H016 N=30 winrate≥55% | PLAN.md L350 |
| H018 | active | 3 | 3 | 6 | +40 | 4-fold symmetry data augmentation (90/180/270° rotation で学習データ 4倍) | 学習収束が 1/4 epoch で同等 loss | PLAN.md B-5 |
| H019 | active | 4 | 4 | 6 | +80 | 戦況分類 (序盤/中盤/終盤/危機) に基づく dynamic dispatch (NN/MCTS/heuristic 切替) | vs Phase 3 N=30 winrate≥55%、LB ≥1300 | PLAN.md L380-381 |
| H020 | active | 4 | 3 | 5 | +50 | endgame all-in mode (残 1 敵かつ自軍 ship 数 > 敵×1.5 で全 fleet 集中派兵) | endgame での勝率 +20% | PLAN.md A-19 |

## In progress

(なし)

## Done

(なし)

## Discarded / Stuck

(なし)
