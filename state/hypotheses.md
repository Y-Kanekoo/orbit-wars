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
| H000 | done | 1 | 5 | 4 | +0 (基盤) | 移植 parity 修復: exp/000b で main_bare.agent への DUMP 仕込み → env.run 経由で agent が呼ばれていない疑い (file 作成されない、直接呼びでは作成)。仮説は「import 共有」より更に上流「agent loader の関数解決」へシフト。次イテレーション入口: kaggle_environments.envs.orbit_wars の source 確認 + main_bare の global を消して毎ターン write 検証 | parity check pass (winrate 0.40-0.60 N=30) | learned_rules.md AVOID phase1_baseline_port_parity_skip / mirror_match_shared_src_import / INFO main_bare_agent_not_invoked_in_env_run; exp/000-parity-fix branch |
| H021 | done | 1 | 5 | 4 | +0 (基盤) | mix-eval infra: tournament.py に `--opponents random,nearest_sniper,prev_best` 追加 (N=30/相手, seat split)。best_score.json schema v2 で mix_eval section。kaggle-quota-guard hook + submit.sh に submit gate enforce。**全 hypothesis の prerequisite** | submit gate (no-regression) = winrate_random≥baseline_random AND winrate_sniper≥baseline_sniper AND winrate_prev_best≥baseline_prev_best+0.05 AND winrate_min≥0.50 AND errors_total=0。baseline は best_score.json mix_eval から動的読込 (hardcode 回避) | exp/021 + exp/021b 完了。baseline mix_eval: random 0.8667 / sniper 0.5667 / prev_best 0.5667。gate を絶対値 (random≥0.90) から no-regression に変更 (絶対値だと baseline 自身が通らない問題、supervisor 通知2) |
| H023 | active | 1 | 5 | 3 | +0 (診断→depth unlock) | beam の opponent model を sniper 風 heuristic (`_simulate_opponents` の `_phase1_decisions`) から legacy-388 の実 move logic (docs/competition/legacy-388/) に置換。深い探索が誤った opponent モデルに過適合する問題の直接テスト | mix-eval no-regression gate。regression 解消→opponent model が真因確定 (depth/MCTS unlock)、継続→bug は downstream (eval/game-tree) | advisor 助言。4 例連続 helps-weak/hurts-strong (territory/projection/threat/depth) の真因切り分け。**H011 MCTS より先** (同 bug を高 fidelity で再現+fast_sim 浪費を防ぐ)。legacy-388 source は docs/competition/legacy-388/ |
| H001 | discarded | 1 | 5 | 3 | +80 | territory control map を eval に加算 (係数 0.3) | (再評価は H002+H003 land 後の combined base 上で mix-eval winrate_min≥0.50 必須、単体 sweep 禁止) | exp/001 + exp 002 で discard。exp 002 は LB regression -17.3 (ローカル 63.3% でも LB 悪化)。learned_rules.md INFO single_opponent_winrate_not_lb_predictive |
| H002 | discarded | 1 | 5 | 4 | +100 | 30 turn 先までの ship 在庫 projection (production + incoming fleet 減算) を eval に加算 (係数 0.3-0.8 sweep) | exp/002-projection で discard。projection_total はスカラー一律加算で係数 sweep 無効 (0.3/0.1 で prev_best 完全同一 0.3667)、production 二重カウントで強相手 regression。learned_rules.md AVOID eval_term_redundant_with_production | PLAN.md L268; projection.py 実装は残置 (default off)、再定式化は H022 |
| H003 | discarded | 1 | 5 | 3 | +120 | 敵 fleet の進行方向考慮 threat (incoming_threat_eta) を eval 守備減算項に統合 (旧 incoming_threat*1.5 を eta*2.5 に置換) | exp/003-threat-eta で discard。mix-eval: random 0.9667↑ / sniper 0.5333↓ / prev_best 0.4667↓、winrate_min 0.4667<0.50 で gate fail。learned_rules.md INFO new_eval_term_helps_random_hurts_strong | incoming_threat_eta + tests は休眠 infra 残置 (eval 未統合、beam.py は baseline 復帰)。再定式化 (別 weight / baseline 併用) は H007 grid search 対象 |
| H022 | discarded | 1 | 4 | 5 | +60 | projection 再定式化 (eval 項追加) | — | **eval 項路線 discard**: H001/H002/H003 が 3 連続で strong opponent regression (new_eval_term_helps_random_hurts_strong)。defer でなく discard で priority sort 再浮上を防ぐ。MCTS で強 baseline が出た段階でのみ再評価検討 |
| H004 | discarded | 1 | 5 | 6 | +60 | beam tuning (search 改善, eval 不変)。**exp/004 で discard**。実証: budget/width は no-op (depth=2/width=16 で beam は max 6ms/budget 300ms、width sweep 16/24/32 は frontier 飽和で timing 完全同一 = a/b/c skip)。唯一の実効レバー depth 2→3 (H004d) を実施 → mix-eval: random 0.8667→0.9333↑ / sniper 0.5667→0.6↑ / **prev_best 0.5667→0.4333↓** (winrate_min 0.4333<0.50 で gate FAIL)。深い探索が誤った opponent モデル (`_phase1_decisions` heuristic) へ過適合し強い相手で regression。eval 項に続き **search depth でも helps-weak/hurts-strong**。learned_rules INFO `deeper_search_overfits_wrong_opponent_model` | mix-eval no-regression gate | beam.py/main.py は baseline (depth=2, budget 0.3) に revert。depth 改善は正しい opponent モデル (H011 MCTS / H013 classifier) 確立後に再検討 |
| H005 | active | 1 | 4 | 4 | +50 | sun blocking aware path planning: fleet 経路が sun を切るか事前判定して候補から除外 | vs H004 N=30 winrate≥55% | PLAN.md L271, B-12 |
| H006 | active | 1 | 4 | 5 | +50 | comet pre-positioning: step 40/140/240/340/440 で comet spawn 予測 zone に事前派兵 | vs H003 N=30 winrate≥55% | PLAN.md L270 |
| H007 | active | 1 | 4 | 6 | +50 | eval 係数 grid search: threat 係数 0.5-1.0 のみ補強 (territory/projection は discard 済で off 据置、eval 項路線回避)。H004 の fallback | best vs baseline、mix-eval no-regression gate | PLAN.md L272; H004 が gate FAIL or LB<+30 時の次手 |
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
