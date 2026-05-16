# /ultraplan 投入プロンプト — Orbit Wars 自律エージェント＋ハーネス構築

Kaggle コンペ **orbit-wars** (https://www.kaggle.com/competitions/orbit-wars) で **Private LB top-9 (一桁順位)** を達成するための、自律実行ハーネス＋精度向上ループの **段階別詳細プラン** を作成してほしい。

---

## 1. コンペ仕様 (取得済み)

**形式**: 通常 ML コンペではなく **agent simulation competition** (Halite / Lux AI 系)。提出物は `main.py` 単体 or `submission.tar.gz` (main.py + helper modules + 学習済み weights)。

**ゲーム概要** (Orbit Wars):
- 100x100 連続空間、中心に太陽 (半径10、フリート横断で消滅)
- 2 or 4 プレイヤー、500 turn で終了
- 各プレイヤー 1 home planet からスタート、ship を蓄積して敵/中立惑星に派兵
- 惑星: 20-40 個。内側は太陽を周回 (角速度 0.025-0.05 rad/turn)、外側は静止
- Comet: step 50/150/250/350/450 で 4 つずつスポーン、楕円軌道を通過
- Fleet 速度は size のログスケール (1 ship=1.0、1000 ship=6.0)
- 戦闘: 最大フリート vs 2番目フリート → 差分が生存 → 守備兵と交戦 → 超えれば占領
- 勝利条件: step 500 時点の総 ship 数最多 OR 単独残存
- **4-fold mirror symmetry** で初期配置の公平性を保証
- 観測フィールド: `planets`, `fleets`, `player`, `angular_velocity`, `initial_planets`, `comets`, `comet_planet_ids`, `remainingOverageTime`
- アクション: `[[from_planet_id, angle_radians, num_ships], ...]`

**重要制約**:
- **actTimeout: 1 秒/turn** ← 計算量の上限
- Code execution: `kaggle-environments>=1.28.0` で **ローカル対戦可能** (`env.run(["main.py", "random"])`)
- Daily submission: 通常 Kaggle 規定 5/日
- 評価: TrueSkill 風レーティング (推定)、リーグ形式で他submissionと自動対戦

**ファイル提出例**:
```bash
tar -czf submission.tar.gz main.py helper.py model_weights.pkl
kaggle competitions submit orbit-wars -f submission.tar.gz -m "v3 MCTS+NN"
```

**API による分析**:
- `kaggle competitions submissions orbit-wars` (履歴・スコア)
- `kaggle competitions episodes <SUB_ID>` (対戦結果)
- `kaggle competitions replay <EP_ID>` (リプレイ JSON)
- `kaggle competitions logs <EP_ID> <player_idx>` (エージェントログ)
- `kaggle competitions leaderboard orbit-wars -s` (LB)

---

## 2. 現状とゴール

- **締切**: 2026-06-23 23:59 (UTC)、本日 2026-05-16 ⇒ **残り 38 日**
- **参加チーム数**: 2,757
- **現在のスコア (user, Y-Kanekoo)**: 388.6 (2026-04-25, "phase1+2 baseline beam depth=2 width=16")
- **9位ボーダー (2026-05-16 時点)**: 1437.2 (kovi)
- **1位**: 1685.3 (Vadasz)
- **必要伸び**: +1048 ポイント
- **実験 budget**: 38 日 × 5 sub/日 = **最大 190 submissions**

---

## 3. 確定済み運用条件

- **ローカル配置**: `~/Projects/orbit-wars` (未作成)
- **Git リモート**: GitHub **private** repo `Y-Kanekoo/orbit-wars` (gh CLI 認証済み、要作成)
- **ブランチ戦略**: main 保護、`exp/<NNN>-<slug>` 命名、PR ベースで最小コミット連打 → merge
- **自律実行**: `claude -p --worktree --max-budget-usd <N>` を **tmux 別セッション**で常駐。本対話セッションは設計・初期構築・オーケストレーションのみ
- **実装担当**: Claude Code 本体 (Codex CLI は 2026-05-09 解約済み)
- **予算**: Claude Max plan quota 内で完結、$上限なし、quota 枯渇時 wait
- **学習基盤**: 第一義は **Kaggle Notebooks** (push&run&pull&submit)。RL self-play 等の GPU heavy 処理は kaggle 上で実行。ローカルでは EDA、heuristic agent ローカルマッチ評価のみ
- **安全境界**: 同一エラー 3 回 → 異 approach、6 回 → skip + `TODO(autonomous):`、テストファイル改変禁止、`--no-verify` 禁止

---

## 4. ハーネス構成要件

### 4-1. Subagents (`.claude/agents/`)
以下 7 個を 配置:
1. **researcher**: kaggle Discussions / 過去類似コンペ (Halite, Lux, Santa) のメタ戦略を調査、hypothesis backlog 生成
2. **feature-engineer**: 観測から派生する戦略特徴 (territory control map, threat map, production projection, fleet ETA matrix) を設計
3. **modeler**: search algorithm (beam/MCTS), evaluation function, NN value head の選定・実装
4. **notebook-builder**: kaggle Notebook 形式に main.py + helper をパッケージ、kernel-metadata.json 生成
5. **kaggle-runner**: `kaggle kernels push/status/output` + `kaggle competitions submit` のラッパー、quota & rate-limit guard
6. **score-analyzer**: local self-play 結果 + LB スコアをパースし、勝率/elo推定/失敗パターン分類
7. **pr-author**: 実験結果サマリ付き PR 作成、merge 判定 (gain ≥ +20 LB or ≥ +5% local winrate)

### 4-2. Hooks (`.claude/hooks/`)
- **PreToolUse(Bash)**: `guard-dangerous-bash.sh` (グローバル流用) + `kaggle-quota-guard.sh` (submission/day カウント、4回到達で次submission阻止)
- **PostToolUse(Edit|Write)**: `auto-lint-fmt.sh` (ruff + black)
- **PostToolUse(Bash, kaggle submit)**: `record-submission.sh` (experiments/ledger.jsonl 追記)
- **Stop**: `auto-commit-on-stop.sh` (グローバル流用) + `analyze-deviations.sh` (グローバル流用) + `trigger-next-iteration.sh` (loop continuation signal)
- **SessionStart**: `load-state.sh` (best_score.json, ledger 末尾 5 件を context に注入)

### 4-3. State 管理 (`state/` & `experiments/`)
- `state/best_score.json`: `{version, lb_score, local_winrate, submission_id, branch, merged_at}`
- `state/quota.json`: `{date, submissions_today, kaggle_kernel_runs_this_week_hours}`
- `state/hypotheses.md`: 優先度付き仮説バックログ (priority/effort/expected_gain 列)
- `experiments/ledger.jsonl`: 1 行 = 1 実験 `{exp_id, branch, hypothesis, cv/local_winrate, lb_score, decision (merge|discard|retry), elapsed_sec, lessons}`
- `experiments/replays/`: 重要対戦リプレイ
- `experiments/logs/`: agent stdout/stderr

### 4-4. Orchestrator ループ
本ループを `claude -p` プロンプトとして実装:
```
while コンペ締切 not 到達 and quota not 枯渇:
  1. state/best_score.json + state/hypotheses.md 読込
  2. 優先度最高の仮説を選択 (researcher 委譲で動的補充)
  3. exp/<NNN>-<slug> ブランチ作成
  4. modeler+feature-engineer に実装委譲、最小コミット連打
  5. ローカル self-play で current_best vs 新agent N=100 戦
  6. winrate ≥ 55% → kaggle 提出 (kaggle-runner)
     else → discard (ledger 記録、branch 削除)
  7. 提出結果監視 (1-3hr 待機、5 submissions/日 quota 注意)
  8. score-analyzer で LB 結果分類 → gain 判定
  9. gain ≥ +20 → pr-author で PR → main merge → 新 best
     else → ledger 記録、branch 残置 (後の ensemble 候補)
  10. lessons を hypotheses.md に反映
  11. goto 1
```

---

## 5. /ultraplan に求める output

以下を **段階別に詳細プラン** として作成してほしい。各 phase で deliverable・所要時間目安・成功基準を明示:

### Phase 0: 環境構築 (Day 1, 半日)
- ディレクトリ scaffold、git init、GitHub private repo、.gitignore、CLAUDE.md
- `.claude/agents/*.md` 7 ファイルの初期定義
- `.claude/hooks/*.sh` の実装
- `.claude/settings.json` (permissions, hooks, model)
- `scripts/kaggle/*.sh` (push_notebook, wait_for_run, pull_output, submit, episodes, replays)
- `scripts/orchestrator/run_iteration.sh`, `tmux_launcher.sh`
- `state/` 初期化、`experiments/ledger.jsonl` 空ファイル
- `docs/HARNESS.md`, `docs/COMPETITION.md`, `docs/HYPOTHESES.md` 雛形
- `requirements.txt` (kaggle, kaggle-environments>=1.28.0, ruff, black, numpy, scipy, pytest, optionally jax/torch)
- 検収: `bash scripts/orchestrator/smoke.sh` でローカル random vs Nearest Sniper が走る

### Phase 1: Strong heuristic baseline (Day 2-5)
- 仮説: beam search depth 6 width 64、評価関数 = 自軍 ships + 0.5*production projection - 敵脅威
- territory control map (各セル最近接 own planet 距離)
- production projection (10 turn 先までの生産予測)
- threat map (敵フリート ETA matrix)
- 検収: ローカル random vs new = winrate ≥ 90%、Nearest Sniper vs new = ≥ 70%、LB ≥ 700

### Phase 2: MCTS + opponent modeling (Day 6-15)
- 仮説: PUCT-MCTS、シミュレーション数 = actTimeout 1 sec 内で 200-1000 rollout
- rollout policy = Phase 1 の heuristic (fast)
- opponent modeling: 直近 episodes log から相手の expansion bias 推定
- 検収: vs Phase 1 = winrate ≥ 70%、LB ≥ 1000

### Phase 3: Neural value/policy head (Day 16-25)
- 仮説: AlphaZero 風、ResNet small (2-4 block) で state → (value, policy_logits)
- self-play data 生成は Kaggle Notebook GPU 上 (T4 x 2)、PPO or AlphaZero 風 update
- inference は CPU で 1 sec 内に収める必要 → ONNX export & quantize
- 検収: vs Phase 2 = winrate ≥ 60%、LB ≥ 1300

### Phase 4: Ensemble & meta-strategy (Day 26-34)
- 仮説: 戦況によって戦略切替 (序盤 = aggressive expansion, 中盤 = territory hold, 終盤 = decisive blitz)
- multiple eval functions の blend
- comet spawn timing aware policy (step 50/150/250/350/450)
- 検収: vs Phase 3 = winrate ≥ 55%、LB ≥ 1450

### Phase 5: Final tuning & defensive submission (Day 35-38)
- hyperparameter sweep (Optuna on Kaggle Notebook)
- final 2 submissions は守備的選択 (lowest variance vs top opponents)
- 検収: LB top-9 達成、private LB 維持

### 各 Phase 共通
- 失敗時 fallback: 1 phase 詰まったら前 phase の hyperparameter 探索に戻る
- quota 枯渇時: ローカル self-play を継続、quota 回復で再提出
- 同一エラー 3 回検出時: `learned_rules.md` に記録、別 approach へ

---

## 6. 加えて回答してほしい問い

1. Phase 3 で AlphaZero 風と PPO どちらを推奨するか (Kaggle GPU 利用効率、1秒/turn の inference 制約 を考慮して)
2. 評価関数の設計で最も差が出る要素は何か (territory control vs production projection vs threat-aware)
3. submission/日 5 回を最大効率で消化する schedule (例: 朝1, 昼2, 夜2 で並列実験)
4. 最終 final submission (Private LB に使われる 2 つ) の選び方
5. risk: 上位陣が後半に強い submission を投入してくる場合の検知と対策
6. ハーネス上での "agent vs agent ローカルマッチ" の並列化方式 (joblib? subprocess? Ray?)
7. external data / pretrained model の利用可否はルール上どう扱われるか (調査タスクとして含めて)

---

最終 deliverable は **markdown 形式の段階別プラン文書** で、そのまま `~/Projects/orbit-wars/docs/PLAN.md` に保存して着手できる粒度であることを期待する。
