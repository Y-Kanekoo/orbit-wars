# /ultraplan 投入 prompt (iter 2: 即時全修正フェーズ)

## 制約 (絶対遵守、ユーザー指示)
- **長期にしない**、**すぐに全て修正**、**後回し禁止**
- supervisor 主導で並列に攻め、autonomous loop と干渉しない
- supervisor worktree (`~/Projects/orbit-wars-watch/`) でのみ作業
- 各修正は **最小単位の commit + PR** で進める (CodeRabbit + CI gate)

## ゴール
**今このセッションで A-M 全 13 タスクを完了させる**。次の autonomous iteration が PR #26 + 本セッションの修正群を pull した状態で開始されるよう、可能な限り早く main へ merge する。

---

## 現状コンテキスト (2026-05-16 21:35 JST)

### 進行中
- Autonomous loop: `~/Projects/orbit-wars/` の tmux session "orbit-wars" で `exp/000-parity-fix` を実装中、tournament.py 実行中 (11 分経過)
- Supervisor: `~/Projects/orbit-wars-watch/` の `supervisor/observe` で本作業
- PR #26 (iter 1 教訓構造化反映) は merge 済み

### Phase 0 完了済み (26 PRs merged)
- harness 全部: 5 agents + 8 hooks + state/ + scripts/ + tests/ + CI + docs
- 締切まで 38 日、LB 388.6 → top-9 ボーダー 1437.2 (+1048 必要)
- 累計コスト $11.51, 1 iter ≈ 1 時間 ≈ $7

### iter 1 で発覚した問題 (≧13 件、全部今すぐ修正)

| # | 問題 | 検出契機 | 緊急度 |
|---|---|---|---|
| A | parity 未達 (src 移植版が legacy-388 と winrate 16.7% で 50% 届かず) | exp/001 ledger | 最高 |
| B | mirror match bug: `env.run([a,a])` 同一 path で 2 step 異常終了 | learned_rules | 高 |
| C | orchestrator prompt 肥大化 → 1 iter 1 時間 (思考時間 58 分) | iter 1 観察 | 高 |
| D | ledger.jsonl が main では空 (state-sync PR で別箇所に書いた可能性) | git log 確認 | 中 |
| E | `.claude/worktrees/` の散らかり (functional-snuggling-cook, mutable-jingling-thompson) | git worktree list | 低 |
| F | settings.json deny rule の `:*| sh` 形式が warning で skip | tmux 起動時 settings warning | 中 |
| G | Phase 1 着手前に src/features/, src/eval/ の scaffold 無し → autonomous が毎回ゼロから | iter 1 で 1 hr 思考 | 最高 |
| H | fast_sim.py 未着手 (Phase 1 必須、test_sim_parity 前提) | PLAN.md | 高 |
| I | tests/test_timing.py 未追加 (PLAN.md Phase 0 完了基準 #2) | smoke ✗ | 中 |
| J | budget ceiling 機構なし ($/quota 暴走防止) | iter 1 で $11 消費 | 高 |
| K | daily_report.sh 未実装 (supervisor 監視自動化) | PLAN.md A-20 | 中 |
| L | .github/workflows の test split (lint/format/pytest/verify を並列) | CI 高速化 | 低 |
| M | run_iteration.sh の implementer/tournament/submit 配線が Phase 0 skeleton のまま | PLAN.md L225 | 高 |

---

## 求める output: 即時実行プラン

A-M 全タスクを **本日中** (= 今のセッションで) supervisor が完了させるための **詳細実行プラン**。出力構造:

### 1. 並列化マップ (依存関係グラフ)
どのタスクがどれと並列実行可能か、どれが直列必須かを mermaid or table で。

### 2. タスク毎詳細
各 A-M について:
- 想定所要時間 (5-45 min)
- ブランチ名 (`fix/<slug>` or `chore/<slug>`)
- 変更ファイル一覧
- 実装方針 (具体的コード or 設定の方向性)
- テスト方法 (smoke / pytest / 手動)
- マージ判定基準 (CodeRabbit + CI pass + 想定された動作確認)

### 3. 推奨実行順序
依存関係を踏まえ、supervisor (1 シリアル worker) が連続実行する場合の順番。並列化したい場合は別 supervisor worktree を追加する案も。

### 4. 各タスクの PR template
タイトル + body の雛形 (Summary / Test plan / 関連 issue or learned_rules link)。

### 5. autonomous loop との干渉回避策
各タスクで main に merge する時に autonomous が pick up するタイミング、conflict 回避方針 (autonomous の exp/000-parity-fix が merge 候補に上がる前に supervisor PR を流す等)。

### 6. 完了判定
全 13 タスク merge 後の supervisor smoke (`bash scripts/orchestrator/smoke.sh` で PASS 数増加、新規 test pass、daily_report 出力例)。

---

## 補足: 解いてほしい設計判断

1. **parity 問題 (A) の root cause**: PLAN.md にもユーザーの仮説 (main.py wrapper の per-turn _safe_act pre-call / sanitize 二重実行 / numpy lazy import overhead / budget 計算式) があるが、最有力候補と修正方針は?
2. **orchestrator prompt 簡素化 (C)**: 1 iter < 30 min を目指す prompt 構造の具体案 (atomic task 列挙 / 中間アウトプット強制 / advisor 呼出制限など)
3. **fast_sim (H)** の scope: kaggle_environments の forward step を 100% 再現する vs combat/sun-blocking のみ近似 — Phase 1 着手にどこまで必要?
4. **budget ceiling (J)**: 単純な $/日 cap か、tokens/iter cap か、両方か。実装案 (hook で claude を kill するのは無理 → wait 注入で実質減速?)
5. **run_iteration 配線 (M)**: Task ツールで subagent を呼ぶ shell から? それとも shell ではなく claude prompt 側に「subagent 呼び出せ」と書く?
6. **autonomous との競合**: autonomous が exp/000-parity-fix で同じ parity 問題を解こうとしている。supervisor が先回りで PR 出した場合、autonomous の作業を破棄する? 残す?

---

## deliverable format
- `docs/PLAN-iter2.md` (1 ファイル) として保存可能な markdown
- 各タスクの実装コード/設定の draft も含めて構わない (autonomous の代わりに supervisor が直接実装する前提)
- 「すぐ全部やる」前提なので **「Phase 2 で〜」「将来〜」は禁止**。今やる/今やらないの 2 値判定のみ
