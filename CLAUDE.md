# orbit-wars プロジェクト指示

> グローバル / `~/Projects/CLAUDE.md` を継承。以下は orbit-wars 固有の Intent のみ。

## Intent: 一桁順位達成
目的: Kaggle Orbit Wars コンペで Private LB top-9 を達成する
制約:
- 締切 2026-06-23 23:59 UTC を絶対遵守
- 全実装 は Claude Code 本体 (Codex 解約済み)
- 実装 → ローカル self-play 評価 → kaggle 提出 の順序を守る
- main 直 push 禁止、必ず `exp/<NNN>-<slug>` ブランチ + PR + merge
成功指標: 2026-06-23 時点で LB rank ≤ 9

## Intent: Kaggle compute 活用
目的: GPU heavy 処理 (self-play, NN training) は Kaggle Notebooks に委譲
制約:
- ローカル GPU 学習禁止 (CPU で kaggle-environments の self-play 評価のみ可)
- Notebook push&run&pull は `scripts/kaggle/*.sh` 経由
- 1 週 30hr の GPU quota を `state/quota.json` で追跡
成功指標: ローカル GPU 利用ゼロ、weekly quota 超過ゼロ

## Intent: 1 秒/turn 制約遵守
目的: 提出 agent は actTimeout 1 秒/turn を絶対超えない
制約:
- ローカル self-play で `assert max_turn_ms < 900` を CI で実施
- NN inference は ONNX export & quantize 必須
- search depth/width は budget 内で動的調整
成功指標: 提出後の disqualification ゼロ

## Intent: submission quota の最大活用
目的: 1 日 5 submissions を浪費せず計画的に消化
制約:
- 提出前に必ず `state/quota.json` を確認 (kaggle-quota-guard hook)
- ローカル winrate ≥ 55% (vs current best, N=100) を満たさない agent は提出しない
- 提出後 1-3hr で score を反映、次提出は score 確認後
成功指標: 無駄提出 (winrate < 50% にもかかわらず提出) ゼロ

## Intent: 学習可能なハーネス
目的: 失敗パターンを自動記録し、再発を防止する
制約:
- 各実験は `experiments/ledger.jsonl` に append (exp_id, hypothesis, score, decision, lessons)
- 同一エラー 3 回検出 → `learned_rules.md` 昇格 → 以降の hypothesis 生成で参照
成功指標: 同パターン再発ゼロ

## Intent: 並列実験の安全実行
目的: worktree を使い複数実験を並走させる
制約:
- worktree は `~/Projects/orbit-wars-worktrees/exp-<NNN>/` 配下
- 同一ファイル並行編集禁止 (主軸 main、各 worktree は独立 branch)
- ローカル self-play は CPU を埋め尽くさない (max workers = nproc / 2)

## Intent: supervisor と autonomous loop の working tree 分離 (iter 1 教訓)
目的: 監視/メタ作業を行う human supervisor が autonomous tmux claude と同じ
working dir を共有しないようにする
背景: 2026-05-16 iter 1 で、supervisor が `git checkout -b fix/...` した
瞬間に autonomous claude も同じ working dir のため意図しないブランチを
見る状態になり、後続作業に混乱を招いた
制約:
- autonomous loop は `~/Projects/orbit-wars/` を占有 (tmux session "orbit-wars" 専用)
- supervisor (人間 + 別 claude セッション) は **必ず別 worktree** で作業
  - 標準パス: `~/Projects/orbit-wars-watch/` (branch `supervisor/observe`)
  - 作成は `bash scripts/orchestrator/setup_supervisor_worktree.sh` で
- supervisor からの fix PR は supervisor worktree 上で作成 → main に merge
- autonomous loop は main の更新を git pull で自動取り込み
成功指標: 同一 working tree で human と autonomous loop が同時に branch
を切り替える事故ゼロ

## 言語・コミット
- 日本語回答、コメント日本語、変数・関数名英語
- コミット形式: `[type] 要約` (feat/fix/refactor/docs/test/chore/ops/exp)
- 実験ブランチコミット: `[exp NNN] 要約`
- 1 コミット 1 目的、最小化
