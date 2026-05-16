# orbit-wars 自律ハーネス

> このドキュメントは Phase 0 構築直後の実態を反映 (2026-05-16)。

## 概要
`claude -p --worktree` を tmux で常駐させ、`docs/PLAN.md` に沿って自律的に Orbit Wars コンペの実験ループを回す。

## 構成要素

### 1. State (`state/`)
| file | 役割 | 更新者 |
|---|---|---|
| `best_score.json` | 現 champion の LB / submission_id / agent_sha | pr-author (merge 時) |
| `quota.json` | 提出 5/日, Notebook GPU 30hr/週 | record-submission hook, score-analyzer |
| `hypotheses.md` | 仮説バックログ (priority desc) | researcher, implementer (status 更新) |
| `learned_rules.md` | 同一エラー 3 回到達で昇格された AVOID rules | analyze-deviations hook |
| `error_counts.json` | recurring_error signature counter | score-analyzer |
| `lb_history.jsonl` | LB top 推移 (24h 急上昇検知 = A-15) | leaderboard.sh |
| `iteration_cooldown` | trigger-next-iteration の発火間隔制御 (60s) | trigger-next-iteration hook |
| `ensemble_pool.json` | Phase 4 召喚候補 (reserve 判定の branch 集) | pr-author |

### 2. Subagents (`.claude/agents/*.md`)
PLAN.md L157-178 に基づく 5 個:
- **researcher** — hypothesis 補充 (週次 or active <3 で起動)
- **implementer** — 仮説 1 件を exp branch で実装 + ローカル test
- **kaggle-runner** — submission pack + verify + submit (quota guard 厳守)
- **score-analyzer** — LB polling + episodes/replays 解析 + recurring_error 検出 + lessons 抽出
- **pr-author** — PR 作成 + CI gate + σ-aware merge 判定 (gain≥+20 AND gain≥2σ AND played≥15)

### 3. Hooks (`.claude/hooks/*.sh`) 8 個
| event | matcher | hook | 概要 |
|---|---|---|---|
| PreToolUse | Bash | guard-dangerous-bash | rm -rf / push --force / --no-verify / pipe-to-shell / secrets 含む git add を block |
| PreToolUse | Bash | kaggle-quota-guard | `kaggle submit/kernels push` の quota chk、5/日 or 30h/週 超で block |
| PostToolUse | Edit\|Write | auto-lint-fmt | *.py に対し ruff --fix + black、残 lint で exit 2 |
| PostToolUse | Bash | record-submission | `kaggle submit` 実行後に ledger.jsonl + quota.json update |
| Stop | "" | analyze-deviations | LB delta / quota 残出力、error_counts 3 到達で learned_rules 昇格 |
| Stop | "" | auto-commit-safe | critical tests pass のみ WIP commit、main 直編集は skip |
| Stop | "" | trigger-next-iteration | tmux session に次イテレーション prompt 送信 (cooldown 60s) |
| SessionStart | "" | load-state | best_score / quota / ledger / hypotheses / learned_rules / 推定 Phase を context に注入 |

### 4. Scripts
| path | 概要 |
|---|---|
| `scripts/orchestrator/smoke.sh` | Phase 0 完了基準診断 (auth/runtime/state/legacy-388 動作確認) |
| `scripts/orchestrator/tmux_launcher.sh` | tmux orbit-wars session で claude -p detach 起動 |
| `scripts/orchestrator/run_iteration.sh` | 1 iteration 本体 (Phase 0 skeleton、Phase 1+ で implementer 配線) |
| `scripts/orchestrator/resume.sh` | クラッシュ復旧 (ledger interrupted 終端、tmux 状況) |
| `scripts/orchestrator/backup.sh` | 日次 git push + state 確認 (A-7) |
| `scripts/selfplay/match.py` | 1 試合 subprocess 実行 (--agent1 --agent2 --seed) |
| `scripts/selfplay/tournament.py` | ProcessPoolExecutor で N 試合並列、winrate 集計 |
| `scripts/kaggle/verify_submission.sh` | 純粋 python 環境で main.agent 動作確認 (A-5) |
| `scripts/kaggle/submit.sh` | verify → tar pack → 再 verify → kaggle submit |
| `scripts/kaggle/leaderboard.sh` | LB pull + lb_history.jsonl append |
| `scripts/kaggle/episodes.sh` | submission の episode 一覧 |
| `scripts/kaggle/replays.sh` | episode の replay JSON pull |
| `scripts/kaggle/push_notebook.sh` | kernel push (Phase 3+) |
| `scripts/kaggle/wait_run.sh` | kernel status 60s polling (max 180min) |
| `scripts/kaggle/pull_output.sh` | kernel output pull (Phase 3+) |

### 5. Source code (`src/`)
- `src/utils/observation.py` (B-9): Planet/Fleet NamedTuple + defensive parser
- `src/utils/action.py` (A-3): sanitize() で不正 move 除外 + 累積 ≤ garrison
- `src/utils/timing.py` (A-1): deadline_iter / Timer
- `src/utils/telemetry.py` (A-14): log/debug/error (LOG_LEVEL env で切替)
- `src/agents/safe_fallback.py`: < 50ms 保証 Nearest-Planet Sniper
- `main.py` (A-1): anytime wrapper (fallback 確保 → core_act deadline 0.90s → 例外/timeout で fallback)

### 6. Tests
- `tests/test_action_legal.py` (12 件): sanitize 仕様 + random hammer
- `tests/test_observation_robustness.py` (10 件): defensive parsing + 4P
- (Phase 1+) `test_timing.py` (1000 turn 全て < 950ms 検証)
- (Phase 1+) `test_sim_parity.py` (fast_sim と kaggle_environments の挙動一致)

## 起動方法

### 初回 (Phase 0 完了確認)
```bash
cd ~/Projects/orbit-wars

# 1. 依存を install
pip install -r requirements.txt

# 2. smoke test で Phase 0 11 項目を確認
bash scripts/orchestrator/smoke.sh

# 3. 全 pytest pass
pytest -q tests/
```

### 自律ループ常駐
```bash
bash scripts/orchestrator/tmux_launcher.sh   # 起動

tmux attach -t orbit-wars                     # 進行確認
tmux capture-pane -p -t orbit-wars | tail -50 # ログ snapshot

tmux kill-session -t orbit-wars               # 停止
```

### Supervisor (監視/メタ作業) は必ず別 worktree (iter 1 教訓)
autonomous tmux claude が `~/Projects/orbit-wars/` を占有する間、human supervisor (or 別 claude session) が同じ working tree で `git checkout` 等を行うと autonomous claude も巻き込まれて state 混乱が起きる。supervisor は **必ず別 worktree** で作業すること。

```bash
# 初回のみ (or 既存なら noop)
bash scripts/orchestrator/setup_supervisor_worktree.sh

# 以降の supervisor 作業はここで
cd ~/Projects/orbit-wars-watch
# fix PR / docs 更新 / state 監視等
```

`~/Projects/orbit-wars/` は autonomous loop 専用、`~/Projects/orbit-wars-watch/` は supervisor 専用。

### 中断復旧
```bash
bash scripts/orchestrator/resume.sh    # ledger 整理 + tmux 状況確認
bash scripts/orchestrator/tmux_launcher.sh   # 再起動
```

## 自己改善ループ (PLAN.md 408-465)

1. **同一エラー 3 回 → learned_rules 昇格**: analyze-deviations が `error_counts.json` を見て自動昇格
2. **仮説 dynamic 補充**: active < 3 で researcher 自動起動
3. **gain plateau で phase 移行**: 直近 5 実験 gain ≤ +10 で Phase X+1 へ
4. **submission backoff & explore/exploit**: 5/日を 3 安全 + 2 探索で割当
5. **失敗 replay の自動回帰**: 負け試合を `experiments/replays/regression/` に保存、新 agent はこれで勝つことを test
6. **cost-aware budget**: Claude API 残で implementer prompt を minimal mode に
7. **dead-branch GC**: 7 日動き無い exp/* を ledger 確認後に削除

## 安全境界

- main 直 push 禁止 (settings.json deny rule + auto-commit-safe hook で二重防止)
- `--no-verify` 禁止 (deny rule + guard hook)
- テスト改変禁止 (implementer subagent 制約)
- 同一エラー 3 回 → 異 approach、6 回 → skip + `TODO(autonomous):`
- Kaggle quota 5/日, GPU 30hr/週 を hook で hard limit
- actTimeout 1 秒/turn を main.py anytime wrapper で物理的に保証

## トラブルシュート

| 症状 | 対処 |
|---|---|
| smoke FAIL: kaggle-environments import 不可 | `pip install -r requirements.txt` |
| tmux launcher が即終了 | `tmux capture-pane -p -t orbit-wars` でログ確認、claude CLI auth を `gh auth status` で chk |
| submit が hook で block | `cat state/quota.json` で残数確認、`date` が当日に更新されていないか |
| 連続エラーで loop 停止 | `cat state/learned_rules.md` で AVOID 確認、追加 hypothesis を researcher で投入 |
| LB が伸びない (連続 5 iter gain≤0) | strategy pivot 通知発火、researcher 強制起動 (A-20) |
| tmux pane が空のまま動かない | `which claude` で `/Applications/cmux.app/...` (GUI) が解決されてないか確認、tmux_launcher.sh は `~/.local/bin/claude` (CLI) を明示解決済 |
| hook で `Permission denied` 連発 | `git ls-files -s .claude/hooks/` で全 hook が `100755` か確認 (iter 1 で guard-dangerous-bash.sh が 100644 のまま merge 事故あり)。smoke.sh が今は検証する |
| supervisor が branch を切ったら autonomous claude が混乱 | supervisor は `~/Projects/orbit-wars-watch/` 別 worktree で作業 (`setup_supervisor_worktree.sh` で作成)、main repo dir を直接触らない |
