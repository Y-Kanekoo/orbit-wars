# state/

自律ループが参照・更新する永続状態を保持する。

## ファイル

### `best_score.json`
現在の champion (main にマージされた最良 agent) の情報。
新しい実験ブランチが提出時に LB スコアを取得し、これを超えた場合のみ PR → merge で更新される。

| field | 説明 |
|---|---|
| `version` | champion バージョン (0 = 新規実装前) |
| `lb_score` | Kaggle LB publicScore |
| `local_winrate` | 直前 champion に対するローカル self-play 勝率 (≥0.55 が PR 条件) |
| `submission_id` | Kaggle submission ID |
| `branch` | merge 元の experiment branch 名 |
| `merged_at` | YYYY-MM-DD |
| `agent_signature` | 戦略の短い説明 |

### `quota.json`
Kaggle 提出 quota と Notebook GPU 時間の追跡。`kaggle-quota-guard.sh` hook が読む。

| field | 説明 |
|---|---|
| `date` | 集計日 (UTC)、日付が変われば submissions_today を 0 にリセット |
| `submissions_today` | 当日提出数 (max 5) |
| `kaggle_kernel_runs_this_week_hours` | 今週の Notebook GPU 累積時間 (max 30) |
| `claude_max_quota_status` | Claude Max plan の quota 推定状態 (ok / low / exhausted) |
