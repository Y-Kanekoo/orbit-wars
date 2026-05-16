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
| - | - | - | - | - | - | (researcher が populate 予定) | - | - |

## In progress

(なし)

## Done

(なし)

## Discarded / Stuck

(なし)
