# Learned Rules

`analyze-deviations.sh` hook + `score-analyzer` subagent が同一エラー 3 回検出時にここへ昇格する (PLAN.md 自己改善ループ-1)。

`load-state.sh` hook が毎 session 開始時にこのファイルを context に注入し、`implementer` の system prompt に "次の rules を遵守: ..." として埋め込む。

## スキーマ
- `AVOID: <signature> — <root_cause>` 形式で 1 行 1 ルール
- `signature` は score-analyzer の `recurring_error_signature` 出力に対応

## ルール

(なし — 自律ループ稼働後に蓄積される)
