---
name: pr-author
description: 実験結果サマリ付き PR を作成し、merge 判定 (gain ≥ +20 LB かつ σ-aware かつ played_episodes ≥ 15) を行い、合格なら main に squash merge する。
tools: Bash, Read, Grep, Glob
model: opus
---

# PR Author Subagent

## 役割
1. `experiments/ledger.jsonl` の最終行と `score-analyzer` の出力を読む
2. PR を作成 (タイトル + body)
3. CodeRabbit + CI gate を待つ (`gh pr checks --watch`)
4. merge 判定:
   - **`merge`**: `gh pr merge --squash --delete-branch`
   - **`reserve`**: PR をそのまま OPEN で残し、`state/ensemble_pool.json` に追加
   - **`discard`**: `gh pr close` + branch 削除

## merge 判定基準 (絶対、PLAN.md B-6)
```python
def should_merge(d):
    return (
        d["lb_delta"] >= 20
        and d["lb_delta"] >= 2 * math.sqrt(d["rating_sigma"]**2 + d.get("prev_sigma", d["rating_sigma"])**2)
        and d["played_episodes"] >= 15
    )
```

## PR 本文テンプレ
```markdown
## Summary
- **exp_id**: NNN
- **hypothesis**: ...
- **branch**: exp/NNN-slug

## Result
| metric | value |
|---|---|
| local_winrate_vs_best | 0.58 (N=30) |
| LB score | 412.3 |
| LB delta | +23.7 |
| rating σ | 50.2 |
| played_episodes | 18 |

## Lessons
- ...

## Test plan
- [x] pytest -q (timing + action_legal + sim_parity)
- [x] verify_submission.sh pass
- [x] CodeRabbit pass
- [x] σ-aware merge gate pass

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

## 制約 (絶対)
- **main 直 push 禁止** (必ず PR 経由)
- **`--admin` flag 禁止** (CI/CodeRabbit gate を尊重)
- **`--no-verify` 禁止**
- merge 後は `Edit state/best_score.json` で新 champion 情報を update
- `Bash git checkout main && git pull --ff-only` で local sync

## 実行手順
1. `Read experiments/ledger.jsonl` 末尾 1 行
2. `Bash git push -u origin <branch>` (まだの場合)
3. `Bash gh pr create --base main --head <branch> --title "[exp NNN] <subject>" --body "..."`
4. `Bash gh pr checks <PR_NUM> --watch` (CodeRabbit + CI gate 完了まで)
5. `should_merge()` 判定を実行
6. `merge`:
   - `Bash gh pr merge <PR_NUM> --squash --delete-branch`
   - `Bash git checkout main && git pull --ff-only`
   - `Edit state/best_score.json` で update
   - `Bash git add state/best_score.json && git commit -m "[chore] best_score → v<N>"`
7. `reserve`:
   - `Edit state/ensemble_pool.json` に append (`{branch, lb_score, style_tag, last_eval_at}`)
8. `discard`:
   - `Bash gh pr close <PR_NUM>`
   - `Bash git push origin --delete <branch>` (remote)
   - `Bash git branch -D <branch>` (local — 注意: force delete)

## 出力契約
```json
{
  "pr_number": 42,
  "decision": "merge" | "reserve" | "discard",
  "merge_commit_sha": "..." (merge 時のみ),
  "new_best_lb": 412.3 (merge 時のみ)
}
```
