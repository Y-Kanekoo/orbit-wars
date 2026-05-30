#!/usr/bin/env bash
# awaiting_lb の OPEN PR を巡回し、live LB と提出経過時間に基づき
# merge/revert/wait の推奨コメントを投稿する。launchd or cron で
# 12h ごとに実行する想定 (前回コメントから状態が動いていなければ skip)。
#
# 推奨ロジック (baseline は state/best_score.json の lb_score か $BASELINE で上書き):
#   LB >= baseline + 5  AND  age >= 24h          -> recommend_merge
#   LB <= baseline - 5  AND  age >= 24h          -> recommend_revert
#   それ以外                                       -> waiting (LB 揺らぎ中 or 早期)
#
# 冪等性: 前回投稿した [auto-LB-monitor] コメントの LB と推奨を比較し、
#   - LB 変動 < 2pt かつ 推奨が同じ -> skip
#   - 上記以外 -> 新規コメント投稿

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# --- 設定 ---
BASELINE="${ORBIT_BASELINE_LB:-}"
if [ -z "$BASELINE" ]; then
  # state/best_score.json から lb_score を抜く (jq 不要、python で)
  BASELINE=$(python3 -c "import json; print(json.load(open('state/best_score.json'))['lb_score'])" 2>/dev/null || echo "414.2")
fi
MARGIN="${ORBIT_LB_MARGIN:-5}"           # baseline ± margin で推奨
MIN_AGE_HOURS="${ORBIT_LB_MIN_AGE_HOURS:-24}"  # 推奨に必要な最低経過時間
LB_DELTA_MIN="${ORBIT_LB_DELTA_MIN:-2}"  # 再投稿の最小 LB 変動

REPO_SLUG="${ORBIT_GITHUB_REPO:-Y-Kanekoo/orbit-wars}"
COMPETITION="${ORBIT_KAGGLE_COMP:-orbit-wars}"
TAG="[auto-LB-monitor]"
DRY_RUN="${ORBIT_DRY_RUN:-0}"  # 1 で gh pr comment を行わずログ出力のみ

NOW_EPOCH=$(date -u +%s)

log() { echo "[recommender] $*" >&2; }

log "baseline=$BASELINE margin=$MARGIN min_age_h=$MIN_AGE_HOURS"

# --- OPEN PR 一覧と本文を取得 ---
PR_JSON=$(gh pr list --repo "$REPO_SLUG" --state open --json number,title,body 2>&1)
if [ $? -ne 0 ]; then
  log "ERROR: gh pr list 失敗: $PR_JSON"
  exit 1
fi

# awaiting_lb を含む PR のみ抽出 (number だけ)
AWAITING_PRS=$(echo "$PR_JSON" | python3 -c "
import json, sys
prs = json.load(sys.stdin)
for p in prs:
    body = (p.get('body') or '').lower()
    if 'awaiting_lb' in body or 'awaiting lb' in body:
        print(p['number'])
")

if [ -z "$AWAITING_PRS" ]; then
  log "awaiting_lb の PR なし。終了。"
  exit 0
fi

log "awaiting_lb PRs: $(echo $AWAITING_PRS | tr '\n' ' ')"

# --- 全 submission を 1 回取得してキャッシュ (各 PR の処理で使い回す) ---
SUBS_RAW=$(kaggle competitions submissions "$COMPETITION" 2>&1)
if [ $? -ne 0 ]; then
  log "ERROR: kaggle CLI 失敗: $SUBS_RAW"
  exit 1
fi

for PR in $AWAITING_PRS; do
  log "--- PR #$PR ---"

  # PR タイトルと本文を取得
  PR_TITLE=$(echo "$PR_JSON" | python3 -c "
import json, sys
prs = json.load(sys.stdin)
for p in prs:
    if p['number'] == $PR:
        print(p.get('title') or '')
        break
")
  PR_BODY=$(echo "$PR_JSON" | python3 -c "
import json, sys
prs = json.load(sys.stdin)
for p in prs:
    if p['number'] == $PR:
        print(p.get('body') or '')
        break
")

  # submission_id 抽出: (1) タイトルの "submission XXX" / "submission_id XXX"、
  # (2) 本文の明示的 "submission_id[: ]+XXX" (loop の auto PR フォーマット)、
  # (3) fallback: 本文の最初の 5XXXXXX 8 桁。本文に複数 submission_id が
  # 表に並ぶ PR (supervisor の比較表) では (1)(2) が優先される設計。
  SUB_ID=$(echo "$PR_TITLE" | grep -oE "[Ss]ubmission(_id)?[ :]+[0-9]{7,9}" | head -1 | grep -oE "[0-9]{7,9}")
  if [ -z "$SUB_ID" ]; then
    SUB_ID=$(echo "$PR_BODY" | grep -oE "submission_id[ :]+\**[0-9]{7,9}" | head -1 | grep -oE "[0-9]{7,9}")
  fi
  if [ -z "$SUB_ID" ]; then
    SUB_ID=$(echo "$PR_BODY" | grep -oE "\b5[0-9]{7}\b" | head -1)
  fi
  if [ -z "$SUB_ID" ]; then
    log "  submission_id を title/body から抽出できず。skip"
    continue
  fi
  log "  submission_id=$SUB_ID"

  # submission の現 LB と提出日時を取得
  SUB_LINE=$(echo "$SUBS_RAW" | grep -E "^${SUB_ID}\s" | head -1)
  if [ -z "$SUB_LINE" ]; then
    log "  Kaggle に submission $SUB_ID が見つからず。skip"
    continue
  fi
  # 列構造: ref fileName date(2 列: YYYY-MM-DD HH:MM:SS.ffffff) description... status publicScore privateScore
  # publicScore は末尾近く。awk で末尾 2-3 トークン目を見る方が安全
  PUBLIC=$(echo "$SUB_LINE" | awk '{
    # 末尾から数えて publicScore を取る (privateScore が空なら最後のトークン、あれば最後から2番目)
    n=NF
    # 数値判定
    if ($n ~ /^[0-9]+(\.[0-9]+)?$/) print $n
    else if ($(n-1) ~ /^[0-9]+(\.[0-9]+)?$/) print $(n-1)
  }')
  SUB_DATE=$(echo "$SUB_LINE" | awk '{print $3" "$4}')
  STATUS=$(echo "$SUB_LINE" | grep -oE "SubmissionStatus\.[A-Z]+" | head -1)

  if [ -z "$PUBLIC" ]; then
    log "  publicScore を抽出できず (status=$STATUS)。skip"
    continue
  fi

  # 提出からの経過時間 (h)
  SUB_EPOCH=$(date -j -u -f "%Y-%m-%d %H:%M:%S" "${SUB_DATE%.*}" +%s 2>/dev/null)
  if [ -z "$SUB_EPOCH" ]; then
    log "  提出日時パース失敗: '$SUB_DATE'。skip"
    continue
  fi
  AGE_H=$(( (NOW_EPOCH - SUB_EPOCH) / 3600 ))

  # 推奨判定
  DELTA=$(python3 -c "print(round($PUBLIC - $BASELINE, 1))")
  RECOMMEND=$(python3 -c "
lb=$PUBLIC; base=$BASELINE; margin=$MARGIN; age=$AGE_H; min_age=$MIN_AGE_HOURS
if age < min_age:
    print('waiting_age')
elif lb >= base + margin:
    print('recommend_merge')
elif lb <= base - margin:
    print('recommend_revert')
else:
    print('waiting_unclear')
")
  log "  LB=$PUBLIC baseline=$BASELINE delta=$DELTA age=${AGE_H}h status=$STATUS -> $RECOMMEND"

  # 既存の最新 [auto-LB-monitor] コメントを取得
  LAST_COMMENT=$(gh pr view "$PR" --repo "$REPO_SLUG" --comments --json comments \
    --jq ".comments[] | select(.body | startswith(\"$TAG\")) | .body" 2>/dev/null | tail -1)

  # 冪等性チェック
  if [ -n "$LAST_COMMENT" ]; then
    LAST_LB=$(echo "$LAST_COMMENT" | grep -oE "current_lb=[0-9]+(\.[0-9]+)?" | head -1 | cut -d= -f2)
    LAST_REC=$(echo "$LAST_COMMENT" | grep -oE "recommendation=[a-z_]+" | head -1 | cut -d= -f2)
    if [ -n "$LAST_LB" ] && [ -n "$LAST_REC" ]; then
      DIFF_ABS=$(python3 -c "print(abs($PUBLIC - $LAST_LB))")
      BELOW_DELTA=$(python3 -c "print(1 if abs($PUBLIC - $LAST_LB) < $LB_DELTA_MIN else 0)")
      if [ "$BELOW_DELTA" = "1" ] && [ "$LAST_REC" = "$RECOMMEND" ]; then
        log "  skip: 前回コメントから LB 変動 $DIFF_ABS < $LB_DELTA_MIN かつ推奨同じ ($LAST_REC)"
        continue
      fi
    fi
  fi

  # コメント投稿
  EMOJI=""
  case "$RECOMMEND" in
    recommend_merge)   EMOJI="🟢" ;;
    recommend_revert)  EMOJI="🔴" ;;
    waiting_*)         EMOJI="⏳" ;;
  esac

  COMMENT_BODY="$TAG ${EMOJI}
- submission_id=$SUB_ID, status=$STATUS, age=${AGE_H}h
- current_lb=$PUBLIC, baseline=$BASELINE, delta=$DELTA
- recommendation=$RECOMMEND (margin=±${MARGIN}, min_age=${MIN_AGE_HOURS}h)

| 条件 | 値 | 閾値 |
|---|---|---|
| age | ${AGE_H}h | >= ${MIN_AGE_HOURS}h |
| LB vs baseline | ${DELTA} | +${MARGIN} で merge / -${MARGIN} で revert |

(自動投稿、12h 間隔。同一推奨かつ LB 変動 < ${LB_DELTA_MIN}pt なら次回 skip)"

  if [ "$DRY_RUN" = "1" ]; then
    log "  DRY-RUN: コメント投稿を skip"
    echo "----- would post to PR #$PR -----"
    echo "$COMMENT_BODY"
    echo "---------------------------------"
  else
    echo "$COMMENT_BODY" | gh pr comment "$PR" --repo "$REPO_SLUG" --body-file - >&2
    log "  コメント投稿 OK"
  fi
done

log "完了"
