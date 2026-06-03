#!/usr/bin/env bash
# awaiting_lb な提出を巡回し、live LB と age に基づき merge/revert/wait 推奨を
# PR コメント (open/closed どちらでも) に冪等投稿する。launchd or cron で 12h ごと実行。
#
# 監視対象 (どちらも対応):
#   (A) OPEN PR で body に "awaiting_lb" を含むもの — 通常の autonomous loop が作る PR
#   (B) ledger.jsonl 末尾 N 件で decision=submitted AND lb_score=null のもの — supervisor
#       の state-only PR が merge された後 OPEN list から消えるケース (PR #79/53299379 で
#       見逃した盲点の修正)
#
# 推奨ロジック (baseline は state/best_score.json か $ORBIT_BASELINE_LB):
#   LB >= baseline + margin AND age >= min_age -> recommend_merge (🟢)
#   LB <= baseline - margin AND age >= min_age -> recommend_revert (🔴)
#   それ以外                                     -> waiting (⏳)
#
# 冪等性: 前回投稿 [auto-LB-monitor] コメントから LB 変動 < LB_DELTA_MIN かつ推奨同じなら skip

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# --- 設定 ---
BASELINE="${ORBIT_BASELINE_LB:-}"
if [ -z "$BASELINE" ]; then
  BASELINE=$(python3 -c "import json; print(json.load(open('state/best_score.json'))['lb_score'])" 2>/dev/null || echo "414.2")
fi
MARGIN="${ORBIT_LB_MARGIN:-5}"
MIN_AGE_HOURS="${ORBIT_LB_MIN_AGE_HOURS:-24}"
LB_DELTA_MIN="${ORBIT_LB_DELTA_MIN:-2}"
LEDGER_SCAN_LIMIT="${ORBIT_LEDGER_SCAN_LIMIT:-50}"     # ledger 末尾 N 件をスキャン
LEDGER_MAX_AGE_HOURS="${ORBIT_LEDGER_MAX_AGE_HOURS:-336}"  # 14日経過は監視打ち切り

REPO_SLUG="${ORBIT_GITHUB_REPO:-Y-Kanekoo/orbit-wars}"
COMPETITION="${ORBIT_KAGGLE_COMP:-orbit-wars}"
TAG="[auto-LB-monitor]"
DRY_RUN="${ORBIT_DRY_RUN:-0}"

NOW_EPOCH=$(date -u +%s)

log() { echo "[recommender] $*" >&2; }

log "baseline=$BASELINE margin=$MARGIN min_age_h=$MIN_AGE_HOURS ledger_scan=$LEDGER_SCAN_LIMIT"

# --- PR 一覧取得 (open + all) ---
OPEN_PR_FILE=$(mktemp /tmp/orbit_pr_open.XXXXXX.json)
ALL_PR_FILE=$(mktemp /tmp/orbit_pr_all.XXXXXX.json)
trap 'rm -f "$OPEN_PR_FILE" "$ALL_PR_FILE"' EXIT

if ! gh pr list --repo "$REPO_SLUG" --state open --json number,title,body --limit 100 > "$OPEN_PR_FILE" 2>&1; then
  log "ERROR: gh pr list (open) 失敗"
  cat "$OPEN_PR_FILE" >&2
  exit 1
fi
if ! gh pr list --repo "$REPO_SLUG" --state all --json number,title,body,state --limit 200 > "$ALL_PR_FILE" 2>&1; then
  log "ERROR: gh pr list (all) 失敗"
  cat "$ALL_PR_FILE" >&2
  exit 1
fi

# --- 監視対象 (submission_id, pr_number, source) を python で構築 ---
TARGETS_FILE=$(mktemp /tmp/orbit_targets.XXXXXX.txt)
python3 <<PYEOF > "$TARGETS_FILE"
import json, re

with open("$OPEN_PR_FILE") as f: open_prs = json.load(f)
with open("$ALL_PR_FILE") as f: all_prs = json.load(f)

def extract_sub_id(title, body):
    title = title or ""
    body = body or ""
    # (1) title の "submission XXX" / "submission_id XXX"
    m = re.search(r"[Ss]ubmission(?:_id)?[ :]+([0-9]{7,9})", title)
    if m: return int(m.group(1))
    # (2) body の明示的 "submission_id[: ]+XXX"
    m = re.search(r"submission_id[ :]+\**([0-9]{7,9})", body)
    if m: return int(m.group(1))
    # (3) fallback: body の最初の 5XXXXXXX 8桁
    m = re.search(r"\b5[0-9]{7}\b", body)
    if m: return int(m.group(0))
    return None

targets = {}  # submission_id -> {pr, source}

# (A) OPEN PR with awaiting_lb
for p in open_prs:
    body = (p.get("body") or "").lower()
    if "awaiting_lb" not in body and "awaiting lb" not in body:
        continue
    sid = extract_sub_id(p.get("title") or "", p.get("body") or "")
    if sid is None: continue
    if sid not in targets:
        targets[sid] = {"pr": p["number"], "source": "open_pr"}

# (B) ledger 末尾スキャン: decision=submitted AND lb_score=null
with open("experiments/ledger.jsonl") as f:
    lines = [l for l in f if l.strip()]
for ln in lines[-$LEDGER_SCAN_LIMIT:]:
    try:
        o = json.loads(ln)
    except Exception:
        continue
    if o.get("decision") != "submitted" or o.get("lb_score") is not None:
        continue
    sid = o.get("submission_id")
    if not sid: continue
    sid = int(sid)
    if sid in targets:
        continue  # OPEN PR 経由で既にカバー済
    # ALL PR から submission_id 一致を探す
    for p in all_prs:
        sid_in_pr = extract_sub_id(p.get("title") or "", p.get("body") or "")
        if sid_in_pr == sid:
            state = p.get("state", "UNKNOWN").lower()
            targets[sid] = {"pr": p["number"], "source": f"ledger_scan_{state}"}
            break
    else:
        # PR が見つからない (orphan submission): ledger だけにある
        targets[sid] = {"pr": None, "source": "ledger_orphan"}

for sid, info in sorted(targets.items()):
    print(f"{sid}|{info['pr'] or ''}|{info['source']}")
PYEOF

if [ ! -s "$TARGETS_FILE" ]; then
  log "監視対象なし。終了。"
  rm -f "$TARGETS_FILE"
  exit 0
fi

log "監視対象 $(wc -l < "$TARGETS_FILE") 件:"
while IFS='|' read -r SID PR_NUM SRC; do
  log "  - sub=$SID pr=#${PR_NUM:--} src=$SRC"
done < "$TARGETS_FILE"

# --- 全 submission を 1 回取得 ---
SUBS_RAW=$(kaggle competitions submissions "$COMPETITION" 2>&1)
if [ $? -ne 0 ]; then
  log "ERROR: kaggle CLI 失敗: $SUBS_RAW"
  rm -f "$TARGETS_FILE"
  exit 1
fi

# --- 各 target を処理 ---
while IFS='|' read -r SUB_ID PR_NUM SRC; do
  log "--- sub=$SUB_ID (pr=#${PR_NUM:--}, src=$SRC) ---"

  if [ -z "$PR_NUM" ]; then
    log "  PR 未発見の orphan submission。skip (将来は state ファイルへ通知化検討)"
    continue
  fi

  SUB_LINE=$(echo "$SUBS_RAW" | grep -E "^${SUB_ID}\s" | head -1)
  if [ -z "$SUB_LINE" ]; then
    log "  Kaggle に submission $SUB_ID が見つからず。skip"
    continue
  fi

  PUBLIC=$(echo "$SUB_LINE" | awk '{
    n=NF
    if ($n ~ /^[0-9]+(\.[0-9]+)?$/) print $n
    else if ($(n-1) ~ /^[0-9]+(\.[0-9]+)?$/) print $(n-1)
  }')
  SUB_DATE=$(echo "$SUB_LINE" | awk '{print $3" "$4}')
  STATUS=$(echo "$SUB_LINE" | grep -oE "SubmissionStatus\.[A-Z]+" | head -1)

  if [ -z "$PUBLIC" ]; then
    log "  publicScore 抽出失敗 (status=$STATUS)。skip"
    continue
  fi

  SUB_EPOCH=$(date -j -u -f "%Y-%m-%d %H:%M:%S" "${SUB_DATE%.*}" +%s 2>/dev/null)
  if [ -z "$SUB_EPOCH" ]; then
    log "  日時パース失敗: '$SUB_DATE'。skip"
    continue
  fi
  AGE_H=$(( (NOW_EPOCH - SUB_EPOCH) / 3600 ))

  # ledger 由来の古い entry (>14d) は監視打ち切り
  if [ "$SRC" != "open_pr" ] && [ "$AGE_H" -gt "$LEDGER_MAX_AGE_HOURS" ]; then
    log "  age=${AGE_H}h > ${LEDGER_MAX_AGE_HOURS}h (ledger scan の age 上限超過)。skip"
    continue
  fi

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

  # 冪等性チェック: 既存 [auto-LB-monitor] コメントと比較
  LAST_COMMENT=$(gh pr view "$PR_NUM" --repo "$REPO_SLUG" --comments --json comments \
    --jq ".comments[] | select(.body | startswith(\"$TAG\")) | .body" 2>/dev/null | tail -1)

  if [ -n "$LAST_COMMENT" ]; then
    LAST_LB=$(echo "$LAST_COMMENT" | grep -oE "current_lb=[0-9]+(\.[0-9]+)?" | head -1 | cut -d= -f2)
    LAST_REC=$(echo "$LAST_COMMENT" | grep -oE "recommendation=[a-z_]+" | head -1 | cut -d= -f2)
    if [ -n "$LAST_LB" ] && [ -n "$LAST_REC" ]; then
      BELOW_DELTA=$(python3 -c "print(1 if abs($PUBLIC - $LAST_LB) < $LB_DELTA_MIN else 0)")
      if [ "$BELOW_DELTA" = "1" ] && [ "$LAST_REC" = "$RECOMMEND" ]; then
        log "  skip: 前回コメントから LB 変動 < $LB_DELTA_MIN かつ推奨同じ ($LAST_REC)"
        continue
      fi
    fi
  fi

  EMOJI=""
  case "$RECOMMEND" in
    recommend_merge)   EMOJI="🟢" ;;
    recommend_revert)  EMOJI="🔴" ;;
    waiting_*)         EMOJI="⏳" ;;
  esac

  SRC_NOTE=""
  case "$SRC" in
    ledger_scan_*)     SRC_NOTE=" (ledger scan, PR は ${SRC#ledger_scan_})" ;;
  esac

  COMMENT_BODY="$TAG ${EMOJI}${SRC_NOTE}
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
    echo "----- would post to PR #$PR_NUM (src=$SRC) -----"
    echo "$COMMENT_BODY"
    echo "------------------------------------------------"
  else
    echo "$COMMENT_BODY" | gh pr comment "$PR_NUM" --repo "$REPO_SLUG" --body-file - >&2
    log "  コメント投稿 OK"
  fi
done < "$TARGETS_FILE"

rm -f "$TARGETS_FILE"
log "完了"
