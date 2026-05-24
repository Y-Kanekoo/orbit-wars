"""mix-eval 結果に対する submit gate 判定 (H021, no-regression 方式)。

使い方:
    python scripts/kaggle/check_mix_gate.py state/last_mix_eval.json \
        [--max-age-sec N] [--baseline-json state/best_score.json]

gate 条件 (全て満たすと exit 0、 1 つでも欠けると exit 1):
    winrate_random             >= baseline_random        (no-regression)
    winrate_nearest_sniper     >= baseline_nearest_sniper (no-regression)
    winrate_min(random,sniper) >= 0.50                    (diverse-opponent 絶対下限)
    prev_best                  >= 0.35                    (collapse tripwire のみ、改善要求なし)
    errors_total               == 0

baseline は --baseline-json (default state/best_score.json) の mix_eval.opponents.*.winrate
から動的に読む (hardcode 回避)。mix_eval section が無い場合のみ保守的な絶対値に fallback。

設計意図 (supervisor 通知3, 2026-05-24): prev_best (= legacy-388 ミラー) は biased low かつ
LB 非予測のため block 条件から除外 (warn only)。6 連続 discard の真因が prev_best gate
(+0.05 改善要求) と判明 — random/sniper は改善していた。prev_best は collapse (< 0.35) のみ
tripwire、winrate_min も prev_best を除外し random/sniper のみで判定。LB regress 時は prev_best
を no-regression (baseline 据置) に revert する (+0.05 版には戻さない、明確に誤りだった)。

stale check: --max-age-sec を指定すると、 mix_eval の measured_at が古すぎる場合 block。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PREV_FLOOR = 0.35  # prev_best (ミラー) collapse tripwire のみ。改善要求せず block 条件から除外
WIN_MIN = 0.50  # random/sniper (diverse opponent) の絶対下限

# best_score.json に mix_eval が無い場合の保守的 fallback (絶対値)
FALLBACK_BASELINE = {
    "random": 0.90,
    "nearest_sniper": 0.60,
    "prev_best": 0.50,
}


def _baselines(baseline_path: Path) -> dict[str, float]:
    """baseline json の mix_eval から 3 相手の baseline winrate を読む。

    mix_eval section が無い (schema v1 等) 場合は FALLBACK_BASELINE を使い、
    prev_best のみ legacy フィールド local_winrate_vs_prev_best を尊重する。
    """
    if not baseline_path.exists():
        return dict(FALLBACK_BASELINE)
    data = json.loads(baseline_path.read_text())

    # mix-eval 結果ファイル形式 (opponents を top-level に持つ) もそのまま読めるように
    mix = data.get("mix_eval") if isinstance(data.get("mix_eval"), dict) else data
    opp = mix.get("opponents", {}) if isinstance(mix, dict) else {}

    result: dict[str, float] = {}
    for key in ("random", "nearest_sniper", "prev_best"):
        o = opp.get(key, {}) if isinstance(opp, dict) else {}
        if isinstance(o, dict) and "winrate" in o:
            result[key] = float(o["winrate"])
        else:
            result[key] = FALLBACK_BASELINE[key]

    # mix_eval が全く無い場合 prev_best は legacy フィールドを尊重
    if not opp:
        legacy = data.get("local_winrate_vs_prev_best")
        if legacy is not None:
            result["prev_best"] = float(legacy)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mix_path")
    ap.add_argument("--max-age-sec", type=int, default=0, help="0 = stale check 無効")
    ap.add_argument(
        "--baseline-json",
        default="state/best_score.json",
        help="baseline winrate を読む json (default state/best_score.json)",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    mix_path = Path(args.mix_path)
    if not mix_path.is_absolute():
        mix_path = repo_root / mix_path
    baseline_path = Path(args.baseline_json)
    if not baseline_path.is_absolute():
        baseline_path = repo_root / baseline_path

    if not mix_path.exists():
        print(f"[mix_gate] BLOCK: {mix_path} が存在しない (mix-eval 未実行)", file=sys.stderr)
        return 1

    mix = json.loads(mix_path.read_text())

    # stale check
    if args.max_age_sec > 0:
        measured = mix.get("measured_at")
        if not measured:
            print("[mix_gate] BLOCK: measured_at 不在 (stale 判定不能)", file=sys.stderr)
            return 1
        try:
            ts = datetime.strptime(measured, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        except ValueError:
            print(f"[mix_gate] BLOCK: measured_at 解析不能: {measured}", file=sys.stderr)
            return 1
        age = (datetime.now(UTC) - ts).total_seconds()
        if age > args.max_age_sec:
            print(
                f"[mix_gate] BLOCK: mix-eval が古い ({int(age)}s > {args.max_age_sec}s)。"
                " 再実行が必要",
                file=sys.stderr,
            )
            return 1

    opp = mix.get("opponents", {})
    try:
        wr_random = opp["random"]["winrate"]
        wr_sniper = opp["nearest_sniper"]["winrate"]
        wr_prev = opp["prev_best"]["winrate"]
    except (KeyError, TypeError):
        print(
            "[mix_gate] BLOCK: opponents に random/nearest_sniper/prev_best が揃っていない",
            file=sys.stderr,
        )
        return 1
    winrate_min = min(wr_random, wr_sniper)  # prev_best (mirror, biased low) は除外
    errors_total = mix.get("errors_total", 1)

    base = _baselines(baseline_path)

    # prev_best (mirror) は non-blocking。collapse tripwire (< PREV_FLOOR) のみ block。
    print(
        f"  [INFO] prev_best (mirror, non-blocking) = {wr_prev} "
        f"(baseline {base['prev_best']}、改善要求なし)",
        file=sys.stderr,
    )

    checks = {
        f"winrate_random ({wr_random}) >= baseline ({base['random']})": wr_random >= base["random"],
        f"winrate_nearest_sniper ({wr_sniper}) >= baseline ({base['nearest_sniper']})": wr_sniper
        >= base["nearest_sniper"],
        f"winrate_min random/sniper ({winrate_min}) >= {WIN_MIN}": winrate_min >= WIN_MIN,
        f"prev_best ({wr_prev}) >= {PREV_FLOOR} (collapse tripwire)": wr_prev >= PREV_FLOOR,
        f"errors_total ({errors_total}) == 0": errors_total == 0,
    }

    failed = [k for k, v in checks.items() if not v]
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}", file=sys.stderr)

    if failed:
        print(f"[mix_gate] BLOCK: {len(failed)} 条件 fail (no-regression)", file=sys.stderr)
        return 1
    print("[mix_gate] PASS: 全 submit gate 条件クリア (no-regression)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
