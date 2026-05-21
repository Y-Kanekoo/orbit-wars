"""mix-eval 結果に対する submit gate 判定 (H021)。

使い方:
    python scripts/kaggle/check_mix_gate.py state/last_mix_eval.json [--max-age-sec N]

gate 条件 (全て満たすと exit 0、 1 つでも欠けると exit 1):
    winrate_random        >= 0.90
    winrate_nearest_sniper>= 0.60
    winrate_prev_best     >= (prev_baseline + 0.05)
    winrate_min           >= 0.50
    errors_total          == 0

prev_baseline は state/best_score.json の現 prev_best winrate (mix_eval セクション優先、
無ければ legacy フィールド local_winrate_vs_prev_best、 それも無ければ 0.50)。

stale check: --max-age-sec を指定すると、 mix_eval の measured_at が古すぎる場合 block。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

WIN_RANDOM_MIN = 0.90
WIN_SNIPER_MIN = 0.60
WIN_PREV_DELTA = 0.05
WIN_MIN = 0.50


def _prev_baseline(repo_root: Path) -> float:
    best_path = repo_root / "state/best_score.json"
    if not best_path.exists():
        return 0.50
    best = json.loads(best_path.read_text())
    mix = best.get("mix_eval")
    if isinstance(mix, dict):
        opp = mix.get("opponents", {}).get("prev_best", {})
        if isinstance(opp, dict) and "winrate" in opp:
            return float(opp["winrate"])
    legacy = best.get("local_winrate_vs_prev_best")
    if legacy is not None:
        return float(legacy)
    return 0.50


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mix_path")
    ap.add_argument("--max-age-sec", type=int, default=0, help="0 = stale check 無効")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    mix_path = Path(args.mix_path)
    if not mix_path.is_absolute():
        mix_path = repo_root / mix_path

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
    winrate_min = mix.get("winrate_min", 0.0)
    errors_total = mix.get("errors_total", 1)

    prev = _prev_baseline(repo_root)
    prev_gate = prev + WIN_PREV_DELTA

    checks = {
        f"winrate_random ({wr_random}) >= {WIN_RANDOM_MIN}": wr_random >= WIN_RANDOM_MIN,
        f"winrate_nearest_sniper ({wr_sniper}) >= {WIN_SNIPER_MIN}": wr_sniper >= WIN_SNIPER_MIN,
        f"winrate_prev_best ({wr_prev}) >= {prev_gate:.4f} (prev {prev}+{WIN_PREV_DELTA})": wr_prev
        >= prev_gate,
        f"winrate_min ({winrate_min}) >= {WIN_MIN}": winrate_min >= WIN_MIN,
        f"errors_total ({errors_total}) == 0": errors_total == 0,
    }

    failed = [k for k, v in checks.items() if not v]
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}", file=sys.stderr)

    if failed:
        print(f"[mix_gate] BLOCK: {len(failed)} 条件 fail", file=sys.stderr)
        return 1
    print("[mix_gate] PASS: 全 submit gate 条件クリア", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
