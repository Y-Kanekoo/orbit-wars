"""並列 self-play tournament (PLAN.md L229, Q6 並列化方式)。

ProcessPoolExecutor で N 試合並列、winrate / 平均 turn / timeout 違反数を集計。

CLI:
    python scripts/selfplay/tournament.py --agent1 main.py --agent2 random --n 30
    python scripts/selfplay/tournament.py --agent1 A --agent2 B --n 100 --seeds 0..99
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _run_one(args_tuple: tuple) -> dict:
    """worker: subprocess で match.py を起動 (kaggle_env のグローバル state を分離)。"""
    agent1, agent2, seed, root = args_tuple
    cmd = [
        sys.executable,
        str(Path(root) / "scripts/selfplay/match.py"),
        "--agent1",
        agent1,
        "--agent2",
        agent2,
        "--seed",
        str(seed),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            return {"seed": seed, "error": proc.stderr[:500] or "non-zero exit"}
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return {"seed": seed, "error": "timeout 600s"}
    except json.JSONDecodeError as e:
        return {"seed": seed, "error": f"json parse: {e}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent1", required=True)
    ap.add_argument("--agent2", required=True)
    ap.add_argument("--n", type=int, default=30, help="seeds 未指定時の試合数 (seed 0..n-1)")
    ap.add_argument("--seeds", default="", help="0..99 形式 or 0,1,2,3 形式")
    ap.add_argument("--max-workers", type=int, default=0, help="0 = cpu_count // 2")
    args = ap.parse_args()

    if args.seeds:
        if ".." in args.seeds:
            a, b = args.seeds.split("..")
            seeds = list(range(int(a), int(b) + 1))
        else:
            seeds = [int(x) for x in args.seeds.split(",")]
    else:
        seeds = list(range(args.n))

    root = Path(__file__).resolve().parents[2]
    workers = args.max_workers if args.max_workers > 0 else max(1, (os.cpu_count() or 2) // 2)

    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_run_one, (args.agent1, args.agent2, s, str(root))) for s in seeds]
        for fut in as_completed(futs):
            results.append(fut.result())

    # 集計
    wins_p1 = sum(1 for r in results if r.get("winner") == 0)
    wins_p2 = sum(1 for r in results if r.get("winner") == 1)
    errors = sum(1 for r in results if "error" in r)
    valid = len(results) - errors
    winrate = wins_p1 / valid if valid > 0 else 0.0

    summary = {
        "agent1": args.agent1,
        "agent2": args.agent2,
        "n_games": len(results),
        "valid_games": valid,
        "wins_p1": wins_p1,
        "wins_p2": wins_p2,
        "errors": errors,
        "winrate_p1": round(winrate, 4),
        "seeds": seeds,
        "workers": workers,
    }
    print(json.dumps(summary, indent=2))
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
