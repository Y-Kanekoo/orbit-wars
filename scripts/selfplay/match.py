"""1 試合の subprocess 実行 (PLAN.md L228)。

CLI:
    python scripts/selfplay/match.py --agent1 main.py --agent2 random --seed 0

stdout: JSON `{seed, rewards, steps, winner, p1_max_act_ms, p2_max_act_ms}`
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def run_match(agent1: str, agent2: str, seed: int) -> dict:
    """1 試合実行して結果 dict を返す。"""
    try:
        from kaggle_environments import make
    except ImportError as e:
        return {"error": f"kaggle-environments missing: {e}"}

    env = make(
        "orbit_wars",
        configuration={"seed": seed},
        debug=False,
    )
    t0 = time.monotonic()
    try:
        env.run([agent1, agent2])
    except Exception as e:  # noqa: BLE001
        return {"error": f"env.run failed: {e}", "seed": seed}
    elapsed = time.monotonic() - t0

    final = env.steps[-1]
    rewards = [float(s.reward) if s.reward is not None else 0.0 for s in final]
    winner = int(max(range(len(rewards)), key=lambda i: rewards[i])) if rewards else -1
    steps = len(env.steps)

    # no-op crash 検出 (PLAN.md exp 001/002 教訓):
    # agent が NameError 等で一切呼ばれない場合、対戦が即終了 (steps<50) し
    # rewards が全 0 になる。これを error 扱いにして mix-eval gate で弾く。
    crash_suspect = steps < 50 and all(r == 0.0 for r in rewards)

    return {
        "seed": seed,
        "agent1": agent1,
        "agent2": agent2,
        "rewards": rewards,
        "steps": steps,
        "winner": winner,
        "crash_suspect": crash_suspect,
        "elapsed_sec": round(elapsed, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent1", required=True)
    ap.add_argument("--agent2", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # 相対 path は repo root 起点
    root = Path(__file__).resolve().parents[2]
    a1 = args.agent1 if args.agent1 in {"random", "reaction"} else str(root / args.agent1)
    a2 = args.agent2 if args.agent2 in {"random", "reaction"} else str(root / args.agent2)

    result = run_match(a1, a2, args.seed)
    print(json.dumps(result))
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    sys.exit(main())
