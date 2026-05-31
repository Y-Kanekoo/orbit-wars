"""並列 self-play tournament (PLAN.md L229, Q6 並列化方式)。

ProcessPoolExecutor で N 試合並列、winrate / 平均 turn / timeout 違反数を集計。

2 モード:
1. 単一対戦 (従来): --agent1 X --agent2 Y --n 30
2. mix-eval (H021): --agent main.py --opponents random,nearest_sniper,prev_best
   各相手 N 試合を seat split (agent=p1 半分 + agent=p2 半分) で測定し、
   相手別 winrate / winrate_min / errors_total を集計。submit gate の入力。

CLI:
    python scripts/selfplay/tournament.py --agent1 main.py --agent2 random --n 30
    python scripts/selfplay/tournament.py --agent main.py \
        --opponents random,nearest_sniper,prev_best --n-per-opponent 30 \
        --out state/last_mix_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

# mix-eval の相手名 → agent file path / 組み込み名 のマッピング
OPPONENT_MAP = {
    "random": "random",
    "nearest_sniper": "docs/competition/competition-starter-main.py",
    "prev_best": "docs/competition/legacy-388/main.py",
}


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


def _parse_seeds(seeds_arg: str, n: int) -> list[int]:
    if seeds_arg:
        if ".." in seeds_arg:
            a, b = seeds_arg.split("..")
            return list(range(int(a), int(b) + 1))
        return [int(x) for x in seeds_arg.split(",")]
    return list(range(n))


def _run_batch(jobs: list[tuple], workers: int) -> list[dict]:
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_run_one, job) for job in jobs]
        for fut in as_completed(futs):
            results.append(fut.result())
    return results


def _eval_vs_opponent(agent: str, opponent: str, n: int, root: str, workers: int) -> dict:
    """agent vs opponent を seat split (agent=p1 半分 + agent=p2 半分) で測定。

    Returns: {opponent, n, wins, losses, errors, winrate, crash_suspects}
    """
    half_p1 = n // 2
    half_p2 = n - half_p1

    # seat A: agent=p1 (agent1), opponent=p2 (agent2) → agent 勝ち = winner==0
    jobs_a = [(agent, opponent, s, root) for s in range(half_p1)]
    # seat B: opponent=p1 (agent1), agent=p2 (agent2) → agent 勝ち = winner==1
    jobs_b = [(opponent, agent, s, root) for s in range(half_p2)]

    res_a = _run_batch(jobs_a, workers)
    res_b = _run_batch(jobs_b, workers)

    wins = 0
    errors = 0
    crash_suspects = 0
    valid = 0
    for r in res_a:
        if "error" in r:
            errors += 1
            continue
        if r.get("crash_suspect"):
            crash_suspects += 1
            errors += 1
            continue
        valid += 1
        if r.get("winner") == 0:
            wins += 1
    for r in res_b:
        if "error" in r:
            errors += 1
            continue
        if r.get("crash_suspect"):
            crash_suspects += 1
            errors += 1
            continue
        valid += 1
        if r.get("winner") == 1:
            wins += 1

    winrate = wins / valid if valid > 0 else 0.0
    return {
        "opponent": opponent,
        "n": n,
        "valid_games": valid,
        "wins": wins,
        "losses": valid - wins,
        "errors": errors,
        "crash_suspects": crash_suspects,
        "winrate": round(winrate, 4),
    }


def _n_for_opponent(name: str, args) -> int:
    """相手別の試合数を返す。

    gate の唯一の decision 軸 prev_best (legacy-388 mirror) は同一コードでも N=30 で
    ±7 試合振れ (`mixeval_prevbest_gate_noisy_misjudges_lb`)、単一測定で gate を両方向に
    誤判定する。一方 random (0.9667 天井) / nearest_sniper (0.6) は識別力が低く N=30 で足りる。
    --prev-best-n>0 のとき prev_best のみ高 N で測り、安価な相手は --n-per-opponent 据置。
    既定 (--prev-best-n=0) は従来どおり全相手一律 (compat 不変)。
    """
    if name == "prev_best" and args.prev_best_n > 0:
        return args.prev_best_n
    return args.n_per_opponent


def _run_mix_eval(args, root: str, workers: int) -> int:
    opp_names = [o.strip() for o in args.opponents.split(",") if o.strip()]
    unknown = [o for o in opp_names if o not in OPPONENT_MAP]
    if unknown:
        print(
            json.dumps({"error": f"unknown opponents: {unknown}", "known": list(OPPONENT_MAP)}),
            file=sys.stderr,
        )
        return 2

    per_opp: dict[str, dict] = {}
    errors_total = 0
    for name in opp_names:
        opp_path = OPPONENT_MAP[name]
        res = _eval_vs_opponent(args.agent, opp_path, _n_for_opponent(name, args), root, workers)
        per_opp[name] = res
        errors_total += res["errors"]

    winrate_min = min((v["winrate"] for v in per_opp.values()), default=0.0)

    summary = {
        "mode": "mix_eval",
        "agent": args.agent,
        "n_per_opponent": args.n_per_opponent,
        "prev_best_n": args.prev_best_n if args.prev_best_n > 0 else args.n_per_opponent,
        "opponents": per_opp,
        "winrate_min": round(winrate_min, 4),
        "errors_total": errors_total,
        "measured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "workers": workers,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.out:
        out_path = Path(root) / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    return 0 if errors_total == 0 else 1


def _run_single(args, root: str, workers: int) -> int:
    seeds = _parse_seeds(args.seeds, args.n)
    jobs = [(args.agent1, args.agent2, s, root) for s in seeds]
    results = _run_batch(jobs, workers)

    wins_p1 = sum(1 for r in results if r.get("winner") == 0)
    wins_p2 = sum(1 for r in results if r.get("winner") == 1)
    errors = sum(1 for r in results if "error" in r)
    crash_suspects = sum(1 for r in results if r.get("crash_suspect"))
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
        "crash_suspects": crash_suspects,
        "winrate_p1": round(winrate, 4),
        "seeds": seeds,
        "workers": workers,
    }
    print(json.dumps(summary, indent=2))
    return 0 if errors == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    # 単一対戦モード
    ap.add_argument("--agent1")
    ap.add_argument("--agent2")
    ap.add_argument("--n", type=int, default=30, help="seeds 未指定時の試合数 (seed 0..n-1)")
    ap.add_argument("--seeds", default="", help="0..99 形式 or 0,1,2,3 形式")
    # mix-eval モード
    ap.add_argument("--agent", help="mix-eval 評価対象 agent (--opponents と併用)")
    ap.add_argument(
        "--opponents",
        default="",
        help="mix-eval 相手 (カンマ区切り): random,nearest_sniper,prev_best",
    )
    ap.add_argument("--n-per-opponent", type=int, default=30)
    ap.add_argument(
        "--prev-best-n",
        type=int,
        default=0,
        help="prev_best (gate decision 軸) のみ高 N で測る override。0 = --n-per-opponent と同一",
    )
    ap.add_argument(
        "--out", default="", help="mix-eval 結果の出力 path (例: state/last_mix_eval.json)"
    )
    ap.add_argument("--max-workers", type=int, default=0, help="0 = cpu_count // 2")
    args = ap.parse_args()

    root = str(Path(__file__).resolve().parents[2])
    workers = args.max_workers if args.max_workers > 0 else max(1, (os.cpu_count() or 2) // 2)

    if args.opponents:
        if not args.agent:
            print(json.dumps({"error": "--opponents requires --agent"}), file=sys.stderr)
            return 2
        return _run_mix_eval(args, root, workers)

    if not args.agent1 or not args.agent2:
        print(
            json.dumps({"error": "single mode requires --agent1 and --agent2"}),
            file=sys.stderr,
        )
        return 2
    return _run_single(args, root, workers)


if __name__ == "__main__":
    sys.exit(main())
