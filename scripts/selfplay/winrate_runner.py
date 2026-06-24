"""Resumable 逐次 winrate runner (`nn_in_mcts_leaf_local_eval_infeasible_in_harness` 解消)。

tournament.py は ProcessPool で N 並列 + 全完了まで結果を親が保持してから集計するため、
NN ロード subprocess を多数並列起動して OOM/reap を起こし、途中で親が死ぬと完了済ゲームの
結果も全ロストする (exp/050: 4 approach 全滅)。本 runner はその throughput/安定性問題を
エンジニアリングで解消する:

- match.py を **逐次** (--workers 既定 1) 起動 = 同時に NN を載せる subprocess は 1 つだけ
  → 8-worker 並列の OOM 圧を排除。
- 1 ゲーム完了ごとに結果を JSONL へ **追記** = 親が途中で死んでも完了分は永続化。
- 再起動時は JSONL の game_id を読んで skip = **resumable** (forward progress 保証)。

NN-in-MCTS は env-gate (`ORBIT_WARS_MCTS=1 ORBIT_WARS_NN_VALUE=1 ORBIT_WARS_NN_VALUE_MODEL=...`)
で起動する。subprocess.run は親 os.environ を継承するため、本 runner を NN env 付きで起動
すれば match.py → main.py → mcts.py まで env が伝播する (tournament.py と同方式)。

CLI:
    ORBIT_WARS_MCTS=1 ORBIT_WARS_NN_VALUE=1 \\
    ORBIT_WARS_NN_VALUE_MODEL=experiments/checkpoints/value_net_c5.int8.onnx \\
    python scripts/selfplay/winrate_runner.py \\
        --agent main.py --opponents nearest_sniper,prev_best \\
        --n-per-opponent 20 \\
        --jsonl state/nn_mcts_games.jsonl \\
        --out state/nn_mcts_winrate.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# tournament.py と同一の相手マッピング (skew 回避のため定義を一致させる)。
OPPONENT_MAP = {
    "random": "random",
    "nearest_sniper": "docs/competition/competition-starter-main.py",
    "prev_best": "docs/competition/legacy-388/main.py",
}


def _build_jobs(agent: str, opponents: list[str], n_per_opp: int, root: str) -> list[dict]:
    """seat split (agent=p1 半分 + agent=p2 半分) で全ゲーム job を生成。

    各 job は安定 game_id を持つ (resume の key)。
    seat="A": agent1=agent (agent が p1) → agent 勝ち = winner==0
    seat="B": agent1=opponent (agent が p2) → agent 勝ち = winner==1
    """
    jobs: list[dict] = []
    for name in opponents:
        opp_path = OPPONENT_MAP[name]
        half_a = n_per_opp // 2
        half_b = n_per_opp - half_a
        for seed in range(half_a):
            jobs.append(
                {
                    "game_id": f"{name}:A:{seed}",
                    "opponent": name,
                    "seat": "A",
                    "seed": seed,
                    "agent1": agent,
                    "agent2": opp_path,
                    "agent_winner_idx": 0,
                }
            )
        for seed in range(half_b):
            jobs.append(
                {
                    "game_id": f"{name}:B:{seed}",
                    "opponent": name,
                    "seat": "B",
                    "seed": seed,
                    "agent1": opp_path,
                    "agent2": agent,
                    "agent_winner_idx": 1,
                }
            )
    return jobs


def _load_done(jsonl_path: Path) -> set[str]:
    """既に JSONL に記録済の game_id を返す (resume 用、error 行も skip し forward progress)。"""
    done: set[str] = set()
    if not jsonl_path.exists():
        return done
    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        gid = rec.get("game_id")
        if gid:
            done.add(gid)
    return done


def _run_one(job: dict, root: str) -> dict:
    """match.py を 1 ゲーム subprocess 起動 (NN env は os.environ 継承)。"""
    cmd = [
        sys.executable,
        str(Path(root) / "scripts/selfplay/match.py"),
        "--agent1",
        job["agent1"],
        "--agent2",
        job["agent2"],
        "--seed",
        str(job["seed"]),
    ]
    rec = {
        "game_id": job["game_id"],
        "opponent": job["opponent"],
        "seat": job["seat"],
        "seed": job["seed"],
    }
    try:
        # timeout は NN-in-MCTS の per-game 実測 (~214s, exp/050) に余裕を持たせる。
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if proc.returncode != 0:
            rec["error"] = (proc.stderr or "non-zero exit")[:500]
            return rec
        match_res = json.loads(proc.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        rec["error"] = "timeout 900s"
        return rec
    except (json.JSONDecodeError, IndexError) as e:
        rec["error"] = f"parse: {e}"
        return rec

    if "error" in match_res:
        rec["error"] = str(match_res["error"])[:500]
        return rec
    if match_res.get("crash_suspect"):
        rec["error"] = "crash_suspect"
        return rec

    winner = match_res.get("winner")
    rec["winner"] = winner
    rec["agent_won"] = winner == job["agent_winner_idx"]
    rec["steps"] = match_res.get("steps")
    rec["elapsed_sec"] = match_res.get("elapsed_sec")
    return rec


def _aggregate(jsonl_path: Path, opponents: list[str], n_per_opp: int) -> dict:
    """JSONL を読んで相手別 winrate を集計。"""
    per_opp: dict[str, dict] = {
        name: {"opponent": name, "n_target": n_per_opp, "valid": 0, "wins": 0, "errors": 0}
        for name in opponents
    }
    if jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = rec.get("opponent")
            if name not in per_opp:
                continue
            agg = per_opp[name]
            if "error" in rec:
                agg["errors"] += 1
                continue
            agg["valid"] += 1
            if rec.get("agent_won"):
                agg["wins"] += 1

    errors_total = 0
    for agg in per_opp.values():
        agg["winrate"] = round(agg["wins"] / agg["valid"], 4) if agg["valid"] > 0 else None
        agg["losses"] = agg["valid"] - agg["wins"]
        errors_total += agg["errors"]

    winrates = [v["winrate"] for v in per_opp.values() if v["winrate"] is not None]
    return {
        "mode": "nn_mcts_winrate_resumable",
        "opponents": per_opp,
        "winrate_min": round(min(winrates), 4) if winrates else None,
        "errors_total": errors_total,
        "nn_env": {
            "ORBIT_WARS_MCTS": os.environ.get("ORBIT_WARS_MCTS", ""),
            "ORBIT_WARS_NN_VALUE": os.environ.get("ORBIT_WARS_NN_VALUE", ""),
            "ORBIT_WARS_NN_VALUE_MODEL": os.environ.get("ORBIT_WARS_NN_VALUE_MODEL", ""),
            "ORBIT_WARS_NN_POLICY": os.environ.get("ORBIT_WARS_NN_POLICY", ""),
            # H017 dual-head eval の model lineage 追跡用 (どの policy model がこの winrate を
            # 出したかを summary だけで辿れるようにする。欠落すると model 取り違えで誤判定)。
            "ORBIT_WARS_NN_POLICY_MODEL": os.environ.get("ORBIT_WARS_NN_POLICY_MODEL", ""),
        },
        "measured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="main.py", help="評価対象 agent (NN は env-gate で起動)")
    ap.add_argument(
        "--opponents",
        default="nearest_sniper,prev_best",
        help="カンマ区切り: random,nearest_sniper,prev_best",
    )
    ap.add_argument("--n-per-opponent", type=int, default=20)
    ap.add_argument(
        "--jsonl",
        default="state/nn_mcts_games.jsonl",
        help="ゲーム結果の追記 JSONL (resume の source)",
    )
    ap.add_argument("--out", default="state/nn_mcts_winrate.json", help="集計 summary 出力 path")
    ap.add_argument(
        "--max-games",
        type=int,
        default=0,
        help="本 invocation で実行する最大ゲーム数 (0=全 pending)。time-bounded 部分実行用",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="同時 subprocess 数 (既定 1 = 逐次、OOM-safe)。>1 は OOM リスク",
    )
    args = ap.parse_args()

    root = str(Path(__file__).resolve().parents[2])
    opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]
    unknown = [o for o in opponents if o not in OPPONENT_MAP]
    if unknown:
        print(json.dumps({"error": f"unknown opponents: {unknown}"}), file=sys.stderr)
        return 2

    jsonl_path = Path(root) / args.jsonl
    out_path = Path(root) / args.out
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    jobs = _build_jobs(args.agent, opponents, args.n_per_opponent, root)
    done = _load_done(jsonl_path)
    pending = [j for j in jobs if j["game_id"] not in done]
    if args.max_games > 0:
        pending = pending[: args.max_games]

    print(
        json.dumps(
            {
                "info": "winrate_runner start",
                "total_jobs": len(jobs),
                "already_done": len(done),
                "pending_this_run": len(pending),
                "workers": args.workers,
            }
        ),
        file=sys.stderr,
    )

    # 逐次実行 + 1 ゲームごとに JSONL 追記 (途中死亡しても永続化)。
    if args.workers <= 1:
        for job in pending:
            t0 = time.monotonic()
            rec = _run_one(job, root)
            rec["wall_sec"] = round(time.monotonic() - t0, 1)
            with jsonl_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(
                json.dumps(
                    {
                        "done": rec["game_id"],
                        "agent_won": rec.get("agent_won"),
                        "error": rec.get("error"),
                        "wall_sec": rec["wall_sec"],
                    }
                ),
                file=sys.stderr,
            )
    else:
        # >1 worker は OOM リスク (exp/050 教訓) だが明示 opt-in なら許可。
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_run_one, job, root): job for job in pending}
            for fut in as_completed(futs):
                rec = fut.result()
                with jsonl_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")

    summary = _aggregate(jsonl_path, opponents, args.n_per_opponent)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["errors_total"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
