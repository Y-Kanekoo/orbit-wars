"""H014 (exp/053) MCTS transposition table premise probe。

目的: transposition table (Zobrist hash で同一 state 再訪コストを排除) が現 MCTS
regime で有効か、**実装前**に premise を機構的に検証する。learned_rules
`mcts_progressive_widening_inert_small_branching` の指示「PW/RAVE 系 (H014/H015) も
同型 inert 可能性、着手前に分岐数 probe で premise を確認する」に従う (exp/051 PW 先例)。

TT が rollout/sec を上げるには search 中に **同一 state が異なる経路で複数回出現**
(= transposition) する必要がある。現 MCTS (src/search/mcts.py) は:
  - root + その children のみの **depth-1 木** (children は孫に展開されない)
  - 各 child = root からの固有 action set 適用後 state
このため木内に transposition は構造的に発生しえない。本 probe はこれを実 obs 上で
定量確認する:
  1. branching: root child 数
  2. within-tree transposition: child state を canonical hash し distinct/total を比較
     (collision = transposition opportunity)
  3. tree-depth: child が孫を持たない (depth-1) ことを確認 → 異経路再訪が原理的に不能
  4. rollout determinism: phase1 rollout は決定論的か (同一 child の再 rollout が同値か)
     = TT でなく「決定論 rollout の memoize」が唯一の冗長性かを切り分け

CLI:
    python scripts/selfplay/probe_mcts_transposition.py --steps 30,80,150,220 --seed 3
"""

from __future__ import annotations

import argparse
import json
import sys

from src.search.beam import (
    ProjectedFleet,
    ProjectedPlanet,
    SearchState,
    _advance_one_turn,
    _expand_turn,
    _parse_obs,
    _simulate_opponents,
)
from src.search.mcts import _rollout_value


def _state_hash(state: SearchState) -> int:
    """SearchState の canonical hash (Zobrist 等価)。

    transposition 判定用に planets/fleets を順序非依存・座標量子化して tuple 化。
    TT は「同一 state」を識別する必要があるので、TT が拾える粒度と同等に正規化する。
    """
    planets = tuple(
        sorted(
            (p.id, p.owner, round(p.x, 3), round(p.y, 3), p.ships, p.production)
            for p in state.planets
        )
    )
    fleets = tuple(
        sorted(
            (
                f.owner,
                round(f.x, 3),
                round(f.y, 3),
                round(f.angle, 3),
                f.ships,
                f.source_planet_id,
            )
            for f in state.fleets
        )
    )
    return hash((planets, fleets, state.step))


def _build_root_children(obs, player: int, time_budget_sec: float = 0.3):
    """mcts.search() の root child 構築を再現 (探索ループ手前まで)。"""
    parsed_player, raw_planets, raw_fleets, step = _parse_obs(obs)
    player = parsed_player
    initial = SearchState(
        planets=[ProjectedPlanet.from_raw(p) for p in raw_planets],
        fleets=[
            ProjectedFleet(
                owner=int(f[1]),
                x=float(f[2]),
                y=float(f[3]),
                angle=float(f[4]),
                ships=int(f[6]),
                source_planet_id=int(f[5]),
                target_planet_id=-1,
                turns_remaining=99,
            )
            for f in raw_fleets
        ],
        step=step,
    )
    import time as _time

    started_at = _time.perf_counter()
    expanded = _expand_turn(
        initial,
        player,
        root_depth=True,
        started_at=started_at,
        time_budget_sec=time_budget_sec,
    )
    if not expanded:
        return player, initial, []
    children = []
    for child_state in expanded:
        child_after = _advance_one_turn(_simulate_opponents(child_state, player))
        child_after.root_actions = child_state.root_actions
        children.append(child_after)
    return player, initial, children


def _collect_observations(seed: int, steps: list[int]) -> list:
    """実 game を random agents で進め、指定 step の player0 観測を集める。"""
    from kaggle_environments import make

    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run(["random", "random"])
    obs_list = []
    max_step = len(env.steps) - 1
    for s in steps:
        if s > max_step:
            continue
        obs = env.steps[s][0].observation
        obs_list.append((s, obs))
    return obs_list


def probe(seed: int, steps: list[int]) -> dict:
    observations = _collect_observations(seed, steps)
    results = []
    for step, obs in observations:
        player, _initial, children = _build_root_children(obs, 0)
        n_children = len(children)
        hashes = [_state_hash(c) for c in children]
        distinct = len(set(hashes))
        # within-tree transposition: child state が重複する数 (TT が拾える対象)
        collisions = n_children - distinct

        # rollout determinism (phase1): 同一 child を 2 回 rollout し値一致を確認。
        # 一致 = 再 rollout は新情報ゼロ = TT でなく memoize の問題 (切り分け)。
        rollout_det = None
        if children:
            c0 = children[0]
            v1 = _rollout_value(c0.clone() if hasattr(c0, "clone") else c0, player)
            v2 = _rollout_value(c0.clone() if hasattr(c0, "clone") else c0, player)
            rollout_det = abs(v1 - v2) < 1e-12

        results.append(
            {
                "step": step,
                "player": player,
                "n_children": n_children,
                "distinct_child_states": distinct,
                "within_tree_transpositions": collisions,
                "phase1_rollout_deterministic": rollout_det,
            }
        )

    total_children = sum(r["n_children"] for r in results)
    total_transpositions = sum(r["within_tree_transpositions"] for r in results)
    return {
        "seed": seed,
        "tree_depth": 1,  # mcts.py: root.children は孫に展開されない (構造的事実)
        "per_obs": results,
        "summary": {
            "obs_probed": len(results),
            "total_root_children": total_children,
            "total_within_tree_transpositions": total_transpositions,
            "tt_hit_opportunity": total_transpositions > 0,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument(
        "--steps",
        type=str,
        default="30,80,150,220",
        help="probe 対象 step (カンマ区切り)",
    )
    args = ap.parse_args()
    steps = [int(x) for x in args.steps.split(",") if x.strip()]
    report = probe(args.seed, steps)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
