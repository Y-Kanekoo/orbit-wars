"""H015 (exp/056) MCTS RAVE / AMAF premise probe。

目的: RAVE/AMAF (rollout で出現した action の value 統計を木全体で共有し、visit が
少ない child の早期推定を改善する) が現 MCTS regime で有効か、**実装前**に premise を
機構的に検証する。learned_rules `mcts_progressive_widening_inert_small_branching` /
`mcts_transposition_table_inert_depth1_tree` の明示指示「H015 (RAVE/AMAF) も着手前に
機構 probe で premise (action 再出現頻度) を確認、おそらく同型 inert」に従う
(H012 exp/051 PW / H014 exp/053 TT 先例)。

RAVE が child 選択を改善する premise (両方が必要):
  P1. branching 過大 & visit 不足 — child 数に対し budget の sim 数が足りず、各 child
      の real visit が少ない時に AMAF の early estimate が効く。逆に少分岐で各 child が
      大量 visit を得るなら AMAF は real visit に即座に上書きされ inert (H012 と同型)。
  P2. action 再出現 — RAVE の AMAF table は action 単位。ある launch action `a` が
      **複数 child の rollout 軌跡に再出現**して初めて、AMAF が a の良し悪しを cross-child
      に伝播でき child ランキングを変えうる。各 child の action が固有で sibling rollout に
      再出現しないなら AMAF は cross-child signal を持たず inert (H014 TT と同型)。

本 probe は実 obs 上で以下を定量する:
  1. branching: root child 数 (= P1 の分岐側)
  2. visit saturation: budget 0.3s で得られる総 sim 数を child 数で割った平均 real visit
     (= P1 の visit 側。十分大なら AMAF の出番なし)
  3. action universe: 全 rollout 軌跡に出現した distinct launch action 数 vs 総出現数
     (= P2。distinct が出現総数に近い = 各 action が 1 回限り = AMAF 集約不能)
  4. cross-child re-occurrence: 各 root child の action set を構成する launch が、
     **他 child の rollout** に何回出現するか (= P2 の核心。0 なら AMAF cross-child 不能)
  5. value-tie: child の rollout value 分布 spread (H012 で確認の「flat = AMAF 並べ替えても
     argmax 不変」を再確認)

action の RAVE key = (source_id, target_id)。ships/angle は派生で AMAF の粒度ではない。

CLI:
    python scripts/selfplay/probe_mcts_rave.py --steps 30,80,150,220 --seed 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

from src.search.beam import (
    ProjectedFleet,
    ProjectedPlanet,
    SearchState,
    _advance_one_turn,
    _apply_actions,
    _expand_turn,
    _parse_obs,
    _phase1_decisions,
    _score_state,
    _simulate_opponents,
)
from src.search.mcts import ROLLOUT_DEPTH, SIGMOID_SCALE


def _action_key(action) -> tuple[int, int | None]:
    """RAVE AMAF key: launch を (source_id, target_id) で識別 (wait は target=None)。"""
    return (int(action.source_id), action.target_id)


def _is_launch(action) -> bool:
    return action.target_id is not None and action.ships > 0


def _build_root_children(obs, time_budget_sec: float = 0.3):
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
    started_at = time.perf_counter()
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


def _rollout_trace(state: SearchState, player: int) -> tuple[float, list]:
    """mcts._rollout_value (phase1 policy) を再現しつつ、rollout 中に取った launch
    action の key 列を記録して返す。value は同一 ([0,1] win-prob)。"""
    traced: list[tuple[int, int | None]] = []
    sim_state = state
    for _ in range(ROLLOUT_DEPTH):
        actions = _phase1_decisions(sim_state, player)
        for a in actions:
            if _is_launch(a):
                traced.append(_action_key(a))
        sim_state = _apply_actions(sim_state, actions, root_depth=False)
        sim_state = _advance_one_turn(_simulate_opponents(sim_state, player))
    score = _score_state(sim_state, player)
    value = 1.0 / (1.0 + math.exp(-score / SIGMOID_SCALE))
    return value, traced


def _budget_sim_count(children: list, player: int, time_budget_sec: float = 0.3) -> int:
    """budget 0.3s × TIME_GUARD_RATIO(0.85) 内に回る rollout 総数を実測 (P1 visit 側)。

    mcts.search の探索ループと同じ rollout コストで sim を回し、deadline までの回数を数える。
    child を round-robin に rollout (UCB1 選択コストは無視できるほど rollout 律速)。
    """
    if not children:
        return 0
    deadline = time.perf_counter() + time_budget_sec * 0.85
    count = 0
    i = 0
    while time.perf_counter() < deadline:
        _rollout_trace(children[i % len(children)], player)
        count += 1
        i += 1
    return count


def probe(seed: int, steps: list[int]) -> dict:
    from kaggle_environments import make

    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run(["random", "random"])
    max_step = len(env.steps) - 1

    results = []
    for step in steps:
        if step > max_step:
            continue
        obs = env.steps[step][0].observation
        player, _initial, children = _build_root_children(obs)
        n_children = len(children)
        if n_children == 0:
            results.append({"step": step, "player": player, "n_children": 0})
            continue

        # 各 child の root action set (launch key) と 1 回 rollout の軌跡を採取。
        child_action_keys: list[set] = []
        child_rollout_keys: list[list] = []
        child_values: list[float] = []
        for c in children:
            root_keys = {_action_key(a) for a in c.root_actions if _is_launch(a)}
            child_action_keys.append(root_keys)
            value, traced = _rollout_trace(c, player)
            child_rollout_keys.append(traced)
            child_values.append(value)

        # action universe (P2): 全 rollout 軌跡の distinct vs total。
        all_rollout_keys = [k for trace in child_rollout_keys for k in trace]
        distinct_actions = len(set(all_rollout_keys))
        total_action_occ = len(all_rollout_keys)

        # cross-child re-occurrence (P2 核心): 各 child の root action が他 child の
        # rollout 軌跡に出現する回数。AMAF が cross-child に伝播できる signal 量。
        cross_child_hits = 0
        for i, root_keys in enumerate(child_action_keys):
            for j, trace in enumerate(child_rollout_keys):
                if i == j:
                    continue
                for k in trace:
                    if k in root_keys:
                        cross_child_hits += 1

        # value-tie (H012 再確認): rollout value の spread。
        vmin, vmax = min(child_values), max(child_values)
        vspread = vmax - vmin

        # visit saturation (P1): budget 内 sim 数 / child 数 = 平均 real visit。
        sim_count = _budget_sim_count(children, player)
        avg_real_visits = sim_count / n_children

        results.append(
            {
                "step": step,
                "player": player,
                "n_children": n_children,
                "budget_sim_count": sim_count,
                "avg_real_visits_per_child": round(avg_real_visits, 1),
                "distinct_rollout_actions": distinct_actions,
                "total_rollout_action_occurrences": total_action_occ,
                "cross_child_action_reoccurrences": cross_child_hits,
                "child_value_spread": round(vspread, 4),
            }
        )

    probed = [r for r in results if r.get("n_children", 0) > 0]
    total_cross = sum(r["cross_child_action_reoccurrences"] for r in probed)
    min_avg_visits = min((r["avg_real_visits_per_child"] for r in probed), default=0.0)
    max_spread = max((r["child_value_spread"] for r in probed), default=0.0)
    return {
        "seed": seed,
        "tree_depth": 1,  # mcts.py: root.children は孫に展開されない (構造的事実)
        "per_obs": results,
        "summary": {
            "obs_probed": len(probed),
            # P1: 各 child が大量 real visit を得るなら AMAF early-estimate は inert。
            "min_avg_real_visits_per_child": min_avg_visits,
            # P2: cross-child action 再出現が 0 なら AMAF は cross-child signal 不能。
            "total_cross_child_reoccurrences": total_cross,
            "amaf_cross_child_signal": total_cross > 0,
            # H012 再確認: value spread が小なら AMAF 並べ替えても argmax 不変。
            "max_child_value_spread": max_spread,
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
