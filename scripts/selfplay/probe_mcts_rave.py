"""H015 (exp/054) MCTS RAVE / AMAF premise probe。

目的: RAVE / AMAF (rollout で出現した move の結果を、その move が選択可能な **他の
決定ノード** にも AMAF 統計として共有し、visit 希少時の value 推定を加速する手法) が
現 MCTS regime で有効か、**実装前**に premise を機構的に検証する。learned_rules
`mcts_progressive_widening_inert_small_branching` / `mcts_transposition_table_inert_depth1_tree`
の指示「PW/RAVE 系 (H014/H015) も着手前に分岐数/機構 probe で premise を確認する」に従う
(exp/051 PW・exp/053 TT 先例)。

RAVE が plain MCTS を上回るには以下が必要:
  (P1) 木に **複数の決定ノード** が存在し、ある rollout の move 結果を他ノードにも共有
       できる (AMAF の本質 = all-moves-as-first-move を木全体に伝播)
  (P2) 各 child の **visit が希少** (UCB1 推定が未収束) で、AMAF の variance 低減が効く
  (P3) rollout で出現する move が root child の action と **重複** し AMAF credit が成立する

現 MCTS (src/search/mcts.py) は root + その children のみの **depth-1 木** (children は
孫に展開されない) = 決定ノードが root 1 個のみ → (P1) が構造的に不成立。さらに PW probe
(exp/051) で root child は 1〜6 個・0.3s budget で ~400 sims = 過剰カバーと判明済 → (P2)
も不成立 (visit 希少でなく value-tie)。本 probe はこれらを実 obs 上で定量確認する:
  1. branching: root child 数 (= 決定ノードの child 数)
  2. n_decision_nodes: 木内で children を持つノード数 (depth-1 なら root のみ = 1) → (P1)
  3. budget coverage: 実 budget で MCTS rollout loop を回し total_sims / per-child visits /
     value spread を計測 → (P2) visit 希少か・value-tie か
  4. AMAF cross-node credit: 各 child から phase1 rollout し出現 move を (source_id,target_id)
     で正規化、他 child の root_actions move と重複するか → (P3) AMAF credit の成立可否

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
    _simulate_opponents,
)
from src.search.mcts import ROLLOUT_DEPTH, _rollout_value


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


def _move_keys(actions) -> set:
    """action 列を AMAF move identity (source_id, target_id) の集合に正規化。

    RAVE は move を位置的に識別 (ships は bucket、本質は「どこからどこへ」)。launch
    しない wait (target_id=None) は move でないため除外する。
    """
    return {(a.source_id, a.target_id) for a in actions if a.target_id is not None}


def _rollout_moves(state: SearchState, player: int) -> set:
    """child から phase1 rollout し、rollout 中に出現した全 move key を集める。

    RAVE の AMAF はこの rollout move 集合を、その move を child action に持つ **他の
    決定ノード** に credit する。本 probe では「他 child の root_actions move と重複するか」で
    AMAF cross-node credit の成立可否を測る。
    """
    moves: set = set()
    sim_state = state
    for _ in range(ROLLOUT_DEPTH):
        actions = _phase1_decisions(sim_state, player)
        moves |= _move_keys(actions)
        sim_state = _apply_actions(sim_state, actions, root_depth=False)
        sim_state = _advance_one_turn(_simulate_opponents(sim_state, player))
    return moves


def _probe_obs(obs, time_budget_sec: float) -> dict:
    player, _initial, children = _build_root_children(obs, time_budget_sec)
    n_children = len(children)
    if n_children == 0:
        return {
            "player": player,
            "n_children": 0,
            "n_decision_nodes": 1,
            "note": "no children (phase1 fallback)",
        }

    # (3) budget coverage: 実 budget で rollout loop を回し visit 分布を測る。
    # mcts.search() の探索ループ (UCB1 child 選択 + rollout backup) を再現。
    visits = [0] * n_children
    value_sum = [0.0] * n_children
    total_sims = 0
    deadline = time.perf_counter() + time_budget_sec * 0.85
    while time.perf_counter() < deadline:
        # UCB1 (mcts._ucb1 と同等): 未訪問は inf 優先
        best_i, best_u = 0, -math.inf
        for i in range(n_children):
            if visits[i] == 0:
                best_i, best_u = i, math.inf
                break
            exploit = value_sum[i] / visits[i]
            explore = 1.4 * math.sqrt(math.log(max(total_sims, 1)) / visits[i])
            u = exploit + explore
            if u > best_u:
                best_i, best_u = i, u
        c = children[best_i]
        v = _rollout_value(c.clone() if hasattr(c, "clone") else c, player)
        visits[best_i] += 1
        value_sum[best_i] += v
        total_sims += 1

    means = [value_sum[i] / visits[i] if visits[i] else 0.0 for i in range(n_children)]
    value_spread = (max(means) - min(means)) if means else 0.0
    min_visits = min(visits)

    # (4) AMAF cross-node credit: child0 の rollout move が他 child の root action move と
    # 重複するか。重複ゼロ = RAVE が他ノードに credit する対象が無い。
    sample = children[0]
    rollout_mv = _rollout_moves(sample.clone() if hasattr(sample, "clone") else sample, player)
    child_action_keys = [_move_keys(c.root_actions) for c in children]
    # child0 の rollout move が「他 child の root action」と重なる数 (AMAF が共有する対象)
    other_union = set().union(*child_action_keys[1:]) if n_children > 1 else set()
    amaf_cross_overlap = len(rollout_mv & other_union)

    return {
        "player": player,
        "n_children": n_children,  # (1)
        "n_decision_nodes": 1,  # (2) depth-1 木: children を持つのは root のみ
        "total_sims": total_sims,  # (3)
        "min_child_visits": min_visits,  # (3) 最小 visit (希少なら RAVE 効く余地)
        "mean_child_visits": round(total_sims / n_children, 1),
        "value_spread": round(value_spread, 4),  # (3) ~0 = value-tie
        "rollout_move_count": len(rollout_mv),  # (4)
        "amaf_cross_node_overlap": amaf_cross_overlap,  # (4) 0 = credit 対象なし
        "other_children_move_count": len(other_union),
    }


def _collect_observations(seed: int, steps: list[int]) -> list:
    from kaggle_environments import make

    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run(["random", "random"])
    obs_list = []
    max_step = len(env.steps) - 1
    for s in steps:
        if s > max_step:
            continue
        obs_list.append((s, env.steps[s][0].observation))
    return obs_list


def probe(seed: int, steps: list[int], time_budget_sec: float) -> dict:
    observations = _collect_observations(seed, steps)
    per_obs = []
    for step, obs in observations:
        r = _probe_obs(obs, time_budget_sec)
        r["step"] = step
        per_obs.append(r)

    valid = [r for r in per_obs if r.get("n_children", 0) > 0]
    total_overlap = sum(r.get("amaf_cross_node_overlap", 0) for r in valid)
    max_spread = max((r.get("value_spread", 0.0) for r in valid), default=0.0)
    min_visits_all = min((r.get("min_child_visits", 0) for r in valid), default=0)
    max_children = max((r.get("n_children", 0) for r in valid), default=0)
    return {
        "seed": seed,
        "time_budget_sec": time_budget_sec,
        "tree_depth": 1,  # mcts.py: root.children は孫に展開されない (構造的事実)
        "per_obs": per_obs,
        "summary": {
            "obs_probed": len(valid),
            "n_decision_nodes": 1,  # 決定ノードは root のみ = RAVE の共有先ゼロ (P1 不成立)
            "max_root_children": max_children,
            "min_child_visits_observed": min_visits_all,  # 大 = visit 過剰 (P2 不成立)
            "max_value_spread": round(max_spread, 4),  # ~0 = value-tie
            "total_amaf_cross_node_overlap": total_overlap,  # 0 = AMAF credit 対象なし (P3)
            "rave_premise_holds": total_overlap > 0 and min_visits_all <= 1,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--steps", type=str, default="30,80,150,220")
    ap.add_argument("--budget", type=float, default=0.3)
    args = ap.parse_args()
    steps = [int(x) for x in args.steps.split(",") if x.strip()]
    report = probe(args.seed, steps, args.budget)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
