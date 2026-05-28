"""H011 minimal PUCT-MCTS (UCB1 prior=uniform、root-level 木 + phase1 rollout)。

設計方針:
- beam の `SearchState` / `_phase1_decisions` / `_score_state` / `_apply_actions` /
  `_advance_one_turn` / `_simulate_opponents` / `_expand_turn` を完全再利用
- root の child 集合 = beam 1 ply 展開 (`_expand_turn` で得る frontier)。
  各 child は 1 turn 分の root_actions を保持
- 各 simulation: 残予算内で 1) UCB1 で child 選択 2) rollout=phase1 で depth=R turn 進める
  3) `_score_state` を sigmoid で [0,1] win-prob 推定 4) backup
- 終了時 root の child を visit-count argmax で選択
- 木の深さは 1 (root のみ branch)。 探索パラダイム置換の効果検証が目的のため、
  深い tree 展開は H012/H014 等の後続改善で追加
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from src.search.beam import (
    ProjectedFleet,
    ProjectedPlanet,
    SearchState,
    _actions_to_moves,
    _advance_one_turn,
    _apply_actions,
    _expand_turn,
    _parse_obs,
    _phase1_decisions,
    _score_state,
    _simulate_opponents,
)

ROLLOUT_DEPTH = 2
UCB_C = 1.4
SIGMOID_SCALE = 100.0
TIME_GUARD_RATIO = 0.85


@dataclass(slots=True)
class MCTSNode:
    state: SearchState
    parent: MCTSNode | None = None
    children: list[MCTSNode] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0


def _ucb1(child: MCTSNode, parent_visits: int) -> float:
    if child.visits == 0:
        return math.inf
    exploit = child.value_sum / child.visits
    explore = UCB_C * math.sqrt(math.log(max(parent_visits, 1)) / child.visits)
    return exploit + explore


def _rollout_value(state: SearchState, player: int) -> float:
    sim_state = state
    for _ in range(ROLLOUT_DEPTH):
        actions = _phase1_decisions(sim_state, player)
        sim_state = _apply_actions(sim_state, actions, root_depth=False)
        sim_state = _advance_one_turn(_simulate_opponents(sim_state, player))
    score = _score_state(sim_state, player)
    return 1.0 / (1.0 + math.exp(-score / SIGMOID_SCALE))


def search(obs: Any, player: int, time_budget_sec: float = 0.8) -> list[list[float]]:
    """MCTS による 1 ターン分の moves。budget 内で root child を visit-count 選択。"""
    parsed_player, raw_planets, raw_fleets, step = _parse_obs(obs)
    if player != parsed_player:
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

    fallback_moves = _actions_to_moves(tuple(_phase1_decisions(initial, player)))
    if time_budget_sec <= 0.0:
        return fallback_moves

    started_at = time.perf_counter()
    expanded = _expand_turn(
        initial,
        player,
        root_depth=True,
        started_at=started_at,
        time_budget_sec=time_budget_sec,
    )
    if expanded is None or not expanded:
        return fallback_moves

    root = MCTSNode(state=initial)
    for child_state in expanded:
        child_after_turn = _advance_one_turn(_simulate_opponents(child_state, player))
        child_after_turn.root_actions = child_state.root_actions
        root.children.append(MCTSNode(state=child_after_turn, parent=root))

    if not root.children:
        return fallback_moves

    deadline = started_at + time_budget_sec * TIME_GUARD_RATIO
    while time.perf_counter() < deadline:
        chosen = max(root.children, key=lambda c: _ucb1(c, root.visits))
        value = _rollout_value(chosen.state, player)
        chosen.visits += 1
        chosen.value_sum += value
        root.visits += 1

    best = max(
        root.children,
        key=lambda c: (
            c.visits,
            c.value_sum / max(c.visits, 1),
        ),
    )
    return _actions_to_moves(best.state.root_actions)
