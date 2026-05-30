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
import os
import random
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
    _candidate_actions_for_planet,
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

# H026 (exp/030) 診断: rollout policy を切替可能化 (default=phase1 で main parity 不変)。
# ORBIT_WARS_MCTS=1 ROLLOUT_POLICY=uniform で exp028 (phase1 rollout) と A/B 比較し、
# strong-opponent regression が phase1 rollout policy 由来かを切り分ける。
ROLLOUT_POLICY = os.environ.get("ROLLOUT_POLICY", "phase1")
# rollout 用 RNG (seed 固定で決定論的、連続 draw で Monte Carlo 多様性を確保)。
_ROLLOUT_RNG = random.Random(0xC0FFEE)

# H016 step 6: leaf 評価を NN value head に差し替える env flag (default OFF = 提出 parity)。
# ORBIT_WARS_NN_VALUE=1 + ORBIT_WARS_NN_VALUE_MODEL=<onnx path> で有効。実モデルは
# Kaggle GPU 学習後に差す (local export は ~74s/game で非現実、supervisor escalate)。
_VALUE_MODEL = None  # lazy cache (ValueModel | None)
_VALUE_MODEL_RESOLVED = False


def reset_value_model() -> None:
    """env を再読込させるため value model cache をクリアする (smoke / test 用)。"""
    global _VALUE_MODEL, _VALUE_MODEL_RESOLVED
    _VALUE_MODEL = None
    _VALUE_MODEL_RESOLVED = False


def _get_value_model():
    """env flag に従い NN value model を解決 (なければ None = rollout 経路)。"""
    global _VALUE_MODEL, _VALUE_MODEL_RESOLVED
    if _VALUE_MODEL_RESOLVED:
        return _VALUE_MODEL
    _VALUE_MODEL_RESOLVED = True
    if os.environ.get("ORBIT_WARS_NN_VALUE") != "1":
        _VALUE_MODEL = None
        return None
    model_path = os.environ.get("ORBIT_WARS_NN_VALUE_MODEL", "")
    if not model_path or not os.path.exists(model_path):
        _VALUE_MODEL = None  # path 未指定/不在は安全に rollout fallback
        return None
    from src.nn.value_infer import ValueModel

    _VALUE_MODEL = ValueModel(model_path)
    return _VALUE_MODEL


def _nn_leaf_value(state: SearchState, player: int, model, ctx) -> float:
    """leaf state を NN value head で評価し [0,1] win-prob に写像する。

    NN value は player 視点 (-1, 1) (+1=勝勢)。MCTS backup は rollout 経路と同一の
    [0,1] スケールを期待する (UCB1 exploit / 最終 argmax) ため (v+1)/2 で線形写像。
    """
    from src.nn.value_infer import encode_search_state

    enc = encode_search_state(state.planets, state.fleets, player, ctx)
    v = model.evaluate(enc)  # (-1, 1)
    return (v + 1.0) / 2.0


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


def _uniform_decisions(state: SearchState, player: int) -> list:
    """uniform rollout policy: 各自軍 planet の候補 action から一様ランダムに 1 つ選ぶ。

    phase1 (greedy heuristic) の代替。`_candidate_actions_for_planet` は wait (do nothing)
    を常に含むため、launch しない選択肢も等確率で評価される。
    """
    actions = []
    for planet in state.planets:
        if planet.owner != player:
            continue
        candidates = _candidate_actions_for_planet(state, planet, player)
        if candidates:
            actions.append(_ROLLOUT_RNG.choice(candidates))
    return actions


def _rollout_value(state: SearchState, player: int) -> float:
    policy = _uniform_decisions if ROLLOUT_POLICY == "uniform" else _phase1_decisions
    sim_state = state
    for _ in range(ROLLOUT_DEPTH):
        actions = policy(sim_state, player)
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

    # H016 step 6: NN value head が有効なら leaf 評価を rollout から差し替える。
    # ctx (comet/angular_velocity/overage) は root obs から 1 度だけ parse して thread。
    value_model = _get_value_model()
    nn_ctx = None
    if value_model is not None:
        from src.nn.value_infer import context_from_observation
        from src.utils.observation import parse

        nn_ctx = context_from_observation(parse(obs))

    deadline = started_at + time_budget_sec * TIME_GUARD_RATIO
    while time.perf_counter() < deadline:
        chosen = max(root.children, key=lambda c: _ucb1(c, root.visits))
        if value_model is not None:
            value = _nn_leaf_value(chosen.state, player, value_model, nn_ctx)
        else:
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
