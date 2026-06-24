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
# H017 (exp/060) PUCT exploration 係数 (AlphaZero 慣習 c_puct ~1-2)。policy prior が
# 有効な時 UCB1 を PUCT に切替え、root children の value-tie (learned_rules
# mcts_progressive_widening_inert_small_branching 等) を prior で割る。
PUCT_C = 1.5
SIGMOID_SCALE = 100.0
TIME_GUARD_RATIO = 0.85

# H026 (exp/030) 診断: rollout policy を切替可能化 (default=phase1 で main parity 不変)。
# ORBIT_WARS_MCTS=1 ROLLOUT_POLICY=uniform で exp028 (phase1 rollout) と A/B 比較し、
# strong-opponent regression が phase1 rollout policy 由来かを切り分ける。
ROLLOUT_POLICY = os.environ.get("ROLLOUT_POLICY", "phase1")
# rollout 用 RNG (seed 固定で決定論的、連続 draw で Monte Carlo 多様性を確保)。
_ROLLOUT_RNG = random.Random(0xC0FFEE)


def _pw_k() -> float:
    """H012 (exp/051) progressive widening 係数。0 (default) = 無効 = exp028 parity。

    env で都度読む (test/diagnostic が monkeypatch せず env 切替で A/B できるよう module
    定数化しない)。値が >0 のとき root child の active 集合を prior 上位から段階解禁する。
    """
    try:
        return float(os.environ.get("MCTS_PW_K", "0"))
    except ValueError:
        return 0.0


def _pw_alpha() -> float:
    try:
        return float(os.environ.get("MCTS_PW_ALPHA", "0.5"))
    except ValueError:
        return 0.5


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


# H017 (exp/060): root の PUCT prior を NN policy head から供給する env flag
# (default OFF = 提出 parity)。ORBIT_WARS_NN_POLICY=1 + ORBIT_WARS_NN_POLICY_MODEL=<onnx>
# で有効。実モデルは Kaggle GPU 学習後に差す (local export 不能、supervisor escalate)。
_POLICY_MODEL = None  # lazy cache (PolicyValueModel | None)
_POLICY_MODEL_RESOLVED = False


def reset_policy_model() -> None:
    """env を再読込させるため policy model cache をクリアする (smoke / test 用)。"""
    global _POLICY_MODEL, _POLICY_MODEL_RESOLVED
    _POLICY_MODEL = None
    _POLICY_MODEL_RESOLVED = False


def _get_policy_model():
    """env flag に従い NN policy model を解決 (なければ None = UCB1 経路)。"""
    global _POLICY_MODEL, _POLICY_MODEL_RESOLVED
    if _POLICY_MODEL_RESOLVED:
        return _POLICY_MODEL
    _POLICY_MODEL_RESOLVED = True
    if os.environ.get("ORBIT_WARS_NN_POLICY") != "1":
        _POLICY_MODEL = None
        return None
    model_path = os.environ.get("ORBIT_WARS_NN_POLICY_MODEL", "")
    if not model_path or not os.path.exists(model_path):
        _POLICY_MODEL = None  # path 未指定/不在は安全に UCB1 fallback
        return None
    from src.nn.value_infer import PolicyValueModel

    _POLICY_MODEL = PolicyValueModel(model_path)
    return _POLICY_MODEL


@dataclass(slots=True)
class MCTSNode:
    state: SearchState
    parent: MCTSNode | None = None
    children: list[MCTSNode] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    prior: float = 0.0  # H017 PUCT prior (policy 有効時のみ意味を持つ)


def _ucb1(child: MCTSNode, parent_visits: int) -> float:
    if child.visits == 0:
        return math.inf
    exploit = child.value_sum / child.visits
    explore = UCB_C * math.sqrt(math.log(max(parent_visits, 1)) / child.visits)
    return exploit + explore


def _puct(child: MCTSNode, parent_visits: int, c_puct: float) -> float:
    """PUCT score = Q + c_puct * P * sqrt(ΣN) / (1 + N_child)。

    UCB1 と違い未訪問 child を inf にせず、prior P で初期探索順序を決める (AlphaZero)。
    parent_visits=0 でも prior が効くよう sqrt(max(ΣN, 1)) とする。
    """
    exploit = child.value_sum / child.visits if child.visits else 0.0
    explore = c_puct * child.prior * math.sqrt(max(parent_visits, 1)) / (1 + child.visits)
    return exploit + explore


def _assign_child_priors(
    root: MCTSNode, initial: SearchState, player: int, policy_model, nn_ctx
) -> None:
    """root state の NN policy で各 child に prior を割当てる (総和 1 に正規化)。

    policy index 規約 (train/serve skew 回避、train_value.py と 1:1):
      i ∈ [0, MAX_PLANETS) = encoder の planet slot i (= ships 降順 sort の i 番目)
      i = MAX_PLANETS = no-op (launch しない)
    各 child の prior は、その root_actions の launch 元 planet の policy prob を平均
    (launch 無し child は no-op prob)。launch 元が encoder の上位 MAX_PLANETS から
    truncate された稀ケースは no-op prob で代替する。
    """
    import numpy as np

    from src.features.encoder import MAX_PLANETS
    from src.nn.value_infer import encode_search_state

    enc = encode_search_state(initial.planets, initial.fleets, player, nn_ctx)
    _, logits = policy_model.evaluate(enc)  # (POLICY_DIM,)
    exp = np.exp(logits - logits.max())
    probs = [float(p) for p in (exp / exp.sum())]
    noop_idx = MAX_PLANETS

    # encoder と同一 sort (ships 降順, 上位 MAX_PLANETS) で planet.id -> slot を再構築
    sorted_planets = sorted(initial.planets, key=lambda p: p.ships, reverse=True)[:MAX_PLANETS]
    id_to_slot = {p.id: i for i, p in enumerate(sorted_planets)}

    raw: list[float] = []
    for child in root.children:
        sources = [
            a.source_id for a in child.state.root_actions if a.target_id is not None and a.ships > 0
        ]
        slots = [id_to_slot[s] for s in sources if s in id_to_slot]
        if slots:
            raw.append(sum(probs[i] for i in slots) / len(slots))
        else:
            raw.append(probs[noop_idx])

    total = sum(raw)
    if total <= 0.0:
        uniform = 1.0 / len(root.children)
        for child in root.children:
            child.prior = uniform
        return
    for child, r in zip(root.children, raw, strict=True):
        child.prior = r / total


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


def _search_root(
    obs: Any, player: int, time_budget_sec: float
) -> tuple[int, SearchState, MCTSNode | None]:
    """MCTS を実行し探索済 root node を返す (展開不可なら root=None)。

    `search()` (move 選択) と `search_root_visit_policy()` (policy target 抽出) で
    本体を共有するための抽出 (exp/061)。返り値は (補正済 player, initial state, root)。
    root=None は budget<=0 / 展開ゼロ / child ゼロ のいずれかで、呼び出し側は
    initial+player から phase1 fallback を生成する (従来 search() の fallback と同一)。
    """
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

    if time_budget_sec <= 0.0:
        return player, initial, None

    started_at = time.perf_counter()
    expanded = _expand_turn(
        initial,
        player,
        root_depth=True,
        started_at=started_at,
        time_budget_sec=time_budget_sec,
    )
    if expanded is None or not expanded:
        return player, initial, None

    root = MCTSNode(state=initial)
    for child_state in expanded:
        child_after_turn = _advance_one_turn(_simulate_opponents(child_state, player))
        child_after_turn.root_actions = child_state.root_actions
        root.children.append(MCTSNode(state=child_after_turn, parent=root))

    if not root.children:
        return player, initial, None

    # H016 step 6 / H017 (exp/060): NN value / policy head が有効なら leaf 評価 (value) /
    # root prior (policy) を NN に差し替える。ctx (comet/angular_velocity/overage) は
    # root obs から 1 度だけ parse して探索ホライズン中 thread する。
    value_model = _get_value_model()
    policy_model = _get_policy_model()
    nn_ctx = None
    if value_model is not None or policy_model is not None:
        from src.nn.value_infer import context_from_observation
        from src.utils.observation import parse

        nn_ctx = context_from_observation(parse(obs))

    # H017 (exp/060): policy 有効時のみ root child に PUCT prior を割当て、選択を
    # UCB1 → PUCT に切替える (value-tie を prior で割る)。policy OFF では従来 UCB1。
    use_puct = policy_model is not None
    if use_puct:
        _assign_child_priors(root, initial, player, policy_model, nn_ctx)

    # H012 (exp/051) progressive widening: prior 上位 child から段階解禁し、予算少時に
    # visit が全 child へ薄く分散する症状 (learned_rules MCTS root-only) を抑える。
    # pw_k=0 (default) では全 child を従来どおり candidate とし exp028 と完全 parity。
    pw_k = _pw_k()
    if pw_k > 0.0:
        # prior = 1 turn 進めた child state の `_score_state` (player 視点、高=有望)。
        root.children.sort(key=lambda c: _score_state(c.state, player), reverse=True)
    pw_alpha = _pw_alpha()

    deadline = started_at + time_budget_sec * TIME_GUARD_RATIO
    while time.perf_counter() < deadline:
        if pw_k > 0.0:
            allowed = max(1, math.ceil(pw_k * (root.visits**pw_alpha)))
            candidates = root.children[:allowed]
        else:
            candidates = root.children
        if use_puct:
            chosen = max(candidates, key=lambda c: _puct(c, root.visits, PUCT_C))
        else:
            chosen = max(candidates, key=lambda c: _ucb1(c, root.visits))
        if value_model is not None:
            value = _nn_leaf_value(chosen.state, player, value_model, nn_ctx)
        else:
            value = _rollout_value(chosen.state, player)
        chosen.visits += 1
        chosen.value_sum += value
        root.visits += 1

    return player, initial, root


def search(obs: Any, player: int, time_budget_sec: float = 0.8) -> list[list[float]]:
    """MCTS による 1 ターン分の moves。budget 内で root child を visit-count 選択。"""
    player, initial, root = _search_root(obs, player, time_budget_sec)
    if root is None:
        return _actions_to_moves(tuple(_phase1_decisions(initial, player)))

    best = max(
        root.children,
        key=lambda c: (
            c.visits,
            c.value_sum / max(c.visits, 1),
        ),
    )
    return _actions_to_moves(best.state.root_actions)


def search_root_visit_policy(obs: Any, player: int, time_budget_sec: float = 0.3) -> list[float]:
    """AlphaZero policy target: MCTS root の visit 分布を policy index 規約に集約・正規化。

    返り値は len POLICY_DIM (= MAX_PLANETS + 1) の確率分布 (行和 1):
      i ∈ [0, MAX_PLANETS) = encoder の planet slot i (= ships 降順 sort の i 番目) を
        launch 元とする child の visit 合計
      i = MAX_PLANETS = no-op (launch なし child) の visit 合計
    index 規約は `_assign_child_priors` (PUCT prior) と 1:1 = train_value.py の
    `policy_target` 並びと一致 (train/serve skew 回避)。`export_value_data.py --policy`
    が各 timestep の教師分布として記録する。

    1 child が複数 planet から launch する compound action は visit を launch 元数で
    均等配分する (`_assign_child_priors` の prob 平均と対称)。encoder 上位 MAX_PLANETS
    から truncate された launch 元しか持たない child は no-op slot に寄せる。
    探索不可 (展開ゼロ) や全 visit ゼロは no-op one-hot を返す。
    """
    from src.features.encoder import MAX_PLANETS

    policy_dim = MAX_PLANETS + 1
    noop_idx = MAX_PLANETS
    target = [0.0] * policy_dim

    player, initial, root = _search_root(obs, player, time_budget_sec)
    if root is None or not root.children:
        target[noop_idx] = 1.0
        return target

    # encoder と同一 sort (ships 降順, 上位 MAX_PLANETS) で planet.id -> slot を再構築
    sorted_planets = sorted(initial.planets, key=lambda p: p.ships, reverse=True)[:MAX_PLANETS]
    id_to_slot = {p.id: i for i, p in enumerate(sorted_planets)}

    for child in root.children:
        if child.visits <= 0:
            continue
        sources = [
            a.source_id for a in child.state.root_actions if a.target_id is not None and a.ships > 0
        ]
        slots = [id_to_slot[s] for s in sources if s in id_to_slot]
        if slots:
            share = child.visits / len(slots)
            for i in slots:
                target[i] += share
        else:
            target[noop_idx] += child.visits

    total = sum(target)
    if total <= 0.0:
        target[noop_idx] = 1.0
        return target
    return [t / total for t in target]


def debug_root_priors(obs: Any, player: int, time_budget_sec: float = 0.3) -> list[float]:
    """root child の PUCT prior 一覧を返す (policy 配線契約の smoke/diagnostic 用)。

    探索は回さず root 展開 + prior 割当のみ実施。policy model 未設定 (env OFF) なら []。
    """
    policy_model = _get_policy_model()
    if policy_model is None:
        return []

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
    started_at = time.perf_counter()
    expanded = _expand_turn(
        initial, player, root_depth=True, started_at=started_at, time_budget_sec=time_budget_sec
    )
    if not expanded:
        return []
    root = MCTSNode(state=initial)
    for child_state in expanded:
        child_after_turn = _advance_one_turn(_simulate_opponents(child_state, player))
        child_after_turn.root_actions = child_state.root_actions
        root.children.append(MCTSNode(state=child_after_turn, parent=root))
    if not root.children:
        return []

    from src.nn.value_infer import context_from_observation
    from src.utils.observation import parse

    nn_ctx = context_from_observation(parse(obs))
    _assign_child_priors(root, initial, player, policy_model, nn_ctx)
    return [c.prior for c in root.children]
