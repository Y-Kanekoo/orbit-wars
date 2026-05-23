"""Lightweight beam search v0 + H001 territory eval.

legacy-388 (LB 388.6) からの移植 + `_score_state` に territory 項を係数 0.3 で加算。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from src.features.projection import IncomingFleet, project_ships, projected_total
from src.features.territory import territory_share
from src.strategy.geometry import angle_to, avoidance_angle, distance, fleet_speed
from src.strategy.targeting import PlanetView, pick_best_target, score_target
from src.strategy.threat import FleetView, incoming_threat

# H004d: depth 2→3 で 1 turn 余分に先読み (eval 不変)。実測 max 27ms で 1s/turn 制約に余裕。
# width は frontier 飽和で 16 超は no-op のため据置。
SEARCH_DEPTH = 3
BEAM_WIDTH = 16
ENEMY_CANDIDATES = 3
NEUTRAL_CANDIDATES = 3
FRIENDLY_CANDIDATES = 1
TIME_GUARD_RATIO = 0.8
TERRITORY_WEIGHT = (
    0.0  # H001 default off (H000 parity 修復前は無効化、territory 計算自体も `if` で短絡)
)
TERRITORY_SCALE = 100.0  # territory_share (0..1) を _score_state スケールに揃える
# H002 discard: projection_total はスカラー加算で sweep 無効 (0.3/0.1 で prev_best 完全同一)、
# production 二重カウントで over-expansion → prev_best regression。default off で再定式化 (H022) 待ち
PROJECTION_WEIGHT = 0.0
PROJECTION_HORIZON = 30  # projection の先読み turn 数


@dataclass(slots=True)
class ProjectedPlanet:
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int

    @classmethod
    def from_raw(cls, row: tuple | list) -> ProjectedPlanet:
        return cls(
            id=int(row[0]),
            owner=int(row[1]),
            x=float(row[2]),
            y=float(row[3]),
            radius=float(row[4]),
            ships=int(row[5]),
            production=int(row[6]),
        )

    def as_view(self) -> PlanetView:
        return PlanetView(
            id=self.id,
            owner=self.owner,
            x=self.x,
            y=self.y,
            radius=self.radius,
            ships=self.ships,
            production=self.production,
        )


@dataclass(slots=True)
class ProjectedFleet:
    owner: int
    x: float
    y: float
    angle: float
    ships: int
    source_planet_id: int
    target_planet_id: int
    turns_remaining: int

    def clone(self) -> ProjectedFleet:
        return ProjectedFleet(
            owner=self.owner,
            x=self.x,
            y=self.y,
            angle=self.angle,
            ships=self.ships,
            source_planet_id=self.source_planet_id,
            target_planet_id=self.target_planet_id,
            turns_remaining=self.turns_remaining,
        )

    def as_view(self, fleet_id: int) -> FleetView:
        return FleetView(
            id=fleet_id,
            owner=self.owner,
            x=self.x,
            y=self.y,
            angle=self.angle,
            from_planet_id=self.source_planet_id,
            ships=self.ships,
        )


@dataclass(slots=True, frozen=True)
class Action:
    source_id: int
    target_id: int | None
    ships: int
    angle: float = 0.0


@dataclass(slots=True)
class SearchState:
    planets: list[ProjectedPlanet]
    fleets: list[ProjectedFleet]
    step: int = 0
    root_actions: tuple[Action, ...] = field(default_factory=tuple)

    def clone(self) -> SearchState:
        return SearchState(
            planets=[
                ProjectedPlanet(
                    id=p.id,
                    owner=p.owner,
                    x=p.x,
                    y=p.y,
                    radius=p.radius,
                    ships=p.ships,
                    production=p.production,
                )
                for p in self.planets
            ],
            fleets=[fleet.clone() for fleet in self.fleets],
            step=self.step,
            root_actions=self.root_actions,
        )


def _parse_obs(obs: Any) -> tuple[int, list, list, int]:
    if isinstance(obs, dict):
        player = int(obs.get("player", 0))
        raw_planets = list(obs.get("planets", []))
        raw_fleets = list(obs.get("fleets", []))
        step = int(obs.get("step", 0))
    else:
        player = int(getattr(obs, "player", 0))
        raw_planets = list(getattr(obs, "planets", []))
        raw_fleets = list(getattr(obs, "fleets", []))
        step = int(getattr(obs, "step", 0))
    return player, raw_planets, raw_fleets, step


def _get_planet(state: SearchState, planet_id: int) -> ProjectedPlanet | None:
    for planet in state.planets:
        if planet.id == planet_id:
            return planet
    return None


def _fleet_views(state: SearchState) -> list[FleetView]:
    return [fleet.as_view(i) for i, fleet in enumerate(state.fleets)]


def _phase1_decisions(state: SearchState, player: int) -> list[Action]:
    planet_views = [planet.as_view() for planet in state.planets]
    fleet_views = _fleet_views(state)
    my_planets = [planet for planet in planet_views if planet.owner == player]
    non_mine = [planet for planet in planet_views if planet.owner != player]
    if not my_planets or not non_mine:
        return []

    decisions: list[Action] = []
    for mine in sorted(my_planets, key=lambda planet: planet.id):
        target = pick_best_target(mine, non_mine, player)
        if target is None:
            continue
        threat = incoming_threat(mine.x, mine.y, player, fleet_views, horizon_turns=20)
        reserve = threat + 1 if threat > 0 else 0
        ships_needed = int(target.ships) + 1
        available = int(mine.ships) - reserve
        if available < ships_needed:
            continue
        decisions.append(
            Action(
                source_id=int(mine.id),
                target_id=int(target.id),
                ships=int(ships_needed),
                angle=float(avoidance_angle(mine.x, mine.y, target.x, target.y, margin=1.0)),
            )
        )
    return decisions


def _actions_to_moves(actions: tuple[Action, ...] | list[Action]) -> list[list[float]]:
    moves: list[list[float]] = []
    for action in sorted(actions, key=lambda item: item.source_id):
        if action.target_id is None or action.ships <= 0:
            continue
        moves.append([int(action.source_id), float(action.angle), int(action.ships)])
    return moves


def _candidate_actions_for_planet(
    state: SearchState,
    source: ProjectedPlanet,
    player: int,
) -> list[Action]:
    fleet_views = _fleet_views(state)
    source_view = source.as_view()
    threat = incoming_threat(source.x, source.y, player, fleet_views, horizon_turns=20)
    reserve = threat + 1 if threat > 0 else 0
    available = source.ships - reserve
    wait = [Action(source_id=source.id, target_id=None, ships=0, angle=0.0)]
    if available <= 0:
        return wait

    enemies = [planet for planet in state.planets if planet.owner not in (-1, player)]
    neutrals = [planet for planet in state.planets if planet.owner == -1]
    friendlies = [
        planet for planet in state.planets if planet.owner == player and planet.id != source.id
    ]

    ranked: list[Action] = []

    def add_ranked(planets: list[ProjectedPlanet], limit: int, friendly: bool = False) -> None:
        if not planets:
            return
        scored = sorted(
            planets,
            key=lambda target: (
                -score_target(source_view, target.as_view(), player),
                target.id,
            ),
        )[:limit]
        for target in scored:
            ships = max(1, available // 2) if friendly else int(target.ships) + 1
            if ships > available:
                continue
            ranked.append(
                Action(
                    source_id=source.id,
                    target_id=target.id,
                    ships=ships,
                    angle=float(
                        avoidance_angle(source.x, source.y, target.x, target.y, margin=1.0)
                    ),
                )
            )

    add_ranked(enemies, ENEMY_CANDIDATES)
    add_ranked(neutrals, NEUTRAL_CANDIDATES)
    add_ranked(friendlies, FRIENDLY_CANDIDATES, friendly=True)

    deduped: list[Action] = []
    seen_targets: set[int | None] = set()
    for action in ranked + wait:
        if action.target_id in seen_targets:
            continue
        seen_targets.add(action.target_id)
        deduped.append(action)
    return deduped


def _apply_action(state: SearchState, action: Action, root_depth: bool = False) -> SearchState:
    next_state = state.clone()
    if root_depth:
        next_state.root_actions = state.root_actions + (action,)
    if action.target_id is None or action.ships <= 0:
        return next_state

    source = _get_planet(next_state, action.source_id)
    target = _get_planet(next_state, action.target_id)
    if source is None or target is None or source.ships < action.ships:
        return next_state

    source.ships -= action.ships
    travel_distance = distance(source.x, source.y, target.x, target.y)
    travel_speed = fleet_speed(action.ships)
    turns_remaining = max(1, math.ceil(travel_distance / max(travel_speed, 1e-6)))
    next_state.fleets.append(
        ProjectedFleet(
            owner=source.owner,
            x=source.x,
            y=source.y,
            angle=angle_to(source.x, source.y, target.x, target.y),
            ships=action.ships,
            source_planet_id=source.id,
            target_planet_id=target.id,
            turns_remaining=turns_remaining,
        )
    )
    return next_state


def _apply_actions(
    state: SearchState,
    actions: tuple[Action, ...] | list[Action],
    root_depth: bool = False,
) -> SearchState:
    next_state = state
    for action in actions:
        next_state = _apply_action(next_state, action, root_depth=root_depth)
    return next_state


def _advance_one_turn(state: SearchState) -> SearchState:
    next_state = state.clone()
    for planet in next_state.planets:
        if planet.owner >= 0:
            planet.ships += planet.production

    arrivals: dict[int, dict[int, int]] = {}
    survivors: list[ProjectedFleet] = []
    for fleet in next_state.fleets:
        speed = fleet_speed(fleet.ships)
        fleet.x += math.cos(fleet.angle) * speed
        fleet.y += math.sin(fleet.angle) * speed
        fleet.turns_remaining -= 1
        if fleet.turns_remaining <= 0:
            arrivals.setdefault(fleet.target_planet_id, {})
            arrivals[fleet.target_planet_id][fleet.owner] = (
                arrivals[fleet.target_planet_id].get(fleet.owner, 0) + fleet.ships
            )
        else:
            survivors.append(fleet)

    next_state.fleets = survivors
    for target_id, arriving in arrivals.items():
        planet = _get_planet(next_state, target_id)
        if planet is None:
            continue
        if planet.ships > 0:
            arriving[planet.owner] = arriving.get(planet.owner, 0) + planet.ships
        ranked = sorted(arriving.items(), key=lambda item: (-item[1], item[0]))
        if len(ranked) == 1:
            winner, ships = ranked[0]
            planet.owner = winner
            planet.ships = ships
            continue
        if ranked[0][1] == ranked[1][1]:
            continue
        winner, top_ships = ranked[0]
        second_ships = ranked[1][1]
        planet.owner = winner
        planet.ships = max(0, top_ships - second_ships)

    next_state.step += 1
    return next_state


def _territory_term(state: SearchState, player: int) -> float:
    """H001: 自軍 vs 敵軍 (中立除く) で territory share の差を取り、スケール変換。"""
    own_xy: list[tuple[float, float]] = []
    other_xy: list[tuple[float, float]] = []
    for p in state.planets:
        if p.owner == player:
            own_xy.append((p.x, p.y))
        elif p.owner >= 0:
            other_xy.append((p.x, p.y))
    if not own_xy or not other_xy:
        return 0.0
    own_share = territory_share(own_xy, other_xy)
    other_share = territory_share(other_xy, own_xy)
    return (own_share - other_share) * TERRITORY_SCALE


def _projection_term(state: SearchState, player: int) -> float:
    """H002: 30 turn 先の自軍 ship 在庫 projection 合計。

    production 積み + incoming fleet 加減算 (target が own planet のもののみ反映、
    raw obs 由来の target=-1 fleet は projection 側で自然に skip される)。
    """
    own_planets = [(p.id, p.production) for p in state.planets if p.owner == player]
    if not own_planets:
        return 0.0
    incoming = [
        IncomingFleet(
            target_planet_id=f.target_planet_id,
            arrival_turn=f.turns_remaining,
            owner=f.owner,
            ships=f.ships,
        )
        for f in state.fleets
    ]
    proj = project_ships(own_planets, incoming, player, n_turns=PROJECTION_HORIZON)
    return float(projected_total(proj))


def _score_state(state: SearchState, player: int) -> float:
    fleet_views = _fleet_views(state)
    planet_views = [planet.as_view() for planet in state.planets]
    targets = [planet for planet in planet_views if planet.owner != player]

    total = 0.0
    for planet in state.planets:
        sign = 1.0 if planet.owner == player else -1.0 if planet.owner >= 0 else -0.35
        total += sign * (planet.production * 8.0 + planet.ships * 1.2)
        if planet.owner == player:
            if targets:
                roi = max(score_target(planet.as_view(), target, player) for target in targets)
                total += max(0.0, roi)
            threat = incoming_threat(planet.x, planet.y, player, fleet_views, horizon_turns=20)
            total -= float(threat) * 1.5

    for fleet in state.fleets:
        if fleet.owner == player:
            total += fleet.ships * 0.6
        elif fleet.owner >= 0:
            total -= fleet.ships * 0.6

    if TERRITORY_WEIGHT > 0.0:
        total += TERRITORY_WEIGHT * _territory_term(state, player)
    if PROJECTION_WEIGHT > 0.0:
        total += PROJECTION_WEIGHT * _projection_term(state, player)
    return total


def _state_signature(state: SearchState) -> tuple:
    planets_sig = tuple((planet.id, planet.owner, planet.ships) for planet in state.planets)
    fleets_sig = tuple(
        sorted(
            (
                fleet.owner,
                fleet.source_planet_id,
                fleet.target_planet_id,
                fleet.ships,
                fleet.turns_remaining,
            )
            for fleet in state.fleets
        )
    )
    return planets_sig, fleets_sig, state.step


def _dedup(states: list[SearchState], player: int) -> list[SearchState]:
    best: dict[tuple, tuple[float, SearchState]] = {}
    for state in states:
        key = _state_signature(state)
        score = _score_state(state, player)
        prev = best.get(key)
        if prev is None or score > prev[0]:
            best[key] = (score, state)
    return [item[1] for item in best.values()]


def _prune(states: list[SearchState], player: int) -> list[SearchState]:
    ranked = sorted(
        states,
        key=lambda state: (
            -_score_state(state, player),
            len(state.fleets),
            len(state.root_actions),
            _state_signature(state),
        ),
    )
    return ranked[:BEAM_WIDTH]


def _expand_turn(
    state: SearchState,
    player: int,
    root_depth: bool,
    started_at: float,
    time_budget_sec: float,
) -> list[SearchState] | None:
    frontier = [state]
    my_planets = sorted(
        [planet for planet in state.planets if planet.owner == player],
        key=lambda planet: planet.id,
    )
    for source in my_planets:
        if time.perf_counter() - started_at > time_budget_sec:
            return None
        expanded: list[SearchState] = []
        for candidate in frontier:
            live_source = _get_planet(candidate, source.id)
            if live_source is None or live_source.owner != player:
                expanded.append(candidate)
                continue
            for action in _candidate_actions_for_planet(candidate, live_source, player):
                expanded.append(_apply_action(candidate, action, root_depth=root_depth))
        frontier = _prune(_dedup(expanded, player), player)
    return frontier


def _simulate_opponents(state: SearchState, player: int) -> SearchState:
    next_state = state
    opponents = sorted(
        {planet.owner for planet in state.planets if planet.owner not in (-1, player)}
    )
    for opponent in opponents:
        for action in _phase1_decisions(next_state, opponent):
            next_state = _apply_action(next_state, action, root_depth=False)
    return next_state


def _rollout_phase1_baseline(
    initial: SearchState,
    player: int,
    depth: int,
) -> SearchState:
    state = initial
    first_actions: tuple[Action, ...] = tuple(_phase1_decisions(state, player))
    state = _apply_actions(state, first_actions, root_depth=True)
    state = _advance_one_turn(_simulate_opponents(state, player))
    for _ in range(max(0, depth - 1)):
        state = _apply_actions(state, _phase1_decisions(state, player), root_depth=False)
        state = _advance_one_turn(_simulate_opponents(state, player))
    if not state.root_actions:
        state.root_actions = first_actions
    return state


def search(obs: Any, player: int, time_budget_sec: float = 0.3) -> list[list[float]]:
    """beam search による 1 ターン分の moves。"""
    parsed_player, raw_planets, raw_fleets, step = _parse_obs(obs)
    if player != parsed_player:
        player = parsed_player

    initial = SearchState(
        planets=[ProjectedPlanet.from_raw(planet) for planet in raw_planets],
        fleets=[
            ProjectedFleet(
                owner=int(fleet[1]),
                x=float(fleet[2]),
                y=float(fleet[3]),
                angle=float(fleet[4]),
                ships=int(fleet[6]),
                source_planet_id=int(fleet[5]),
                target_planet_id=-1,
                turns_remaining=99,
            )
            for fleet in raw_fleets
        ],
        step=step,
    )

    fallback_actions = tuple(_phase1_decisions(initial, player))
    fallback_moves = _actions_to_moves(fallback_actions)
    if SEARCH_DEPTH <= 1 or time_budget_sec <= 0.0:
        return fallback_moves

    started_at = time.perf_counter()
    beam = [initial]
    best_frontier = [_rollout_phase1_baseline(initial, player, SEARCH_DEPTH)]

    for depth in range(SEARCH_DEPTH):
        if time.perf_counter() - started_at > time_budget_sec:
            return fallback_moves
        expanded_states: list[SearchState] = []
        for state in beam:
            expanded = _expand_turn(
                state,
                player,
                root_depth=depth == 0,
                started_at=started_at,
                time_budget_sec=time_budget_sec,
            )
            if expanded is None:
                return fallback_moves
            for candidate in expanded:
                projected = _advance_one_turn(_simulate_opponents(candidate, player))
                expanded_states.append(projected)
        if not expanded_states:
            break
        beam = _prune(_dedup(expanded_states, player), player)
        best_frontier = beam
        if time.perf_counter() - started_at > time_budget_sec * TIME_GUARD_RATIO:
            break

    if not best_frontier:
        return fallback_moves
    best_state = max(best_frontier, key=lambda state: _score_state(state, player))
    if not best_state.root_actions:
        return fallback_moves
    best_moves = _actions_to_moves(best_state.root_actions)
    return best_moves or fallback_moves
