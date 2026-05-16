"""Orbit Wars 提出エージェント — Phase 1 + Phase 2 beam search v0."""

from __future__ import annotations

import os
from typing import Any

try:
    # package import (pytest / `from agent.main import agent`)
    from .search.beam import search as beam_search
    from .strategy.geometry import avoidance_angle
    from .strategy.targeting import PlanetView, pick_best_target
    from .strategy.threat import FleetView, incoming_threat
except (ImportError, KeyError):
    # kaggle_environments は agent/main.py を __name__ 無しの空 globals で exec するため、
    # relative import が KeyError("'__name__' not in globals") を投げる。ImportError ではない。
    from search.beam import search as beam_search  # type: ignore[no-redef]
    from strategy.geometry import avoidance_angle  # type: ignore[no-redef]
    from strategy.targeting import PlanetView, pick_best_target  # type: ignore[no-redef]
    from strategy.threat import FleetView, incoming_threat  # type: ignore[no-redef]


BEAM_TIME_BUDGET_SEC_DEFAULT = 0.3


def _beam_enabled() -> bool:
    # submission 時はデフォルト有効。テスト高速化用に環境変数で切り替え可能。
    return os.getenv("ORBIT_WARS_DISABLE_BEAM", "0") != "1"


def _beam_time_budget_sec() -> float:
    # 環境変数で上書き可能（テストで小さく、本番で既定値）。
    raw = os.getenv("ORBIT_WARS_BEAM_TIME_BUDGET_SEC")
    if raw is None:
        return BEAM_TIME_BUDGET_SEC_DEFAULT
    try:
        return float(raw)
    except (TypeError, ValueError):
        return BEAM_TIME_BUDGET_SEC_DEFAULT


def _parse_obs(obs: Any) -> tuple[int, list, list]:
    if isinstance(obs, dict):
        player = obs.get("player", 0)
        raw_planets = obs.get("planets", [])
        raw_fleets = obs.get("fleets", [])
    else:
        player = getattr(obs, "player", 0)
        raw_planets = getattr(obs, "planets", [])
        raw_fleets = getattr(obs, "fleets", [])
    return player, list(raw_planets), list(raw_fleets)


def _phase1_moves_from_parsed(
    player: int, raw_planets: list, raw_fleets: list
) -> list[list[float]]:
    planets = [PlanetView.from_raw(p) for p in raw_planets]
    fleets = [FleetView.from_raw(f) for f in raw_fleets]
    my_planets = [p for p in planets if p.owner == player]
    non_mine = [p for p in planets if p.owner != player]
    if not my_planets or not non_mine:
        return []

    moves: list[list[float]] = []
    for mine in my_planets:
        target = pick_best_target(mine, non_mine, player)
        if target is None:
            continue
        threat = incoming_threat(mine.x, mine.y, player, fleets, horizon_turns=20)
        reserve = threat + 1 if threat > 0 else 0
        ships_needed = int(target.ships) + 1
        available = int(mine.ships) - reserve
        if available < ships_needed:
            continue
        angle = avoidance_angle(mine.x, mine.y, target.x, target.y, margin=1.0)
        moves.append([int(mine.id), float(angle), int(ships_needed)])

    return moves


def phase1_moves(obs: Any) -> list[list[float]]:
    """Phase 1 の決定論的 baseline moves を返す。"""
    player, raw_planets, raw_fleets = _parse_obs(obs)
    return _phase1_moves_from_parsed(player, raw_planets, raw_fleets)


def agent(obs: Any) -> list[list[float]]:
    """1 ターン分の moves を返す。

    戻り値: [[from_planet_id, direction_angle, num_ships], ...]
    """
    player, raw_planets, raw_fleets = _parse_obs(obs)
    fallback = _phase1_moves_from_parsed(player, raw_planets, raw_fleets)

    if not _beam_enabled():
        return fallback

    try:
        return beam_search(obs, player, time_budget_sec=_beam_time_budget_sec())
    except Exception:
        return fallback
