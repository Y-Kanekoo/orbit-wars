"""Orbit Wars observation parser (B-9 defensive parsing)."""

from __future__ import annotations

from typing import Any, NamedTuple


class Planet(NamedTuple):
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int


class Fleet(NamedTuple):
    id: int
    owner: int
    x: float
    y: float
    angle: float
    from_planet_id: int
    ships: int


class Observation(NamedTuple):
    player: int
    planets: list[Planet]
    fleets: list[Fleet]
    angular_velocity: float
    initial_planets: list[Planet]
    comet_planet_ids: list[int]
    remaining_overage_time: float
    raw: Any  # 元 obs を保持 (researcher / debug 用途)


def _get(obs: Any, key: str, default: Any = None) -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def parse(obs: Any) -> Observation:
    player = int(_get(obs, "player", 0))
    raw_planets = _get(obs, "planets", []) or []
    raw_fleets = _get(obs, "fleets", []) or []
    angular_velocity = float(_get(obs, "angular_velocity", 0.0) or 0.0)
    raw_initial = _get(obs, "initial_planets", raw_planets) or []
    comet_ids = _get(obs, "comet_planet_ids", []) or []
    remaining = float(_get(obs, "remainingOverageTime", 0.0) or 0.0)

    planets: list[Planet] = []
    for p in raw_planets:
        if not p or len(p) < 7:
            continue
        planets.append(
            Planet(
                id=int(p[0]),
                owner=int(p[1]),
                x=float(p[2]),
                y=float(p[3]),
                radius=float(p[4]),
                ships=int(p[5]),
                production=int(p[6]),
            )
        )

    fleets: list[Fleet] = []
    for f in raw_fleets:
        if not f or len(f) < 7:
            continue
        fleets.append(
            Fleet(
                id=int(f[0]),
                owner=int(f[1]),
                x=float(f[2]),
                y=float(f[3]),
                angle=float(f[4]),
                from_planet_id=int(f[5]),
                ships=int(f[6]),
            )
        )

    initial_planets: list[Planet] = []
    for p in raw_initial:
        if not p or len(p) < 7:
            continue
        initial_planets.append(
            Planet(
                id=int(p[0]),
                owner=int(p[1]),
                x=float(p[2]),
                y=float(p[3]),
                radius=float(p[4]),
                ships=int(p[5]),
                production=int(p[6]),
            )
        )

    return Observation(
        player=player,
        planets=planets,
        fleets=fleets,
        angular_velocity=angular_velocity,
        initial_planets=initial_planets,
        comet_planet_ids=[int(c) for c in comet_ids],
        remaining_overage_time=remaining,
        raw=obs,
    )


def own_planets(o: Observation) -> list[Planet]:
    return [p for p in o.planets if p.owner == o.player]


def enemy_planets(o: Observation) -> list[Planet]:
    return [p for p in o.planets if p.owner not in (o.player, -1)]


def neutral_planets(o: Observation) -> list[Planet]:
    return [p for p in o.planets if p.owner == -1]
