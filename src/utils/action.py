"""Action sanitizer (PLAN.md A-3): 不正 action を除外し合計派兵 ≤ garrison。"""

from __future__ import annotations

import math
from typing import Any


def sanitize(moves: list[list[Any]], obs: Any) -> list[list[Any]]:
    """全 agent return 直前で必ず通すこと。

    - 自軍以外の惑星からの発射を除外
    - ships <= 0 を除外
    - 同一惑星からの累積 ships が garrison を超えないよう動的に切り詰め
    - angle を有限 float に
    """
    if not moves:
        return []

    if isinstance(obs, dict):
        player = int(obs.get("player", 0))
        raw_planets = obs.get("planets", []) or []
    else:
        player = int(getattr(obs, "player", 0))
        raw_planets = getattr(obs, "planets", []) or []

    own: dict[int, int] = {}
    for p in raw_planets:
        if not p or len(p) < 7:
            continue
        if int(p[1]) == player:
            own[int(p[0])] = int(p[5])

    spent: dict[int, int] = {}
    valid: list[list[Any]] = []

    for mv in moves:
        if not mv or len(mv) < 3:
            continue
        try:
            from_id = int(mv[0])
            angle = float(mv[1])
            ships = int(mv[2])
        except (TypeError, ValueError):
            continue

        if from_id not in own:
            continue
        if ships <= 0:
            continue
        if not math.isfinite(angle):
            continue

        remaining = own[from_id] - spent.get(from_id, 0)
        if remaining <= 0:
            continue

        ships = min(ships, remaining)
        if ships < 1:
            continue

        spent[from_id] = spent.get(from_id, 0) + ships
        valid.append([from_id, angle, ships])

    return valid
