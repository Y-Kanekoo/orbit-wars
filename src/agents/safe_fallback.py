"""Safe fallback agent (PLAN.md A-1): must complete < 50ms.

Strategy: Nearest-Planet Sniper.
  各自軍惑星から最近の非自軍惑星に garrison+1 を送る。ships 不足なら見送り。
"""

from __future__ import annotations

import math
from typing import Any

from src.utils import action as action_util
from src.utils import observation as obs_util


def act(obs: Any) -> list[list[Any]]:
    """1 turn の moves を返す。`[[from_planet_id, angle, ships], ...]`。"""
    o = obs_util.parse(obs)
    my = obs_util.own_planets(o)
    targets = [p for p in o.planets if p.owner != o.player]
    if not my or not targets:
        return []

    moves: list[list[Any]] = []
    for mine in my:
        nearest = min(
            targets,
            key=lambda t: (mine.x - t.x) ** 2 + (mine.y - t.y) ** 2,
        )
        need = nearest.ships + 1
        if mine.ships >= need:
            angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
            moves.append([mine.id, angle, need])

    return action_util.sanitize(moves, obs)
