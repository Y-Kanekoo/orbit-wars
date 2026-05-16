"""Production projection: N turn 先までの ship 在庫を予測 (PLAN.md L254, H002)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import NamedTuple


class IncomingFleet(NamedTuple):
    """projection 入力で使う incoming fleet 情報。"""

    target_planet_id: int
    arrival_turn: int  # 0 = この turn 着、1 = 次 turn 着、...
    owner: int
    ships: int


def project_ships(
    own_planets: Iterable[tuple[int, int]],
    incoming: Iterable[IncomingFleet],
    own_player: int,
    n_turns: int = 30,
) -> dict[int, int]:
    """N turn 先の自軍 planet ごとの ships 在庫を予測。

    Parameters
    ----------
    own_planets : iterable of (planet_id, production_per_turn)
        現時点での自軍 planet と毎 turn 生産量。
    incoming : iterable of IncomingFleet
        将来 turn に到達する fleet (自軍 + 敵軍混在)。
    own_player : int
        自分の player ID。
    n_turns : int
        予測 horizon。default 30。

    Returns
    -------
    dict[planet_id, projected_ships]
        各 own planet の n_turns 後の予測 ships。
        負の値 = 占領されている可能性、0 にクリップ。

    Notes
    -----
    - 簡易モデル: production を直線的に積み、incoming arriving 時に減算/加算。
    - 戦闘判定は単純化 (敵 fleet を含む incoming は全部 ship 数を相殺)。
    - 占領後の所有権変化は **無視** (n_turns 内の浅い予測なので近似で十分)。
    """
    inventory: dict[int, int] = {}
    production_map: dict[int, int] = {}
    for planet_id, prod in own_planets:
        inventory[planet_id] = 0  # 起点は 0 ships (deltaを返す)
        production_map[planet_id] = max(0, int(prod))

    # turn 別に incoming を整理
    arrivals: dict[int, list[IncomingFleet]] = {}
    for fl in incoming:
        if fl.arrival_turn < 0 or fl.arrival_turn > n_turns:
            continue
        arrivals.setdefault(fl.arrival_turn, []).append(fl)

    for t in range(n_turns):
        # 1. arrival 処理 (turn 開始時)
        for fl in arrivals.get(t, []):
            if fl.target_planet_id not in inventory:
                continue
            if fl.owner == own_player:
                inventory[fl.target_planet_id] += fl.ships
            else:
                inventory[fl.target_planet_id] -= fl.ships

        # 2. production
        for pid, prod in production_map.items():
            inventory[pid] += prod

    return {pid: max(0, ships) for pid, ships in inventory.items()}


def projected_total(projection: dict[int, int]) -> int:
    """projection 辞書から合計 ships を返す (eval_fn で使う集約)。"""
    return sum(projection.values())
