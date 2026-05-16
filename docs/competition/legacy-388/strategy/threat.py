"""脅威評価: 敵fleetの軌道と自軍惑星の防衛必要量を推定する。

Phase2以降で main.py から利用する。現状はインターフェース定義と、
最小限のベースライン実装（最近接敵fleetが何ターンで到達するか）を置く。
"""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import distance, fleet_speed


@dataclass(slots=True, frozen=True)
class FleetView:
    id: int
    owner: int
    x: float
    y: float
    angle: float
    from_planet_id: int
    ships: int

    @classmethod
    def from_raw(cls, row: tuple) -> "FleetView":
        return cls(
            id=int(row[0]),
            owner=int(row[1]),
            x=float(row[2]),
            y=float(row[3]),
            angle=float(row[4]),
            from_planet_id=int(row[5]),
            ships=int(row[6]),
        )


def turns_to_reach(fleet: FleetView, target_x: float, target_y: float) -> float:
    """fleet が (target_x, target_y) に到達するまでの概算ターン数。

    実際には fleet は angle方向に直進するが、簡易化して距離/速度で見積もる。
    """
    d = distance(fleet.x, fleet.y, target_x, target_y)
    v = fleet_speed(fleet.ships)
    return d / max(v, 1e-6)


def incoming_threat(
    defender_x: float,
    defender_y: float,
    my_player: int,
    enemy_fleets: list[FleetView],
    horizon_turns: int = 20,
) -> int:
    """horizon_turns以内に防衛側に突っ込んでくる敵shipsの合計（粗い上限）。"""
    total = 0
    for f in enemy_fleets:
        if f.owner == my_player:
            continue
        if turns_to_reach(f, defender_x, defender_y) <= horizon_turns:
            total += f.ships
    return total
