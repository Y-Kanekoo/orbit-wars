"""脅威評価: 敵 fleet が horizon_turns 以内に到達する ships 合計を見積る。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.strategy.geometry import distance, fleet_speed


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
    def from_raw(cls, row: tuple) -> FleetView:
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
    total = 0
    for f in enemy_fleets:
        if f.owner == my_player:
            continue
        if turns_to_reach(f, defender_x, defender_y) <= horizon_turns:
            total += f.ships
    return total


def incoming_threat_eta(
    defender_x: float,
    defender_y: float,
    my_player: int,
    enemy_fleets: list[FleetView],
    horizon_turns: int = 20,
) -> float:
    """方向考慮 threat (H003)。

    `incoming_threat` は直線距離のみで判定するため、planet から離れていく敵 fleet も
    脅威に数える欠陥がある。本関数は fleet の進行方向 (angle) と planet 方向の cos を取り、
    向かってくる成分のみを脅威として ships に重み付けする。

    - align = cos(heading_error): 進行方向が planet を向くほど 1 に近い。<=0 (離反) は除外。
    - eta = (距離 × align) / speed: 進行方向に沿った最接近までの turn 数。horizon 超過は除外。
    - 戻り値 = Σ ships × align (向かってくる敵 fleet のみ)。

    NOTE (discard): exp/003 で eval に weight 2.5 で統合 (`incoming_threat * 1.5` 置換) したが、
    mix-eval gate で強い相手に regression し discard (ledger exp/003 参照)。本関数は eval 未統合の
    休眠 infra として残置。再定式化 (別 weight / baseline incoming_threat との併用) は H007 grid
    search 対象。
    """
    total = 0.0
    for f in enemy_fleets:
        if f.owner == my_player or f.owner < 0:
            continue
        dx = defender_x - f.x
        dy = defender_y - f.y
        d = math.hypot(dx, dy)
        if d < 1e-6:
            total += float(f.ships)
            continue
        align = (math.cos(f.angle) * dx + math.sin(f.angle) * dy) / d
        if align <= 0.0:
            continue
        eta = (d * align) / max(fleet_speed(f.ships), 1e-6)
        if eta > horizon_turns:
            continue
        total += float(f.ships) * align
    return total
