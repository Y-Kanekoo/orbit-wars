"""目標惑星の選定ロジック（ROI・生産性・距離の重み付け）。

Phase2以降で main.py から利用する。現状は stub 的実装で、
Nearest Sniper と同等の選定を書き直しつつ、拡張ポイントを示す。
"""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import distance, fleet_speed


@dataclass(slots=True, frozen=True)
class PlanetView:
    """observation の生配列を扱いやすい形に包む。"""

    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int

    @classmethod
    def from_raw(cls, row: tuple) -> "PlanetView":
        return cls(
            id=int(row[0]),
            owner=int(row[1]),
            x=float(row[2]),
            y=float(row[3]),
            radius=float(row[4]),
            ships=int(row[5]),
            production=int(row[6]),
        )


def score_target(
    source: PlanetView,
    target: PlanetView,
    my_player: int,
    w_distance: float = 1.0,
    w_production: float = 2.0,
    w_garrison: float = 1.5,
) -> float:
    """`source` から `target` を取るROIスコア（高いほど優先）。

    - 距離が近いほど到達早くて良い
    - production が高いほど取得価値が高い
    - garrisonが大きいほどコスト高（ペナルティ）
    """
    d = distance(source.x, source.y, target.x, target.y)
    # travel_turns 推定：大きな艦隊なら速いので、garrison+1 の速度を仮定
    ships_sim = max(1, target.ships + 1)
    v = fleet_speed(ships_sim)
    turns = d / max(v, 1e-6)
    garrison_cost = target.ships if target.owner != my_player else 0
    value = target.production * w_production
    return value - w_distance * turns - w_garrison * garrison_cost


def pick_best_target(
    source: PlanetView, candidates: list[PlanetView], my_player: int
) -> PlanetView | None:
    """最もROIの高い候補を返す。候補ゼロなら None。"""
    if not candidates:
        return None
    return max(candidates, key=lambda t: score_target(source, t, my_player))
