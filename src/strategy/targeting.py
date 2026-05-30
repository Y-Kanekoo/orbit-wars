"""目標惑星の選定ロジック (ROI スコア = production - 距離コスト - garrison)。"""

from __future__ import annotations

from dataclasses import dataclass

from src.strategy.geometry import distance, fleet_speed

# H006 (exp/039): comet target bias。comet は spawn 時に neutral planet として
# 可視化されるが現状 eval で完全に無視されている (transient だが安価に capture でき
# 敵への denial 価値もある)。score_target に comet 選択 bias を加える (production への
# additive scalar でなく target 選択軸 = eval_term_redundant_with_production 回避)。
COMET_TARGET_BONUS = 6.0


@dataclass(slots=True, frozen=True)
class PlanetView:
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int
    is_comet: bool = False

    @classmethod
    def from_raw(cls, row: tuple) -> PlanetView:
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
    d = distance(source.x, source.y, target.x, target.y)
    ships_sim = max(1, target.ships + 1)
    v = fleet_speed(ships_sim)
    turns = d / max(v, 1e-6)
    garrison_cost = target.ships if target.owner != my_player else 0
    value = target.production * w_production
    # comet は自軍以外かつ on-board (off-board sentinel x=-99 を除外) のとき
    # capture/denial 優先 bias を付与。距離 penalty は据置なので遠方 comet は追わない。
    if target.is_comet and target.owner != my_player and target.x >= 0.0:
        value += COMET_TARGET_BONUS
    return value - w_distance * turns - w_garrison * garrison_cost


def pick_best_target(
    source: PlanetView, candidates: list[PlanetView], my_player: int
) -> PlanetView | None:
    if not candidates:
        return None
    return max(candidates, key=lambda t: score_target(source, t, my_player))
