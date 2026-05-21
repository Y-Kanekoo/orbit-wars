"""threat.incoming_threat_eta の方向考慮挙動を検証 (H003).

直線距離のみの incoming_threat との差分 (離反 fleet を脅威に数えない) を中心に確認する。
"""

from __future__ import annotations

import math

from src.strategy.threat import FleetView, incoming_threat, incoming_threat_eta

PLANET = (50.0, 50.0)


def _fleet(x, y, angle, ships, owner=1):
    return FleetView(id=0, owner=owner, x=x, y=y, angle=angle, from_planet_id=0, ships=ships)


def test_fleet_heading_toward_counts_full() -> None:
    # planet の左 (40,50) から +x 方向 (angle=0) で planet に直進 → align=1.0
    f = _fleet(40.0, 50.0, 0.0, 10)
    assert incoming_threat_eta(*PLANET, my_player=0, enemy_fleets=[f]) == 10.0


def test_fleet_heading_away_counts_zero() -> None:
    # 同位置だが -x 方向 (angle=pi) で planet から離反 → 除外。
    # incoming_threat は距離のみで判定するため 10 を数えてしまう (これが H003 の修正点)。
    f = _fleet(40.0, 50.0, math.pi, 10)
    assert incoming_threat_eta(*PLANET, my_player=0, enemy_fleets=[f]) == 0.0
    assert incoming_threat(*PLANET, my_player=0, enemy_fleets=[f]) == 10


def test_perpendicular_fleet_counts_zero() -> None:
    # planet の左から +y 方向 (angle=pi/2) → 接近成分 0 → 除外 (float 誤差は許容)
    f = _fleet(40.0, 50.0, math.pi / 2, 10)
    assert incoming_threat_eta(*PLANET, my_player=0, enemy_fleets=[f]) < 1e-9


def test_beyond_horizon_counts_zero() -> None:
    # 1 ship は speed 1.0、距離 50 → eta 50 > horizon 20 → 除外
    f = _fleet(0.0, 50.0, 0.0, 1)
    assert incoming_threat_eta(*PLANET, my_player=0, enemy_fleets=[f], horizon_turns=20) == 0.0


def test_own_fleet_ignored() -> None:
    f = _fleet(40.0, 50.0, 0.0, 10, owner=0)
    assert incoming_threat_eta(*PLANET, my_player=0, enemy_fleets=[f]) == 0.0


def test_diagonal_partial_alignment() -> None:
    # planet 方向と 60° ずれた heading → align=cos(60°)=0.5 → ships*0.5
    # fleet at (40,50), planet 方向 bearing=0。heading=60°。
    f = _fleet(40.0, 50.0, math.radians(60.0), 20)
    result = incoming_threat_eta(*PLANET, my_player=0, enemy_fleets=[f])
    assert math.isclose(result, 20 * math.cos(math.radians(60.0)), rel_tol=1e-6)
