"""projection.project_ships の動作検証."""

from __future__ import annotations

from src.features.projection import IncomingFleet, project_ships, projected_total


def test_no_incoming_production_only() -> None:
    own = [(0, 2), (1, 3)]
    result = project_ships(own, [], own_player=0, n_turns=10)
    assert result == {0: 20, 1: 30}


def test_own_incoming_adds() -> None:
    own = [(0, 1)]
    incoming = [IncomingFleet(target_planet_id=0, arrival_turn=5, owner=0, ships=50)]
    result = project_ships(own, incoming, own_player=0, n_turns=10)
    # 10 turn 生産 = 10、 + arrival 50 = 60
    assert result == {0: 60}


def test_enemy_incoming_subtracts() -> None:
    own = [(0, 1)]
    incoming = [IncomingFleet(target_planet_id=0, arrival_turn=3, owner=1, ships=5)]
    result = project_ships(own, incoming, own_player=0, n_turns=10)
    # production 10 - 5 = 5
    assert result == {0: 5}


def test_enemy_overwhelm_clipped_to_zero() -> None:
    own = [(0, 1)]
    incoming = [IncomingFleet(target_planet_id=0, arrival_turn=2, owner=1, ships=100)]
    result = project_ships(own, incoming, own_player=0, n_turns=10)
    # production 10 - 100 = -90 → clip 0
    assert result == {0: 0}


def test_arrival_beyond_horizon_ignored() -> None:
    own = [(0, 1)]
    incoming = [IncomingFleet(target_planet_id=0, arrival_turn=100, owner=0, ships=999)]
    result = project_ships(own, incoming, own_player=0, n_turns=10)
    assert result == {0: 10}


def test_unknown_target_ignored() -> None:
    own = [(0, 1)]
    incoming = [IncomingFleet(target_planet_id=99, arrival_turn=2, owner=0, ships=50)]
    result = project_ships(own, incoming, own_player=0, n_turns=10)
    # planet 99 は own_planets に無い → projection に乗らない
    assert result == {0: 10}


def test_multi_incoming_ordered() -> None:
    own = [(0, 0)]  # production 0
    incoming = [
        IncomingFleet(0, 1, owner=0, ships=10),
        IncomingFleet(0, 2, owner=1, ships=3),
        IncomingFleet(0, 3, owner=0, ships=5),
    ]
    result = project_ships(own, incoming, own_player=0, n_turns=5)
    # t1: +10  t2: -3  t3: +5  → 12
    assert result == {0: 12}


def test_projected_total() -> None:
    proj = {0: 10, 1: 20, 2: 30}
    assert projected_total(proj) == 60


def test_projected_total_empty() -> None:
    assert projected_total({}) == 0
