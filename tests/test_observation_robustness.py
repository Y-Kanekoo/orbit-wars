"""observation.parse() defensive parsing 検証 (PLAN.md B-9)。"""

from __future__ import annotations

from types import SimpleNamespace

from src.utils import observation


def test_parse_empty() -> None:
    o = observation.parse({})
    assert o.player == 0
    assert o.planets == []
    assert o.fleets == []
    assert o.comet_planet_ids == []


def test_parse_dict_minimal() -> None:
    o = observation.parse({"player": 1, "planets": [[0, 0, 10.0, 10.0, 1.0, 5, 1]], "fleets": []})
    assert o.player == 1
    assert len(o.planets) == 1
    assert o.planets[0].id == 0
    assert o.planets[0].ships == 5


def test_parse_namespace() -> None:
    raw = SimpleNamespace(player=2, planets=[[3, 1, 80.0, 80.0, 1.0, 7, 2]], fleets=[])
    o = observation.parse(raw)
    assert o.player == 2
    assert o.planets[0].id == 3
    assert o.planets[0].owner == 1


def test_parse_handles_missing_fields() -> None:
    o = observation.parse({"player": 0})
    assert o.planets == []
    assert o.fleets == []
    assert o.angular_velocity == 0.0


def test_parse_handles_none_fleets() -> None:
    o = observation.parse({"player": 0, "planets": [], "fleets": None})
    assert o.fleets == []


def test_parse_skips_malformed_planet() -> None:
    # 長さ < 7 の planet は skip
    o = observation.parse({"player": 0, "planets": [[0, 0, 10.0], [1, 0, 10.0, 10.0, 1.0, 5, 1]]})
    assert len(o.planets) == 1
    assert o.planets[0].id == 1


def test_parse_skips_malformed_fleet() -> None:
    o = observation.parse({"player": 0, "planets": [], "fleets": [[], [1, 0, 10.0, 10.0, 0.5, 0, 3]]})
    assert len(o.fleets) == 1
    assert o.fleets[0].ships == 3


def test_parse_planet_types_coerced() -> None:
    o = observation.parse({"player": 0, "planets": [["5", "1", "10", "10", "1", "20", "2"]]})
    p = o.planets[0]
    assert isinstance(p.id, int) and p.id == 5
    assert isinstance(p.owner, int) and p.owner == 1
    assert isinstance(p.x, float) and p.x == 10.0
    assert isinstance(p.ships, int) and p.ships == 20


def test_helpers() -> None:
    o = observation.parse(
        {
            "player": 0,
            "planets": [
                [0, 0, 10.0, 10.0, 1.0, 5, 1],
                [1, 1, 80.0, 80.0, 1.0, 7, 1],
                [2, -1, 50.0, 50.0, 1.0, 3, 1],
            ],
            "fleets": [],
        }
    )
    assert len(observation.own_planets(o)) == 1
    assert len(observation.enemy_planets(o)) == 1
    assert len(observation.neutral_planets(o)) == 1


def test_parse_4p_observation() -> None:
    o = observation.parse(
        {
            "player": 2,
            "planets": [
                [0, 0, 10.0, 10.0, 1.0, 5, 1],
                [1, 1, 90.0, 10.0, 1.0, 5, 1],
                [2, 2, 10.0, 90.0, 1.0, 5, 1],
                [3, 3, 90.0, 90.0, 1.0, 5, 1],
            ],
            "fleets": [],
        }
    )
    assert o.player == 2
    assert len(observation.own_planets(o)) == 1
    assert len(observation.enemy_planets(o)) == 3
