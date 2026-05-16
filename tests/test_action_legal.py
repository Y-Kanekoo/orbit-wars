"""action.sanitize() の legality 検証 (PLAN.md A-3)。"""

from __future__ import annotations

import math
import random

from src.utils import action


def _mk_obs(player: int, planets: list[list]) -> dict:
    return {"player": player, "planets": planets, "fleets": []}


def test_sanitize_empty_moves() -> None:
    obs = _mk_obs(0, [[0, 0, 10.0, 10.0, 1.0, 10, 1]])
    assert action.sanitize([], obs) == []


def test_sanitize_valid_single_move() -> None:
    obs = _mk_obs(0, [[0, 0, 10.0, 10.0, 1.0, 10, 1]])
    result = action.sanitize([[0, 1.5, 5]], obs)
    assert result == [[0, 1.5, 5]]


def test_sanitize_drops_negative_ships() -> None:
    obs = _mk_obs(0, [[0, 0, 10.0, 10.0, 1.0, 10, 1]])
    assert action.sanitize([[0, 0.0, -5]], obs) == []


def test_sanitize_drops_zero_ships() -> None:
    obs = _mk_obs(0, [[0, 0, 10.0, 10.0, 1.0, 10, 1]])
    assert action.sanitize([[0, 0.0, 0]], obs) == []


def test_sanitize_drops_enemy_planet_launch() -> None:
    obs = _mk_obs(0, [[0, 0, 10.0, 10.0, 1.0, 10, 1], [1, 1, 50.0, 50.0, 1.0, 20, 1]])
    # 自分は player=0、planet id=1 は player=1 のもの
    assert action.sanitize([[1, 0.0, 5]], obs) == []


def test_sanitize_drops_unknown_planet() -> None:
    obs = _mk_obs(0, [[0, 0, 10.0, 10.0, 1.0, 10, 1]])
    assert action.sanitize([[99, 0.0, 5]], obs) == []


def test_sanitize_cumulative_truncation() -> None:
    """同一惑星から複数回派兵 → 累積 ships ≤ garrison になるよう動的に切り詰める。"""
    obs = _mk_obs(0, [[0, 0, 10.0, 10.0, 1.0, 10, 1]])
    result = action.sanitize([[0, 0.0, 6], [0, 1.0, 6]], obs)
    # 6+6=12 だが garrison=10 → 2 個目は 4 に切り詰め
    assert len(result) == 2
    assert result[0][2] == 6
    assert result[1][2] == 4


def test_sanitize_cumulative_truncation_drops_overflow() -> None:
    """累積で garrison 使い切ったら以降の move は drop。"""
    obs = _mk_obs(0, [[0, 0, 10.0, 10.0, 1.0, 10, 1]])
    result = action.sanitize([[0, 0.0, 10], [0, 1.0, 5]], obs)
    assert result == [[0, 0.0, 10]]  # 2 個目は丸ごと drop


def test_sanitize_drops_nan_angle() -> None:
    obs = _mk_obs(0, [[0, 0, 10.0, 10.0, 1.0, 10, 1]])
    assert action.sanitize([[0, math.nan, 5]], obs) == []


def test_sanitize_drops_inf_angle() -> None:
    obs = _mk_obs(0, [[0, 0, 10.0, 10.0, 1.0, 10, 1]])
    assert action.sanitize([[0, math.inf, 5]], obs) == []


def test_sanitize_drops_malformed_move() -> None:
    obs = _mk_obs(0, [[0, 0, 10.0, 10.0, 1.0, 10, 1]])
    # 長さ 2 の move (3 要素必須)
    assert action.sanitize([[0, 1.0]], obs) == []  # type: ignore[list-item]
    # None 含む
    assert action.sanitize([None, [0, 1.0, 5]], obs) == [[0, 1.0, 5]]  # type: ignore[list-item]


def test_sanitize_hammer_random() -> None:
    """ランダム入力でも常に legal を返すこと (例外を投げないこと)。"""
    rng = random.Random(42)
    for _ in range(100):
        n = rng.randint(1, 10)
        planets = []
        for i in range(n):
            owner = rng.choice([-1, 0, 1, 2, 3])
            planets.append(
                [i, owner, rng.uniform(0, 100), rng.uniform(0, 100), 1.0, rng.randint(0, 50), 1]
            )
        obs = _mk_obs(0, planets)

        moves = []
        for _ in range(rng.randint(0, 20)):
            moves.append([rng.randint(-5, n + 5), rng.uniform(-10, 10), rng.randint(-10, 100)])

        result = action.sanitize(moves, obs)
        # 結果は必ず list of [int, float, positive_int]
        for mv in result:
            assert isinstance(mv, list)
            assert len(mv) == 3
            assert isinstance(mv[0], int)
            assert isinstance(mv[1], float)
            assert isinstance(mv[2], int)
            assert mv[2] > 0
