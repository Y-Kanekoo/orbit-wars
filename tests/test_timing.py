"""main.agent() の 1 秒/turn 制約検証 (PLAN.md A-1)."""

from __future__ import annotations

import importlib
import random
import time

import pytest

# main.py を import (sys.path は conftest.py で repo root 注入済)
main = importlib.import_module("main")


def _mock_obs(n_planets: int = 20, n_fleets: int = 5, seed: int = 0) -> dict:
    rng = random.Random(seed)
    planets = []
    for i in range(n_planets):
        owner = rng.choice([-1, 0, 1, 2, 3])
        planets.append(
            [
                i,
                owner,
                rng.uniform(5, 95),
                rng.uniform(5, 95),
                1.0 + rng.uniform(0, 1.5),
                rng.randint(0, 80),
                rng.randint(1, 5),
            ]
        )
    fleets = []
    for i in range(n_fleets):
        fleets.append(
            [
                i,
                rng.choice([0, 1, 2, 3]),
                rng.uniform(5, 95),
                rng.uniform(5, 95),
                rng.uniform(0, 6.28),
                rng.randint(0, n_planets - 1),
                rng.randint(1, 100),
            ]
        )
    return {
        "player": 0,
        "step": rng.randint(0, 500),
        "planets": planets,
        "fleets": fleets,
        "angular_velocity": 0.03,
        "remainingOverageTime": 60.0,
        "comet_planet_ids": [],
    }


def _measure_one(obs: dict) -> float:
    t0 = time.monotonic()
    result = main.agent(obs)
    elapsed = time.monotonic() - t0
    assert isinstance(result, list), f"agent must return list, got {type(result)}"
    return elapsed * 1000.0  # ms


def test_single_turn_within_budget() -> None:
    """1 turn が < 950ms (margin under 1s)."""
    obs = _mock_obs(n_planets=20, n_fleets=5, seed=0)
    elapsed_ms = _measure_one(obs)
    assert elapsed_ms < 950.0, f"agent took {elapsed_ms:.1f}ms (>= 950ms)"


def test_warmup_then_steady() -> None:
    """warmup 1 turn 後、N=10 turn 全て < 950ms。"""
    # warmup (ONNX 等の cold start を吸収)
    _measure_one(_mock_obs(seed=999))

    max_ms = 0.0
    for seed in range(10):
        obs = _mock_obs(n_planets=20, n_fleets=5, seed=seed)
        ms = _measure_one(obs)
        max_ms = max(max_ms, ms)

    assert max_ms < 950.0, f"warm max = {max_ms:.1f}ms (>= 950ms)"


def test_dense_observation() -> None:
    """惑星多め (40) + fleet 多め (20) の状況でも < 950ms。"""
    obs = _mock_obs(n_planets=40, n_fleets=20, seed=42)
    elapsed_ms = _measure_one(obs)
    assert elapsed_ms < 950.0, f"dense obs took {elapsed_ms:.1f}ms"


def test_empty_planets_robust() -> None:
    """planets=[] でも crash しないこと (timing 計測対象外、健全性のみ)。"""
    obs = {"player": 0, "step": 0, "planets": [], "fleets": []}
    result = main.agent(obs)
    assert result == []


@pytest.mark.parametrize("seed", list(range(20)))
def test_param_seeds_under_950ms(seed: int) -> None:
    """20 seed で偏ったケースが無いことを確認 (warmup 後)。"""
    if seed == 0:
        # warmup
        _measure_one(_mock_obs(seed=999))
    obs = _mock_obs(n_planets=25, n_fleets=10, seed=seed)
    elapsed_ms = _measure_one(obs)
    assert elapsed_ms < 950.0, f"seed={seed} took {elapsed_ms:.1f}ms"
