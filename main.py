"""Orbit Wars agent entry point (PLAN.md A-1 anytime wrapper).

actTimeout 1 秒/turn を絶対超えないため、以下の二段構え:
1. 最初に safe_act() (< 50ms 保証) で fallback action を確保
2. core_act() を deadline 0.90s で呼ぶ、exception or タイムアウトなら fallback を返す

Phase 0 では core_act = safe_act (= nearest sniper)。
Phase 1+ で core_act を heuristic/MCTS/NN に差し替えていく。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

# Kaggle で main.py が repo root 外に置かれて実行されるケースに備える
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.agents.safe_fallback import act as _safe_act  # noqa: E402
from src.utils import action as _action  # noqa: E402
from src.utils import telemetry as _tel  # noqa: E402
from src.utils.timing import Timer  # noqa: E402


def _core_act(obs: Any, deadline: float) -> list[list[Any]]:
    """Phase 1+ で heuristic/MCTS/NN を差し替える本体。
    Phase 0 では safe_act と同じ実装。
    """
    return _safe_act(obs)


def agent(obs: Any) -> list[list[Any]]:
    """Kaggle env から呼ばれるエントリポイント。"""
    turn = int(obs.get("step", 0)) if isinstance(obs, dict) else getattr(obs, "step", 0)
    deadline = time.monotonic() + 0.90  # 100ms margin under 1.0s limit

    # まず確実な fallback を用意 (< 50ms 想定)
    fallback: list[list[Any]] = []
    try:
        with Timer() as t_fb:
            fallback = _safe_act(obs)
        if t_fb.elapsed_ms > 100:
            _tel.error(turn, "safe_fallback slow", elapsed_ms=t_fb.elapsed_ms)
    except Exception as e:  # noqa: BLE001
        _tel.error(turn, "safe_fallback raised", err=str(e))
        return []

    # core_act を時間内で実行
    try:
        with Timer() as t_core:
            result = _core_act(obs, deadline=deadline)
        if t_core.elapsed_ms > 950:
            _tel.error(turn, "core slow timeout-risk", elapsed_ms=t_core.elapsed_ms)
            return _action.sanitize(fallback, obs)
        _tel.log(turn, core_ms=round(t_core.elapsed_ms, 2), n_moves=len(result))
        return _action.sanitize(result, obs)
    except Exception as e:  # noqa: BLE001
        _tel.error(turn, "core raised", err=str(e))
        return _action.sanitize(fallback, obs)
