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

# Kaggle で main.py が repo root 外に置かれて実行されるケースに備える。
# 注: kaggle_environments の agent loader は空 globals で exec() するため
# __file__ が env に注入されない。NameError を捕捉し、その場合は
# kaggle agent loader が既に exec_dir を sys.path に append 済 (agent.py L53)
# なので sys.path 操作は不要。
try:
    _ROOT = Path(__file__).resolve().parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
except NameError:
    # kaggle_environments exec context: __file__ 不在
    pass

from src.agents.safe_fallback import act as _safe_act  # noqa: E402
from src.search.beam import search as _beam_search  # noqa: E402
from src.utils import action as _action  # noqa: E402
from src.utils import telemetry as _tel  # noqa: E402
from src.utils.timing import Timer  # noqa: E402

# H004: depth=3 でも実測 max 27ms のため headroom only。万一の spike 時に
# deadline 0.90s 内で beam が完走できる余地を確保 (挙動は depth 変更が主因)。
_BEAM_TIME_BUDGET_SEC = 0.7


def _core_act(obs: Any, deadline: float) -> list[list[Any]]:
    """Phase 1+ heuristic 本体: legacy-388 移植 beam + H001 territory eval。

    deadline (monotonic 絶対時刻) から残予算を計算し、beam に渡す。
    残予算が 0 以下なら safe_act にフォールバック。
    """
    remaining = max(0.0, deadline - time.monotonic())
    budget = min(_BEAM_TIME_BUDGET_SEC, remaining)
    if budget <= 0.05:
        return _safe_act(obs)
    player = int(obs.get("player", 0)) if isinstance(obs, dict) else int(getattr(obs, "player", 0))
    return _beam_search(obs, player, time_budget_sec=budget)


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
