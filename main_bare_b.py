"""main_bare.py の同等 copy (mirror 用に file path を別にして kaggle env の同一 path 衝突を回避)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.search.beam import search as _beam_search  # noqa: E402


def agent(obs: Any) -> list[list[float]]:
    player = int(obs.get("player", 0)) if isinstance(obs, dict) else int(getattr(obs, "player", 0))
    return _beam_search(obs, player, time_budget_sec=0.3)
