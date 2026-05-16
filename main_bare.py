"""Bare agent: main.py wrapper layer を完全 bypass し、直接 src/search/beam.search を呼ぶ。

H000 parity 検証用。legacy-388 vs N=30 で 50% に近づけば、 wrapper layer が劣化要因。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# kaggle_environments の agent loader は空 globals で exec() するため __file__ 不在。
# NameError を捕捉し、その場合は kaggle agent loader 側で exec_dir が既に
# sys.path に append 済 (agent.py L53) なので何もしない。
try:
    _ROOT = Path(__file__).resolve().parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
except NameError:
    pass

from src.search.beam import search as _beam_search  # noqa: E402


def agent(obs: Any) -> list[list[float]]:
    player = int(obs.get("player", 0)) if isinstance(obs, dict) else int(getattr(obs, "player", 0))
    return _beam_search(obs, player, time_budget_sec=0.3)
