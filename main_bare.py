"""Bare agent: main.py wrapper layer を完全 bypass し、直接 src/search/beam.search を呼ぶ。

H000 parity 検証用。legacy-388 vs N=30 で 50% に近づけば、 wrapper layer が劣化要因。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.search.beam import search as _beam_search  # noqa: E402

_DEBUG_DUMP_DONE = False


def agent(obs: Any) -> list[list[float]]:
    global _DEBUG_DUMP_DONE
    if not _DEBUG_DUMP_DONE:
        _DEBUG_DUMP_DONE = True
        player_val = (
            obs.get("player", "?") if isinstance(obs, dict) else getattr(obs, "player", "?")
        )
        keys = [k for k in sys.modules if "search" in k or "beam" in k or "strategy" in k]
        try:
            with open("/tmp/main_bare_dump.log", "a") as f:
                f.write(
                    f"[main_bare DUMP] player={player_val}"
                    f" sys.path[:3]={sys.path[:3]}"
                    f" modules={sorted(keys)}\n"
                )
        except OSError:
            pass
    player = int(obs.get("player", 0)) if isinstance(obs, dict) else int(getattr(obs, "player", 0))
    return _beam_search(obs, player, time_budget_sec=0.3)
