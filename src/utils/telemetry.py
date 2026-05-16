"""Agent internal logging (PLAN.md A-14)."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

_LEVEL = os.environ.get("LOG_LEVEL", "normal").lower()  # silent | normal | verbose


def _enabled(level: str) -> bool:
    order = {"silent": 0, "normal": 1, "verbose": 2}
    return order.get(_LEVEL, 1) >= order.get(level, 1)


def log(turn: int, **data: Any) -> None:
    """通常 log (kaggle log size を意識)。LOG_LEVEL=silent では出力しない。"""
    if not _enabled("normal"):
        return
    sys.stderr.write(json.dumps({"turn": turn, **data}, default=str) + "\n")
    sys.stderr.flush()


def debug(turn: int, **data: Any) -> None:
    """詳細 log。LOG_LEVEL=verbose のときのみ出力。"""
    if not _enabled("verbose"):
        return
    sys.stderr.write(json.dumps({"turn": turn, "level": "debug", **data}, default=str) + "\n")
    sys.stderr.flush()


def error(turn: int, msg: str, **data: Any) -> None:
    """error log (常に出力、silent でも出す)。"""
    sys.stderr.write(
        json.dumps({"turn": turn, "level": "error", "msg": msg, **data}, default=str) + "\n"
    )
    sys.stderr.flush()
