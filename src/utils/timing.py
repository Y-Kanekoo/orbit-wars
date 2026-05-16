"""Timing utilities (PLAN.md A-1 anytime wrapper)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def deadline_iter(seconds: float) -> Iterator[float]:
    """`with deadline_iter(0.9) as deadline:` 形式で deadline 時刻 (monotonic) を返す。

    呼び出し側は内部ループで `time.monotonic() < deadline` を確認すること。
    """
    deadline = time.monotonic() + seconds
    try:
        yield deadline
    finally:
        pass


def time_remaining(deadline: float) -> float:
    """deadline までの残秒数 (負なら過ぎている)。"""
    return deadline - time.monotonic()


def has_time_left(deadline: float, margin: float = 0.05) -> bool:
    """margin (秒) を含めてまだ余裕があるか。"""
    return time_remaining(deadline) > margin


class Timer:
    """`with Timer() as t:` で elapsed_ms を計測。"""

    def __init__(self) -> None:
        self.start: float = 0.0
        self.end: float = 0.0

    def __enter__(self) -> Timer:
        self.start = time.monotonic()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.end = time.monotonic()

    @property
    def elapsed_ms(self) -> float:
        end = self.end or time.monotonic()
        return (end - self.start) * 1000.0
