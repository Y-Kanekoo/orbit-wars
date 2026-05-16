"""Territory control map (H001).

各 cell の最近接 own/敵 planet を比較し、own が勝つ cell 比率を返す。
beam の `_score_state` から呼ばれるため numpy ベクトル化で sub-ms に抑える。
"""

from __future__ import annotations

import numpy as np

_BOARD_SIZE = 100.0


def territory_share(
    own_xy: list[tuple[float, float]] | np.ndarray,
    other_xy: list[tuple[float, float]] | np.ndarray,
    resolution: int = 40,
) -> float:
    """own が最近接の cell 比率を 0..1 で返す。

    own_xy: 自軍惑星座標 (n, 2)
    other_xy: 自軍以外で owned (中立は除外) 惑星座標 (m, 2)
    resolution: 1 辺の cell 数。40 → 1600 cells。

    own が無い or other が無い場合は 0.0 を返す (territory 概念が成立しない)。
    """
    own_arr = np.asarray(own_xy, dtype=np.float64).reshape(-1, 2)
    other_arr = np.asarray(other_xy, dtype=np.float64).reshape(-1, 2)
    if own_arr.size == 0 or other_arr.size == 0:
        return 0.0

    step = _BOARD_SIZE / resolution
    coords = (np.arange(resolution, dtype=np.float64) + 0.5) * step
    xx, yy = np.meshgrid(coords, coords)
    cells = np.stack([xx.ravel(), yy.ravel()], axis=-1)  # (R*R, 2)

    d_own = np.min(((cells[:, None, :] - own_arr[None, :, :]) ** 2).sum(-1), axis=1)
    d_other = np.min(((cells[:, None, :] - other_arr[None, :, :]) ** 2).sum(-1), axis=1)
    own_cells = int((d_own < d_other).sum())
    return own_cells / float(resolution * resolution)
