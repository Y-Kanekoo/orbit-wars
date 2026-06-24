"""4-fold rotational symmetry data augmentation (H018 / exp052)。

NN value head の学習データ (`export_value_data.py` が書き出す encode 済 tensor)
を、盤面の回転対称性を使って 4 倍に増やす。実 self-play game を増やさずに有効
データ量を稼ぐ (data-gen のローカル生成は実用規模で非現実的 =
learned_rules `local_cpu_selfplay_export_infeasible_at_scale`、Kaggle escalate 済)。

## 対称性の根拠
盤面は 100×100 の **正方形**、sun は中心 (50,50) の **radial gravity**。盤面全体を
中心まわりに 90/180/270° 回転した配置は、元と完全に等価な game state (同一 value)
かつ等確率に出現する。よって encode 済 tensor の座標・方向ベクトルを回転すれば、
value target を変えずに 3 つの追加サンプルが得られる。

**rotation のみ (mirror 不可)**: 鏡映は planet の公転方向 (`angular_velocity` の符号)
を反転させ、元と異なる dynamics の state になる。よって 90° 刻み回転 4 通りのみ。

## 実装方針
- encoder.py の列定義 (`encode_observation`) に厳密対応した列を回転:
  - planet token: 位置 (nx, ny) = cols [3], [4]
  - fleet token:  位置 (nx, ny) = cols [2], [3]、方向 (cos a, sin a) = cols [4], [5]
  - owner channel / radius / log ships|prod / comet flag / mask / value / game_id:
    回転不変 (そのまま)。global_features も owner 比・angular_velocity 等のスカラーで不変。
- 90° 刻みは回転行列の成分が 0, ±1 のため **整数演算で exact** (float drift ゼロ)、
  trig を使わず determinism を保証する。
- **副作用なし / agent 非関与**: main.py / beam.py からは import しない純データ変換。
"""

from __future__ import annotations

import numpy as np

from src.features.encoder import (
    FLEET_FEATURES,
    GLOBAL_FEATURES,
    PLANET_FEATURES,
    EncodedState,
)

# encoder.py の列定義に対応した座標・方向ベクトルの列 index。
# encoder を変更したらここも更新すること (smoke が次元一致を assert する)。
_PLANET_XY = (3, 4)
_FLEET_XY = (2, 3)
_FLEET_DIR = (4, 5)  # (cos angle, sin angle)

# 回転を適用するキーと「回転不変でそのまま複製する」キー。
_TOKEN_KEYS = ("planet_tokens", "fleet_tokens")
_PASSTHROUGH_KEYS = ("planet_mask", "fleet_mask", "global_features", "value", "game_id")


def _rot_xy(x: np.ndarray, y: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) を中心まわりに反時計回り k*90° 回転 (k=0..3、整数演算で exact)。"""
    k %= 4
    if k == 0:
        return x, y
    if k == 1:  # 90°:  (x, y) -> (-y,  x)
        return -y, x
    if k == 2:  # 180°: (x, y) -> (-x, -y)
        return -x, -y
    return y, -x  # 270°: (x, y) -> ( y, -x)


def rotate_tokens(
    planet_tokens: np.ndarray, fleet_tokens: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """planet/fleet token 行列を k*90° 回転した **コピー** を返す (元は不変)。

    任意の leading batch 次元に対応 (..., FEATURES)。位置と fleet 方向ベクトルのみ
    回転、その他の列は据え置き。
    """
    pt = np.array(planet_tokens, dtype=np.float32, copy=True)
    ft = np.array(fleet_tokens, dtype=np.float32, copy=True)
    if k % 4 == 0:
        return pt, ft

    # pt[..., c] は view を返すため、回転前に値を snapshot してから書き戻す
    # (RHS の view が LHS 代入で自己破壊されるのを防ぐ)。
    px, py = _PLANET_XY
    pt[..., px], pt[..., py] = _rot_xy(pt[..., px].copy(), pt[..., py].copy(), k)

    fx, fy = _FLEET_XY
    ft[..., fx], ft[..., fy] = _rot_xy(ft[..., fx].copy(), ft[..., fy].copy(), k)
    dx, dy = _FLEET_DIR
    ft[..., dx], ft[..., dy] = _rot_xy(ft[..., dx].copy(), ft[..., dy].copy(), k)
    return pt, ft


def rotate_encoded(state: EncodedState, k: int) -> EncodedState:
    """EncodedState 1 件を k*90° 回転した新しい EncodedState を返す。"""
    pt, ft = rotate_tokens(state.planet_tokens, state.fleet_tokens, k)
    return EncodedState(
        planet_tokens=pt,
        planet_mask=state.planet_mask,
        fleet_tokens=ft,
        fleet_mask=state.fleet_mask,
        global_features=state.global_features,
    )


def augment_4fold(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """batched value dataset を 90/180/270° 回転で 4 倍に拡張した dict を返す。

    入力 data は `export_value_data.py` の npz 互換 (planet_tokens / fleet_tokens /
    planet_mask / fleet_mask / global_features / value / game_id を batch 軸 0 で持つ)。
    出力は同キーで、各キーが [k=0, k=1, k=2, k=3] の順に縦結合された 4N サンプル。

    game_id は 4 コピー間で**同一値のまま**保持する。grouped split (train_value.py)
    後の train サブセットにのみ適用する想定なので、回転コピーが train/val を跨いで
    leak することはない (同一 game の全回転は同じ side に留まる)。
    """
    rotated: dict[str, list[np.ndarray]] = {key: [] for key in (*_TOKEN_KEYS, *_PASSTHROUGH_KEYS)}
    for k in range(4):
        pt, ft = rotate_tokens(data["planet_tokens"], data["fleet_tokens"], k)
        rotated["planet_tokens"].append(pt)
        rotated["fleet_tokens"].append(ft)
        for key in _PASSTHROUGH_KEYS:
            rotated[key].append(np.asarray(data[key]))
    return {key: np.concatenate(parts, axis=0) for key, parts in rotated.items()}


def _smoke() -> int:
    """回転の正当性を検証 (encoder smoke 規約に倣う in-module CPU smoke)。"""
    # 列 index が encoder の次元内に収まっていること (encoder 変更検知)
    assert max(_PLANET_XY) < PLANET_FEATURES, "planet 列 index が PLANET_FEATURES 超過"
    assert max(_FLEET_XY + _FLEET_DIR) < FLEET_FEATURES, "fleet 列 index が FLEET_FEATURES 超過"
    assert GLOBAL_FEATURES > 0

    rng = np.random.default_rng(0)
    n = 5
    pt = rng.standard_normal((n, 4, PLANET_FEATURES)).astype(np.float32)
    ft = rng.standard_normal((n, 3, FLEET_FEATURES)).astype(np.float32)

    # (1) 4 回 90° 回転で元に戻る (k を累積適用しても、k=4 ≡ k=0)
    p4, f4 = pt, ft
    for _ in range(4):
        p4, f4 = rotate_tokens(p4, f4, 1)
    assert np.allclose(p4, pt) and np.allclose(f4, ft), "90°×4 で元に戻らない"

    # (2) 180° は座標符号反転、回転不変列は不変
    p180, f180 = rotate_tokens(pt, ft, 2)
    px, py = _PLANET_XY
    assert np.allclose(p180[..., px], -pt[..., px]) and np.allclose(p180[..., py], -pt[..., py])
    inv_cols = [c for c in range(PLANET_FEATURES) if c not in _PLANET_XY]
    assert np.allclose(p180[..., inv_cols], pt[..., inv_cols]), "planet 不変列が変化"

    # (3) fleet 方向ベクトルが位置と同じ回転を受ける = 単位ノルム保存
    fx, fy = _FLEET_XY
    dx, dy = _FLEET_DIR
    norm_before = np.hypot(ft[..., dx], ft[..., dy])
    for k in range(4):
        _, fk = rotate_tokens(pt, ft, k)
        assert np.allclose(np.hypot(fk[..., dx], fk[..., dy]), norm_before), "方向ノルム不保存"

    # (4) 90° 回転が位置と方向で一致 (同一回転行列): fleet 位置と方向の回転結果が
    #     入力 (px,py)/(dx,dy) に同じ写像を適用していること
    _, f90 = rotate_tokens(pt, ft, 1)
    assert np.allclose(f90[..., fx], -ft[..., fy]) and np.allclose(f90[..., fy], ft[..., fx])
    assert np.allclose(f90[..., dx], -ft[..., dy]) and np.allclose(f90[..., dy], ft[..., dx])

    # (5) 元配列が破壊されない (copy セマンティクス)
    pt_orig = pt.copy()
    rotate_tokens(pt, ft, 3)
    assert np.allclose(pt, pt_orig), "入力が in-place 破壊された"

    # (6) augment_4fold: 4N サンプル・value/game_id/mask が 4 連結で保存
    data = {
        "planet_tokens": pt,
        "fleet_tokens": ft,
        "planet_mask": rng.integers(0, 2, (n, 4)).astype(np.float32),
        "fleet_mask": rng.integers(0, 2, (n, 3)).astype(np.float32),
        "global_features": rng.standard_normal((n, GLOBAL_FEATURES)).astype(np.float32),
        "value": np.where(np.arange(n) % 2 == 0, 1.0, -1.0).astype(np.float32),
        "game_id": np.arange(n, dtype=np.int32),
    }
    aug = augment_4fold(data)
    assert aug["value"].shape[0] == 4 * n, "拡張後サンプル数が 4N でない"
    # k=0 ブロックは原データと一致 (恒等回転)
    assert np.allclose(aug["planet_tokens"][:n], pt), "k=0 ブロックが原データと不一致"
    # value/game_id が 4 タイル
    assert np.array_equal(aug["value"], np.tile(data["value"], 4))
    assert np.array_equal(aug["game_id"], np.tile(data["game_id"], 4))
    # 不変な global_features も 4 タイル (回転対象外)
    assert np.allclose(aug["global_features"], np.tile(data["global_features"], (4, 1)))

    print("symmetry smoke OK:")
    print("  90°×4 = identity / 180° sign-flip / 方向ノルム保存 / copy 安全")
    print(f"  augment_4fold: {n} -> {aug['value'].shape[0]} samples (value/game_id 保存)")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_smoke())
