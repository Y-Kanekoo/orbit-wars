"""State -> tensor encoder (H016 infra step 1).

`Observation` を NN value head が消費できる固定サイズ tensor に変換する。
planets / fleets を token 行列 + padding mask、加えて盤面 global vector を出力する
(PLAN.md L328-334: planets+fleets を token、Transformer encoder への入力)。

設計方針:
- **player-relative**: owner は絶対 player id でなく self/enemy/neutral の 3 channel に
  変換。これで seat (player 0/1) 非依存となり対称性が保たれる。
- **deterministic / 副作用なし**: agent 挙動を変えない純関数。beam / main からは
  import しない (この iter は +0 基盤、実学習・推論統合は後続 iter)。
- **正規化**: 座標は board (0..100) を [-1, 1] に、ships/production は log1p で
  裾の重い分布を圧縮。NN 学習の数値安定性のため。
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from src.utils.observation import Observation

# 盤面定数 (territory.py の _BOARD_SIZE と整合)
_BOARD_SIZE = 100.0

# token 行列の最大長。実 obs は planets ~6-12 / fleets 可変。
# padding mask で実体数を区別するため余裕を持たせる。
MAX_PLANETS = 16
MAX_FLEETS = 32

# 1 planet token の特徴次元 (下記 _planet_token と一致させること)
PLANET_FEATURES = 9
# 1 fleet token の特徴次元 (下記 _fleet_token と一致させること)
FLEET_FEATURES = 7
# global vector 次元 (下記 _global_features と一致させること)
GLOBAL_FEATURES = 6


class EncodedState(NamedTuple):
    """encode_observation の出力 bundle。

    planet_tokens: (MAX_PLANETS, PLANET_FEATURES) float32
    planet_mask:   (MAX_PLANETS,) float32  実 token=1.0 / padding=0.0
    fleet_tokens:  (MAX_FLEETS, FLEET_FEATURES) float32
    fleet_mask:    (MAX_FLEETS,) float32
    global_features: (GLOBAL_FEATURES,) float32
    """

    planet_tokens: np.ndarray
    planet_mask: np.ndarray
    fleet_tokens: np.ndarray
    fleet_mask: np.ndarray
    global_features: np.ndarray


def _norm_xy(x: float, y: float) -> tuple[float, float]:
    """座標を [-1, 1] に正規化 (board 中心原点)。"""
    return (2.0 * x / _BOARD_SIZE - 1.0, 2.0 * y / _BOARD_SIZE - 1.0)


def _owner_channels(owner: int, player: int) -> tuple[float, float, float]:
    """owner を (is_self, is_enemy, is_neutral) の one-hot に変換。"""
    if owner == player:
        return (1.0, 0.0, 0.0)
    if owner < 0:
        return (0.0, 0.0, 1.0)
    return (0.0, 1.0, 0.0)


def _planet_token(planet, player: int, comet_ids: set[int]) -> list[float]:
    is_self, is_enemy, is_neutral = _owner_channels(planet.owner, player)
    nx, ny = _norm_xy(planet.x, planet.y)
    return [
        is_self,
        is_enemy,
        is_neutral,
        nx,
        ny,
        planet.radius / _BOARD_SIZE,
        math.log1p(max(0.0, float(planet.ships))),
        math.log1p(max(0.0, float(planet.production))),
        1.0 if planet.id in comet_ids else 0.0,
    ]


def _fleet_token(fleet, player: int) -> list[float]:
    is_self, is_enemy, _ = _owner_channels(fleet.owner, player)
    nx, ny = _norm_xy(fleet.x, fleet.y)
    return [
        is_self,
        is_enemy,
        nx,
        ny,
        math.cos(fleet.angle),
        math.sin(fleet.angle),
        math.log1p(max(0.0, float(fleet.ships))),
    ]


def _global_features(o: Observation) -> list[float]:
    n_self = sum(1 for p in o.planets if p.owner == o.player)
    n_enemy = sum(1 for p in o.planets if p.owner not in (o.player, -1))
    n_neutral = sum(1 for p in o.planets if p.owner < 0)
    n_planets = max(1, len(o.planets))
    return [
        n_self / n_planets,
        n_enemy / n_planets,
        n_neutral / n_planets,
        math.log1p(max(0.0, float(len(o.fleets)))),
        o.angular_velocity,
        math.log1p(max(0.0, o.remaining_overage_time)),
    ]


def select_planet_tokens(planets: list, player: int) -> list:
    """token に残す planet を選び、全体 ships 降順の順序で返す (<= MAX_PLANETS)。

    H017 (exp/062): policy head は per-token logit (= encoder token 位置 1:1) で、
    launch 元は必ず自軍 planet。素朴な「全 planet ships 降順 top-MAX_PLANETS」では
    自軍 planet が敵/中立の大艦隊に押し出され mid/late game で頻繁に truncate され、
    その launch が policy index に乗らず no-op に collapse する欠陥があった
    (learned_rules `policy_index_truncates_own_planets`、e2e policy_noop_frac=0.67)。

    対策: 切り詰めが起きる (planet 数 > MAX_PLANETS) 場合のみ **自軍 planet を優先保持**
    し、残り slot を ships 降順の非自軍で充填する。**token の順序自体は全体 ships 降順を
    維持** (membership のみ変更) するため、value head (順序非依存の permutation-invariant
    pooling) は planet 数 <= MAX_PLANETS の局面で完全不変。

    encoder と MCTS serve (`_assign_child_priors` / `search_root_visit_policy`) が
    **本 helper を共有**することで policy index の train/serve skew を構造的にゼロにする。
    planet は `.id` / `.ships` / `.owner` を持てばよい (Planet / ProjectedPlanet 共通)。
    """
    by_ships = sorted(planets, key=lambda p: p.ships, reverse=True)
    if len(by_ships) <= MAX_PLANETS:
        return by_ships
    # 自軍を ships 降順で先に確保 (by_ships は既に ships 降順)。自軍数が MAX_PLANETS 超の
    # 極稀ケースは ships 上位 MAX_PLANETS のみ残す。
    own_ids = [p.id for p in by_ships if p.owner == player]
    kept_ids = set(own_ids[:MAX_PLANETS])
    # 残り slot を ships 降順 (自軍含む既存はそのまま) で充填
    for p in by_ships:
        if len(kept_ids) >= MAX_PLANETS:
            break
        kept_ids.add(p.id)
    return [p for p in by_ships if p.id in kept_ids]


def encode_observation(o: Observation) -> EncodedState:
    """Observation を固定サイズ tensor bundle に変換する。

    MAX_PLANETS / MAX_FLEETS を超える要素は ships 降順で上位を残し切り詰める
    (盤面支配に効く大艦隊を優先保持)。ただし planet は `select_planet_tokens` で
    自軍 launch 元を優先保持する (policy index truncation 回避、H017 exp/062)。
    padding は 0 埋め + mask=0。
    """
    comet_ids = set(o.comet_planet_ids)

    planets = select_planet_tokens(o.planets, o.player)
    planet_tokens = np.zeros((MAX_PLANETS, PLANET_FEATURES), dtype=np.float32)
    planet_mask = np.zeros((MAX_PLANETS,), dtype=np.float32)
    for i, planet in enumerate(planets):
        planet_tokens[i] = _planet_token(planet, o.player, comet_ids)
        planet_mask[i] = 1.0

    fleets = sorted(o.fleets, key=lambda f: f.ships, reverse=True)[:MAX_FLEETS]
    fleet_tokens = np.zeros((MAX_FLEETS, FLEET_FEATURES), dtype=np.float32)
    fleet_mask = np.zeros((MAX_FLEETS,), dtype=np.float32)
    for i, fleet in enumerate(fleets):
        fleet_tokens[i] = _fleet_token(fleet, o.player)
        fleet_mask[i] = 1.0

    global_features = np.asarray(_global_features(o), dtype=np.float32)

    return EncodedState(
        planet_tokens=planet_tokens,
        planet_mask=planet_mask,
        fleet_tokens=fleet_tokens,
        fleet_mask=fleet_mask,
        global_features=global_features,
    )


if __name__ == "__main__":  # CPU smoke (tiny sample で end-to-end 検証)
    from src.utils.observation import Fleet, Planet

    sample = Observation(
        player=1,
        planets=[
            Planet(id=0, owner=0, x=10.0, y=10.0, radius=3.0, ships=50, production=5),
            Planet(id=1, owner=1, x=90.0, y=90.0, radius=3.0, ships=30, production=4),
            Planet(id=2, owner=-1, x=50.0, y=50.0, radius=2.0, ships=10, production=2),
        ],
        fleets=[
            Fleet(id=0, owner=1, x=80.0, y=80.0, angle=0.5, from_planet_id=1, ships=15),
        ],
        angular_velocity=0.02,
        initial_planets=[],
        comet_planet_ids=[2],
        remaining_overage_time=58.0,
        raw=None,
    )
    enc = encode_observation(sample)
    assert enc.planet_tokens.shape == (MAX_PLANETS, PLANET_FEATURES)
    assert enc.fleet_tokens.shape == (MAX_FLEETS, FLEET_FEATURES)
    assert enc.global_features.shape == (GLOBAL_FEATURES,)
    assert enc.planet_mask.sum() == 3.0
    assert enc.fleet_mask.sum() == 1.0
    # player=1 の自軍 planet (id=1) は is_self channel=1
    self_idx = int(np.argmax(enc.planet_mask * enc.planet_tokens[:, 0]))
    assert enc.planet_tokens[self_idx, 0] == 1.0
    print("encoder smoke OK:")
    print("  planet_tokens", enc.planet_tokens.shape, "mask sum", enc.planet_mask.sum())
    print("  fleet_tokens", enc.fleet_tokens.shape, "mask sum", enc.fleet_mask.sum())
    print("  global", enc.global_features)

    # --- H017 (exp/062): >MAX_PLANETS 局面で自軍 launch 元が truncate されない検証 ---
    # 低 ships の自軍 3 個 + 高 ships の敵/中立を多数並べ、素朴な top-MAX_PLANETS では
    # 自軍が押し出される状況を作る (policy_index_truncates_own_planets の再現条件)。
    many = [
        Planet(id=900 + k, owner=1, x=5.0, y=5.0, radius=2.0, ships=2, production=1)
        for k in range(3)
    ]
    many += [
        Planet(id=k, owner=0, x=50.0, y=50.0, radius=3.0, ships=100 + k, production=3)
        for k in range(MAX_PLANETS + 5)
    ]
    own_ids = {p.id for p in many if p.owner == 1}
    naive = {p.id for p in sorted(many, key=lambda p: p.ships, reverse=True)[:MAX_PLANETS]}
    assert not (own_ids & naive), "テスト前提崩れ: 素朴 top-N で自軍が既に残ってしまう"
    retained = {p.id for p in select_planet_tokens(many, player=1)}
    assert (
        own_ids <= retained
    ), f"自軍 launch 元が truncate された: own={own_ids} retained={retained}"
    assert len(retained) == MAX_PLANETS, f"retained 数不正: {len(retained)}"
    # 順序は全体 ships 降順を維持 (retained 内)
    ordered = select_planet_tokens(many, player=1)
    assert [p.ships for p in ordered] == sorted(
        (p.ships for p in ordered), reverse=True
    ), "token 順序が ships 降順でない"
    # planet 数 <= MAX_PLANETS では従来挙動と完全一致 (value path 不変保証)
    few = many[:3] + many[3:6]
    assert select_planet_tokens(few, player=1) == sorted(
        few, key=lambda p: p.ships, reverse=True
    ), "<= MAX_PLANETS で従来 (素朴 ships 降順) と不一致"
    print(
        f"  own-retention OK: 自軍 {own_ids} が >MAX_PLANETS 局面でも全保持 (retained {len(retained)})"
    )
