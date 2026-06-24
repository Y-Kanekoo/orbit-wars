"""ONNX value 推論ラッパ + SearchState→tensor 変換 (H016 infra step 6)。

MCTS の leaf 評価を rollout+`_score_state`+sigmoid から **NN value head** に差し替える
ための推論側ヘルパ。学習 (Kaggle GPU) で得た ONNX value model を読み、探索内の
`SearchState` を encoder と同一経路で tensor 化して value ∈ (-1, 1) を返す。

設計方針:
- **train/serve skew ゼロ**: `SearchState` を直接 token 化せず、一旦 `Observation` を
  再構成して `encode_observation` (= 学習データ生成と同一関数) に渡す。token の正規化・
  owner channel・log1p 圧縮ロジックを二重実装しないため、学習分布と推論分布が構造的に一致。
- **global vector のみ近似**: `SearchState` は comet / angular_velocity / overage を持たない
  ため、root observation を `parse()` して得た値を探索ホライズン中 **定数として thread** する
  (comet planet id・angular_velocity は board 固定 / 一定、overage は低 signal の clock)。
- **agent 不変**: main.py / beam からは import しない。MCTS leaf 統合は env flag
  (`ORBIT_WARS_NN_VALUE=1`) でのみ有効、提出物 (beam, MCTS OFF) は byte 不変。
- **1秒/turn**: onnxruntime は CPUExecutionProvider・intra_op=1 で決定論的に。
  exp/036 計測で FP32 ValueNet は ~0.34ms/query → MCTS の数百〜千 leaf eval/turn が予算内。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.features.encoder import EncodedState, encode_observation
from src.utils.observation import Fleet, Observation, Planet

if TYPE_CHECKING:  # 循環 import 回避 (beam を runtime import しない型注釈のみ)
    from src.search.beam import ProjectedFleet, ProjectedPlanet

# model.py / train_value.py の ONNX export と **完全一致**させること (serve skew 回避)。
# src から scripts を import しないため値を複製 (順序が ONNX 入力束縛)。
INPUT_NAMES = (
    "planet_tokens",
    "planet_mask",
    "fleet_tokens",
    "fleet_mask",
    "global_features",
)


@dataclass(frozen=True, slots=True)
class GlobalContext:
    """探索ホライズン中 定数として持ち回る盤面 global 情報 (root obs 由来)。"""

    comet_ids: frozenset[int]
    angular_velocity: float
    remaining_overage_time: float


def context_from_observation(o: Observation) -> GlobalContext:
    """root の `Observation` から探索中 thread する global context を作る。"""
    return GlobalContext(
        comet_ids=frozenset(o.comet_planet_ids),
        angular_velocity=float(o.angular_velocity),
        remaining_overage_time=float(o.remaining_overage_time),
    )


def encode_search_state(
    planets: list[ProjectedPlanet],
    fleets: list[ProjectedFleet],
    player: int,
    ctx: GlobalContext,
) -> EncodedState:
    """探索内 `SearchState` の planets/fleets を encoder と同一経路で tensor 化する。

    `SearchState` の各フィールドは `Observation` の Planet/Fleet と 1:1 対応する
    (ProjectedPlanet: id/owner/x/y/radius/ships/production、ProjectedFleet:
    owner/x/y/angle/source_planet_id/ships)。`Observation` を再構成して
    `encode_observation` に渡すことで token 正規化を学習側と完全共有する。
    """
    obs = Observation(
        player=player,
        planets=[
            Planet(
                id=p.id,
                owner=p.owner,
                x=p.x,
                y=p.y,
                radius=p.radius,
                ships=p.ships,
                production=p.production,
            )
            for p in planets
        ],
        fleets=[
            Fleet(
                id=i,
                owner=f.owner,
                x=f.x,
                y=f.y,
                angle=f.angle,
                from_planet_id=f.source_planet_id,
                ships=f.ships,
            )
            for i, f in enumerate(fleets)
        ],
        angular_velocity=ctx.angular_velocity,
        initial_planets=[],
        comet_planet_ids=list(ctx.comet_ids),
        remaining_overage_time=ctx.remaining_overage_time,
        raw=None,
    )
    return encode_observation(obs)


class ValueModel:
    """ONNX value model の薄い推論ラッパ (lazy session, CPU・決定論的)。"""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._sess = None  # lazy (onnxruntime import を初回 evaluate まで遅延)

    def _session(self):
        if self._sess is None:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1  # 1秒/turn 決定論性 (exp/036 と同一設定)
            opts.inter_op_num_threads = 1
            self._sess = ort.InferenceSession(
                self.model_path, sess_options=opts, providers=["CPUExecutionProvider"]
            )
        return self._sess

    def evaluate(self, encoded: EncodedState) -> float:
        """単一 `EncodedState` の value ∈ (-1, 1) を返す (player 視点、+1=勝勢)。"""
        feeds = {
            "planet_tokens": encoded.planet_tokens[None, ...],
            "planet_mask": encoded.planet_mask[None, ...],
            "fleet_tokens": encoded.fleet_tokens[None, ...],
            "fleet_mask": encoded.fleet_mask[None, ...],
            "global_features": encoded.global_features[None, ...],
        }
        out = self._session().run(["value"], feeds)[0]
        return float(np.asarray(out).reshape(-1)[0])


class PolicyValueModel:
    """PolicyValueNet ONNX (dual head) の薄い推論ラッパ (H017 infra step)。

    value ∈ (-1, 1) と policy_logits (POLICY_DIM,) を返す。MCTS root の PUCT prior に
    policy_logits を配線するためのもので、leaf 評価 (value) は既存 `ValueModel` 経路と
    独立に composable とするため本ラッパは prior 用 policy のみを駆動する想定 (value も
    返すが MCTS 側は現状未使用、value+policy 単一モデル統合は後続 step)。

    ValueModel と同一の CPU・intra_op=1 決定論設定 (1秒/turn 予算遵守)。
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._sess = None  # lazy (onnxruntime import を初回 evaluate まで遅延)

    def _session(self):
        if self._sess is None:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            self._sess = ort.InferenceSession(
                self.model_path, sess_options=opts, providers=["CPUExecutionProvider"]
            )
        return self._sess

    def evaluate(self, encoded: EncodedState) -> tuple[float, np.ndarray]:
        """単一 `EncodedState` の (value ∈ (-1, 1), policy_logits (POLICY_DIM,)) を返す。

        policy_logits は **masked raw logit** (padding planet は _MASK_NEG)。softmax は
        呼び出し側 (MCTS prior) で適用する。
        """
        feeds = {
            "planet_tokens": encoded.planet_tokens[None, ...],
            "planet_mask": encoded.planet_mask[None, ...],
            "fleet_tokens": encoded.fleet_tokens[None, ...],
            "fleet_mask": encoded.fleet_mask[None, ...],
            "global_features": encoded.global_features[None, ...],
        }
        value, policy_logits = self._session().run(["value", "policy_logits"], feeds)
        return (
            float(np.asarray(value).reshape(-1)[0]),
            np.asarray(policy_logits, dtype=np.float64).reshape(-1),
        )


def _smoke() -> int:
    """untrained model で配線契約を end-to-end 検証 (tests/ 改変禁止のため __main__)。

    検証項目:
    1. encode_search_state が encode_observation と **token 一致** (skew ゼロ証明)
    2. untrained ValueNet を ONNX export → ValueModel.evaluate が有限・|value|<1
    3. MCTS leaf 統合 (ORBIT_WARS_NN_VALUE) が budget 内で valid moves を返す
    """
    import tempfile
    from pathlib import Path

    import torch

    from src.nn.model import ValueNet
    from src.search.beam import ProjectedFleet, ProjectedPlanet

    # --- 1. skew ゼロ: SearchState 経由 と Observation 直接 で token が一致すること ---
    planets = [
        ProjectedPlanet(id=0, owner=0, x=10.0, y=10.0, radius=3.0, ships=50, production=5),
        ProjectedPlanet(id=1, owner=1, x=90.0, y=90.0, radius=3.0, ships=30, production=4),
        ProjectedPlanet(id=2, owner=-1, x=50.0, y=50.0, radius=2.0, ships=10, production=2),
    ]
    fleets = [
        ProjectedFleet(
            owner=1,
            x=80.0,
            y=80.0,
            angle=0.5,
            ships=15,
            source_planet_id=1,
            target_planet_id=-1,
            turns_remaining=99,
        ),
    ]
    ctx = GlobalContext(
        comet_ids=frozenset({2}), angular_velocity=0.02, remaining_overage_time=58.0
    )
    enc_ss = encode_search_state(planets, fleets, player=1, ctx=ctx)

    ref_obs = Observation(
        player=1,
        planets=[
            Planet(
                id=p.id,
                owner=p.owner,
                x=p.x,
                y=p.y,
                radius=p.radius,
                ships=p.ships,
                production=p.production,
            )
            for p in planets
        ],
        fleets=[Fleet(id=0, owner=1, x=80.0, y=80.0, angle=0.5, from_planet_id=1, ships=15)],
        angular_velocity=0.02,
        initial_planets=[],
        comet_planet_ids=[2],
        remaining_overage_time=58.0,
        raw=None,
    )
    enc_ref = encode_observation(ref_obs)
    for field in ("planet_tokens", "planet_mask", "fleet_tokens", "fleet_mask", "global_features"):
        a = getattr(enc_ss, field)
        b = getattr(enc_ref, field)
        assert np.array_equal(a, b), f"skew 検出: {field} が encode_observation と不一致"

    # --- 2. untrained ValueNet を ONNX export → evaluate が有限・|value|<1 ---
    with tempfile.TemporaryDirectory() as td:
        onnx_path = Path(td) / "value_smoke.onnx"
        torch.manual_seed(0)
        model = ValueNet().eval()
        dummy = (
            torch.from_numpy(enc_ss.planet_tokens[None, ...]),
            torch.from_numpy(enc_ss.planet_mask[None, ...]),
            torch.from_numpy(enc_ss.fleet_tokens[None, ...]),
            torch.from_numpy(enc_ss.fleet_mask[None, ...]),
            torch.from_numpy(enc_ss.global_features[None, ...]),
        )
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=list(INPUT_NAMES),
            output_names=["value"],
            dynamic_axes={name: {0: "batch"} for name in INPUT_NAMES} | {"value": {0: "batch"}},
            opset_version=17,
        )
        vm = ValueModel(str(onnx_path))
        v = vm.evaluate(enc_ss)
        assert np.isfinite(v), f"value が非有限: {v}"
        assert abs(v) < 1.0, f"tanh 範囲外: {v}"
        print(f"value_infer smoke: skew=0 (token 一致) / untrained value={v:.5f} (|v|<1 OK)")

        # --- 3. MCTS leaf 統合が budget 内で valid moves を返す ---
        import os

        from src.search import mcts

        sample_obs = {
            "player": 0,
            "planets": [
                [0, 0, 30.0, 30.0, 3.0, 100, 2],
                [1, 1, 70.0, 70.0, 3.0, 10, 2],
                [2, -1, 50.0, 50.0, 2.0, 5, 1],
            ],
            "fleets": [],
            "angular_velocity": 0.02,
            "comet_planet_ids": [],
            "remaining_overage_time": 58.0,
        }
        os.environ["ORBIT_WARS_NN_VALUE"] = "1"
        os.environ["ORBIT_WARS_NN_VALUE_MODEL"] = str(onnx_path)
        mcts.reset_value_model()  # env 反映のため cache クリア
        moves = mcts.search(sample_obs, player=0, time_budget_sec=0.3)
        assert isinstance(moves, list), f"moves が list でない: {type(moves)}"
        for m in moves:
            assert isinstance(m, list) and len(m) == 3, f"move 形式不正: {m}"
        os.environ.pop("ORBIT_WARS_NN_VALUE", None)
        os.environ.pop("ORBIT_WARS_NN_VALUE_MODEL", None)
        mcts.reset_value_model()
        print(f"value_infer smoke: MCTS NN leaf 統合 OK (moves={len(moves)} 件, budget 内)")

    return 0


def _policy_smoke() -> int:
    """H017 PUCT prior 配線契約を untrained PolicyValueNet で end-to-end 検証。

    検証項目:
    1. untrained PolicyValueNet を dual ONNX export → PolicyValueModel.evaluate が
       value 有限・|v|<1 / policy_logits shape (POLICY_DIM,) / softmax 後 padding slot ~0
    2. MCTS PUCT 統合 (ORBIT_WARS_NN_POLICY) が budget 内で valid moves を返す
    3. root child prior が確率分布 (総和 1) として割当たり、value-tie でも child を
       differentiate する (prior の min != max)
    """
    import os
    import tempfile
    from pathlib import Path

    import torch

    from src.nn.model import POLICY_DIM, PolicyValueNet
    from src.search import mcts

    sample_obs = {
        "player": 0,
        "planets": [
            [0, 0, 30.0, 30.0, 3.0, 100, 2],
            [1, 1, 70.0, 70.0, 3.0, 10, 2],
            [2, -1, 50.0, 50.0, 2.0, 5, 1],
            [3, 0, 20.0, 80.0, 3.0, 40, 3],
        ],
        "fleets": [],
        "angular_velocity": 0.02,
        "comet_planet_ids": [],
        "remaining_overage_time": 58.0,
    }

    with tempfile.TemporaryDirectory() as td:
        onnx_path = Path(td) / "pv_smoke.onnx"
        torch.manual_seed(0)
        model = PolicyValueNet().eval()
        # encoder 出力 shape の dummy 入力 (smoke 用に encode_search_state で 1 件作る)
        from src.search.beam import ProjectedPlanet

        planets = [
            ProjectedPlanet(id=0, owner=0, x=30.0, y=30.0, radius=3.0, ships=100, production=2),
            ProjectedPlanet(id=1, owner=1, x=70.0, y=70.0, radius=3.0, ships=10, production=2),
        ]
        ctx = GlobalContext(
            comet_ids=frozenset(), angular_velocity=0.02, remaining_overage_time=58.0
        )
        enc = encode_search_state(planets, [], player=0, ctx=ctx)
        dummy = (
            torch.from_numpy(enc.planet_tokens[None, ...]),
            torch.from_numpy(enc.planet_mask[None, ...]),
            torch.from_numpy(enc.fleet_tokens[None, ...]),
            torch.from_numpy(enc.fleet_mask[None, ...]),
            torch.from_numpy(enc.global_features[None, ...]),
        )
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=list(INPUT_NAMES),
            output_names=["value", "policy_logits"],
            dynamic_axes={name: {0: "batch"} for name in INPUT_NAMES}
            | {"value": {0: "batch"}, "policy_logits": {0: "batch"}},
            opset_version=17,
        )

        pvm = PolicyValueModel(str(onnx_path))
        v, logits = pvm.evaluate(enc)
        assert np.isfinite(v) and abs(v) < 1.0, f"value 範囲外: {v}"
        assert logits.shape == (POLICY_DIM,), f"policy_logits shape 不正: {logits.shape}"
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        assert np.isclose(probs.sum(), 1.0), "softmax 総和 != 1"
        # planets=2 件 → slot 2..15 は padding (_MASK_NEG) で prob ~0
        assert probs[2 : POLICY_DIM - 1].sum() < 1e-6, "padding slot の prob が ~0 でない"
        print(
            f"policy_infer smoke: value={v:.5f} (|v|<1 OK) / policy softmax 総和=1 / "
            f"padding slot prob={probs[2:POLICY_DIM - 1].sum():.2e}"
        )

        # MCTS PUCT 統合 (env-gated)
        os.environ["ORBIT_WARS_NN_POLICY"] = "1"
        os.environ["ORBIT_WARS_NN_POLICY_MODEL"] = str(onnx_path)
        mcts.reset_policy_model()
        priors = mcts.debug_root_priors(sample_obs, player=0, time_budget_sec=0.3)
        moves = mcts.search(sample_obs, player=0, time_budget_sec=0.3)
        os.environ.pop("ORBIT_WARS_NN_POLICY", None)
        os.environ.pop("ORBIT_WARS_NN_POLICY_MODEL", None)
        mcts.reset_policy_model()

        assert isinstance(moves, list), f"moves が list でない: {type(moves)}"
        for m in moves:
            assert isinstance(m, list) and len(m) == 3, f"move 形式不正: {m}"
        assert priors, "root prior が割当たっていない"
        assert np.isclose(sum(priors), 1.0), f"prior 総和 != 1: {sum(priors)}"
        assert max(priors) > min(priors), "prior が全 child 同一 (value-tie を割れない)"
        print(
            f"policy_infer smoke: MCTS PUCT 統合 OK (moves={len(moves)} 件 / "
            f"prior 総和=1 / spread={max(priors) - min(priors):.4f} > 0)"
        )

    return 0


if __name__ == "__main__":
    import sys

    rc = _smoke()
    if rc == 0:
        rc = _policy_smoke()
    sys.exit(rc)
