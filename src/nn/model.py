"""NN value head (H016 infra step 3 / Phase 3.2)。

`src/features/encoder.py` の `EncodedState` (planets/fleets token + mask + global) を
入力に、局面の **player 視点 value** ∈ (-1, 1) を出力する Transformer encoder。
PLAN.md L322-334 のアーキテクチャ (token embedding → 4-layer transformer (heads=4,
d=64) → mean+attention pooling → value head) に対応。policy head は H017 で追加する
(本 step は value head のみ、exp/032 ledger の段取り)。

設計方針:
- **agent 不変**: main.py / beam.py からは import しない純 module。推論統合は後続 step。
  この iter は +0 基盤 (encoder/exporter step と同じ infra_merge ライン)。
- **mask を attention / pooling に必ず適用**: padding token (mask=0) を
  key_padding_mask で attention から除外し、pooling も valid token のみ平均する。
  mask 無視は「shape は通るが padding 数で出力が変わる」典型バグなので smoke で検証。
- **global token を常に 1 つ付与**: 全 planet/fleet が空でも sequence に有効 token が
  必ず 1 つ残り、全 key padding による attention NaN を回避する。
- **ONNX export 可能**: 提出時は ONNX 化必須 (CLAUDE.md)。FP32 round-trip を smoke で
  確認し、export 不能な op を後続 Kaggle step より前に潰す。FP16 quantize は別 step。

学習は Kaggle GPU Notebook に委譲 (ローカル GPU 学習禁止)。本 module は CPU forward の
shape/mask/export 検証のみを担う。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from src.features.encoder import (
    FLEET_FEATURES,
    GLOBAL_FEATURES,
    MAX_PLANETS,
    PLANET_FEATURES,
)

if TYPE_CHECKING:
    from src.features.encoder import EncodedState

# アーキテクチャ定数 (PLAN.md L331-334: d=64, heads=4, 4 layer)
MODEL_DIM = 64
N_HEADS = 4
N_LAYERS = 4
FF_DIM = 4 * MODEL_DIM  # transformer feed-forward 中間次元

# entity type id (planet / fleet / global の 3 種を type embedding で区別)
_TYPE_PLANET = 0
_TYPE_FLEET = 1
_TYPE_GLOBAL = 2
_N_TYPES = 3


class ValueNet(nn.Module):
    """EncodedState バッチ → scalar value ∈ (-1, 1)。

    forward は encoder 出力をそのまま batched で受ける:
        planet_tokens (B, P, PLANET_FEATURES) / planet_mask (B, P)
        fleet_tokens  (B, F, FLEET_FEATURES)  / fleet_mask  (B, F)
        global_features (B, GLOBAL_FEATURES)
    すべての planet/fleet/global token を 1 sequence に連結し、type embedding を
    加えて self-attention で相互作用させる。
    """

    def __init__(
        self,
        d: int = MODEL_DIM,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        ff_dim: int = FF_DIM,
    ) -> None:
        super().__init__()
        self.d = d
        # entity ごとの token embedding (特徴次元 → d)
        self.planet_embed = nn.Linear(PLANET_FEATURES, d)
        self.fleet_embed = nn.Linear(FLEET_FEATURES, d)
        self.global_embed = nn.Linear(GLOBAL_FEATURES, d)
        # entity 種別を区別する type embedding
        self.type_embed = nn.Embedding(_N_TYPES, d)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # norm_first=True では nested tensor 最適化が無効なため明示 off (warning 抑制)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        # attention pooling 用の学習可能 query (valid token の重み付き和)
        self.pool_query = nn.Linear(d, 1)

        # value head: [mean_pool ; attn_pool] (2d) → scalar。tanh で (-1, 1) に。
        self.value_head = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )

    def forward(
        self,
        planet_tokens: Tensor,
        planet_mask: Tensor,
        fleet_tokens: Tensor,
        fleet_mask: Tensor,
        global_features: Tensor,
    ) -> Tensor:
        b = planet_tokens.shape[0]
        device = planet_tokens.device

        # 各 entity を d 次元に embedding + type embedding 付与
        p_tok = self.planet_embed(planet_tokens) + self.type_embed(
            torch.full((1,), _TYPE_PLANET, dtype=torch.long, device=device)
        )
        f_tok = self.fleet_embed(fleet_tokens) + self.type_embed(
            torch.full((1,), _TYPE_FLEET, dtype=torch.long, device=device)
        )
        # global は 1 token。常に有効 (mask=1) で全 key padding を防ぐ。
        g_tok = (
            self.global_embed(global_features)
            + self.type_embed(torch.full((1,), _TYPE_GLOBAL, dtype=torch.long, device=device))
        ).unsqueeze(
            1
        )  # (B, 1, d)
        g_mask = torch.ones((b, 1), dtype=planet_mask.dtype, device=device)

        # 全 token を 1 sequence に連結
        tokens = torch.cat([p_tok, f_tok, g_tok], dim=1)  # (B, S, d)
        mask = torch.cat([planet_mask, fleet_mask, g_mask], dim=1)  # (B, S) 1=valid

        # key_padding_mask: True=無視 (padding)。valid token のみ attention 対象に。
        key_padding_mask = mask < 0.5
        encoded = self.transformer(tokens, src_key_padding_mask=key_padding_mask)

        pooled = self._pool(encoded, mask)  # (B, 2d)
        value = torch.tanh(self.value_head(pooled))  # (B, 1) ∈ (-1, 1)
        return value.squeeze(-1)  # (B,)

    def _pool(self, encoded: Tensor, mask: Tensor) -> Tensor:
        """valid token のみで masked mean pool + masked attention pool を連結。"""
        m = mask.unsqueeze(-1)  # (B, S, 1)
        denom = m.sum(dim=1).clamp_min(1.0)  # (B, 1) valid token 数 (>=1)
        mean_pool = (encoded * m).sum(dim=1) / denom  # (B, d)

        # attention pool: padding を -inf で除外した softmax 重み
        scores = self.pool_query(encoded).squeeze(-1)  # (B, S)
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(mask < 0.5, neg_inf)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, S, 1)
        attn_pool = (encoded * weights).sum(dim=1)  # (B, d)

        return torch.cat([mean_pool, attn_pool], dim=1)  # (B, 2d)


# policy action space (H017): launch 元 planet ごとの logit + 末尾 1 個の no-op (wait) logit。
# 学習 target は MCTS の root visit 分布を「launch 元 planet」に集約した分布 (no-op は
# 「どの planet からも launch しない」visit 比率)。target_planet / ships の細粒度は本 step
# では持たず、MCTS expansion order の prior (どの planet を先に展開するか) を与える設計。
POLICY_DIM = MAX_PLANETS + 1
# masked logit に入れる大きな負値。-inf でなく有限大負値にして ONNX export / softmax を安定化。
_MASK_NEG = -1.0e9


class PolicyValueNet(nn.Module):
    """EncodedState バッチ → (value ∈ (-1, 1), policy_logits (B, POLICY_DIM))。

    AlphaZero 風の dual head。trunk (token embedding + transformer + masked pool) は
    ValueNet と同構造だが、**ValueNet の state_dict key / ONNX 契約を凍結する**ため独立
    クラスとして実装する (既存 checkpoint `value_net_c5.*` と value_infer.py が ValueNet に
    依存)。共有 trunk への refactor は parameter key を変えて既存 asset を壊すため不可。

    policy head は encoder の planet token 構造と 1:1 対応:
        planet 位置 (sequence 先頭 MAX_PLANETS 個) の encoded 表現 → 各 1 logit
        global token (sequence 末尾) の encoded 表現 → no-op logit
        → cat して (B, MAX_PLANETS+1)。padding planet は planet_mask で _MASK_NEG に。
    no-op logit は常に valid (全 planet が padding でも有効な選択肢が 1 つ残る)。

    返すのは **masked raw logits** (softmax は loss / 推論側で適用)。padding planet は
    _MASK_NEG なので log_softmax / softmax 後に確率 ~0 となる。
    """

    def __init__(
        self,
        d: int = MODEL_DIM,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        ff_dim: int = FF_DIM,
    ) -> None:
        super().__init__()
        self.d = d
        self.planet_embed = nn.Linear(PLANET_FEATURES, d)
        self.fleet_embed = nn.Linear(FLEET_FEATURES, d)
        self.global_embed = nn.Linear(GLOBAL_FEATURES, d)
        self.type_embed = nn.Embedding(_N_TYPES, d)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        self.pool_query = nn.Linear(d, 1)
        self.value_head = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )
        # policy head: planet token / global token から per-token logit
        self.planet_policy_head = nn.Linear(d, 1)
        self.noop_policy_head = nn.Linear(d, 1)

    def forward(
        self,
        planet_tokens: Tensor,
        planet_mask: Tensor,
        fleet_tokens: Tensor,
        fleet_mask: Tensor,
        global_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        b = planet_tokens.shape[0]
        device = planet_tokens.device

        p_tok = self.planet_embed(planet_tokens) + self.type_embed(
            torch.full((1,), _TYPE_PLANET, dtype=torch.long, device=device)
        )
        f_tok = self.fleet_embed(fleet_tokens) + self.type_embed(
            torch.full((1,), _TYPE_FLEET, dtype=torch.long, device=device)
        )
        g_tok = (
            self.global_embed(global_features)
            + self.type_embed(torch.full((1,), _TYPE_GLOBAL, dtype=torch.long, device=device))
        ).unsqueeze(
            1
        )  # (B, 1, d)
        g_mask = torch.ones((b, 1), dtype=planet_mask.dtype, device=device)

        tokens = torch.cat([p_tok, f_tok, g_tok], dim=1)  # (B, S, d)
        mask = torch.cat([planet_mask, fleet_mask, g_mask], dim=1)  # (B, S)
        key_padding_mask = mask < 0.5
        encoded = self.transformer(tokens, src_key_padding_mask=key_padding_mask)

        # --- value head (ValueNet と同手順) ---
        pooled = self._pool(encoded, mask)
        value = torch.tanh(self.value_head(pooled)).squeeze(-1)  # (B,)

        # --- policy head ---
        n_p = planet_tokens.shape[1]  # = MAX_PLANETS
        planet_encoded = encoded[:, :n_p, :]  # (B, P, d)
        planet_logits = self.planet_policy_head(planet_encoded).squeeze(-1)  # (B, P)
        # padding planet を _MASK_NEG に (softmax 後 ~0)
        planet_logits = planet_logits.masked_fill(planet_mask < 0.5, _MASK_NEG)
        # global token = sequence 末尾 (S-1) の encoded から no-op logit
        noop_logit = self.noop_policy_head(encoded[:, -1, :])  # (B, 1)
        policy_logits = torch.cat([planet_logits, noop_logit], dim=1)  # (B, P+1)

        return value, policy_logits

    def _pool(self, encoded: Tensor, mask: Tensor) -> Tensor:
        """ValueNet._pool と同一 (masked mean + masked attention pool)。"""
        m = mask.unsqueeze(-1)
        denom = m.sum(dim=1).clamp_min(1.0)
        mean_pool = (encoded * m).sum(dim=1) / denom

        scores = self.pool_query(encoded).squeeze(-1)
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(mask < 0.5, neg_inf)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        attn_pool = (encoded * weights).sum(dim=1)

        return torch.cat([mean_pool, attn_pool], dim=1)


def batch_from_encoded(states: list[EncodedState]) -> dict[str, Tensor]:
    """EncodedState のリストを ValueNet.forward の kwargs 用 tensor dict に束ねる。

    学習 (Kaggle) / 推論統合 step の双方から再利用する変換ヘルパ。
    """
    import numpy as np

    return {
        "planet_tokens": torch.from_numpy(np.stack([s.planet_tokens for s in states])),
        "planet_mask": torch.from_numpy(np.stack([s.planet_mask for s in states])),
        "fleet_tokens": torch.from_numpy(np.stack([s.fleet_tokens for s in states])),
        "fleet_mask": torch.from_numpy(np.stack([s.fleet_mask for s in states])),
        "global_features": torch.from_numpy(np.stack([s.global_features for s in states])),
    }


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":  # CPU smoke (shape / mask 適用 / ONNX round-trip 検証)
    import numpy as np

    from src.features.encoder import (
        FLEET_FEATURES as _FF,
    )
    from src.features.encoder import (
        MAX_FLEETS,
        MAX_PLANETS,
        encode_observation,
    )
    from src.features.encoder import (
        PLANET_FEATURES as _PF,
    )
    from src.utils.observation import Fleet, Observation, Planet

    torch.manual_seed(0)
    model = ValueNet().eval()
    n_params = count_params(model)
    print(f"ValueNet params = {n_params:,} (PLAN spec ~500K)")
    assert n_params < 1_000_000, f"param 過多: {n_params}"

    # --- (1) encoder 出力を batched でそのまま消費 ---
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
    batch = batch_from_encoded([enc, enc])  # B=2
    with torch.no_grad():
        out = model(**batch)
    assert out.shape == (2,), f"value shape 不正: {out.shape}"
    assert torch.isfinite(out).all(), "value に非有限値"
    assert (out.abs() < 1.0).all(), "tanh 範囲外 (|value| >= 1)"
    print(f"  forward OK: value shape {tuple(out.shape)} = {out.tolist()}")

    # --- (2) mask 適用検証 (最重要): padding slot に garbage を入れても出力不変 ---
    # 実 token はそのまま、padding 部 (mask=0) にゼロでない値を注入する。mask が
    # 効いていれば出力は変わらない。効いていなければ padding が pooled 表現を汚す。
    poisoned = {k: v.clone() for k, v in batch.items()}
    pmask = poisoned["planet_mask"][0]  # (MAX_PLANETS,)
    fmask = poisoned["fleet_mask"][0]
    pad_p = pmask < 0.5
    pad_f = fmask < 0.5
    poisoned["planet_tokens"][0, pad_p] = 7.0  # garbage を padding token に
    poisoned["fleet_tokens"][0, pad_f] = -5.0
    with torch.no_grad():
        out_poison = model(**poisoned)
    # sample[0] は padding を汚染、sample[1] は無汚染 → mask が効けば [0] は不変
    delta = (out_poison[0] - out[0]).abs().item()
    assert delta < 1e-5, f"mask 無効: padding garbage で出力が {delta} 変化"
    assert (out_poison[1] - out[1]).abs().item() < 1e-6
    print(f"  mask 適用 OK: padding garbage による出力変化 {delta:.2e} (< 1e-5)")
    assert pad_p.any() and pad_f.any(), "padding slot が無く mask 検証が空振り"

    # --- (3) ONNX export + onnxruntime round-trip (FP32, atol 緩め) ---
    import io

    dummy = batch
    input_names = [
        "planet_tokens",
        "planet_mask",
        "fleet_tokens",
        "fleet_mask",
        "global_features",
    ]
    buf = io.BytesIO()
    torch.onnx.export(
        model,
        tuple(dummy[name] for name in input_names),
        buf,
        input_names=input_names,
        output_names=["value"],
        dynamic_axes={name: {0: "batch"} for name in input_names} | {"value": {0: "batch"}},
        opset_version=17,
    )
    import onnxruntime as ort

    sess = ort.InferenceSession(buf.getvalue(), providers=["CPUExecutionProvider"])
    ort_out = sess.run(["value"], {name: dummy[name].numpy() for name in input_names})[0]
    max_diff = float(np.abs(ort_out - out.numpy()).max())
    assert max_diff < 1e-4, f"ONNX round-trip 不一致: max_diff={max_diff}"
    print(f"  ONNX round-trip OK: torch vs onnxruntime max_diff {max_diff:.2e}")
    # 念のため shape 定数が encoder と一致
    assert (_PF, _FF) == (PLANET_FEATURES, FLEET_FEATURES)
    assert batch["planet_tokens"].shape[1:] == (MAX_PLANETS, _PF)
    assert batch["fleet_tokens"].shape[1:] == (MAX_FLEETS, _FF)
    print("ValueNet smoke OK")

    # ================= PolicyValueNet (H017 dual head) smoke =================
    print("\n--- PolicyValueNet (H017) ---")
    pv = PolicyValueNet().eval()
    pv_params = count_params(pv)
    print(f"PolicyValueNet params = {pv_params:,} (PLAN spec ~500K)")
    assert pv_params < 1_000_000, f"param 過多: {pv_params}"

    # --- (1) dual output shape ---
    with torch.no_grad():
        v_out, p_logits = pv(**batch)
    assert v_out.shape == (2,), f"value shape 不正: {v_out.shape}"
    assert p_logits.shape == (2, POLICY_DIM), f"policy shape 不正: {p_logits.shape}"
    assert POLICY_DIM == MAX_PLANETS + 1
    assert torch.isfinite(v_out).all() and (v_out.abs() < 1.0).all(), "value 不正"
    print(f"  dual output OK: value {tuple(v_out.shape)} / policy {tuple(p_logits.shape)}")

    # --- (2) padding planet は masked (softmax 後 ~0) ---
    # sample[0] は実 planet 3 個 (id 0/1/2)、残り 13 slot は padding。
    pmask0 = batch["planet_mask"][0] < 0.5  # padding planet 位置
    pad_logits = p_logits[0, :MAX_PLANETS][pmask0]
    assert pmask0.any(), "padding planet が無く mask 検証が空振り"
    assert (pad_logits <= _MASK_NEG + 1.0).all(), "padding planet logit が masked されていない"
    probs0 = torch.softmax(p_logits[0], dim=0)
    assert probs0[:MAX_PLANETS][pmask0].max().item() < 1e-6, "padding planet の prob が非ゼロ"
    assert abs(float(probs0.sum()) - 1.0) < 1e-5, "policy 確率が正規化されていない"
    # no-op (末尾) は常に valid
    assert torch.isfinite(p_logits[0, -1]), "no-op logit が非有限"
    print(f"  policy mask OK: padding planet prob max {probs0[:MAX_PLANETS][pmask0].max():.2e}")

    # --- (3) padding token の garbage が valid policy logit を汚さない ---
    pv_poison = {k: v.clone() for k, v in batch.items()}
    pv_poison["planet_tokens"][0, pmask0] = 7.0
    fmask0 = batch["fleet_mask"][0] < 0.5
    pv_poison["fleet_tokens"][0, fmask0] = -5.0
    with torch.no_grad():
        v_p, p_p = pv(**pv_poison)
    valid_idx = ~pmask0
    d_valid = (p_p[0, :MAX_PLANETS][valid_idx] - p_logits[0, :MAX_PLANETS][valid_idx]).abs().max()
    assert d_valid.item() < 1e-4, f"mask 無効: valid policy logit が {d_valid} 変化"
    assert (v_p[0] - v_out[0]).abs().item() < 1e-4, "mask 無効: value が変化"
    print(f"  policy/value mask 適用 OK: valid logit 変化 {d_valid:.2e}")

    # --- (4) ONNX export + round-trip (dual output) ---
    buf2 = io.BytesIO()
    torch.onnx.export(
        pv,
        tuple(dummy[name] for name in input_names),
        buf2,
        input_names=input_names,
        output_names=["value", "policy_logits"],
        dynamic_axes={name: {0: "batch"} for name in input_names}
        | {"value": {0: "batch"}, "policy_logits": {0: "batch"}},
        opset_version=17,
    )
    sess2 = ort.InferenceSession(buf2.getvalue(), providers=["CPUExecutionProvider"])
    ort_v, ort_p = sess2.run(
        ["value", "policy_logits"], {name: dummy[name].numpy() for name in input_names}
    )
    dv = float(np.abs(ort_v - v_out.numpy()).max())
    dp = float(np.abs(ort_p - p_logits.numpy()).max())
    assert dv < 1e-4, f"ONNX value 不一致: {dv}"
    # _MASK_NEG (大絶対値) を含むため絶対 atol でなく相対許容で判定
    assert dp < 1e-2, f"ONNX policy 不一致: {dp}"
    print(f"  ONNX dual round-trip OK: value max_diff {dv:.2e} / policy max_diff {dp:.2e}")
    print("model smoke OK")
