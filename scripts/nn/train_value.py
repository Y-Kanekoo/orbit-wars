"""NN value head 学習 harness (H016 infra step 4)。

`export_value_data.py` が書き出した `.npz` (encode 済 tensor + value target + game_id)
を読み込み、`src/nn/model.py` の `ValueNet` を MSE で学習する。学習済み重みを
torch checkpoint に保存し、推論時と同一の入力署名で ONNX export まで行う
(PLAN.md L328-334 Transformer value head、提出は ONNX 必須 = CLAUDE.md)。

設計方針:
- **本番学習は Kaggle GPU Notebook に委譲** (CLAUDE.md「ローカル GPU 学習禁止」)。本
  harness は device 自動選択で CPU/GPU 両対応。ローカルは `--smoke` で tiny synthetic
  data の CPU forward/学習収束/ONNX round-trip のみ検証する (agent 不変・no-submit)。
- **val split は game 単位 (grouped)**: MC-outcome value は同一 game・同一 player の全
  timestep が **完全に同一の target**。timestep をランダム分割すると train/val に同一
  game の相関サンプルが混ざり val loss が leakage で楽観化し、汎化判定が壊れる
  (advisor 指摘)。よって `game_id` で game ごとに train/val に振り分ける。
- **train/serve skew 回避**: ONNX の input_names 順序・dynamic_axes を model.py smoke と
  完全一致させ、学習済み net で torch vs onnxruntime round-trip を再 assert する。
- **副作用なし**: main.py / beam.py からは import しない。MCTS への value 統合は後続 step。

CLI:
    .venv/bin/python scripts/nn/train_value.py \
        --data data/value_dataset.npz --epochs 30 --batch-size 256 \
        --out experiments/checkpoints/value_net
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features.symmetry import augment_4fold  # noqa: E402
from src.nn.model import POLICY_DIM, PolicyValueNet, ValueNet, count_params  # noqa: E402

# ValueNet.forward の引数順と一致させること (ONNX serve skew 回避の要)。
INPUT_NAMES = [
    "planet_tokens",
    "planet_mask",
    "fleet_tokens",
    "fleet_mask",
    "global_features",
]


def _grouped_split(
    game_id: np.ndarray, val_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """game_id 単位で train/val のサンプル index を返す (timestep leakage 回避)。

    game をシャッフルして val_frac 分を val に割り当て、各 game の全 timestep を
    まるごと同じ側に置く。val game が 0 になる場合は最低 1 game を val に回す。
    """
    games = np.unique(game_id)
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    n_val = max(1, int(round(len(games) * val_frac))) if len(games) > 1 else 0
    val_games = set(games[:n_val].tolist())
    is_val = np.array([g in val_games for g in game_id])
    val_idx = np.flatnonzero(is_val)
    train_idx = np.flatnonzero(~is_val)
    return train_idx, val_idx


def _slice_numpy(
    data: dict, idx: np.ndarray, extra_keys: tuple[str, ...] = ()
) -> dict[str, np.ndarray]:
    """npz の指定 index 群を augment/tensor 化前の numpy 部分集合 dict に切り出す。

    extra_keys に policy_target 等の追加 target を渡すと併せて切り出す (PolicyValueNet 用)。
    既定 () で従来の ValueNet path は不変。
    """
    keys = (*INPUT_NAMES, "value", "game_id", *extra_keys)
    return {name: np.asarray(data[name])[idx] for name in keys}


def _tensors_from_numpy(
    sub: dict[str, np.ndarray], device: torch.device, extra_float_keys: tuple[str, ...] = ()
) -> dict[str, Tensor]:
    """numpy 部分集合を NN forward 用 tensor dict + target に束ねる。

    extra_float_keys は float32 target として追加変換する (例: policy_target)。
    """
    out = {
        name: torch.from_numpy(np.ascontiguousarray(sub[name])).to(device) for name in INPUT_NAMES
    }
    out["value"] = torch.from_numpy(np.asarray(sub["value"]).astype(np.float32)).to(device)
    for key in extra_float_keys:
        out[key] = torch.from_numpy(
            np.ascontiguousarray(np.asarray(sub[key]).astype(np.float32))
        ).to(device)
    return out


def _load_tensors(data: dict, idx: np.ndarray, device: torch.device) -> dict[str, Tensor]:
    """npz の指定 index 群を ValueNet forward 用 tensor dict + target に束ねる。"""
    return _tensors_from_numpy(_slice_numpy(data, idx), device)


def _epoch(
    model: ValueNet,
    tensors: dict[str, Tensor],
    batch_size: int,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """1 epoch 実行。optimizer=None で eval (no grad)。(mean_loss, sign_acc) を返す。"""
    n = tensors["value"].shape[0]
    order = np.arange(n)
    train = optimizer is not None
    if train:
        rng.shuffle(order)
    model.train(train)

    total_loss = 0.0
    correct = 0
    counted = 0
    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for start in range(0, n, batch_size):
            sel = order[start : start + batch_size]
            batch = {name: tensors[name][sel] for name in INPUT_NAMES}
            target = tensors["value"][sel]
            pred = model(**batch)
            loss = loss_fn(pred, target)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss) * len(sel)
            # sign accuracy: draw (target=0) は符号判定対象外
            nz = target != 0
            if nz.any():
                correct += int(((pred[nz] > 0) == (target[nz] > 0)).sum())
                counted += int(nz.sum())
    mean_loss = total_loss / max(1, n)
    sign_acc = correct / counted if counted else float("nan")
    return mean_loss, sign_acc


def export_onnx(model: ValueNet, sample: dict[str, Tensor], out_path: Path) -> float:
    """学習済み model を ONNX export し、onnxruntime との round-trip max_diff を返す。"""
    model.eval()
    cpu = {name: sample[name].detach().cpu() for name in INPUT_NAMES}
    with torch.no_grad():
        ref = model(**cpu).numpy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        tuple(cpu[name] for name in INPUT_NAMES),
        str(out_path),
        input_names=INPUT_NAMES,
        output_names=["value"],
        dynamic_axes={name: {0: "batch"} for name in INPUT_NAMES} | {"value": {0: "batch"}},
        opset_version=17,
    )
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(["value"], {name: cpu[name].numpy() for name in INPUT_NAMES})[0]
    return float(np.abs(ort_out - ref).max())


def train(
    data: dict,
    epochs: int,
    batch_size: int,
    lr: float,
    val_frac: float,
    seed: int,
    out_prefix: Path,
    augment: bool = False,
) -> dict:
    """value dataset から ValueNet を学習し checkpoint + ONNX を書き出す。

    augment=True で train サブセットのみ 90/180/270° 回転で 4 倍に拡張する (H018)。
    val は honest な汎化測定のため**拡張しない** (回転コピーは新情報を持たず val loss を
    水増しするだけ)。回転は grouped split **後** の train_idx にのみ適用するため、同一
    game の全回転は同じ side に留まり train/val leakage は起きない。
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    game_id = np.asarray(data["game_id"])
    train_idx, val_idx = _grouped_split(game_id, val_frac, seed)
    train_np = _slice_numpy(data, train_idx)
    n_train_raw = int(len(train_idx))
    if augment:
        train_np = augment_4fold(train_np)
    train_t = _tensors_from_numpy(train_np, device)
    val_t = _load_tensors(data, val_idx, device) if len(val_idx) else None

    model = ValueNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(seed)

    best_val = float("inf")
    ckpt_path = out_prefix.with_suffix(".pt")
    history = []
    last_tr_acc = float("nan")
    last_va_acc = float("nan")
    for ep in range(epochs):
        tr_loss, last_tr_acc = _epoch(model, train_t, batch_size, loss_fn, optimizer, rng)
        if val_t is not None:
            va_loss, last_va_acc = _epoch(model, val_t, batch_size, loss_fn, None, rng)
        else:
            va_loss = float("nan")
        history.append((tr_loss, va_loss))
        # val があれば val、なければ train を基準に best を保存
        score = va_loss if val_t is not None else tr_loss
        if score < best_val:
            best_val = score
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"state_dict": model.state_dict(), "epoch": ep, "val_loss": va_loss},
                ckpt_path,
            )

    # ONNX は最終 model で export (本番は best 復元してから export する運用も可)
    onnx_path = out_prefix.with_suffix(".onnx")
    sample = (
        {name: val_t[name][:2] for name in INPUT_NAMES}
        if val_t is not None
        else {name: train_t[name][:2] for name in INPUT_NAMES}
    )
    onnx_diff = export_onnx(model, sample, onnx_path)

    first_tr, last_tr = history[0][0], history[-1][0]
    return {
        "n_train_raw": n_train_raw,
        "n_train": int(train_t["value"].shape[0]),
        "augment": bool(augment),
        "n_val": int(len(val_idx)),
        "n_games": int(np.unique(game_id).size),
        "params": count_params(model),
        "epochs": epochs,
        "train_loss_first": round(first_tr, 6),
        "train_loss_last": round(last_tr, 6),
        "val_loss_best": round(best_val, 6) if val_t is not None else None,
        "train_sign_acc": round(last_tr_acc, 4),
        "val_sign_acc": round(last_va_acc, 4) if val_t is not None else None,
        "onnx_round_trip_max_diff": onnx_diff,
        "checkpoint": str(ckpt_path),
        "onnx": str(onnx_path),
    }


# ============================================================================
# PolicyValueNet (H017 dual head) 学習 path
# ============================================================================
#
# value MSE + policy cross-entropy の combined loss で `PolicyValueNet` を学習する。
# policy target は MCTS root visit 分布を「launch 元 planet」に集約した確率分布
# (npz key `policy_target`, shape (N, POLICY_DIM) = MAX_PLANETS launch-source + 1 no-op)。
# index 規約: i ∈ [0, MAX_PLANETS) は encoder の planet slot i (= planet_tokens[i]) に対応、
# index MAX_PLANETS は no-op。padding planet slot の target は 0、行は valid 範囲で総和 1。
# この index 規約は model.PolicyValueNet の policy_logits 並びと 1:1 (train/serve skew 回避)。
#
# 既存 ValueNet path (train/_epoch/export_onnx) は一切変更せず、PV は独立 sibling として
# 追加する (checkpoint `value_net_c5.*` と value_infer.py の ValueNet 依存を保護)。

# value/policy の混合比 (AlphaZero 慣習: 両 loss を同オーダで加算)。
DEFAULT_POLICY_WEIGHT = 1.0

PV_INPUT_NAMES = INPUT_NAMES  # forward の入力署名は value-only と共通


def _policy_cross_entropy(logits: Tensor, target: Tensor) -> Tensor:
    """masked policy logit に対する soft-target cross-entropy (= KL + target entropy)。

    target は確率分布 (行和 1)。padding slot の target は 0 ゆえ、model 側で _MASK_NEG
    (有限大負値) に詰められた logit の log_softmax (有限) と 0 を掛けて寄与 0 になる
    (-inf を使わないため NaN は発生しない)。
    """
    log_probs = torch.log_softmax(logits, dim=1)  # (B, POLICY_DIM)
    return -(target * log_probs).sum(dim=1).mean()


def _epoch_pv(
    model: PolicyValueNet,
    tensors: dict[str, Tensor],
    batch_size: int,
    policy_weight: float,
    optimizer: torch.optim.Optimizer | None,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """PV 1 epoch。optimizer=None で eval。(mean_total_loss, value_sign_acc, mean_policy_ce)。"""
    n = tensors["value"].shape[0]
    order = np.arange(n)
    train = optimizer is not None
    if train:
        rng.shuffle(order)
    model.train(train)

    total_loss = 0.0
    total_pce = 0.0
    correct = 0
    counted = 0
    value_loss_fn = nn.MSELoss()
    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for start in range(0, n, batch_size):
            sel = order[start : start + batch_size]
            batch = {name: tensors[name][sel] for name in PV_INPUT_NAMES}
            v_target = tensors["value"][sel]
            p_target = tensors["policy_target"][sel]
            v_pred, p_logits = model(**batch)
            v_loss = value_loss_fn(v_pred, v_target)
            p_ce = _policy_cross_entropy(p_logits, p_target)
            loss = v_loss + policy_weight * p_ce
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss) * len(sel)
            total_pce += float(p_ce) * len(sel)
            nz = v_target != 0
            if nz.any():
                correct += int(((v_pred[nz] > 0) == (v_target[nz] > 0)).sum())
                counted += int(nz.sum())
    mean_loss = total_loss / max(1, n)
    mean_pce = total_pce / max(1, n)
    sign_acc = correct / counted if counted else float("nan")
    return mean_loss, sign_acc, mean_pce


def export_onnx_pv(
    model: PolicyValueNet, sample: dict[str, Tensor], out_path: Path
) -> tuple[float, float]:
    """学習済み PV を dual-output ONNX export し (value_diff, policy_diff) を返す。"""
    model.eval()
    cpu = {name: sample[name].detach().cpu() for name in PV_INPUT_NAMES}
    with torch.no_grad():
        ref_v, ref_p = model(**cpu)
        ref_v = ref_v.numpy()
        ref_p = ref_p.numpy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        tuple(cpu[name] for name in PV_INPUT_NAMES),
        str(out_path),
        input_names=PV_INPUT_NAMES,
        output_names=["value", "policy_logits"],
        dynamic_axes={name: {0: "batch"} for name in PV_INPUT_NAMES}
        | {"value": {0: "batch"}, "policy_logits": {0: "batch"}},
        opset_version=17,
    )
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_v, ort_p = sess.run(
        ["value", "policy_logits"], {name: cpu[name].numpy() for name in PV_INPUT_NAMES}
    )
    return float(np.abs(ort_v - ref_v).max()), float(np.abs(ort_p - ref_p).max())


def train_pv(
    data: dict,
    epochs: int,
    batch_size: int,
    lr: float,
    val_frac: float,
    seed: int,
    out_prefix: Path,
    policy_weight: float = DEFAULT_POLICY_WEIGHT,
) -> dict:
    """value+policy dataset から PolicyValueNet を学習し checkpoint + dual ONNX を書き出す。

    grouped split は ValueNet path と同一 (game 単位で leakage 回避)。augment は policy
    target の planet index 置換が必要なため本 step では非対応 (将来 step)。
    """
    if "policy_target" not in data:
        raise KeyError(
            "policy_target が dataset に無い (PolicyValueNet 学習には visit 分布 target が必要)"
        )
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    game_id = np.asarray(data["game_id"])
    train_idx, val_idx = _grouped_split(game_id, val_frac, seed)
    extra = ("policy_target",)
    train_t = _tensors_from_numpy(_slice_numpy(data, train_idx, extra), device, extra)
    val_t = (
        _tensors_from_numpy(_slice_numpy(data, val_idx, extra), device, extra)
        if len(val_idx)
        else None
    )

    model = PolicyValueNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    best_val = float("inf")
    ckpt_path = out_prefix.with_suffix(".pt")
    history = []
    last_tr_acc = float("nan")
    last_va_acc = float("nan")
    first_pce = last_pce = float("nan")
    val_pce = float("nan")
    for ep in range(epochs):
        tr_loss, last_tr_acc, tr_pce = _epoch_pv(
            model, train_t, batch_size, policy_weight, optimizer, rng
        )
        if ep == 0:
            first_pce = tr_pce
        last_pce = tr_pce
        if val_t is not None:
            va_loss, last_va_acc, val_pce = _epoch_pv(
                model, val_t, batch_size, policy_weight, None, rng
            )
        else:
            va_loss = float("nan")
        history.append((tr_loss, va_loss))
        score = va_loss if val_t is not None else tr_loss
        if score < best_val:
            best_val = score
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"state_dict": model.state_dict(), "epoch": ep, "val_loss": va_loss},
                ckpt_path,
            )

    onnx_path = out_prefix.with_suffix(".onnx")
    sample = (
        {name: val_t[name][:2] for name in PV_INPUT_NAMES}
        if val_t is not None
        else {name: train_t[name][:2] for name in PV_INPUT_NAMES}
    )
    onnx_v_diff, onnx_p_diff = export_onnx_pv(model, sample, onnx_path)

    first_tr, last_tr = history[0][0], history[-1][0]
    return {
        "model": "PolicyValueNet",
        "policy_weight": policy_weight,
        "n_train": int(train_t["value"].shape[0]),
        "n_val": int(len(val_idx)),
        "n_games": int(np.unique(game_id).size),
        "params": count_params(model),
        "epochs": epochs,
        "train_loss_first": round(first_tr, 6),
        "train_loss_last": round(last_tr, 6),
        "val_loss_best": round(best_val, 6) if val_t is not None else None,
        "train_policy_ce_first": round(first_pce, 6),
        "train_policy_ce_last": round(last_pce, 6),
        "val_policy_ce_last": round(val_pce, 6) if val_t is not None else None,
        "train_sign_acc": round(last_tr_acc, 4),
        "val_sign_acc": round(last_va_acc, 4) if val_t is not None else None,
        "onnx_value_max_diff": onnx_v_diff,
        "onnx_policy_max_diff": onnx_p_diff,
        "checkpoint": str(ckpt_path),
        "onnx": str(onnx_path),
    }


def _make_synthetic(n_games: int, per_game: int, seed: int, with_policy: bool = False) -> dict:
    """smoke 用 tiny synthetic dataset。学習可能な信号を仕込み収束を確認する。

    with_policy=True で PolicyValueNet 用の policy_target (visit 分布を模した確率分布) も
    付与する (valid planet の feature[0] 最大 slot に質量集中 = NN が学習可能な信号)。

    各 game に固定 value (-1/+1) を割り当て、global_features[0] にその符号を弱く
    埋め込む (NN が拾える線形信号)。同一 game の全 timestep は同一 value =
    実 exporter の MC-outcome 設計と同じ性質を持たせ grouped split を検証可能にする。
    """
    from src.features.encoder import (
        FLEET_FEATURES,
        GLOBAL_FEATURES,
        MAX_FLEETS,
        MAX_PLANETS,
        PLANET_FEATURES,
    )

    rng = np.random.default_rng(seed)
    pt, pm, ft, fm, gf, val, gid, ptg = [], [], [], [], [], [], [], []
    for g in range(n_games):
        v = 1.0 if g % 2 == 0 else -1.0
        for _ in range(per_game):
            planet_tokens = rng.standard_normal((MAX_PLANETS, PLANET_FEATURES)).astype(np.float32)
            planet_mask = np.zeros((MAX_PLANETS,), dtype=np.float32)
            n_valid = int(rng.integers(1, MAX_PLANETS))
            planet_mask[:n_valid] = 1.0
            fleet_tokens = rng.standard_normal((MAX_FLEETS, FLEET_FEATURES)).astype(np.float32)
            fleet_mask = np.zeros((MAX_FLEETS,), dtype=np.float32)
            fleet_mask[: rng.integers(0, MAX_FLEETS)] = 1.0
            g_feat = rng.standard_normal((GLOBAL_FEATURES,)).astype(np.float32)
            g_feat[0] = v * 0.8  # 学習可能な信号 (value の符号を弱く反映)
            pt.append(planet_tokens)
            pm.append(planet_mask)
            ft.append(fleet_tokens)
            fm.append(fleet_mask)
            gf.append(g_feat)
            val.append(v)
            gid.append(g)
            if with_policy:
                # 学習可能な policy 信号: valid planet のうち feature[0] 最大の slot に質量
                # 0.8、no-op に 0.1、残り 0.1 を valid planet に一様。padding slot は 0。
                target = np.zeros((POLICY_DIM,), dtype=np.float32)
                valid_feat = planet_tokens[:n_valid, 0]
                best = int(np.argmax(valid_feat))
                target[:n_valid] = 0.1 / n_valid
                target[best] += 0.8
                target[MAX_PLANETS] = 0.1  # no-op
                target /= target.sum()
                ptg.append(target)
    out = {
        "planet_tokens": np.stack(pt),
        "planet_mask": np.stack(pm),
        "fleet_tokens": np.stack(ft),
        "fleet_mask": np.stack(fm),
        "global_features": np.stack(gf),
        "value": np.asarray(val, dtype=np.float32),
        "game_id": np.asarray(gid, dtype=np.int32),
    }
    if with_policy:
        out["policy_target"] = np.stack(ptg)
    return out


def _smoke_pv() -> None:
    """PolicyValueNet dual-head 学習を tiny synthetic data で検証 (H017)。

    value/policy 両 head が収束し (loss 減少)、policy_target の masked slot 安全性
    (NaN なし) と dual ONNX round-trip を assert する。
    """
    import tempfile

    from src.features.encoder import MAX_PLANETS

    data = _make_synthetic(n_games=8, per_game=12, seed=0, with_policy=True)
    assert "policy_target" in data, "policy_target が生成されていない"
    pt = data["policy_target"]
    assert pt.shape[1] == POLICY_DIM == MAX_PLANETS + 1, f"policy_target 次元不正: {pt.shape}"
    # 行和 1 / padding slot は 0 (mask 整合)
    assert np.allclose(pt.sum(axis=1), 1.0, atol=1e-5), "policy_target が確率分布でない"
    pad = data["planet_mask"] < 0.5
    assert float(pt[:, :MAX_PLANETS][pad].max()) == 0.0, "padding planet slot の target が非ゼロ"

    with tempfile.TemporaryDirectory() as tmp:
        stats = train_pv(
            data,
            epochs=60,
            batch_size=32,
            lr=1e-3,
            val_frac=0.25,
            seed=0,
            out_prefix=Path(tmp) / "policy_value_net",
        )
        assert stats["model"] == "PolicyValueNet"
        # value MSE + policy CE の combined loss が明確に減少 (両 head 学習可能)
        assert (
            stats["train_loss_last"] < stats["train_loss_first"] * 0.7
        ), f"combined loss 収束せず: {stats['train_loss_first']} -> {stats['train_loss_last']}"
        # policy CE 単独でも減少 (policy head が信号を学習)
        assert (
            stats["train_policy_ce_last"] < stats["train_policy_ce_first"] * 0.9
        ), f"policy CE 収束せず: {stats['train_policy_ce_first']} -> {stats['train_policy_ce_last']}"
        assert np.isfinite(stats["train_loss_last"]), "loss に NaN (masked CE が壊れている)"
        assert Path(stats["checkpoint"]).exists() and Path(stats["onnx"]).exists()
        assert (
            stats["onnx_value_max_diff"] < 1e-4
        ), f"ONNX value 不一致: {stats['onnx_value_max_diff']}"
        # _MASK_NEG (大絶対値) を含むため policy は相対許容
        assert (
            stats["onnx_policy_max_diff"] < 1e-2
        ), f"ONNX policy 不一致: {stats['onnx_policy_max_diff']}"
    print(
        f"  PolicyValueNet OK: combined loss {stats['train_loss_first']:.4f} -> "
        f"{stats['train_loss_last']:.4f}, policy CE {stats['train_policy_ce_first']:.4f} -> "
        f"{stats['train_policy_ce_last']:.4f}, params {stats['params']:,}, "
        f"ONNX v/p diff {stats['onnx_value_max_diff']:.2e}/{stats['onnx_policy_max_diff']:.2e}"
    )


def _smoke() -> int:
    """tiny synthetic data で pipeline (grouped split → 学習収束 → ONNX) を検証。"""
    import tempfile

    data = _make_synthetic(n_games=8, per_game=12, seed=0)

    # grouped split が game をまたがず分割していること (leakage 検証)
    tr_idx, va_idx = _grouped_split(data["game_id"], val_frac=0.25, seed=0)
    tr_games = set(data["game_id"][tr_idx].tolist())
    va_games = set(data["game_id"][va_idx].tolist())
    assert tr_games and va_games, "split が片側空"
    assert tr_games.isdisjoint(va_games), f"game leakage: {tr_games & va_games}"
    assert len(va_idx) > 0, "val が空"
    print(f"  grouped split OK: train_games={len(tr_games)} val_games={len(va_games)} (disjoint)")

    with tempfile.TemporaryDirectory() as tmp:
        stats = train(
            data,
            epochs=40,
            batch_size=32,
            lr=1e-3,
            val_frac=0.25,
            seed=0,
            out_prefix=Path(tmp) / "value_net",
        )
        # 学習収束: 信号を仕込んだので train loss が明確に下がること
        assert (
            stats["train_loss_last"] < stats["train_loss_first"] * 0.7
        ), f"学習が収束せず: {stats['train_loss_first']} -> {stats['train_loss_last']}"
        # checkpoint / onnx が実在
        assert Path(stats["checkpoint"]).exists(), "checkpoint 未保存"
        assert Path(stats["onnx"]).exists(), "ONNX 未保存"
        # ONNX round-trip (serve skew 検証)
        assert (
            stats["onnx_round_trip_max_diff"] < 1e-4
        ), f"ONNX round-trip 不一致: {stats['onnx_round_trip_max_diff']}"

        # --augment 検証 (H018): train が 4 倍・収束する・val game は非拡張のまま
        aug = train(
            data,
            epochs=40,
            batch_size=32,
            lr=1e-3,
            val_frac=0.25,
            seed=0,
            out_prefix=Path(tmp) / "value_net_aug",
            augment=True,
        )
        assert (
            aug["n_train"] == 4 * aug["n_train_raw"]
        ), f"augment 後 train が 4N でない: {aug['n_train_raw']} -> {aug['n_train']}"
        assert aug["n_val"] == stats["n_val"], "val が augment で変化 (拡張対象外のはず)"
        assert aug["augment"] is True
        assert (
            aug["train_loss_last"] < aug["train_loss_first"] * 0.7
        ), f"augment 学習が収束せず: {aug['train_loss_first']} -> {aug['train_loss_last']}"
        print(
            f"  augment OK: n_train {aug['n_train_raw']} -> {aug['n_train']} (4x), "
            f"val={aug['n_val']} unchanged, loss {aug['train_loss_first']:.4f} -> {aug['train_loss_last']:.4f}"
        )
    import json

    print(
        f"  train OK: {json.dumps({k: stats[k] for k in ('train_loss_first','train_loss_last','val_loss_best','onnx_round_trip_max_diff')})}"
    )

    # PolicyValueNet (H017 dual head) の学習 path も併せて検証
    _smoke_pv()

    print("train_value smoke OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/value_dataset.npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/checkpoints/value_net")
    ap.add_argument(
        "--augment",
        action="store_true",
        help="train サブセットを 90/180/270° 回転で 4 倍に拡張 (H018、val は非拡張)",
    )
    ap.add_argument(
        "--policy",
        action="store_true",
        help="PolicyValueNet (dual head) を value MSE + policy CE で学習 (H017、要 policy_target、Kaggle GPU)",
    )
    ap.add_argument(
        "--policy-weight",
        type=float,
        default=DEFAULT_POLICY_WEIGHT,
        help="policy CE の混合比 (--policy 時のみ)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny synthetic data で grouped split・学習収束・ONNX round-trip を検証 (書き出しは tmp)",
    )
    args = ap.parse_args()

    if args.smoke:
        return _smoke()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = _ROOT / data_path
    npz = np.load(data_path)
    data = {k: npz[k] for k in npz.files}
    out_prefix = _ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    if args.policy:
        stats = train_pv(
            data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_frac=args.val_frac,
            seed=args.seed,
            out_prefix=out_prefix,
            policy_weight=args.policy_weight,
        )
    else:
        stats = train(
            data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_frac=args.val_frac,
            seed=args.seed,
            out_prefix=out_prefix,
            augment=args.augment,
        )
    import json

    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
