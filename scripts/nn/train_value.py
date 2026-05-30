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

from src.nn.model import ValueNet, count_params  # noqa: E402

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


def _load_tensors(data: dict, idx: np.ndarray, device: torch.device) -> dict[str, Tensor]:
    """npz の指定 index 群を ValueNet forward 用 tensor dict + target に束ねる。"""
    out = {name: torch.from_numpy(np.asarray(data[name])[idx]).to(device) for name in INPUT_NAMES}
    out["value"] = torch.from_numpy(np.asarray(data["value"])[idx].astype(np.float32)).to(device)
    return out


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
) -> dict:
    """value dataset から ValueNet を学習し checkpoint + ONNX を書き出す。"""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    game_id = np.asarray(data["game_id"])
    train_idx, val_idx = _grouped_split(game_id, val_frac, seed)
    train_t = _load_tensors(data, train_idx, device)
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
        "n_train": int(len(train_idx)),
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


def _make_synthetic(n_games: int, per_game: int, seed: int) -> dict:
    """smoke 用 tiny synthetic dataset。学習可能な信号を仕込み収束を確認する。

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
    pt, pm, ft, fm, gf, val, gid = [], [], [], [], [], [], []
    for g in range(n_games):
        v = 1.0 if g % 2 == 0 else -1.0
        for _ in range(per_game):
            planet_tokens = rng.standard_normal((MAX_PLANETS, PLANET_FEATURES)).astype(np.float32)
            planet_mask = np.zeros((MAX_PLANETS,), dtype=np.float32)
            planet_mask[: rng.integers(1, MAX_PLANETS)] = 1.0
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
    return {
        "planet_tokens": np.stack(pt),
        "planet_mask": np.stack(pm),
        "fleet_tokens": np.stack(ft),
        "fleet_mask": np.stack(fm),
        "global_features": np.stack(gf),
        "value": np.asarray(val, dtype=np.float32),
        "game_id": np.asarray(gid, dtype=np.int32),
    }


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
    import json

    print(
        f"  train OK: {json.dumps({k: stats[k] for k in ('train_loss_first','train_loss_last','val_loss_best','onnx_round_trip_max_diff')})}"
    )
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
    stats = train(
        data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_frac=args.val_frac,
        seed=args.seed,
        out_prefix=out_prefix,
    )
    import json

    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
