"""ONNX value head の量子化 harness (H016 infra step 5)。

`train_value.py` が export した FP32 ONNX (`ValueNet`) を **int8 dynamic quantization**
で量子化し、CPU 推論の latency / size / 精度劣化を計測する。提出 agent は CPU・
1秒/turn 制約下で走る (CLAUDE.md「NN inference は ONNX export & quantize 必須」)。

なぜ FP16 でなく int8 か (advisor 指摘):
- ONNX Runtime の **CPUExecutionProvider は FP16 ネイティブ kernel が乏しい**。FP16
  model を読むと各 op の前後に Cast(FP16↔FP32) が挿入され、**latency は縮まらず
  むしろ悪化** することが多い。効くのは file size 半減のみ。
- ValueNet は Linear / MatMul (attention + feed-forward) 主体で、これは int8 dynamic
  quantization が最も効く形状。CPU で size 約 1/4 + speedup が狙える。
- このステップの存在意義は「NN を 1秒/turn 内に載せられるか」の検証なので、smoke は
  **CPU 推論 latency を必ず計測** する (size + round-trip だけでは無回答)。

実測の結論 (exp/036 smoke、最重要 — MCTS 統合 step はこのデータで serving 形式を選ぶ):
- ValueNet は 210K params (d=64) と **小さく、FP32 でも CPU batch=1 で ~0.34ms/query**。
  1秒/turn 予算 (1000ms) に対し桁違いの余裕で、MCTS で数百〜千 state/turn の value 評価が
  可能。**NN を 1秒/turn 内に載せられるか = 余裕で YES** (H016 の生死問題に回答)。
- この規模では **int8 dynamic quantization は逆効果**: 量子化ノード
  (DynamicQuantizeLinear / MatMulInteger + scale/zero-point) のオーバーヘッドが weight
  削減を上回り、ONNX が **大きく (size ratio ~1.23)・遅く (speedup ~0.92)** なる。
  int8 が効くのは数 M params 以上の大型 model。本 head には不要。
- よって smoke は「量子化が必ず縮む/速い」を assert **しない** (実測に反する偽の期待)。
  代わりに (a) FP32 latency が turn 予算内 (b) int8 ONNX が生成され有限出力を返す
  (c) 比較計測が走る、を検証する。MCTS 統合 step は **FP32 ONNX を採用** が現状の推奨。

精度判定 (int8 の現実的 tolerance):
- int8 量子化の round-trip 誤差は FP32 round-trip (1e-4) には収まらない。abs diff の
  絶対値より、MCTS が value で state を ranking する用途上 **符号保存 (sign agreement)**
  が本質なので参考計測する (untrained synthetic は出力が 0 近傍で符号 noise が大きいため
  smoke では情報出力に留め、本番学習済み model で実測する)。

設計方針:
- **agent 不変**: main.py / beam.py からは import しない純 module。MCTS への value
  統合は後続 step。この iter は +0 基盤 (encoder/exporter/model/train step と同じ
  infra_merge ライン)。
- **二段構え** (train_value.py 踏襲): 本番は Kaggle 学習済み実 ONNX を CLI で量子化、
  ローカルは `--smoke` で tiny synthetic ONNX を export → 量子化 → 計測まで検証。

CLI:
    .venv/bin/python scripts/nn/quantize_onnx.py \
        --in experiments/checkpoints/value_net.onnx \
        --out experiments/checkpoints/value_net.int8.onnx
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 1秒/turn 制約 (CLAUDE.md)。NN value は探索 budget の一部なので、単 query latency が
# この上限を大きく下回ることを smoke で保証する (margin を取り 50ms を check 閾値とする)。
TURN_BUDGET_MS = 1000.0
SMOKE_LATENCY_CEIL_MS = 50.0

# train_value.py と同一の input 署名 (serve skew 回避の要)。
INPUT_NAMES = [
    "planet_tokens",
    "planet_mask",
    "fleet_tokens",
    "fleet_mask",
    "global_features",
]


def quantize_int8(in_path: Path, out_path: Path) -> None:
    """FP32 ONNX を int8 dynamic quantization で量子化し out_path に書き出す。"""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(in_path),
        model_output=str(out_path),
        weight_type=QuantType.QInt8,
    )


def _make_feed(rng: np.random.Generator, batch: int) -> dict[str, np.ndarray]:
    """ValueNet forward 用の合成入力 feed を作る (encoder の shape 定数に従う)。"""
    from src.features.encoder import (
        FLEET_FEATURES,
        GLOBAL_FEATURES,
        MAX_FLEETS,
        MAX_PLANETS,
        PLANET_FEATURES,
    )

    def mask(n_max: int, lo: int) -> np.ndarray:
        m = np.zeros((batch, n_max), dtype=np.float32)
        for b in range(batch):
            m[b, : rng.integers(lo, n_max + 1)] = 1.0
        return m

    return {
        "planet_tokens": rng.standard_normal((batch, MAX_PLANETS, PLANET_FEATURES)).astype(
            np.float32
        ),
        "planet_mask": mask(MAX_PLANETS, 1),
        "fleet_tokens": rng.standard_normal((batch, MAX_FLEETS, FLEET_FEATURES)).astype(np.float32),
        "fleet_mask": mask(MAX_FLEETS, 0),
        "global_features": rng.standard_normal((batch, GLOBAL_FEATURES)).astype(np.float32),
    }


def _measure_latency(
    sess_path: Path, feed: dict[str, np.ndarray], iters: int, warmup: int
) -> tuple[np.ndarray, float]:
    """CPUExecutionProvider で推論し (出力, 1 回あたり平均 ms) を返す。"""
    import onnxruntime as ort

    so = ort.SessionOptions()
    # latency 計測の再現性を上げるため intra-op 並列を 1 に固定 (turn 予算は単 query)
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(sess_path), sess_options=so, providers=["CPUExecutionProvider"])
    out = sess.run(["value"], feed)[0]
    for _ in range(warmup):
        sess.run(["value"], feed)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = sess.run(["value"], feed)[0]
    elapsed_ms = (time.perf_counter() - t0) / iters * 1000.0
    return out, elapsed_ms


def _onnx_output_names(path: Path) -> list[str]:
    """ONNX graph の出力名一覧を返す (dual-head=PolicyValueNet 検出用)。"""
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    return [o.name for o in model.graph.output]


def _run_session(path: Path, feed: dict[str, np.ndarray], output_names: list[str]) -> list:
    """CPUExecutionProvider で指定出力を 1 回推論する (latency 計測なし)。"""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    return sess.run(output_names, feed)


def _verify_policy_survives(fp32_path: Path, int8_path: Path, feed: dict[str, np.ndarray]) -> dict:
    """dual-head ONNX の policy_logits が int8 量子化後も健全か検証する。

    MCTS PUCT prior は policy_logits を softmax して root child prior に配線するため、
    int8 量子化で policy head が壊れる (NaN/Inf・masked padding の崩壊) と、value だけを
    見る既存検証では検出できず stage3 が success を返し、supervisor の GPU run が壊れた
    prior を serve してしまう。本検証は value 検証の盲点 (policy 出力) を埋める。

    policy index 規約 (train_value.py / encoder と 1:1): slot [0:MAX_PLANETS) は planet
    token に対応し planet_mask==0 (padding) は _MASK_NEG で softmax 後 ~0、slot MAX_PLANETS
    は no-op (常時 valid)。
    """
    fp32_logits = np.asarray(_run_session(fp32_path, feed, ["policy_logits"])[0], dtype=np.float64)
    int8_logits = np.asarray(_run_session(int8_path, feed, ["policy_logits"])[0], dtype=np.float64)

    def _softmax(x: np.ndarray) -> np.ndarray:
        z = x - x.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    int8_probs = _softmax(int8_logits)
    planet_mask = np.asarray(feed["planet_mask"])  # (batch, MAX_PLANETS)
    n_planet = planet_mask.shape[1]
    pad = planet_mask == 0.0  # padding planet = policy slot で prob ~0 であるべき
    padding_prob = (int8_probs[:, :n_planet] * pad).sum(axis=-1)
    return {
        "policy_dim": int(int8_logits.shape[-1]),
        "policy_int8_finite": bool(np.isfinite(int8_logits).all()),
        "policy_softmax_sum_dev": round(float(np.abs(int8_probs.sum(axis=-1) - 1.0).max()), 6),
        "policy_padding_prob_max": round(float(padding_prob.max()) if pad.any() else 0.0, 6),
        "policy_max_abs_diff": round(float(np.abs(int8_logits - fp32_logits).max()), 5),
    }


def evaluate(
    fp32_path: Path,
    int8_path: Path,
    batch: int,
    iters: int,
    warmup: int,
    seed: int,
) -> dict:
    """FP32 と int8 ONNX を同一 feed で比較し latency/size/精度を集計する。

    dual-head ONNX (PolicyValueNet: value + policy_logits) を検出した場合は policy_logits
    の int8 量子化生存も併せて検証する (`_verify_policy_survives`)。
    """
    rng = np.random.default_rng(seed)
    feed = _make_feed(rng, batch)

    fp32_out, fp32_ms = _measure_latency(fp32_path, feed, iters, warmup)
    int8_out, int8_ms = _measure_latency(int8_path, feed, iters, warmup)

    fp32_out = np.asarray(fp32_out).reshape(-1)
    int8_out = np.asarray(int8_out).reshape(-1)
    max_abs_diff = float(np.abs(int8_out - fp32_out).max())
    # sign agreement: 0 近傍 (|fp32|<0.05) は符号が不安定なので判定対象外
    decisive = np.abs(fp32_out) >= 0.05
    if decisive.any():
        sign_agree = float((np.sign(int8_out[decisive]) == np.sign(fp32_out[decisive])).mean())
    else:
        sign_agree = float("nan")

    fp32_kb = fp32_path.stat().st_size / 1024.0
    int8_kb = int8_path.stat().st_size / 1024.0
    result = {
        "batch": batch,
        "iters": iters,
        "fp32_latency_ms": round(fp32_ms, 4),
        "int8_latency_ms": round(int8_ms, 4),
        "speedup": round(fp32_ms / int8_ms, 3) if int8_ms > 0 else None,
        "fp32_size_kb": round(fp32_kb, 1),
        "int8_size_kb": round(int8_kb, 1),
        "size_ratio": round(int8_kb / fp32_kb, 3) if fp32_kb > 0 else None,
        "max_abs_diff": round(max_abs_diff, 5),
        "sign_agreement": round(sign_agree, 4),
        "n_decisive": int(decisive.sum()),
        # 1秒/turn 予算で評価可能な state 数 (FP32 単 query 基準の目安)
        "fp32_evals_per_turn": int(TURN_BUDGET_MS / fp32_ms) if fp32_ms > 0 else None,
    }

    # dual-head ONNX (PolicyValueNet) なら policy_logits の量子化生存も検証する
    if "policy_logits" in _onnx_output_names(int8_path):
        result.update(_verify_policy_survives(fp32_path, int8_path, feed))
    return result


def _export_fp32_synthetic(out_path: Path, seed: int) -> None:
    """smoke 用: ValueNet を初期化し FP32 ONNX を export (train_value.py と同署名)。"""
    import torch

    from src.nn.model import ValueNet

    torch.manual_seed(seed)
    model = ValueNet().eval()
    rng = np.random.default_rng(seed)
    feed = _make_feed(rng, batch=2)
    args = tuple(torch.from_numpy(feed[name]) for name in INPUT_NAMES)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        args,
        str(out_path),
        input_names=INPUT_NAMES,
        output_names=["value"],
        dynamic_axes={name: {0: "batch"} for name in INPUT_NAMES} | {"value": {0: "batch"}},
        opset_version=17,
    )


def _export_dualhead_synthetic(out_path: Path, seed: int) -> None:
    """smoke 用: PolicyValueNet を初期化し dual-head (value + policy_logits) FP32 ONNX を
    export (H017。notebook stage3 が POLICY=1 で量子化する model と同署名)。"""
    import torch

    from src.nn.model import PolicyValueNet

    torch.manual_seed(seed)
    model = PolicyValueNet().eval()
    rng = np.random.default_rng(seed)
    feed = _make_feed(rng, batch=2)
    args = tuple(torch.from_numpy(feed[name]) for name in INPUT_NAMES)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        args,
        str(out_path),
        input_names=INPUT_NAMES,
        output_names=["value", "policy_logits"],
        dynamic_axes={name: {0: "batch"} for name in INPUT_NAMES}
        | {"value": {0: "batch"}, "policy_logits": {0: "batch"}},
        opset_version=17,
    )


def _smoke() -> int:
    """tiny synthetic ONNX で 量子化 → latency/size/sign-agreement を検証。"""
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        fp32_path = Path(tmp) / "value_net.onnx"
        int8_path = Path(tmp) / "value_net.int8.onnx"
        _export_fp32_synthetic(fp32_path, seed=0)
        quantize_int8(fp32_path, int8_path)
        assert int8_path.exists(), "int8 ONNX 未生成"

        # turn 推論は通常 batch=1。これが 1秒/turn 検証の本命なので batch=1 で測る。
        stats = evaluate(fp32_path, int8_path, batch=1, iters=80, warmup=10, seed=1)

        # (1) コア: FP32 単 query latency が turn 予算を大きく下回る (NN を 1秒/turn 内に
        #     載せられるか = H016 の生死問題)。値は ledger に記録。
        assert stats["fp32_latency_ms"] > 0, "FP32 latency 計測失敗"
        assert (
            stats["fp32_latency_ms"] < SMOKE_LATENCY_CEIL_MS
        ), f"FP32 latency が予算超過: {stats['fp32_latency_ms']}ms >= {SMOKE_LATENCY_CEIL_MS}ms"
        # (2) int8 ONNX が生成され有限出力を返す (量子化 capability の健全性)
        assert int8_path.exists(), "int8 ONNX 未生成"
        assert stats["int8_latency_ms"] > 0, "int8 latency 計測失敗"
        assert np.isfinite(stats["max_abs_diff"]), "int8 出力が非有限"
        # (3) FP32 vs int8 の乖離が破滅的でない (壊れた量子化を検出する緩い上限。
        #     int8 が「精度的に使える/使えない」の最終判定は本番学習済み model で行う)
        assert stats["max_abs_diff"] < 0.5, f"int8 誤差が破滅的: {stats['max_abs_diff']}"
        # NOTE: size_ratio / speedup は assert しない — この規模 (210K params) では int8 が
        # 逆効果 (size ratio ~1.23 / speedup ~0.92)。量子化が必ず効くという偽の期待を
        # smoke に焼き込まないため、これらは情報出力に留める (docstring「実測の結論」)。

        # (4) dual-head (H017 PolicyValueNet): policy_logits が int8 量子化後も生存するか
        #     検証する (PUCT prior の serve 健全性。value だけ見る上記検証の盲点を埋める)。
        dh_fp32 = Path(tmp) / "pv_net.onnx"
        dh_int8 = Path(tmp) / "pv_net.int8.onnx"
        _export_dualhead_synthetic(dh_fp32, seed=0)
        quantize_int8(dh_fp32, dh_int8)
        assert "policy_logits" in _onnx_output_names(dh_int8), "dual-head 出力が int8 で消失"
        # padding slot を確実に作るため planet 3 件のみ active な feed で検証
        dh_feed = _make_feed(np.random.default_rng(2), batch=1)
        dh_feed["planet_mask"] = np.zeros_like(dh_feed["planet_mask"])
        dh_feed["planet_mask"][:, :3] = 1.0
        pol = _verify_policy_survives(dh_fp32, dh_int8, dh_feed)
        assert pol["policy_int8_finite"], "int8 policy_logits が非有限"
        assert (
            pol["policy_softmax_sum_dev"] < 1e-4
        ), f"int8 policy softmax 総和が 1 でない: {pol['policy_softmax_sum_dev']}"
        assert (
            pol["policy_padding_prob_max"] < 1e-3
        ), f"int8 padding slot prob が ~0 でない (masked planet 漏れ): {pol['policy_padding_prob_max']}"

    print(f"  quantize OK: {json.dumps(stats)}")
    print(f"  dual-head policy survives int8: {json.dumps(pol)}")
    print("quantize_onnx smoke OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="experiments/checkpoints/value_net.onnx")
    ap.add_argument("--out", dest="out_path", default="experiments/checkpoints/value_net.int8.onnx")
    ap.add_argument("--batch", type=int, default=1, help="latency 計測の batch (turn 推論は通常 1)")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="tiny synthetic ONNX で量子化・latency・sign agreement を検証",
    )
    args = ap.parse_args()

    if args.smoke:
        return _smoke()

    in_path = Path(args.in_path)
    if not in_path.is_absolute():
        in_path = _ROOT / in_path
    out_path = Path(args.out_path)
    if not out_path.is_absolute():
        out_path = _ROOT / out_path

    quantize_int8(in_path, out_path)
    stats = evaluate(in_path, out_path, args.batch, args.iters, args.warmup, args.seed)
    import json

    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
