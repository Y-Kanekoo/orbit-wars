"""軽量NNポリシーの推論ラッパ（Phase4 以降で使用）。

提出エージェントから import される側。学習側（training/）には依存しない。
- CPU推論前提
- `models/*.pt` または `*.onnx` を遅延ロード
- actTimeout=1s 制限下で 10ms/inference を目標
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_MODEL_CACHE: dict[str, Any] = {}


def _model_path(name: str) -> Path:
    """提出tar.gz展開後のmodels/を優先し、無ければproject ./models/ を見る。"""
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent.parent / "models" / name,  # project root ./models/
        here.parent / "models" / name,  # tar.gz 内 agent/models/ に同梱する場合
        Path(os.environ.get("ORBIT_WARS_MODELS", "/kaggle/input/models")) / name,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"model not found: {name} (checked: {candidates})")


def load_torch_model(name: str) -> Any:
    """torch モデルを CPU で遅延ロード（初回のみ）。"""
    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name]
    import torch  # 遅延 import（提出サイズ/起動時間対策）

    model = torch.load(_model_path(name), map_location="cpu", weights_only=True)
    model.eval()
    _MODEL_CACHE[name] = model
    return model


def predict(model_name: str, features) -> Any:
    """特徴量→行動スコア。Phase4 で実装、現状は NotImplemented。"""
    raise NotImplementedError(
        "policy.inference.predict is a Phase4 placeholder. "
        "Use agent.strategy.* for Phase1-3."
    )
