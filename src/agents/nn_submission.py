"""route (b) NN-in-MCTS 提出 bundle 用 agent factory (H038/exp073)。

H037 (exp/072) 確定次手 (ii): NN-in-MCTS (c5 int8 value leaf) を実 submission path に
wire し beam 435.3 baseline と **LB で答え合わせ**する。local strong-opponent eval は
両方向 noisy (NN value leaf は plain-MCTS regression を治すが beam baseline を超えない =
`nn_value_leaf_heals_mcts_strong_opponent_regression`) ゆえ真の judge は LB delta 単独
(`mixeval_prevbest_gate_noisy_misjudges_lb`)。

なぜ factory が要るか:
  live main.py は NN-in-MCTS を env var (`ORBIT_WARS_MCTS=1` 等) でしか有効化できないが、
  Kaggle 提出に env を渡す手段が無い。本 factory は bundle 専用 main.py から呼ばれ、
  env var 無しで NN-in-MCTS を起動する (NN env を自前で os.environ に set)。

downside-bounded 設計 (DQ / regression ゼロが絶対制約):
  - 初期化時に onnxruntime import + ONNX session 生成を probe し、不可なら **beam-only**
    agent を返す (= LB 435.3 baseline を完全再現)。Kaggle 実機の onnxruntime 有無は
    local 検証不能だが、不在なら beam に落ちるため downside は 435.3 に bound される。
  - NN 利用可能時も、per-turn で NN-in-MCTS が例外を投げたら beam → safe_act の順に
    fallback (safe_act 単独に落とさず beam を挟むことで弱化を防ぐ)。
  - actTimeout 1s/turn: deadline 0.90s (100ms margin)、core が 950ms 超過なら fallback。

agent 不変: live main.py は本 module を import しない。本 module を使うのは
build_nn_submission.py が生成する bundle 専用 main.py のみ (no-submit infra)。
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.agents.safe_fallback import act as _safe_act
from src.search.beam import search as _beam_search
from src.search.mcts import search as _mcts_search
from src.utils import action as _action
from src.utils import telemetry as _tel
from src.utils.timing import Timer

Agent = Callable[[Any], list]

# bundle 同梱モデルの既定ファイル名。build_nn_submission.py が bundle root にこの名前で
# c5 int8 onnx を配置する。repo 内実行 (smoke) では experiments/checkpoints 配下も探索。
_MODEL_FILENAME = "value_net_c5.int8.onnx"
_MCTS_TIME_BUDGET_SEC = 0.3
_TURN_DEADLINE_SEC = 0.90  # 100ms margin under 1.0s actTimeout


def _resolve_model_path() -> str | None:
    """NN value model の path を解決する (env override > bundle root > repo checkpoints)。"""
    env_path = os.environ.get("ORBIT_WARS_NN_VALUE_MODEL", "").strip()
    if env_path and Path(env_path).is_file():
        return env_path

    candidates: list[Path] = []
    try:
        here = Path(__file__).resolve()
        # bundle layout: <root>/main.py, <root>/src/agents/nn_submission.py, <root>/<model>
        bundle_root = here.parents[2]
        candidates.append(bundle_root / _MODEL_FILENAME)
        candidates.append(bundle_root / "experiments" / "checkpoints" / _MODEL_FILENAME)
    except NameError:
        # kaggle exec context: __file__ 不在。cwd 基準で探す。
        candidates.append(Path.cwd() / _MODEL_FILENAME)

    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def _nn_available(model_path: str) -> bool:
    """onnxruntime import + ONNX session 生成が可能かを probe する (downside bound)。"""
    try:
        import onnxruntime as ort  # noqa: F401

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        return True
    except Exception:  # noqa: BLE001 — import/load 失敗は全て beam fallback に倒す
        return False


def _beam_agent(obs: Any) -> list:
    """LB 435.3 baseline と同等の beam-only agent (NN unavailable 時の fallback path)。"""
    turn = int(obs.get("step", 0)) if isinstance(obs, dict) else getattr(obs, "step", 0)
    deadline = time.monotonic() + _TURN_DEADLINE_SEC
    player = int(obs.get("player", 0)) if isinstance(obs, dict) else int(getattr(obs, "player", 0))

    fallback: list = []
    try:
        fallback = _safe_act(obs)
    except Exception as e:  # noqa: BLE001
        _tel.error(turn, "safe_fallback raised", err=str(e))
        return []

    try:
        with Timer() as t_core:
            remaining = max(0.0, deadline - time.monotonic())
            result = _beam_search(obs, player, time_budget_sec=min(0.3, remaining))
        if t_core.elapsed_ms > 950:
            _tel.error(turn, "beam slow timeout-risk", elapsed_ms=t_core.elapsed_ms)
            return _action.sanitize(fallback, obs)
        return _action.sanitize(result, obs)
    except Exception as e:  # noqa: BLE001
        _tel.error(turn, "beam raised", err=str(e))
        return _action.sanitize(fallback, obs)


def _nn_mcts_agent(obs: Any) -> list:
    """NN-in-MCTS agent。例外時は beam → safe_act の順に fallback (downside bound)。"""
    turn = int(obs.get("step", 0)) if isinstance(obs, dict) else getattr(obs, "step", 0)
    deadline = time.monotonic() + _TURN_DEADLINE_SEC
    player = int(obs.get("player", 0)) if isinstance(obs, dict) else int(getattr(obs, "player", 0))

    fallback: list = []
    try:
        fallback = _safe_act(obs)
    except Exception as e:  # noqa: BLE001
        _tel.error(turn, "safe_fallback raised", err=str(e))
        return []

    try:
        with Timer() as t_core:
            remaining = max(0.0, deadline - time.monotonic())
            budget = min(_MCTS_TIME_BUDGET_SEC, remaining)
            if budget <= 0.05:
                return _action.sanitize(fallback, obs)
            result = _mcts_search(obs, player, time_budget_sec=budget)
        if t_core.elapsed_ms > 950:
            _tel.error(turn, "nn-mcts slow timeout-risk", elapsed_ms=t_core.elapsed_ms)
            return _action.sanitize(fallback, obs)
        _tel.log(turn, core_ms=round(t_core.elapsed_ms, 2), n_moves=len(result))
        return _action.sanitize(result, obs)
    except Exception as e:  # noqa: BLE001
        # NN-in-MCTS が壊れても safe_act でなく beam に落として弱化を防ぐ。
        _tel.error(turn, "nn-mcts raised, beam fallback", err=str(e))
        return _beam_agent(obs)


def make_agent() -> Agent:
    """提出用 agent を返す。NN が使えれば NN-in-MCTS、使えなければ beam-only。

    NN 使用時は MCTS / NN value の env flag を os.environ に set する (mcts.py は
    呼出時に os.environ.get で読むため、ここで set すれば env var 無しで有効化される)。
    policy は OFF: c5 は value-only モデルで、H037 が「治すが超えない」と測定した構成
    (ORBIT_WARS_NN_POLICY="") を提出 path で再現する。
    """
    model_path = _resolve_model_path()
    if model_path is None or not _nn_available(model_path):
        return _beam_agent

    os.environ["ORBIT_WARS_MCTS"] = "1"
    os.environ["ORBIT_WARS_NN_VALUE"] = "1"
    os.environ["ORBIT_WARS_NN_VALUE_MODEL"] = model_path
    os.environ.setdefault("ORBIT_WARS_NN_POLICY", "")  # value-only (H037 構成)
    return _nn_mcts_agent
