"""route (b) NN-in-MCTS 提出 bundle agent factory の downside-bound 検証 (H038/exp073)。

検証対象 (DQ / regression ゼロが絶対制約):
- onnxruntime / model 不在時に make_agent() が beam-only agent に fallback する
  (= LB 435.3 baseline を再現、NN crash で agent が死なない)。
- いずれの path でも agent 出力が legal な move list 形式である。
- env.run の実走 (errors=0 / latency) は build_nn_submission.py の env.run smoke が担当
  (slow なため本 unit test には含めない)。
"""

from __future__ import annotations

from typing import Any

from src.agents import nn_submission


def _mk_obs() -> dict[str, Any]:
    return {
        "player": 0,
        "step": 0,
        "planets": [
            [0, 0, 10.0, 10.0, 1.0, 10, 1],
            [1, 1, 90.0, 90.0, 1.0, 10, 1],
            [2, -1, 50.0, 50.0, 1.0, 3, 1],
        ],
        "fleets": [],
        "angular_velocity": 0.03,
        "remainingOverageTime": 60.0,
    }


def _assert_legal_moves(result: Any) -> None:
    assert isinstance(result, list)
    for mv in result:
        assert isinstance(mv, list) and len(mv) == 3, f"bad move format: {mv}"


def test_falls_back_to_beam_when_model_missing(monkeypatch) -> None:
    """model path が解決できない時は beam-only agent を返す。"""
    monkeypatch.setattr(nn_submission, "_resolve_model_path", lambda: None)
    agent = nn_submission.make_agent()
    assert agent is nn_submission._beam_agent
    _assert_legal_moves(agent(_mk_obs()))


def test_falls_back_to_beam_when_onnxruntime_unavailable(monkeypatch) -> None:
    """onnxruntime import / session 生成が失敗する時は beam-only agent を返す。"""
    monkeypatch.setattr(nn_submission, "_resolve_model_path", lambda: "/tmp/does-not-exist.onnx")
    monkeypatch.setattr(nn_submission, "_nn_available", lambda _p: False)
    agent = nn_submission.make_agent()
    assert agent is nn_submission._beam_agent
    _assert_legal_moves(agent(_mk_obs()))


def test_beam_agent_returns_legal_moves() -> None:
    """beam fallback path 単体が legal な move を返す (LB 435.3 path 健全性)。"""
    _assert_legal_moves(nn_submission._beam_agent(_mk_obs()))


def test_nn_agent_returns_legal_moves_when_available() -> None:
    """NN が使える環境 (local venv) では NN-in-MCTS agent が legal な move を返す。

    onnxruntime / c5 モデルが無い CI では beam-only に fallback し、いずれにせよ
    legal moves を返すことを確認する (どちらの path も DQ しない)。
    """
    agent = nn_submission.make_agent()
    assert agent in (nn_submission._nn_mcts_agent, nn_submission._beam_agent)
    _assert_legal_moves(agent(_mk_obs()))
