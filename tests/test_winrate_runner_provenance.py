"""winrate_runner の summary provenance 完全性検証 (H039/exp074)。

検証対象:
- `_aggregate()` が出力する `nn_env` ブロックが NN-in-MCTS の全 env var を記録する。
  特に H017 dual-head eval の model lineage 追跡に必須な `ORBIT_WARS_NN_POLICY_MODEL`
  が欠落しないこと (欠落すると「どの policy model がこの winrate を出したか」が summary
  だけで辿れず model 取り違えで誤判定する)。
"""

from __future__ import annotations

from pathlib import Path

from scripts.selfplay import winrate_runner

# NN-in-MCTS の起動に関わる env var (summary が全て記録すべき集合)。
_EXPECTED_NN_ENV_KEYS = {
    "ORBIT_WARS_MCTS",
    "ORBIT_WARS_NN_VALUE",
    "ORBIT_WARS_NN_VALUE_MODEL",
    "ORBIT_WARS_NN_POLICY",
    "ORBIT_WARS_NN_POLICY_MODEL",
}


def test_nn_env_records_all_keys(tmp_path: Path) -> None:
    """nn_env が NN env var 5 個を全て記録する (policy model 欠落 = lineage 追跡不能)。"""
    jsonl = tmp_path / "games.jsonl"  # 空 = ゲーム未実行でも provenance は出る
    summary = winrate_runner._aggregate(jsonl, ["random"], n_per_opp=1)
    assert "nn_env" in summary, "summary に nn_env provenance ブロックが無い"
    assert (
        set(summary["nn_env"].keys()) == _EXPECTED_NN_ENV_KEYS
    ), f"nn_env の記録 env var が想定と不一致: {sorted(summary['nn_env'].keys())}"


def test_nn_env_reflects_policy_model_value(tmp_path: Path, monkeypatch) -> None:
    """ORBIT_WARS_NN_POLICY_MODEL の実値が summary に反映される (dual-head eval の trace)。"""
    model = "experiments/checkpoints/policy_value_net.int8.onnx"
    monkeypatch.setenv("ORBIT_WARS_NN_POLICY", "1")
    monkeypatch.setenv("ORBIT_WARS_NN_POLICY_MODEL", model)
    jsonl = tmp_path / "games.jsonl"
    summary = winrate_runner._aggregate(jsonl, ["random"], n_per_opp=1)
    assert summary["nn_env"]["ORBIT_WARS_NN_POLICY"] == "1"
    assert summary["nn_env"]["ORBIT_WARS_NN_POLICY_MODEL"] == model
