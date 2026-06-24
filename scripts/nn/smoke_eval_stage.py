"""notebook stage-4 (H033/exp068, EVAL_RUN=1) の NN-in-MCTS winrate seam を tiny model で
end-to-end 実走して de-risk する smoke。

背景 (なぜ必要か):
  exp/066 が notebook chain の stage1-3 (export→train→quantize) を tiny synthetic data で
  end-to-end 検証したのに対し、exp/068 で追加した stage-4 (`EVAL_RUN=1`: trained int8 を
  NN-in-MCTS に wire し tournament.py で winrate を測る go/no-go stage) は **py_compile +
  CLI 引数確認のみ**で実走検証されていない。残る未検証 seam:

    tournament.py (subprocess, NN env を親 os.environ に set)
      → kaggle_environments env.run([main.py, opponent])
        → main.py (ORBIT_WARS_MCTS=1 を honor)
          → src/search/mcts.py (ORBIT_WARS_NN_VALUE / ORBIT_WARS_NN_POLICY を os.environ で読む)
            → nn_mcts_winrate.json

  この seam が壊れていると、supervisor の single GPU run (POLICY=1 EVAL_RUN=1) が model を
  得ても winrate signal が出ず GPU quota を浪費する。本 smoke は tiny dual-head int8 を
  最小 scale (random / N=1 / workers=1) で NN-in-MCTS に通し、winrate JSON が期待 key を持って
  生成されることを assert することで、GPU を焼く前にこの seam を local で固める。

  注: full N=30×3 mix-eval は `nn_in_mcts_leaf_local_eval_infeasible_in_harness` で local 不能
  (Kaggle escalate)。本 smoke は wiring 確認の最小 1 opponent / N=1 のみで、winrate の go/no-go
  判定はしない (それは Kaggle GPU run の役割)。

使い方:
  python scripts/nn/smoke_eval_stage.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# stage-4 と train/serve skew ゼロを保つため、notebook stage3 が量子化するのと同じ
# dual-head export / int8 helper を再利用する。
from scripts.nn.quantize_onnx import (  # noqa: E402
    _export_dualhead_synthetic,
    quantize_int8,
)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fp32 = tmp_path / "value_net.onnx"
        int8 = tmp_path / "value_net.int8.onnx"
        out_json = tmp_path / "nn_mcts_winrate.json"

        # --- 1. notebook stage3 相当: tiny dual-head int8 を生成 ---
        _export_dualhead_synthetic(fp32, seed=0)
        quantize_int8(fp32, int8)
        assert int8.exists(), "dual-head int8 ONNX 未生成"

        # --- 2. notebook stage4 相当: NN-in-MCTS env を親プロセスに set し
        #        tournament.py を最小 scale で実走 (wiring 確認のみ、winrate 判定はしない) ---
        env = dict(os.environ)
        env["ORBIT_WARS_MCTS"] = "1"
        env["ORBIT_WARS_NN_VALUE"] = "1"
        env["ORBIT_WARS_NN_VALUE_MODEL"] = str(int8)
        # dual-head int8 は policy_logits も持つ → PUCT prior path も配線して検証
        env["ORBIT_WARS_NN_POLICY"] = "1"
        env["ORBIT_WARS_NN_POLICY_MODEL"] = str(int8)

        cmd = [
            sys.executable,
            "scripts/selfplay/tournament.py",
            "--agent",
            "main.py",
            "--opponents",
            "random",  # 最小: 1 opponent のみ (full mix は Kaggle escalate)
            "--n-per-opponent",
            "1",
            "--max-workers",
            "1",
            "--out",
            str(out_json),
        ]
        print(f"$ [NN-in-MCTS PUCT-prior+value-leaf model={int8.name}] {' '.join(cmd)}", flush=True)
        # 1 game の NN-in-MCTS は per-turn 0.3s budget 律速で bounded だが余裕を持って timeout 設定。
        proc = subprocess.run(
            cmd, cwd=str(_ROOT), env=env, timeout=600, capture_output=True, text=True
        )
        if proc.returncode != 0:
            print("STDOUT:\n" + proc.stdout, file=sys.stderr)
            print("STDERR:\n" + proc.stderr, file=sys.stderr)
            raise AssertionError(f"stage-4 tournament が rc={proc.returncode} で失敗")

        # --- 3. winrate JSON が期待 key を持って生成されたか assert ---
        assert out_json.exists(), "nn_mcts_winrate.json が生成されなかった"
        summary = json.loads(out_json.read_text())
        for key in ("mode", "agent", "opponents", "winrate_min", "errors_total"):
            assert key in summary, f"winrate JSON に key '{key}' が欠落"
        assert summary["mode"] == "mix_eval", f"mode 異常: {summary['mode']}"
        assert "random" in summary["opponents"], "opponents に random が無い"
        rnd = summary["opponents"]["random"]
        for key in ("winrate", "wins", "n", "errors"):
            assert key in rnd, f"opponent 結果に key '{key}' が欠落"
        # wiring 検証: errors=0 (NN model が valid に load され MCTS が valid moves を返した)。
        assert summary["errors_total"] == 0, (
            f"errors_total={summary['errors_total']} = NN-in-MCTS path が "
            "agent エラーを出した (model load 失敗 / invalid moves)"
        )

    print(
        "smoke_eval_stage OK: stage-4 NN-in-MCTS seam 実走完了 "
        f"(winrate={rnd['winrate']} wins={rnd['wins']}/{rnd['n']} errors=0、"
        "winrate 値自体は wiring 確認用の最小 N=1 で go/no-go 判定対象でない)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
