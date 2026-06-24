"""route (b) NN-in-MCTS 提出 bundle ビルダー + env.run de-risk smoke (H038/exp073)。

H037 (exp/072) 確定次手 (ii)「NN-in-MCTS を実 submission path に wire し beam 435.3 と
LB で答え合わせ」を supervisor が 1 提出で実行できるよう、提出 bundle を組み立てて
kaggle_environments env.run で実走検証する。

現状 blocker (本ビルダーが解消):
  submit.sh は main.py+src/ のみ pack し ONNX モデル/onnxruntime を bundle せず、live
  main.py は NN-in-MCTS を env var でしか有効化できないが Kaggle 提出に env を渡す手段が
  無い → route (b) の提出 bundle がそもそも存在しなかった。

bundle layout (<out>/):
  main.py                  # 薄い entry: from src.agents.nn_submission import make_agent
  src/                     # repo src/ ツリー
  value_net_c5.int8.onnx   # c5 int8 value model (distribution shift 解消済, H037 測定構成)

downside bound (DQ ゼロが絶対制約):
  bundle 専用 main.py は make_agent() を使う。onnxruntime が Kaggle 実機に**無い**場合は
  beam-only に graceful fallback し LB 435.3 baseline を再現する (nn_submission.py 参照)。
  本ビルダーの env.run smoke は **local venv (onnxruntime あり)** で走るため NN path を
  実走検証するが、Kaggle 実機の onnxruntime 有無は local 検証不能 = supervisor が route (b)
  提出 1 回で答え合わせする残存 unknown (downside は fallback で bound 済)。

使い方:
  python scripts/kaggle/build_nn_submission.py            # build + env.run smoke
  python scripts/kaggle/build_nn_submission.py --out /path/to/bundle --tar
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import tarfile
import time
from contextlib import redirect_stderr
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_MODEL_SRC = _ROOT / "experiments" / "checkpoints" / "value_net_c5.int8.onnx"
_MODEL_NAME = "value_net_c5.int8.onnx"
_LATENCY_HARD_MS = 900.0  # actTimeout 1s/turn の安全閾 (main.py deadline 0.90s と同じ margin)

_BUNDLE_MAIN_PY = '''\
"""NN-in-MCTS 提出 bundle entry (H038/exp073、build_nn_submission.py 生成)。

live repo の main.py (beam, LB 435.3) とは別物。make_agent() が onnxruntime/モデルを
probe し、使えれば NN-in-MCTS、使えなければ beam-only (= 435.3) に graceful fallback する。
"""

from __future__ import annotations

import sys
from pathlib import Path

# kaggle agent loader は空 globals で exec するため __file__ 不在のことがある。
try:
    _ROOT = Path(__file__).resolve().parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
except NameError:
    pass

from src.agents.nn_submission import make_agent  # noqa: E402

agent = make_agent()
'''


def build_bundle(out_dir: Path) -> Path:
    """bundle を out_dir に組み立てて path を返す。"""
    if not _MODEL_SRC.is_file():
        raise FileNotFoundError(f"c5 int8 model が無い: {_MODEL_SRC}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    (out_dir / "main.py").write_text(_BUNDLE_MAIN_PY)
    shutil.copytree(
        _ROOT / "src",
        out_dir / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(_MODEL_SRC, out_dir / _MODEL_NAME)
    return out_dir


def _obs_of(step_agent) -> object:
    """env.steps[i][0] から observation を取り出す (Struct / dict 両対応)。"""
    return getattr(step_agent, "observation", None) or step_agent["observation"]


def _latency_replay(observations: list, n_sample: int = 40) -> dict:
    """env.steps の player-0 観測を make_agent() で in-process replay し per-turn 実 latency。

    kaggle_environments は agent stderr を抑制し core_ms telemetry を捕捉できないため、
    submission-safe (max<900ms) を実証する経路として real states を直接 replay する。
    late/dense turn (encoder token 多) が worst-case なので末尾 10 turn を必ず含める。
    """
    from src.agents.nn_submission import make_agent

    agent = make_agent()
    n = len(observations)
    if n == 0:
        return {"path": agent.__name__, "n_timed": 0, "max_ms": None, "warmup_ms": None}

    stride = max(1, n // n_sample)
    idx = sorted(set(list(range(0, n, stride)) + list(range(max(0, n - 10), n))))

    # warmup: 初回呼び出しは ONNX session の cold load を含むため別計上 (submission の
    # 1 turn 目に相当、deadline 0.90s 内なら可)。
    w0 = time.monotonic()
    agent(observations[idx[0]])
    warmup_ms = (time.monotonic() - w0) * 1000.0

    timed: list[float] = []
    for i in idx[1:]:
        t = time.monotonic()
        agent(observations[i])
        timed.append((time.monotonic() - t) * 1000.0)

    return {
        "path": agent.__name__,
        "n_timed": len(timed),
        "max_ms": round(max(timed), 1) if timed else None,
        "median_ms": round(sorted(timed)[len(timed) // 2], 1) if timed else None,
        "warmup_ms": round(warmup_ms, 1),
    }


def env_run_smoke(bundle_dir: Path, seed: int = 0) -> dict:
    """bundle 専用 main.py を file-path agent として env.run で 1 game 実走 (vs random)。

    提出と同じ「file path を agent として渡す」mode で、モデル load + NN-in-MCTS が
    valid moves を返し errors=0 で完走するかを検証し、env.steps の観測を in-process
    replay して per-turn latency が submission-safe (<900ms) かを実測する。
    """
    from kaggle_environments import make

    agent_path = str(bundle_dir / "main.py")
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)

    buf = io.StringIO()
    t0 = time.monotonic()
    with redirect_stderr(buf):
        env.run([agent_path, "random"])
    elapsed = time.monotonic() - t0
    stderr_text = buf.getvalue()

    final = env.steps[-1]
    rewards = [float(s.reward) if s.reward is not None else 0.0 for s in final]
    steps = len(env.steps)
    # agent が一切呼ばれない crash (NameError 等) は steps<50 で全 reward 0 になる。
    crash_suspect = steps < 50 and all(r == 0.0 for r in rewards)

    # telemetry が error を出していないか (model load 失敗 / invalid moves)。
    error_lines = [ln for ln in stderr_text.splitlines() if '"level": "error"' in ln]

    # player-0 観測を in-process replay して実 latency を計測 (stderr 抑制対策)。
    obs0 = [_obs_of(st[0]) for st in env.steps]
    latency = _latency_replay([o for o in obs0 if o is not None])

    return {
        "seed": seed,
        "agent_path": agent_path,
        "steps": steps,
        "rewards": rewards,
        "crash_suspect": crash_suspect,
        "elapsed_sec": round(elapsed, 1),
        "latency": latency,
        "error_lines": error_lines[:5],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_ROOT / "build" / "nn_submission"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tar", action="store_true", help="bundle を tar.gz に固める")
    ap.add_argument("--no-smoke", action="store_true", help="env.run 実走検証を skip")
    args = ap.parse_args()

    out_dir = Path(args.out)
    print(f"[build] bundle 組み立て -> {out_dir}", flush=True)
    build_bundle(out_dir)
    print(f"[build] OK: main.py + src/ + {_MODEL_NAME} ({_MODEL_SRC.stat().st_size} bytes)")

    if not args.no_smoke:
        print(f"[smoke] env.run (file-path agent vs random, seed={args.seed}) ...", flush=True)
        result = env_run_smoke(out_dir, seed=args.seed)
        print("[smoke] " + json.dumps(result, ensure_ascii=False))

        if result["crash_suspect"]:
            print("[smoke] FAIL: crash_suspect (agent が呼ばれず即終了)", file=sys.stderr)
            return 1
        if result["error_lines"]:
            print("[smoke] FAIL: agent が error telemetry を出した", file=sys.stderr)
            return 1
        if result["steps"] < 50:
            print(f"[smoke] FAIL: steps={result['steps']} < 50 (ゲーム未完走)", file=sys.stderr)
            return 1
        lat = result["latency"]
        mx = lat["max_ms"]
        if mx is None:
            print("[smoke] FAIL: latency replay で 1 turn も計測できず", file=sys.stderr)
            return 1
        if mx >= _LATENCY_HARD_MS:
            print(
                f"[smoke] FAIL: max per-turn={mx}ms >= {_LATENCY_HARD_MS} (1s/turn DQ リスク)",
                file=sys.stderr,
            )
            return 1
        print(
            f"[smoke] OK: per-turn max={mx}ms median={lat['median_ms']}ms "
            f"warmup={lat['warmup_ms']}ms < {_LATENCY_HARD_MS} (submission-safe, "
            f"path={lat['path']}, n_timed={lat['n_timed']})"
        )

    if args.tar:
        tar_path = out_dir.with_suffix(".tar.gz")
        with tarfile.open(tar_path, "w:gz") as tf:
            for p in sorted(out_dir.rglob("*")):
                if "__pycache__" in p.parts or p.suffix == ".pyc":
                    continue
                tf.add(p, arcname=str(p.relative_to(out_dir)))
        print(f"[build] tar -> {tar_path}")

    print(
        "[build] DONE. route (b) bundle 準備完了。\n"
        "  残存 unknown: Kaggle 実機の onnxruntime 有無 (local 検証不能)。\n"
        "  不在時は beam-only に graceful fallback し LB 435.3 を再現 (downside bound 済)。\n"
        "  supervisor は本 bundle を kaggle submit して route (b) LB 答え合わせ可能。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
