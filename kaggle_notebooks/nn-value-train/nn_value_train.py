# orbit-wars NN value head: Kaggle GPU 学習 notebook (H016 step 7, exp/041)
#
# 目的: `local_cpu_selfplay_export_infeasible_at_scale` のボトルネック
# (self-play data-gen ~74s/game、ローカル CPU で実用規模 ~20h+ 非現実的) を
# Kaggle の複数 CPU core での並列 data-gen + GPU 学習に委譲し、
# 「NN value が MCTS strong-opponent regression (sniper 0.20 / prev_best 0.17) を
#  baseline 付近に戻すか」という H016 のコア signal を出す **first-signal モデル**を
# 1 run で生成する。
#
# パイプライン (全て repo の既存 step 2/4/6 スクリプトを呼ぶだけ。学習ロジック重複なし):
#   1. scripts/selfplay/export_value_data.py --workers=<cpu>  (並列 self-play → npz)
#   2. scripts/nn/train_value.py                              (ValueNet 学習 → .pt + .onnx)
#   3. scripts/nn/quantize_onnx.py                            (int8 量子化 + CPU latency)
#   4. scripts/selfplay/tournament.py (EVAL_RUN=1, H033/exp068)  NN-in-MCTS winrate
#      (trained int8 を MCTS leaf value (+POLICY 時 PUCT prior) に wire し strong-opponent
#       winrate を測定 = H016/H017 の go/no-go signal。`nn_in_mcts_leaf_local_eval_infeasible_in_harness`
#       で local 不能ゆえ Kaggle 側で取得。既定 OFF = 従来 train→quantize で完全不変)
#
# === H017 dual-head mode (POLICY=1, exp/065) ===============================
# 環境変数 POLICY=1 で PolicyValueNet (value + policy prior) の dual-head 学習に
# 切替: stage1 export が --policy-budget で MCTS visit policy_target を記録、
# stage2 train が --policy で value MSE + launch-row policy CE (H032: 死 row 除外)
# を学習。stage3 quantize は value 出力名指しで dual-head ONNX も無改修で動作。
# 既定 (POLICY=0) は従来 H016 value-only path で完全不変。
# 注意: policy_target は per-timestep に POLICY_BUDGET 秒の MCTS search を足すため
# data-gen が value-only より重い。first-signal は GAMES_PER_OPPONENT を下げて短縮。
# 成果物は /kaggle/working に出力し、"Save Version" で Kaggle dataset 化 →
# 次の MCTS 統合 iter で `value_net.int8.onnx` を ORBIT_WARS_NN_VALUE_MODEL に差す。
#
# === supervisor 起動手順 (handoff) =========================================
# このカーネルは orbit-wars repo の src/ を必要とする。以下のいずれかで供給:
#   (A) repo を Kaggle dataset 化して attach (推奨・再現性高):
#       cd ~/Projects/orbit-wars && kaggle datasets create -p . -r zip \
#         (dataset-metadata.json で slug 指定) → kernel-metadata.json の
#         dataset_sources に "<user>/<slug>" を追加 → push。
#   (B) internet 有効 (本 metadata は enable_internet=true) なので環境変数
#       ORBIT_WARS_GIT_URL に repo の git URL を渡して clone (private repo は
#       PAT 付き URL)。
# push: bash scripts/kaggle/push_notebook.sh nn-value-train
# 進捗/取得: bash scripts/kaggle/wait_run.sh / pull_output.sh
#
# === feasibility 試算 (exp/041 ローカル計測ベース) ==========================
#   per-game ~74s (single-thread CPU, baseline beam vs mix)。Kaggle GPU notebook は
#   通常 4 CPU core → 4-way 並列で実効 ~18.5s/game。GPU 9h 上限のうち data-gen に
#   ~2.5h 割けば ~500 game (first-signal 最小)、学習+量子化は GPU で <30min。
#   → 1 run で first-signal モデルが出る。規模拡大 (数千 game) が要る場合は
#     GAMES_PER_OPPONENT を上げ、出力 npz を dataset 化して APPEND_NPZ で継ぎ足す
#     (resumable accumulate、下記)。
# ===========================================================================

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# --- パラメータ (環境変数で上書き可能、first-signal 向け既定) ---------------
OPPONENTS = os.environ.get("OPPONENTS", "random,nearest_sniper,prev_best")
GAMES_PER_OPPONENT = int(os.environ.get("GAMES_PER_OPPONENT", "170"))  # 3 相手 ×170 ≈ 510 game
STRIDE = int(os.environ.get("STRIDE", "4"))
EPOCHS = int(os.environ.get("EPOCHS", "40"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "256"))
WORKERS = int(os.environ.get("WORKERS", str(os.cpu_count() or 4)))
APPEND_NPZ = os.environ.get("APPEND_NPZ", "")  # 既存 dataset npz を継ぎ足す場合の path
# H017 dual-head (PolicyValueNet): POLICY=1 で stage1 に policy_target export、
# stage2 を dual-head 学習 (value MSE + launch-row policy CE) に切替。既定 OFF =
# 従来 H016 value-only path で完全不変。
POLICY = os.environ.get("POLICY", "0") == "1"
# policy_target は各 timestep で MCTS visit 探索を回すため value-only より重い
# (per-timestep ~POLICY_BUDGET 秒の追加 search)。first-signal は小 budget で十分。
POLICY_BUDGET = float(os.environ.get("POLICY_BUDGET", "0.1"))
# H033 (exp/068): trained model の go/no-go signal = NN-in-MCTS の strong-opponent
# winrate。`nn_in_mcts_leaf_local_eval_infeasible_in_harness` (local harness が
# 8-worker 並列下で OOM/reap、~466s で親死亡) ゆえ Kaggle 側で測定する。
# EVAL_RUN=1 opt-in (既定 OFF = 従来 train→quantize で完全不変)。
EVAL_RUN = os.environ.get("EVAL_RUN", "0") == "1"
EVAL_N = int(os.environ.get("EVAL_N", "30"))  # 相手あたり試合数
# local OOM の正体は高 worker 並列下の onnxruntime/torch メモリ圧 → 既定で低 worker。
EVAL_WORKERS = int(os.environ.get("EVAL_WORKERS", "2"))
# 既定は本 run が生成した int8 を測定。dataset mount 済の既存 model を指す場合に上書き可。
EVAL_MODEL = os.environ.get("EVAL_MODEL", "")

WORK = Path("/kaggle/working")
DATA_NPZ = WORK / "value_dataset.npz"
CKPT = WORK / "value_net"  # train が .pt / .onnx を生成
INT8 = WORK / "value_net.int8.onnx"
EVAL_JSON = WORK / "nn_mcts_winrate.json"


def run(cmd: list[str], cwd: Path) -> None:
    """サブプロセス実行 (stdout 直結、失敗で即 raise)。"""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def locate_repo() -> Path:
    """orbit-wars repo root を特定する (dataset mount 探索 → git clone fallback)。"""
    marker = Path("src/nn/model.py")
    # (A) attach された dataset を探索 (marker file を持つ最初のディレクトリ)
    for base in [Path("/kaggle/input"), Path("/kaggle/working")]:
        if not base.exists():
            continue
        for cand in sorted(base.iterdir()):
            if (cand / marker).exists():
                print(f"[locate_repo] dataset mount で repo 発見: {cand}")
                return cand
    # (B) internet 経由 git clone (public repo のため PAT 不要で default URL を使う、env で上書き可)
    git_url = os.environ.get("ORBIT_WARS_GIT_URL", "https://github.com/Y-Kanekoo/orbit-wars.git")
    if git_url:
        dest = WORK / "orbit-wars-src"
        if not (dest / marker).exists():
            run(["git", "clone", "--depth", "1", git_url, str(dest)], cwd=WORK)
        return dest
    raise SystemExit(
        "repo source 未供給: kernel-metadata.json の dataset_sources に repo dataset を"
        " 追加するか、ORBIT_WARS_GIT_URL を設定してください (notebook 冒頭 handoff 参照)。"
    )


def ensure_deps() -> None:
    """Kaggle イメージに無い依存のみ install (torch/onnx は GPU image に同梱)。"""
    try:
        import kaggle_environments  # noqa: F401
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q", "kaggle-environments"], cwd=WORK)
    for mod, pkg in [("onnx", "onnx"), ("onnxruntime", "onnxruntime")]:
        try:
            __import__(mod)
        except ImportError:
            run([sys.executable, "-m", "pip", "install", "-q", pkg], cwd=WORK)


def merge_append(repo: Path) -> None:
    """APPEND_NPZ 指定時、既存 dataset npz を今回生成分の前に結合 (resumable accumulate)。

    game_id は既存の最大 +1 で offset し衝突回避 (grouped split の安定性維持)。
    """
    if not APPEND_NPZ or not Path(APPEND_NPZ).exists():
        return
    print(f"[merge_append] {APPEND_NPZ} を結合")
    prev = np.load(APPEND_NPZ)
    cur = np.load(DATA_NPZ)
    offset = int(prev["game_id"].max()) + 1 if len(prev["game_id"]) else 0
    merged = {k: np.concatenate([prev[k], cur[k]], axis=0) for k in cur.files if k != "game_id"}
    merged["game_id"] = np.concatenate([prev["game_id"], cur["game_id"] + offset], axis=0)
    np.savez_compressed(DATA_NPZ, **merged)
    print(
        f"[merge_append] merged samples={len(merged['value'])} games={int(merged['game_id'].max())+1}"
    )


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    repo = locate_repo()
    ensure_deps()

    timings = {}

    # --- stage 1: 並列 self-play data-gen ---
    t = time.time()
    export_cmd = [
        sys.executable,
        "scripts/selfplay/export_value_data.py",
        "--agent",
        "main.py",
        "--opponents",
        OPPONENTS,
        "--games-per-opponent",
        str(GAMES_PER_OPPONENT),
        "--stride",
        str(STRIDE),
        "--workers",
        str(WORKERS),
        "--out",
        str(DATA_NPZ),
    ]
    if POLICY:
        # H017: 各 timestep の MCTS visit policy target を npz key policy_target に記録
        export_cmd += ["--policy-budget", str(POLICY_BUDGET)]
    run(export_cmd, cwd=repo)
    timings["export_s"] = round(time.time() - t, 1)
    merge_append(repo)

    # --- stage 2: 学習 → torch ckpt + ONNX (POLICY 時は PolicyValueNet dual-head) ---
    t = time.time()
    train_cmd = [
        sys.executable,
        "scripts/nn/train_value.py",
        "--data",
        str(DATA_NPZ),
        "--epochs",
        str(EPOCHS),
        "--batch-size",
        str(BATCH_SIZE),
        "--out",
        str(CKPT),
    ]
    if POLICY:
        # H017: dual head (value MSE + launch-row policy CE、H032 で死 row は CE 除外が既定)
        train_cmd += ["--policy"]
    run(train_cmd, cwd=repo)
    timings["train_s"] = round(time.time() - t, 1)

    # --- stage 3: int8 量子化 + CPU latency (提出は CPU 1秒/turn 制約下) ---
    t = time.time()
    run(
        [
            sys.executable,
            "scripts/nn/quantize_onnx.py",
            "--in",
            str(CKPT) + ".onnx",
            "--out",
            str(INT8),
        ],
        cwd=repo,
    )
    timings["quantize_s"] = round(time.time() - t, 1)

    # --- stage 4 (H033, EVAL_RUN=1): NN-in-MCTS winrate (Kaggle 側 go/no-go) ---
    # trained int8 model を MCTS leaf value (+ POLICY 時 PUCT prior) に wire し、
    # tournament.py で random/nearest_sniper/prev_best vs NN-in-MCTS の winrate を測定。
    # これが H016/H017 全体の go/no-go signal (`nn_in_mcts_leaf_local_eval_infeasible_in_harness`
    # で local 不能、ここで取得する)。
    if EVAL_RUN:
        t = time.time()
        model_path = EVAL_MODEL or str(INT8)
        # 親 env を継承し NN-in-MCTS path を有効化 (main.py は ORBIT_WARS_MCTS=1 を honor)。
        eval_env = dict(os.environ)
        eval_env["ORBIT_WARS_MCTS"] = "1"
        eval_env["ORBIT_WARS_NN_VALUE"] = "1"
        eval_env["ORBIT_WARS_NN_VALUE_MODEL"] = model_path
        if POLICY:
            # dual-head int8 は policy_logits も持つ → PUCT prior を有効化
            eval_env["ORBIT_WARS_NN_POLICY"] = "1"
            eval_env["ORBIT_WARS_NN_POLICY_MODEL"] = model_path
        eval_cmd = [
            sys.executable,
            "scripts/selfplay/tournament.py",
            "--agent",
            "main.py",
            "--opponents",
            OPPONENTS,
            "--n-per-opponent",
            str(EVAL_N),
            "--max-workers",
            str(EVAL_WORKERS),
            "--out",
            str(EVAL_JSON),
        ]
        mode = "PUCT-prior+value-leaf" if POLICY else "value-leaf"
        print(
            f"\n$ [ORBIT_WARS_MCTS=1 NN-in-MCTS({mode}) model={model_path}] "
            f"{' '.join(eval_cmd)}",
            flush=True,
        )
        subprocess.run(eval_cmd, cwd=str(repo), check=True, env=eval_env)
        timings["winrate_eval_s"] = round(time.time() - t, 1)

    n_games = GAMES_PER_OPPONENT * len(OPPONENTS.split(","))
    print(
        f"\n[DONE] mode={'dual-head(H017)' if POLICY else 'value-only(H016)'} "
        f"workers={WORKERS} games≈{n_games} timings={timings} "
        f"per_game≈{round(timings['export_s']/max(n_games,1),1)}s "
        f"outputs={sorted(p.name for p in WORK.glob('value_net*'))}"
    )
    if EVAL_RUN:
        print(
            f"[H033] NN-in-MCTS winrate → {EVAL_JSON.name} (per-opponent N={EVAL_N}, "
            f"workers={EVAL_WORKERS})。go/no-go: strong opponent (nearest_sniper/prev_best) が "
            "baseline (0.6 / 0.51) 付近に戻るか。winner=baseline (sniper 0.20/prev 0.17) からの回復が H016/H017 のコア signal。"
        )
    elif POLICY:
        print(
            "次 step (H017 go/no-go): EVAL_RUN=1 で本 notebook stage 4 を有効化 → trained "
            "dual-head int8 を NN-in-MCTS (PUCT prior + value leaf) に wire し strong-opponent "
            "winrate を Kaggle 側で測定 (`nn_in_mcts_leaf_local_eval_infeasible_in_harness`)。"
        )
    else:
        print(
            "次 step (H016 go/no-go): EVAL_RUN=1 で本 notebook stage 4 を有効化 → trained int8 を "
            "NN-in-MCTS (value leaf) に wire し strong-opponent regression が baseline 付近に戻るか"
            "Kaggle 側で測定 (H016 コア signal)。"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
