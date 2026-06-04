# orbit-wars NN value: legacy-388 (prev_best) 戦の data 増量 (H016 cycle 4)
#
# 目的: H016 cycle 2-3 で確認した `nn_value_distribution_shift_legacy388_mirror` を解消。
# cycle 1 で学習した NN は train_acc 0.97 / val_acc 0.76 で legacy-388 mirror state に対し
# 劣勢評価が不正確 → 強敵相手の劣勢 state を 1000 games 追加収録して再学習させる。
#
# 構成:
#   - data-gen のみ (train + quantize は P100 incompat 経験から local CPU で実施)
#   - enable_gpu=false (P100 incompat 回避、CPU 9h 上限内で完結)
#   - OPPONENTS=prev_best (legacy-388 戦のみ、distribution shift 解消)
#   - GAMES_PER_OPPONENT=1000 (cycle 1 の prev_best 170 → 1000 で大幅増量)
#
# 完了後 local merge:
#   - 既存 cycle 1 npz (510 games, mix) + 本 cycle npz (1000 games, prev_best only)
#   - = total 1510 games、legacy-388 戦は 170+1000=1170 で全体の 78%
#
# Kaggle CPU notebook 上限 9h: 4-way 並列で ~18.5s/game × 1000 games = ~5h、余裕。

import os
import subprocess
import sys
import time
from pathlib import Path

# --- パラメータ (cycle 4 専用、hardcode) ----------------------------------
OPPONENTS = "prev_best"  # legacy-388 戦のみ
GAMES_PER_OPPONENT = int(os.environ.get("GAMES_PER_OPPONENT", "1000"))
STRIDE = int(os.environ.get("STRIDE", "4"))
WORKERS = int(os.environ.get("WORKERS", str(os.cpu_count() or 4)))

WORK = Path("/kaggle/working")
DATA_NPZ = WORK / "value_dataset_prev_only.npz"


def run(cmd: list[str], cwd: Path) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def locate_repo() -> Path:
    """orbit-wars repo root を git clone で取得 (public repo)。"""
    git_url = os.environ.get("ORBIT_WARS_GIT_URL", "https://github.com/Y-Kanekoo/orbit-wars.git")
    dest = WORK / "orbit-wars-src"
    marker = Path("src/nn/model.py")
    if not (dest / marker).exists():
        run(["git", "clone", "--depth", "1", git_url, str(dest)], cwd=WORK)
    return dest


def ensure_deps() -> None:
    """data-gen に必要な依存のみ (kaggle_environments + numpy。torch/onnx は不要)。"""
    try:
        import kaggle_environments  # noqa: F401
    except ImportError:
        run(
            [sys.executable, "-m", "pip", "install", "-q", "kaggle-environments"],
            cwd=WORK,
        )


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    repo = locate_repo()
    ensure_deps()

    t = time.time()
    run(
        [
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
        ],
        cwd=repo,
    )
    export_s = round(time.time() - t, 1)

    # 出力統計
    import numpy as np

    data = np.load(DATA_NPZ)
    n_samples = len(data["value"])
    n_games = int(data["game_id"].max()) + 1 if "game_id" in data.files else "N/A"
    print(
        f"\n[DONE] workers={WORKERS} games={n_games} samples={n_samples} "
        f"export_s={export_s} ({export_s / 60:.1f} min) "
        f"avg_per_game={export_s / max(n_games, 1):.2f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
