"""H021 mix-eval submit gate のテスト (check_mix_gate.py)。"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts/kaggle/check_mix_gate.py"


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_mix(
    tmp_path: Path,
    *,
    random_wr: float,
    sniper_wr: float,
    prev_wr: float,
    winrate_min: float,
    errors_total: int,
    measured_at: str | None = None,
) -> Path:
    mix = {
        "mode": "mix_eval",
        "opponents": {
            "random": {"winrate": random_wr},
            "nearest_sniper": {"winrate": sniper_wr},
            "prev_best": {"winrate": prev_wr},
        },
        "winrate_min": winrate_min,
        "errors_total": errors_total,
        "measured_at": measured_at or _now(),
    }
    p = tmp_path / "mix.json"
    p.write_text(json.dumps(mix))
    return p


def _run_gate(mix_path: Path, *extra: str) -> int:
    r = subprocess.run(
        [sys.executable, str(GATE), str(mix_path), *extra],
        capture_output=True,
        text=True,
    )
    return r.returncode


def test_gate_pass(tmp_path: Path) -> None:
    # prev_best 0.70 は repo の prev baseline (~0.53) + 0.05 を確実に超える
    mix = _write_mix(
        tmp_path,
        random_wr=0.95,
        sniper_wr=0.65,
        prev_wr=0.70,
        winrate_min=0.65,
        errors_total=0,
    )
    assert _run_gate(mix) == 0


def test_gate_fail_low_random(tmp_path: Path) -> None:
    mix = _write_mix(
        tmp_path,
        random_wr=0.85,  # < 0.90
        sniper_wr=0.65,
        prev_wr=0.70,
        winrate_min=0.65,
        errors_total=0,
    )
    assert _run_gate(mix) == 1


def test_gate_fail_low_sniper(tmp_path: Path) -> None:
    mix = _write_mix(
        tmp_path,
        random_wr=0.95,
        sniper_wr=0.55,  # < 0.60
        prev_wr=0.70,
        winrate_min=0.55,
        errors_total=0,
    )
    assert _run_gate(mix) == 1


def test_gate_fail_winrate_min(tmp_path: Path) -> None:
    mix = _write_mix(
        tmp_path,
        random_wr=0.95,
        sniper_wr=0.65,
        prev_wr=0.70,
        winrate_min=0.45,  # < 0.50
        errors_total=0,
    )
    assert _run_gate(mix) == 1


def test_gate_fail_errors(tmp_path: Path) -> None:
    mix = _write_mix(
        tmp_path,
        random_wr=0.95,
        sniper_wr=0.65,
        prev_wr=0.70,
        winrate_min=0.65,
        errors_total=2,  # != 0
    )
    assert _run_gate(mix) == 1


def test_gate_fail_prev_no_improvement(tmp_path: Path) -> None:
    # prev_best 0.50 は repo baseline (~0.53) + 0.05 を超えない
    mix = _write_mix(
        tmp_path,
        random_wr=0.95,
        sniper_wr=0.65,
        prev_wr=0.50,
        winrate_min=0.50,
        errors_total=0,
    )
    assert _run_gate(mix) == 1


def test_gate_stale(tmp_path: Path) -> None:
    old = (datetime.now(UTC) - timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    mix = _write_mix(
        tmp_path,
        random_wr=0.95,
        sniper_wr=0.65,
        prev_wr=0.70,
        winrate_min=0.65,
        errors_total=0,
        measured_at=old,
    )
    # stale check 無効ならpass、 有効 (max-age 1h) なら block
    assert _run_gate(mix) == 0
    assert _run_gate(mix, "--max-age-sec", "3600") == 1


def test_gate_missing_file(tmp_path: Path) -> None:
    assert _run_gate(tmp_path / "nonexistent.json") == 1


def test_opponent_map_paths_exist() -> None:
    """OPPONENT_MAP の agent file が実在することを確認 (random は組み込みなので除外)。"""
    sys.path.insert(0, str(ROOT / "scripts/selfplay"))
    import tournament

    for name, path in tournament.OPPONENT_MAP.items():
        if name == "random":
            assert path == "random"
            continue
        assert (ROOT / path).exists(), f"{name} agent file 不在: {path}"
