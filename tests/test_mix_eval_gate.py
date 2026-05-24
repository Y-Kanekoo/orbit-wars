"""H021 mix-eval submit gate のテスト (check_mix_gate.py, no-regression 方式)。

baseline は --baseline-json で明示的に渡し、repo state 非依存にする。
baseline 値: random 0.8667 / nearest_sniper 0.5667 / prev_best 0.5667 (現 main parity)。
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts/kaggle/check_mix_gate.py"

BASE_RANDOM = 0.8667
BASE_SNIPER = 0.5667
BASE_PREV = 0.5667


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_baseline(tmp_path: Path) -> Path:
    best = {
        "schema_version": 2,
        "mix_eval": {
            "opponents": {
                "random": {"winrate": BASE_RANDOM},
                "nearest_sniper": {"winrate": BASE_SNIPER},
                "prev_best": {"winrate": BASE_PREV},
            }
        },
    }
    p = tmp_path / "baseline.json"
    p.write_text(json.dumps(best))
    return p


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


def _run_gate(mix_path: Path, baseline_path: Path, *extra: str) -> int:
    r = subprocess.run(
        [
            sys.executable,
            str(GATE),
            str(mix_path),
            "--baseline-json",
            str(baseline_path),
            *extra,
        ],
        capture_output=True,
        text=True,
    )
    return r.returncode


def test_gate_pass(tmp_path: Path) -> None:
    base = _write_baseline(tmp_path)
    # random/sniper baseline 以上、prev_best は non-blocking
    mix = _write_mix(
        tmp_path,
        random_wr=0.90,
        sniper_wr=0.60,
        prev_wr=0.65,
        winrate_min=0.60,
        errors_total=0,
    )
    assert _run_gate(mix, base) == 0


def test_gate_pass_exactly_baseline_random(tmp_path: Path) -> None:
    # random が baseline ちょうど (no-regression は >= なので pass)
    base = _write_baseline(tmp_path)
    mix = _write_mix(
        tmp_path,
        random_wr=BASE_RANDOM,
        sniper_wr=0.60,
        prev_wr=0.65,
        winrate_min=BASE_RANDOM,
        errors_total=0,
    )
    assert _run_gate(mix, base) == 0


def test_gate_fail_random_regression(tmp_path: Path) -> None:
    base = _write_baseline(tmp_path)
    mix = _write_mix(
        tmp_path,
        random_wr=0.85,  # < baseline 0.8667
        sniper_wr=0.60,
        prev_wr=0.65,
        winrate_min=0.60,
        errors_total=0,
    )
    assert _run_gate(mix, base) == 1


def test_gate_fail_sniper_regression(tmp_path: Path) -> None:
    base = _write_baseline(tmp_path)
    mix = _write_mix(
        tmp_path,
        random_wr=0.90,
        sniper_wr=0.55,  # < baseline 0.5667
        prev_wr=0.65,
        winrate_min=0.55,
        errors_total=0,
    )
    assert _run_gate(mix, base) == 1


def test_gate_fail_winrate_min(tmp_path: Path) -> None:
    # 新 gate: winrate_min = min(random, sniper)。sniper を 0.49 に落とすと
    # winrate_min < 0.50 で block (sniper no-regression も同時に fail するが block は確実)
    base = _write_baseline(tmp_path)
    mix = _write_mix(
        tmp_path,
        random_wr=0.90,
        sniper_wr=0.49,  # winrate_min=min(0.90,0.49)=0.49 < 0.50
        prev_wr=0.65,
        winrate_min=0.49,
        errors_total=0,
    )
    assert _run_gate(mix, base) == 1


def test_gate_fail_prev_collapse(tmp_path: Path) -> None:
    # prev_best collapse (< PREV_FLOOR 0.35) は tripwire で block (mirror で壊滅的破綻)
    base = _write_baseline(tmp_path)
    mix = _write_mix(
        tmp_path,
        random_wr=0.90,
        sniper_wr=0.60,
        prev_wr=0.30,  # < 0.35
        winrate_min=0.60,
        errors_total=0,
    )
    assert _run_gate(mix, base) == 1


def test_gate_fail_errors(tmp_path: Path) -> None:
    base = _write_baseline(tmp_path)
    mix = _write_mix(
        tmp_path,
        random_wr=0.90,
        sniper_wr=0.60,
        prev_wr=0.65,
        winrate_min=0.60,
        errors_total=2,  # != 0
    )
    assert _run_gate(mix, base) == 1


def test_gate_pass_prev_low_non_blocking(tmp_path: Path) -> None:
    # 新 gate: prev_best は non-blocking。改善なし (mirror regression 0.4667) でも
    # >= 0.35 かつ random/sniper OK なら pass (H024 の実数値ケース)
    base = _write_baseline(tmp_path)
    mix = _write_mix(
        tmp_path,
        random_wr=0.9333,
        sniper_wr=0.60,
        prev_wr=0.4667,  # baseline 0.5667 から下落だが non-blocking
        winrate_min=0.4667,
        errors_total=0,
    )
    assert _run_gate(mix, base) == 0


def test_gate_stale(tmp_path: Path) -> None:
    base = _write_baseline(tmp_path)
    old = (datetime.now(UTC) - timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    mix = _write_mix(
        tmp_path,
        random_wr=0.90,
        sniper_wr=0.60,
        prev_wr=0.65,
        winrate_min=0.60,
        errors_total=0,
        measured_at=old,
    )
    assert _run_gate(mix, base) == 0  # stale check 無効
    assert _run_gate(mix, base, "--max-age-sec", "3600") == 1  # stale block


def test_gate_missing_file(tmp_path: Path) -> None:
    base = _write_baseline(tmp_path)
    assert _run_gate(tmp_path / "nonexistent.json", base) == 1


def test_baseline_fallback_when_no_mix_eval(tmp_path: Path) -> None:
    """baseline json に mix_eval が無い場合は FALLBACK_BASELINE (random>=0.90) を使う。"""
    base = tmp_path / "best_v1.json"
    base.write_text(json.dumps({"schema_version": 1, "local_winrate_vs_prev_best": 0.50}))
    # random 0.88 は fallback baseline 0.90 を下回る → fail
    mix = _write_mix(
        tmp_path,
        random_wr=0.88,
        sniper_wr=0.65,
        prev_wr=0.60,
        winrate_min=0.60,
        errors_total=0,
    )
    assert _run_gate(mix, base) == 1


def test_opponent_map_paths_exist() -> None:
    """OPPONENT_MAP の agent file が実在することを確認 (random は組み込みなので除外)。"""
    sys.path.insert(0, str(ROOT / "scripts/selfplay"))
    import tournament

    for name, path in tournament.OPPONENT_MAP.items():
        if name == "random":
            assert path == "random"
            continue
        assert (ROOT / path).exists(), f"{name} agent file 不在: {path}"
