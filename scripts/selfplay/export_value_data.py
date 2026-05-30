"""Self-play value 学習データ exporter (H016 infra step 2)。

self-play trajectory の各 timestep を `(encoded_state, value_target)` サンプルに
変換し `.npz` で書き出す。NN value head (PLAN.md L328-334 Transformer encoder) の
教師データ生成。agent 挙動は不変 (+0 基盤、H021/encoder step 1 の先例)。

value 設計 (Monte-Carlo outcome、時間割引なし):
- kaggle env の最終 reward は既に **player 視点の勝敗** ({-1, 0, +1}) で与えられる
  (確認: random vs random で final rewards = [-1, 1])。
- 各 timestep の観測は `env.steps[t][p].observation` (player=p の完全 obs) を
  推論時と同一の `src.utils.observation.parse` で Observation 化する
  (train/serve encoding skew 回避)。
- value target = `rewards[p]`。encoder は player-relative (self/enemy/neutral) で、
  観測 slice と reward を同じ player index p で引くため **符号は構造的に正しい**
  (player 0 のサンプルと player 1 のサンプルで自動的に符号が反転する)。

データ生成分布: agent (既定 baseline beam main.py) vs opponents mix。両 player の
状態を収集する (value は局面の善し悪しであり policy 非依存、AlphaZero 同様)。
mirror self-play (env.run([main.py, main.py]) 同一 path) は learned_rules
`mirror_match_shared_src_import` で異常終了するため使わない。

CLI:
    .venv/bin/python scripts/selfplay/export_value_data.py \
        --agent main.py --opponents random,nearest_sniper,prev_best \
        --games-per-opponent 4 --stride 4 --out data/value_dataset.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features.encoder import encode_observation  # noqa: E402
from src.utils.observation import parse  # noqa: E402

# tournament.py と同一の相手マッピング (重複定義だが import 依存を避け self-contained に)
OPPONENT_MAP = {
    "random": "random",
    "nearest_sniper": "docs/competition/competition-starter-main.py",
    "prev_best": "docs/competition/legacy-388/main.py",
}

# crash suspect 判定 (match.py と同一基準): 全 agent が呼ばれず即終了した試合は捨てる
_CRASH_MAX_STEPS = 50


class Sample(NamedTuple):
    """1 timestep の学習サンプル (encode 済 tensor + value target + meta)。"""

    planet_tokens: np.ndarray
    planet_mask: np.ndarray
    fleet_tokens: np.ndarray
    fleet_mask: np.ndarray
    global_features: np.ndarray
    value: float
    player: int
    step: int
    game_id: int


def _resolve_agent(name: str) -> str:
    """相手名 / agent file path を match.py と同じ規約で解決する。"""
    if name in OPPONENT_MAP:
        name = OPPONENT_MAP[name]
    if name in {"random", "reaction"}:
        return name
    return str(_ROOT / name)


def collect_game(
    agent1: str, agent2: str, seed: int, stride: int = 1, game_id: int = 0
) -> tuple[list[Sample], list[float]]:
    """1 試合を実行し、各 player・各 timestep のサンプル列と最終 rewards を返す。

    stride>1 で timestep を間引く (連続局面の相関を抑え dataset を圧縮)。
    crash suspect 試合は空サンプル + rewards を返す (呼び出し側で除外)。
    game_id は train/val を **game 単位で grouped split** するための識別子
    (MC-outcome は同一 game の全 timestep が同一 value target のため、timestep
    random split は val loss を leakage で楽観化させる → 学習側で game_id 分割が必須)。
    """
    from kaggle_environments import make

    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([agent1, agent2])

    final = env.steps[-1]
    rewards = [float(s["reward"]) if s["reward"] is not None else 0.0 for s in final]
    n_steps = len(env.steps)
    if n_steps < _CRASH_MAX_STEPS and all(r == 0.0 for r in rewards):
        return [], rewards  # crash suspect: agent が呼ばれず信号なし

    samples: list[Sample] = []
    for t in range(0, n_steps, stride):
        step = env.steps[t]
        for p in range(len(step)):
            obs = step[p].get("observation")
            if not obs or not obs.get("planets"):
                continue
            o = parse(obs)
            # value は **その観測の player 視点** の最終勝敗。obs slice と reward を
            # 同一 index p で引くため符号は自動整合 (player-relative encoder と一致)。
            value = rewards[p] if p < len(rewards) else 0.0
            enc = encode_observation(o)
            samples.append(
                Sample(
                    planet_tokens=enc.planet_tokens,
                    planet_mask=enc.planet_mask,
                    fleet_tokens=enc.fleet_tokens,
                    fleet_mask=enc.fleet_mask,
                    global_features=enc.global_features,
                    value=value,
                    player=o.player,
                    step=t,
                    game_id=game_id,
                )
            )
    return samples, rewards


class _Task(NamedTuple):
    """1 game の実行指示 (並列 worker へ渡す picklable な引数束)。"""

    a1: str
    a2: str
    seed: int
    stride: int
    task_idx: int  # 投入順の安定 id (crash 除外後に game_id へ dense 再採番)


def _run_task(task: _Task) -> tuple[int, list[Sample], list[float]]:
    """worker entry point: 1 game を収集し (task_idx, samples, rewards) を返す。

    game_id は collect_game に task_idx を渡しておき、呼び出し側で crash 除外後に
    dense 再採番する (並列実行で投入順と完了順がズレるため task_idx で安定整列)。
    """
    samples, rewards = collect_game(task.a1, task.a2, task.seed, task.stride, game_id=task.task_idx)
    return task.task_idx, samples, rewards


def export(
    agent: str,
    opponents: list[str],
    games_per_opponent: int,
    stride: int,
    seed_base: int,
    out_path: Path,
    workers: int = 1,
) -> dict:
    """opponents mix で self-play し、value dataset を npz に書き出す。

    seat split: 偶数 game は agent=p1、奇数 game は agent=p2 で seat bias を平準化。
    workers>1 で game 単位に multiprocess 並列化する (各 game は独立に env を make する
    self-contained 処理のため fan-out 可能、Kaggle の複数 CPU core を活用)。結果は
    task_idx で投入順に整列し、crash 除外後に game_id を dense 再採番するため
    workers 値に依存せず決定的 (workers=1 は従来逐次挙動と同一)。
    """
    a_agent = _resolve_agent(agent)

    # 投入順 task list を構築 (opponent 外ループ × game 内ループ、seat swap は従来同一)
    tasks: list[_Task] = []
    task_idx = 0
    for opp in opponents:
        a_opp = _resolve_agent(opp)
        for g in range(games_per_opponent):
            seed = seed_base + g
            if g % 2 == 0:
                a1, a2 = a_agent, a_opp
            else:
                a1, a2 = a_opp, a_agent  # seat swap
            tasks.append(_Task(a1=a1, a2=a2, seed=seed, stride=stride, task_idx=task_idx))
            task_idx += 1

    # task_idx -> (samples, rewards) を収集 (workers>1 は ProcessPool で fan-out)
    results: dict[int, list[Sample]] = {}
    if workers <= 1:
        for task in tasks:
            _, samples, _ = _run_task(task)
            results[task.task_idx] = samples
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            for idx, samples, _ in pool.map(_run_task, tasks):
                results[idx] = samples

    # task 投入順に走査し、非空 (=non-crash) game に game_id を dense 再採番
    all_samples: list[Sample] = []
    games_used = 0
    games_skipped = 0
    game_id = 0
    for task in tasks:
        samples = results.get(task.task_idx, [])
        if not samples:
            games_skipped += 1
            continue
        # collect_game には task_idx を game_id として渡したため dense id に書き換える
        all_samples.extend(s._replace(game_id=game_id) for s in samples)
        games_used += 1
        game_id += 1

    if not all_samples:
        raise RuntimeError("有効サンプルゼロ (全 game が crash suspect の可能性)")

    planet_tokens = np.stack([s.planet_tokens for s in all_samples])
    planet_mask = np.stack([s.planet_mask for s in all_samples])
    fleet_tokens = np.stack([s.fleet_tokens for s in all_samples])
    fleet_mask = np.stack([s.fleet_mask for s in all_samples])
    global_features = np.stack([s.global_features for s in all_samples])
    value = np.asarray([s.value for s in all_samples], dtype=np.float32)
    player = np.asarray([s.player for s in all_samples], dtype=np.int32)
    step = np.asarray([s.step for s in all_samples], dtype=np.int32)
    game_id = np.asarray([s.game_id for s in all_samples], dtype=np.int32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        planet_tokens=planet_tokens,
        planet_mask=planet_mask,
        fleet_tokens=fleet_tokens,
        fleet_mask=fleet_mask,
        global_features=global_features,
        value=value,
        player=player,
        step=step,
        game_id=game_id,
    )
    return {
        "n_samples": len(all_samples),
        "games_used": games_used,
        "games_skipped": games_skipped,
        "n_games": int(game_id.max()) + 1 if len(game_id) else 0,
        "value_min": float(value.min()),
        "value_max": float(value.max()),
        "win_frac": float((value > 0).mean()),
        "loss_frac": float((value < 0).mean()),
        "draw_frac": float((value == 0).mean()),
        "out": str(out_path),
    }


def _smoke() -> int:
    """符号正当性 + shape を assert する CPU smoke (tests/ 改変禁止のため __main__ 内)。"""
    from src.features.encoder import (
        FLEET_FEATURES,
        GLOBAL_FEATURES,
        MAX_FLEETS,
        MAX_PLANETS,
        PLANET_FEATURES,
    )

    samples, rewards = collect_game("random", "random", seed=1, stride=20)
    assert samples, "smoke: サンプルが空 (env 異常)"
    # 符号正当性 (advisor 点1): 各サンプルの value は **その player 視点** の最終勝敗。
    for s in samples:
        assert (
            s.value == rewards[s.player]
        ), f"符号不整合: player={s.player} value={s.value} rewards={rewards}"
        assert s.value in (-1.0, 0.0, 1.0), f"value が {{-1,0,1}} 外: {s.value}"
        assert s.planet_tokens.shape == (MAX_PLANETS, PLANET_FEATURES)
        assert s.fleet_tokens.shape == (MAX_FLEETS, FLEET_FEATURES)
        assert s.global_features.shape == (GLOBAL_FEATURES,)
    # 両 player のサンプルが存在し、勝者/敗者で value 符号が反転していること
    players_seen = {s.player for s in samples}
    if rewards[0] != rewards[1]:
        winner = 0 if rewards[0] > rewards[1] else 1
        win_vals = {s.value for s in samples if s.player == winner}
        lose_vals = {s.value for s in samples if s.player == (1 - winner)}
        assert win_vals == {1.0}, f"勝者 value が +1 でない: {win_vals}"
        assert lose_vals == {-1.0}, f"敗者 value が -1 でない: {lose_vals}"
    print(
        f"export_value_data smoke OK: n={len(samples)} players={sorted(players_seen)} "
        f"rewards={rewards}"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="main.py")
    ap.add_argument("--opponents", default="random,nearest_sniper,prev_best")
    ap.add_argument("--games-per-opponent", type=int, default=4)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="game 単位の multiprocess 並列数 (Kaggle で os.cpu_count() 推奨、1 で逐次)",
    )
    ap.add_argument("--out", default="data/value_dataset.npz")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="1 game vs random を収集し符号正当性・shape を assert (書き出しなし)",
    )
    args = ap.parse_args()

    if args.smoke:
        return _smoke()

    opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]
    stats = export(
        agent=args.agent,
        opponents=opponents,
        games_per_opponent=args.games_per_opponent,
        stride=args.stride,
        seed_base=args.seed_base,
        out_path=_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out),
        workers=args.workers,
    )
    import json

    print(json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
