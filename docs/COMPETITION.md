# Orbit Wars コンペ仕様 (要約)

詳細は `docs/competition/competition-README.md` (Kaggle 配布物の原本)。本ファイルは戦略立案に直結する要点のみ。

## 概要
- 連続 2D 100x100 空間で惑星を占領する RTS agent competition
- 2 or 4 プレイヤー、500 turn 上限
- 中心に半径 10 の太陽 (fleet 横断で消滅)
- 4-fold mirror symmetry で初期配置の公平性保証
- 評価は TrueSkill 風 (推定)、リーグ形式

## 制約 (絶対)
| 制約 | 値 | 影響 |
|---|---|---|
| `actTimeout` | 1 秒/turn | 計算量の絶対上限、超過で disqualify |
| daily submission | 5 / 日 | quota-guard hook で hard limit |
| Notebook GPU (推定) | ~30 hr/週 | Phase 3+ で意識、quota-guard 監視 |

## ゲームメカニクス

### 惑星
`[id, owner, x, y, radius, ships, production]`
- owner: 0-3 / -1 (中立)
- radius: `1 + ln(production)` (=1.0-2.6)
- production: 1-5 (turn 毎の ship 生産量)
- 内側 (orbital_radius + planet_radius < 50): 角速度 0.025-0.05 rad/turn で太陽周回
- 外側: 静止
- 全 20-40 個、5-10 対称グループ × 4

### Fleet
`[id, owner, x, y, angle, from_planet_id, ships]`
- 速度 = `1.0 + (maxSpeed - 1.0) * (log(ships) / log(1000)) ^ 1.5`
  - 1 ship = 1.0 unit/turn
  - 1000 ship ≈ 6.0 unit/turn (max)
- 直線移動、毎 turn で out_of_bounds / sun crossing / planet collision check
- 衝突 = 戦闘トリガー

### Comet (重要)
- step 50/150/250/350/450 に 4 個ずつ spawn
- 楕円軌道、radius 1.0、production 1
- 終わったら離脱 (掴んだ ship も道連れ)
- spawn step が既知 → pre-positioning 戦略可能 (H006)

### 戦闘
1. 同 turn に同 planet へ到着した fleet を owner 別に集約
2. 最大 vs 2 番目 → 差分が生存 (tie で全滅)
3. 生存 attacker vs 守備 → 超えれば占領、守備が生き残れば防御成功

### 勝利条件
- step 500 時点の **総 ship 数 (planets + fleets)** が最多 → win
- 単独残存 (他全員 0 planet 0 fleet) → 即 win

## Observation Field
| field | 型 | 説明 |
|---|---|---|
| `player` | int | 自分の player ID (0-3) |
| `planets` | list | 全惑星 (comets 含む) |
| `fleets` | list | アクティブ fleet |
| `angular_velocity` | float | 内側惑星の回転速度 |
| `initial_planets` | list | 初期配置 (回転予測に使う) |
| `comets` | list | comet グループ data (paths, path_index) |
| `comet_planet_ids` | list[int] | comet な planet IDs |
| `remainingOverageTime` | float | 残 overage budget (秒) |

## Action Format
`[[from_planet_id, angle_radians, num_ships], ...]`
- 1 turn で複数 launch 可 (B-3 multi-launch 対応)
- 空 list `[]` = no action (B-8 で必ず候補に含める)
- 不正 action は静かに棄却 (= turn 何もしない、A-3 で防御)

## Submission

```bash
# 単一ファイル
kaggle competitions submit orbit-wars -f main.py -m "v1"

# 多ファイル (src/ helper 同梱)
tar -czf submission.tar.gz main.py src/
kaggle competitions submit orbit-wars -f submission.tar.gz -m "v2"
```

`scripts/kaggle/submit.sh` でラップ。

## 評価 / 分析 API
```bash
kaggle competitions submissions orbit-wars               # 履歴
kaggle competitions episodes <SUB_ID>                    # 対戦リスト
kaggle competitions replay <EP_ID> -p ./replays          # replay JSON
kaggle competitions logs <EP_ID> <player_idx>            # agent stderr
kaggle competitions leaderboard orbit-wars -s            # LB
```

## 現状 (2026-05-16)
- 締切まで残 **38 日** (2026-06-23 23:59 UTC)
- 参加 **2,757 teams**
- top-9 ボーダー **1437.2** (kovi)
- user 現在: **388.6** (legacy-388, "phase1+2 baseline beam depth=2 width=16")
- 必要伸び: **+1048**

## external data / pretrained
- Phase 0 では researcher 未起動。Kaggle rules を読み込んで本ファイルに追記する責務は researcher (Phase 1 開始時)
- 暫定方針: **external なしで戦う** (Orbit Wars 専用 env、汎用 pretrained では転移無効)
