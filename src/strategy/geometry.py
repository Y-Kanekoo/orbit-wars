"""幾何ユーティリティ: 距離・角度・太陽回避・軌道予測。

all public functions are pure and allocation-free where practical,
since they are called in the 1秒/turn 制限下。
"""

from __future__ import annotations

import math

BOARD_SIZE = 100.0
SUN_CENTER = (50.0, 50.0)
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0  # orbital_radius + planet_radius < 50 → 公転


def distance(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def angle_to(ax: float, ay: float, bx: float, by: float) -> float:
    return math.atan2(by - ay, bx - ax)


def segment_hits_sun(x0: float, y0: float, x1: float, y1: float, margin: float = 0.5) -> bool:
    cx, cy = SUN_CENTER
    r = SUN_RADIUS + margin
    dx, dy = x1 - x0, y1 - y0
    fx, fy = x0 - cx, y0 - cy
    a = dx * dx + dy * dy
    if a < 1e-9:
        return fx * fx + fy * fy < r * r
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4 * a * c
    if disc < 0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2 * a)
    t2 = (-b + sq) / (2 * a)
    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0)


def fleet_speed(ships: int, max_speed: float = 6.0) -> float:
    if ships <= 1:
        return 1.0
    return 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5


def predict_planet_position(
    cx: float,
    cy: float,
    initial_x: float,
    initial_y: float,
    angular_velocity: float,
    turns_ahead: int,
) -> tuple[float, float]:
    dx = initial_x - cx
    dy = initial_y - cy
    radius = math.hypot(dx, dy)
    theta0 = math.atan2(dy, dx)
    theta = theta0 + angular_velocity * turns_ahead
    return cx + radius * math.cos(theta), cy + radius * math.sin(theta)


def avoidance_angle(
    from_x: float, from_y: float, to_x: float, to_y: float, margin: float = 1.0
) -> float:
    direct = angle_to(from_x, from_y, to_x, to_y)
    if not segment_hits_sun(from_x, from_y, to_x, to_y, margin=margin):
        return direct
    # 太陽の左右のどちらを回るかを、始点が太陽のどちら側にあるかで決める
    cx, cy = SUN_CENTER
    cross = (to_x - from_x) * (cy - from_y) - (to_y - from_y) * (cx - from_x)
    sign = 1.0 if cross > 0 else -1.0
    # fleet は固定 heading の直線移動かつ sun-crossing で破壊されるため、
    # 単一 12° では near-tangent 経路を抜けきれず fleet を太陽に捨てる。
    # 偏向後の直線経路が太陽を抜けるまで段階的にオフセットを増やす (本格版)。
    # 12° で抜けるケースは従来と完全同一出力 (失敗ケースのみ強い偏向を返す)。
    dist = distance(from_x, from_y, to_x, to_y)
    angle = direct + sign * math.radians(12.0)
    for step_deg in range(12, 91, 12):
        angle = direct + sign * math.radians(float(step_deg))
        end_x = from_x + dist * math.cos(angle)
        end_y = from_y + dist * math.sin(angle)
        if not segment_hits_sun(from_x, from_y, end_x, end_y, margin=margin):
            return angle
    # 90° でも抜けない場合は best effort (最大偏向) を返す
    return angle
