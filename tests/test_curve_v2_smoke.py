import time

import numpy as np


def _synthetic_abs_locs(t0: float, n: int = 12, dt: float = 0.05):
    """生成一段简单的合成轨迹，用于冒烟测试。

    说明：
        - z 线性减小（表示向机器人飞来），用于满足 legacy 的 z 速度检查。
        - y 为抛物线（固定重力 -4.9），与 curve2 的约束拟合一致。
    """

    for i in range(n):
        t = i * dt
        x = 0.2
        z = 10.0 - 10.0 * t
        y = 1.2 + 5.0 * t - 4.9 * t * t
        yield np.array([x, y, z, t0 + t], dtype=float)


def test_curve_v2_curve_add_frame_smoke():
    from curve.curve_v2 import Curve

    c = Curve()
    t0 = time.time()

    last = None
    for loc in _synthetic_abs_locs(t0=t0):
        last = c.add_frame(loc.tolist(), is_bot_fire=-1)

    # legacy 行为：可能返回 None（点不够/抑制），也可能返回接球点列表。
    assert last is None or last == -1 or isinstance(last, list)


def test_curve_v2_ball_tracer_smoke():
    from curve.curve_v2.ball_tracer import BallTracer, Status

    tracer = BallTracer()

    # 强制进入 RETURN_BALL，直接走 curve.add_frame 路径做冒烟验证。
    tracer.status = Status.RETURN_BALL
    t0 = time.time()
    tracer.first_return_ball_time = t0

    last_data = None
    for loc in _synthetic_abs_locs(t0=t0):
        last_data = tracer.calc_target_loc_with_abs_ballloc(loc)

    assert isinstance(last_data, dict)
    assert "abs_loc" in last_data
    assert "land_speed" in last_data
