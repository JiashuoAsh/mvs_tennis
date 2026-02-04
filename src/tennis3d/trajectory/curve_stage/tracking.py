from __future__ import annotations

import math

from tennis3d.trajectory.curve_stage.models import _Track


def _dist3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    dz = float(a[2] - b[2])
    return float(math.sqrt(dx * dx + dy * dy + dz * dz))


def _predict_track_pos(tr: _Track, *, t_abs: float, min_dt_s: float) -> tuple[float, float, float] | None:
    """预测当前时刻 track 的位置（用于数据关联）。

    说明：
        - 纯距离最近邻在多球接近/交叉时容易 swap。
        - 这里使用“最近两次观测”估计常速度，并外推到当前时刻做 gating。
        - 若历史不足或 dt 异常，则回退到 last_pos，保持行为可解释。
    """

    if tr.last_pos is None or tr.last_t_abs is None:
        return None

    # 历史不足：无法估速度，回退到 last_pos。
    if tr.prev_pos is None or tr.prev_t_abs is None:
        return tr.last_pos

    dt_hist = float(tr.last_t_abs - tr.prev_t_abs)
    if dt_hist < float(min_dt_s):
        return tr.last_pos

    vx = float(tr.last_pos[0] - tr.prev_pos[0]) / dt_hist
    vy = float(tr.last_pos[1] - tr.prev_pos[1]) / dt_hist
    vz = float(tr.last_pos[2] - tr.prev_pos[2]) / dt_hist

    dt = float(t_abs - tr.last_t_abs)
    return (
        float(tr.last_pos[0] + vx * dt),
        float(tr.last_pos[1] + vy * dt),
        float(tr.last_pos[2] + vz * dt),
    )
