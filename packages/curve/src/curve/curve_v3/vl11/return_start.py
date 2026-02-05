"""参考 legacy/ball_tracer.py::is_user_return_ball 的“回球开始”启发式。

说明：
    这个启发式用于在一个 shot 的点序列中定位“真正开始向机器人飞来”的起点，
    以便在离线抽取/可视化时裁剪掉前面明显无关的杂点段（例如长时间的平台/误识别）。

注意：
    - 该逻辑依赖 z 方向“回球时持续变近”（即 z 递减）。如果你的坐标系相反，
      需要调整判定或在上层关闭该启发式。
    - 离线 DB 通常缺少机器人位姿，因此 ball_bot_dis 相关阈值使用 bot_z_m 常量近似。
"""

from __future__ import annotations

from ..types import BallObservation
from .types import ReturnStartConfig


def find_return_start_index(points: list[BallObservation], cfg: ReturnStartConfig) -> int | None:
    """在点序列中寻找首次满足“回球开始”判定的窗口起点索引。"""

    n = len(points)
    k = int(cfg.return_len)
    if n < k or k < 2:
        return None

    for end in range(k - 1, n):
        start = end - (k - 1)
        window = points[start : end + 1]

        # 条件2：连续接近（z 递减）且高度足够。
        ok = True
        for i in range(k - 1):
            z_curr = float(window[i].z)
            z_next = float(window[i + 1].z)
            y_curr = float(window[i].y)
            if z_curr < z_next:
                ok = False
                break
            if y_curr < float(cfg.min_y_m):
                ok = False
                break
        if not ok:
            continue

        # 条件3：离机器人不能太近。
        ball_bot_dis = float(window[-1].z - float(cfg.bot_z_m))
        if ball_bot_dis < float(cfg.min_ball_bot_dis_m):
            continue

        # 条件4：近期（最多 12 帧）最大飞行距离。
        start_i = max(0, end - 11)
        z_last = float(points[end].z)
        max_dis = 0.0
        for i in range(start_i, end + 1):
            max_dis = max(max_dis, float(points[i].z - z_last))
        if max_dis <= float(cfg.begin_dis_m):
            continue

        # 条件5：时间窗口。
        time_gap = float(window[-1].t - window[0].t)
        if time_gap >= float(cfg.max_window_s):
            continue

        # 条件6：首帧间距。
        first_gap = float(window[0].z - window[1].z)
        if first_gap <= float(cfg.first_gap_m):
            continue

        return int(start)

    return None
