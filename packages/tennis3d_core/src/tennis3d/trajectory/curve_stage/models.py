from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from typing import Any


@dataclass
class _BallMeas:
    ball_id: int
    x: float
    y: float
    z: float
    quality: float
    num_views: int | None = None
    median_reproj_error_px: float | None = None
    ball_3d_std_m_max: float | None = None


@dataclass
class _RecentObs:
    """track 内部缓存的最近观测（已应用 y 变换）。

    说明：
        - 该缓存用于 episode 起始判定与（可选）重放给 curve 拟合器。
        - t_abs 采用 stage 选出的绝对时间轴（capture_t_abs/created_at）。
    """

    t_abs: float
    x: float
    y: float
    z: float
    conf: float | None

    @property
    def pos(self) -> tuple[float, float, float]:
        return (float(self.x), float(self.y), float(self.z))


@dataclass
class _Track:
    track_id: int
    created_t_abs: float | None = None
    last_t_abs: float | None = None
    last_pos: tuple[float, float, float] | None = None
    prev_t_abs: float | None = None
    prev_pos: tuple[float, float, float] | None = None
    n_obs: int = 0

    # 运动学诊断（便于下游做可解释筛选）
    last_speed_mps: float | None = None
    speed_ewma_mps: float | None = None
    last_motion_t_abs: float | None = None

    # episode 状态
    episode_id: int = 0
    episode_active: bool = False
    episode_start_t_abs: float | None = None
    episode_end_t_abs: float | None = None
    episode_end_reason: str | None = None

    # v3 可用的“落地时间”缓存（用于 episode 结束判定）
    predicted_land_time_abs: float | None = None

    # v3 可用的“第二次落地时间”缓存（用于 episode 覆盖两段后再结束）
    predicted_second_land_time_abs: float | None = None

    # 最近观测缓存（用于 episode 判定/重放）
    recent: deque[_RecentObs] = field(default_factory=deque)

    # 拟合器实例（按需创建）
    v3: Any | None = None
    v3_legacy: Any | None = None
    v2: Any | None = None
