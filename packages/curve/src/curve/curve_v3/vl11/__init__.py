"""curve_v3 的离线抽取/评测工具子包（最初来源于 v-l11 数据集）。

说明：
    该子包用于离线从 sqlite DB 抽取网球轨迹片段（shots），并提供分段/过滤等纯函数。
    主要入口：
        - DB 抽取：`extract_db_shots`
        - 类型：`ShotTrajectory`、`TrajectoryFilterConfig`、`ReturnStartConfig`
        - 纯函数：shot 切分与过滤、回球起点启发式

注意：
    这些模块主要用于离线评测/数据准备，不是 `docs/curve.md` 的核心在线算法路径。
"""

from .db import extract_db_shots, load_abs_loc_points
from .return_start import find_return_start_index
from .split import (
    find_bounce_index,
    split_by_gap_threshold,
    split_points_into_shots,
    split_shot_pre_post,
)
from .types import ReturnStartConfig, ShotTrajectory, TrajectoryFilterConfig

__all__ = [
    "ReturnStartConfig",
    "ShotTrajectory",
    "TrajectoryFilterConfig",
    "extract_db_shots",
    "find_return_start_index",
    "find_bounce_index",
    "load_abs_loc_points",
    "split_by_gap_threshold",
    "split_points_into_shots",
    "split_shot_pre_post",
]
