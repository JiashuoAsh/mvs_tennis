from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CurveStageConfig:
    """Curve stage 配置。

    说明：
        该配置刻意保持“少而关键”。curve_v3 内部的细粒度参数仍使用其默认值，
        后续若要暴露再逐步增加。
    """

    enabled: bool = False

    # 主输出算法：
    # - v3：输出落点/落地时间/走廊（默认）
    # - v2：输出 legacy 的接球点候选（curve2）
    # - v3_legacy：输出 v3 的 legacy 适配层接球点候选
    primary: str = "v3"  # v3|v2|v3_legacy

    # 对比开关：用于离线评测 v2 vs v3。
    compare_v2: bool = False
    compare_v3_legacy: bool = False

    # 多球关联/track 管理
    max_tracks: int = 5
    association_dist_m: float = 0.6
    max_missed_s: float = 0.15
    min_dt_s: float = 1e-6

    # 走廊输出：按 y 平面穿越输出一组统计（用于规划挑选击球高度范围）
    corridor_y_min: float = 0.6
    corridor_y_max: float = 1.6
    corridor_y_step: float = 0.1

    # 点级置信度来源
    conf_from: str = "quality"  # quality|constant
    constant_conf: float = 1.0

    # 可选：对输入到 curve 的坐标做轻量变换。
    # 说明：
    # - 仅影响“传给 curve_v3/v2/legacy 的观测值”，不会修改上游定位输出 balls[*].ball_3d_world。
    # - 默认不变换（保持旧行为）。
    # - 若你的坐标系需要把 y 轴翻转并做零点校正，可用：y' = -(y - y_offset_m)。
    y_offset_m: float = 0.0
    y_negate: bool = False

    # legacy 需要的字段（沿用旧接口约定）
    is_bot_fire: int = -1

    # ----------------------------
    # 观测过滤（在进入关联/拟合之前）
    # ----------------------------
    # 说明：
    # - 这些过滤使用 run_localization_pipeline 的输出诊断字段（若存在）。
    # - 默认全部关闭，保持旧行为。
    obs_min_views: int = 0
    obs_min_quality: float = 0.0
    obs_max_median_reproj_error_px: float | None = None
    obs_max_ball_3d_std_m: float | None = None

    # ----------------------------
    # 轨迹有效性/片段（episode）判定
    # ----------------------------
    episode_enabled: bool = False

    # episode 起始判定（唯一方案）：分轴(z方向) + y轴重力一致性。
    episode_buffer_s: float = 0.6
    episode_min_obs: int = 5

    # z 方向门控。
    # - 0：不限制方向，仅用 |dz| / |vz|
    # - +1：要求 dz 为正（z 增大）
    # - -1：要求 dz 为负（z 减小）
    episode_z_dir: int = 0
    episode_min_abs_dz_m: float = 0.25
    episode_min_abs_vz_mps: float = 1.0

    # y 轴重力一致性检查。
    episode_gravity_mps2: float = 9.8
    episode_gravity_tol_mps2: float = 3.0

    # episode 结束判定：
    episode_stationary_speed_mps: float = 0.25
    episode_end_if_stationary_s: float = 0.35
    episode_end_after_predicted_land_s: float = 0.2

    # episode 行为。
    reset_predictor_on_episode_start: bool = True
    reset_predictor_on_episode_end: bool = True
    feed_curve_only_when_episode_active: bool = False
    episode_lock_single_track: bool = False
