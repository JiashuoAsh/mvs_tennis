"""v-l11.db 离线抽取相关的数据结构。

说明：
    这些类型被 DB 抽取、过滤与切分逻辑共同使用。
    单独成文件的目的：避免模块之间产生循环依赖，同时让 DB 抽取代码
    更聚焦在“DB IO + 高层抽取 API”。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from curve_v3.types import BallObservation


@dataclass(frozen=True)
class ShotTrajectory:
    """从 DB 中提取的一段 shot 轨迹。"""

    shot_index: int
    points: list[BallObservation]
    bounce_index: int | None
    pre_points: list[BallObservation]
    post_points: list[BallObservation]

    # 保留“整段轨迹”的原始点序列（在噪点过滤/后处理之前）。
    # 说明：
    #   - points/pre_points/post_points 可能会因为 filter_config 而显著减少；
    #   - all_* 字段用于“想要完整轨迹(起止时间+中间所有点)”的场景。
    all_points: list[BallObservation] = field(default_factory=list)
    all_bounce_index: int | None = None
    all_pre_points: list[BallObservation] = field(default_factory=list)
    all_post_points: list[BallObservation] = field(default_factory=list)


@dataclass(frozen=True)
class TrajectoryFilterConfig:
    """用于离线抽取的轻量噪点过滤配置。

    目标：
        在 DB 存在“手里乱挥/误识别”等杂点时，提高 shot 切分与 bounce 检测的鲁棒性。

    说明：
        默认关闭（enabled=False），避免在不知情的情况下改变既有行为。
        建议通过脚本参数（例如 --filter-noise）显式开启。
    """

    enabled: bool = False

    # 点级门禁：基于 dt 与速度范围的硬过滤。
    min_dt_s: float = 0.005
    max_dt_s: float = 0.200
    max_speed_m_s: float = 40.0
    z_speed_range: tuple[float, float] = (0.5, 27.0)

    # 段级一致性：z 方向应大多一致（不强制要求 vz>0）。
    min_forward_ratio: float = 0.70

    # 段级一致性：y 轴应体现“整体向下的加速度”。
    # 约定：要求 a_y <= -min_downward_acc_m_s2 才认为满足“向下加速度”。
    min_downward_acc_m_s2: float = 2.0
    gravity_mad_k: float = 4.0
    max_gravity_mad_ratio: float = 1.2

    # 从一个粗分组中提取“最佳连续段”时，允许跳过的异常点数量。
    max_skip_points: int = 3

    # 过滤后可用于后续逻辑的最少点数。
    min_run_points: int = 10

    # 如果粗分组很多，按分数选 top-K；None 表示用 expected_num_shots。
    select_top_k: int | None = None


@dataclass(frozen=True)
class ReturnStartConfig:
    """参考 is_user_return_ball 的回球开始启发式配置。

    说明：
        该启发式用于在一个 shot 内裁剪掉“回球真正开始之前”的无关点段。
        它依赖 z 方向的“持续变近”(z 递减)判定。
    """

    return_len: int = 4
    min_y_m: float = 0.6
    max_window_s: float = 0.5

    # 以下阈值在真实系统中来自 BotMotionConfig；离线抽取使用保守默认值。
    min_ball_bot_dis_m: float = 0.3
    begin_dis_m: float = 0.6
    first_gap_m: float = 0.1

    # ball_bot_dis = ball_z - bot_z；离线 DB 缺少 bot 位姿时，用常量近似。
    bot_z_m: float = 0.0
