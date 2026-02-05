"""curve_v2 所需的最小 BotMotionConfig。

该 legacy 代码最初依赖外部工程（例如 `hit.cfgs.bot_motion_config.BotMotionConfig`）。
为了让 `curve_v2` 在本仓库内可独立运行，这里提供一份“最小可用”的常量集合。

注意：
    - 这些值主要用于让算法流程跑通与回放对比。
    - 若你要做严格的物理/策略对齐，请用真实业务侧的配置替换这些默认值。
"""

from __future__ import annotations


class BotMotionConfig:
    """legacy 轨迹预测与过滤用常量。

    这里只保留 `curve2.py` / `ball_tracer.py` 中被引用到的字段。
    """

    # -------------------- legacy：反弹速度经验模型 --------------------
    # 反弹速度预测为 vx' = vx / (1 + a*vx)
    # legacy 代码中只把它作为常量挂载（部分版本会使用）。
    BOUNCE_VX_A: float = 0.05

    # X/Z 方向反弹：v_new = v_old * k + b
    # 说明：这里给出一组偏“温和”的默认值，避免反弹后速度夸张。
    BOUNCE_X_COEFF: tuple[float, float] = (0.64, 0.0)
    BOUNCE_Z_COEFF: tuple[float, float] = (0.64, 0.0)

    # Y 方向反弹：vy_new = -(vy_old * k + b)
    BOUNCE_Y_COEFF: tuple[float, float] = (0.67, 0.0)

    # -------------------- legacy：接球高度区间（过网/击球区） --------------------
    NET_HEIGHT_1: float = 0.40
    NET_HEIGHT_2: float = 1.10

    # -------------------- ball_tracer：用户发球/回球检测启发式 --------------------
    IS_USER_SERVE: bool = False

    # 回球检测：球与机器人之间最小距离（m）
    RETURN_BALL_MINIMUS_DIS: float = 1.0

    # 回球检测：在最近窗口内，球必须“远->近”至少变化这么多（m）
    RETURN_BALL_BEGIN_DIS: float = 0.6

    # 回球检测：第一帧与第二帧的距离差必须超过该阈值（m）
    RETURN_BALL_FIRST_GAP: float = 0.12
