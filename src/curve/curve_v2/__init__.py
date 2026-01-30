"""curve_v2：legacy 网球轨迹拟合与过滤逻辑。

说明：
    - 该目录用于保留一份可运行的 legacy 版本，便于与 `curve_v3` 做对比。
    - `curve2.Curve` 是主要入口（对外方法 `add_frame`）。
    - `ball_tracer.BallTracer` 提供状态机与回球检测启发式（可选）。

注意：
    - 本模块不依赖项目外部的 `hit/` 包；必要常量在 `bot_motion_config.py` 中提供。
"""

from curve.curve_v2.ball_tracer import BallTracer
from curve.curve_v2.curve2 import Curve

__all__ = [
    "BallTracer",
    "Curve",
]
