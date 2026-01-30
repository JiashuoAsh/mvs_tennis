"""curve_v3 包入口。

导出内容：
    - CurvePredictorV3：两阶段（prior + posterior）新预测器 API。
    - Curve：兼容旧版（legacy/curve2.py）Curve 的适配器。
"""

from curve_v3.config import CurveV3Config
from curve_v3.core import CurvePredictorV3
from curve_v3.legacy import Curve
from curve_v3.types import (
    BallObservation,
    BounceEvent,
    Candidate,
    CorridorOnPlane,
    CorridorByTime,
    FusionInfo,
    PrefitFreezeInfo,
    PosteriorState,
)

__all__ = [
    "BallObservation",
    "BounceEvent",
    "Candidate",
    "CorridorByTime",
    "CorridorOnPlane",
    "Curve",
    "CurvePredictorV3",
    "CurveV3Config",
    "FusionInfo",
    "PrefitFreezeInfo",
    "PosteriorState",
]
