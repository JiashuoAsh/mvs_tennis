from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tennis3d.trajectory.curve_stage.config import CurveStageConfig


@dataclass
class _CurveImports:
    """curve 算法实现的延迟 import 结果（按需填充）。"""

    CurvePredictorV3: Any | None = None
    BallObservation: Any | None = None
    CurveV2: Any | None = None
    CurveV3Legacy: Any | None = None


def _ensure_curve_imports(cfg: CurveStageConfig) -> _CurveImports:
    """按配置决定需要哪些 curve 模块，并延迟导入。

    说明：
        - 该逻辑集中在一个模块里，便于清晰表达依赖边界。
        - 只在 cfg.enabled=True 且真正处理记录时才会触发。
    """

    need_v3 = str(cfg.primary) == "v3"
    need_v2 = str(cfg.primary) == "v2" or bool(cfg.compare_v2)
    need_v3_legacy = str(cfg.primary) == "v3_legacy" or bool(cfg.compare_v3_legacy)

    out = _CurveImports()

    if need_v3:
        from curve.curve_v3.core import CurvePredictorV3
        from curve.curve_v3.types import BallObservation

        out.CurvePredictorV3 = CurvePredictorV3
        out.BallObservation = BallObservation

    if need_v2:
        from curve.curve_v2 import Curve as CurveV2

        out.CurveV2 = CurveV2

    if need_v3_legacy:
        from curve.curve_v3.legacy import Curve as CurveV3Legacy

        out.CurveV3Legacy = CurveV3Legacy

    return out
