"""curve_v3 的通用工具模块。

说明：
    - 这里放与 prior/posterior/corridor 无关、但被多处复用的工具：
      数学小函数、日志工具、分段检测器等。
"""

from curve_v3.utils.bounce_detector import BounceTransitionDetector
from curve_v3.utils.math_utils import (
    constrained_quadratic_fit,
    polyder_val,
    polyval,
    real_roots_of_quadratic,
    weighted_linear_fit,
    weighted_quantile_1d,
    weighted_quantiles_1d,
)
from curve_v3.utils.predictor_logging import default_logger

__all__ = [
    "BounceTransitionDetector",
    "constrained_quadratic_fit",
    "default_logger",
    "polyder_val",
    "polyval",
    "real_roots_of_quadratic",
    "weighted_linear_fit",
    "weighted_quantile_1d",
    "weighted_quantiles_1d",
]
