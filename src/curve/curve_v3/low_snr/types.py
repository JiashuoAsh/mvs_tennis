"""低 SNR（低信噪比）场景下的轻量决策类型。

设计目标：
    - 只描述“如何处理噪声”的策略结果，不包含任何轨迹预测的业务逻辑。
    - 不依赖 `curve_v3.core`，避免循环依赖。
    - 保持非常小的 API 面：只提供 mode 与诊断指标，方便上游记录/排障。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AxisName = Literal["x", "y", "z"]

# 说明：
# - FULL：正常估计。
# - FREEZE_A：冻结该轴加速度（等价于用线性模型/或在后验里把对应加速度列置零）。
# - STRONG_PRIOR_V：速度强贴先验（或更强的 MAP 先验）。
# - IGNORE_AXIS：该轴观测不进入拟合（w=0），输出由先验传播给出。
AxisMode = Literal["FULL", "FREEZE_A", "STRONG_PRIOR_V", "IGNORE_AXIS"]


@dataclass(frozen=True)
class AxisDecision:
    """单轴的低 SNR 决策结果。"""

    mode: AxisMode

    # 诊断指标：
    # - delta：窗口内 max-min（米）。
    # - sigma_mean：窗口内观测噪声尺度均值（米）。
    # - n：参与统计的点数。
    delta: float
    sigma_mean: float
    n: int


@dataclass(frozen=True)
class WindowDecisions:
    """三轴同步的低 SNR 决策结果。"""

    x: AxisDecision
    y: AxisDecision
    z: AxisDecision

    def modes(self) -> tuple[AxisMode, AxisMode, AxisMode]:
        return self.x.mode, self.y.mode, self.z.mode
