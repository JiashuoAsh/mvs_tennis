"""低 SNR（低信噪比）专用包。

该包聚焦“噪声处理策略”，不关心反弹预测的业务细节：
    - 由 conf 估计观测噪声尺度 σ 与权重 w。
    - 用窗口内可辨识性指标（Δu vs \bar{σ}）做退化判别，输出 mode_x/mode_y/mode_z。

使用方式（上层）：
    - prefit/posterior 在构造线性系统前，调用 :func:`analyze_window` 得到决策。
    - 根据 mode 决定：冻结加速度/速度强先验/忽略轴等动作。

注意：
    - 该包不依赖 `curve_v3.core`，避免耦合与循环依赖。
"""

from curve_v3.low_snr.policy import LowSnrPolicyParams, analyze_window, sigma_from_conf, weights_from_conf
from curve_v3.low_snr.types import AxisDecision, AxisMode, WindowDecisions

__all__ = [
    "AxisDecision",
    "AxisMode",
    "WindowDecisions",
    "LowSnrPolicyParams",
    "sigma_from_conf",
    "weights_from_conf",
    "analyze_window",
]
