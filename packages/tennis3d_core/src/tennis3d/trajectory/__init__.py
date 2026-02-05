"""轨迹相关的后处理模块。

说明：
    `tennis3d.pipeline` 只负责“检测 + 多视角定位”产出 3D 点。
    轨迹拟合（curve_v2/v3）属于下游状态机/规划的输入准备，放在独立子包中，
    以便按需组合到在线/离线入口，而不侵入定位核心。
"""

from __future__ import annotations

from tennis3d.trajectory.curve_stage import CurveStage, CurveStageConfig, apply_curve_stage

__all__ = [
    "CurveStage",
    "CurveStageConfig",
    "apply_curve_stage",
]
