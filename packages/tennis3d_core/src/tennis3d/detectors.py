"""旧检测器模块已移除。

说明：
- detectors 已拆分到独立顶层包 `tennis3d_detectors`。
- 核心协议 `Detector` 已下沉到 `tennis3d.models.Detector`。
"""

from __future__ import annotations


raise ImportError(
    "`tennis3d.detectors` 已拆分为 `tennis3d_detectors`；"
    "Detector 协议在 `tennis3d.models.Detector`。"
)

