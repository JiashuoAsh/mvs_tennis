"""tennis3d：网球检测 + 多相机 3D 定位业务库。

说明：
- 采集与相机 SDK 封装在顶层包 `mvs`（src/mvs）。
- 本包提供几何（geometry）、融合（localization）、在线/离线共享流水线（pipeline）以及应用入口（apps）。

对外推荐从 `tennis3d.api` 导入少量稳定入口函数，避免外部项目依赖内部目录结构。
"""

from tennis3d.api import build_calibration, build_detector, iter_localization_from_captures

__all__ = [
    "build_calibration",
    "build_detector",
    "iter_localization_from_captures",
]

