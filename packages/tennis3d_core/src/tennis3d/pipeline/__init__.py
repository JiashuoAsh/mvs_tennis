"""在线/离线共用的 3D 定位流水线。

该包提供可复用的流水线“积木”：
- `sources_offline`/`sources_online`：从在线 MVS 取流或离线 captures 中产出每组图像 (meta, images_by_camera)
- `core`：对每组图像执行检测 + 多视角定位，并产出可 JSON 序列化的记录

设计目标：让 `tennis3d.apps.*` 只承担命令行参数解析与 I/O（保持入口脚本尽量薄）。
"""

from .core import run_localization_pipeline
from .sources_offline import iter_capture_image_groups
from .sources_online import OnlineGroupWaitTimeout, iter_mvs_image_groups

__all__ = [
    "iter_capture_image_groups",
    "iter_mvs_image_groups",
    "OnlineGroupWaitTimeout",
    "run_localization_pipeline",
]
