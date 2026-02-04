"""sources 模块已拆分（breaking=1）。

请改用：
- `tennis3d.pipeline.sources_offline.iter_capture_image_groups`
- `tennis3d.pipeline.sources_online.iter_mvs_image_groups` / `OnlineGroupWaitTimeout`
- `tennis3d.pipeline.time_utils`（时间戳/中位数等纯函数）

保留该文件仅用于在误引用旧入口时尽早失败，避免新旧实现并存导致逻辑漂移。
"""

from __future__ import annotations


raise RuntimeError(
    "sources 模块已拆分并移除：请改用 sources_offline / sources_online / time_utils，或通过 tennis3d.pipeline 的公共出口导入。"
)
