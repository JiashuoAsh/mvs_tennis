# -*- coding: utf-8 -*-

"""旧的采集会话模块入口已移除（breaking=1）。

说明：
- 类型/配置已迁移到 `mvs.capture_session_types`。
- 落盘录制实现已迁移到 `mvs.capture_session_recording`。

本模块不再提供任何旧符号（不转发），用于尽早暴露误引用旧入口的问题。
"""

from __future__ import annotations


raise RuntimeError(
    "mvs.capture_session 已拆分并移除：请改用 mvs.capture_session_types 与 mvs.capture_session_recording。"
)
