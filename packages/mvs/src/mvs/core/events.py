# -*- coding: utf-8 -*-

"""事件结构定义（用于事件队列）。

说明：
- 本包内部会把一些“诊断事件”写入 `event_queue`（例如相机事件回调、软触发下发记录）。
- 事件使用 dict 的好处是易序列化（JSONL）；但缺点是结构不自解释。
- 这里用 TypedDict 给事件字段“定名”，让 IDE/静态检查更友好，同时不引入运行时依赖。
"""

from __future__ import annotations

from typing import Literal, TypedDict


class CameraEvent(TypedDict):
    """相机事件（来自 MV_CC_RegisterEventCallBackEx）。"""

    type: Literal["camera_event"]
    created_at: float
    host_monotonic: float
    serial: str

    event_name: str
    requested_event_name: str
    event_id: int
    stream_channel: int
    block_id: int
    event_timestamp: int


class SoftTriggerSendEvent(TypedDict):
    """软触发下发事件（诊断用）。"""

    type: Literal["soft_trigger_send"]
    seq: int
    created_at: float
    host_monotonic: float
    targets: list[str]


MvsEvent = CameraEvent | SoftTriggerSendEvent
