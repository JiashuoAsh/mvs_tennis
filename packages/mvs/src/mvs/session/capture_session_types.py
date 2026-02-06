# -*- coding: utf-8 -*-

"""采集会话类型定义。

职责：
- 只定义数据结构与类型别名（配置/结果/枚举型字符串）。

边界：
- 该模块不做任何 SDK 操作，不做任何文件 I/O，也不打印日志。
- 需要执行“落盘采集”的实现请使用 `mvs.run_capture_session()`（见 `mvs.session.capture_session_recording`）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mvs.capture.triggering import TriggerPlan


GroupBy = Literal["frame_num", "sequence"]
SaveMode = Literal["none", "sdk-bmp", "raw"]


@dataclass(frozen=True)
class CaptureSessionConfig:
    """一次采集会话的配置（用于可复用调用）。"""

    serials: list[str]
    trigger_plan: TriggerPlan
    trigger_activation: str
    trigger_cache_enable: bool
    timeout_ms: int
    group_timeout_ms: int
    max_pending_groups: int
    group_by: GroupBy
    save_mode: SaveMode
    output_dir: Path
    max_groups: int
    bayer_method: int
    max_wait_seconds: float
    idle_log_seconds: float
    camera_event_names: list[str]

    # master/slave
    master_serial: str
    master_line_out: str
    master_line_source: str
    master_line_mode: str

    # 带宽估算
    expected_fps: float
    soft_trigger_fps: float

    # 图像参数
    pixel_format: str
    image_width: int | None
    image_height: int | None
    image_offset_x: int
    image_offset_y: int

    # 曝光/增益
    exposure_auto: str
    exposure_time_us: float
    gain_auto: str
    gain: float


@dataclass(frozen=True)
class CaptureSessionResult:
    """一次采集会话的结果（便于外部程序做后续处理/验收）。"""

    exit_code: int
    groups_done: int
    output_dir: Path
    metadata_path: Path
