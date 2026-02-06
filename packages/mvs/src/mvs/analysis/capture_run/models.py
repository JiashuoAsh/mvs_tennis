# -*- coding: utf-8 -*-

"""采集运行分析：数据模型。

说明：
- 这里放“结构化结果”的 dataclass，避免 compute/report 之间用一堆松散 dict 传参。
- 为了保持 CLI 与历史输出稳定，字段命名与含义尽量沿用旧实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True, slots=True)
class RunSummary:
    """一次采集运行的核心统计结果（便于 JSON 输出）。"""

    jsonl_lines: int
    records: int
    num_cameras_observed: int
    cameras: List[int]
    serials: List[str]

    # 完整性
    groups_complete: int
    groups_incomplete: int
    frames_per_group_min: int
    frames_per_group_median: float
    frames_per_group_max: int

    # 输出目录结构
    group_dirs: int

    # 图像格式一致性
    width_unique: int
    height_unique: int
    pixel_type_unique: int

    # 丢包
    lost_packet_total: int
    lost_packet_max: int
    groups_with_lost_packet: int

    # 时间（主机侧）
    host_spread_ms_min: int
    host_spread_ms_median: float
    host_spread_ms_max: int

    # 时间（相机侧，单位取决于机型/配置）
    dev_spread_raw_min: int
    dev_spread_raw_median: float
    dev_spread_raw_max: int
    dev_spread_norm_min: int
    dev_spread_norm_median: float
    dev_spread_norm_max: int

    # 频率
    created_dt_s_median: Optional[float]
    approx_fps_median: Optional[float]

    # 频率（更接近“相机实际出图/触发频率”：基于 Grabber 记录的 arrival_monotonic）
    arrival_dt_s_median: Optional[float]
    arrival_fps_median: Optional[float]
    camera_arrival_fps_min: Optional[float]
    camera_arrival_fps_median: Optional[float]
    camera_arrival_fps_max: Optional[float]

    # 频率：发命令（soft trigger）
    soft_trigger_sends: int
    soft_trigger_dt_s_median: Optional[float]
    soft_trigger_fps_median: Optional[float]

    # 频率：曝光（相机事件 ExposureStart/ExposureEnd）
    exposure_events: int
    exposure_event_name: str
    exposure_dt_s_median_host: Optional[float]
    exposure_fps_median_host: Optional[float]
    camera_exposure_fps_min: Optional[float]
    camera_exposure_fps_median: Optional[float]
    camera_exposure_fps_max: Optional[float]
    exposure_dt_ticks_median: Optional[float]

    # 文件
    missing_files: int

    # frame_num 对齐（当使用 frame_num/sequence 分组时更关键）
    frame_num_norm_spread_min: int
    frame_num_norm_spread_median: float
    frame_num_norm_spread_max: int
    groups_with_frame_num_norm_mismatch: int


@dataclass(frozen=True, slots=True)
class RunComputed:
    """分析过程的中间结果（用于渲染报告与构建 payload）。"""

    output_dir: Path
    meta_path: Path
    group_by_values: List[str]

    expected_fps: Optional[float]
    fps_tolerance_ratio: float

    summary: RunSummary
    checks: List[Tuple[str, bool, str]]

    frame_num_continuity_lines: List[str]
    frame_num_continuity_payload: Dict[str, Any]

    per_camera: Dict[str, Any]
    per_serial_exposure: Dict[str, Any]

    soft_trigger_targets: Dict[str, int]
