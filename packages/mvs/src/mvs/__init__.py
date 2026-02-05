# -*- coding: utf-8 -*-

"""MVS 采集封装包。

目标：把相机初始化、取流、保存/处理流程做成可复用的 API。

注意：
- 该包依赖海康 MVS 的 Python ctypes 示例绑定（MvImport 目录）。
- 运行机器需要能找到 MvCameraControl.dll（或通过参数/环境变量提供）。
"""

from mvs.sdk.binding import MvsBinding, MvsDllNotFoundError, load_mvs_binding
from mvs.sdk.camera import (
    MvsCamera,
    MvsError,
    MvsSdk,
    configure_exposure,
    configure_pixel_format,
    configure_resolution,
    configure_trigger,
)
from mvs.sdk.devices import DeviceDesc, enumerate_devices
from mvs.core.events import MvsEvent
from mvs.capture.grab import FramePacket, Grabber
from mvs.capture.grouping import TriggerGroupAssembler
from mvs.capture.pipeline import QuadCapture, open_quad_capture
from mvs.capture.save import save_frame_as_bmp
from mvs.capture.soft_trigger import SoftwareTriggerLoop
from mvs.capture.bandwidth import BandwidthEstimate, estimate_camera_bandwidth, format_bandwidth_report
from mvs.session.capture_session_recording import run_capture_session
from mvs.session.capture_session_types import CaptureSessionConfig, CaptureSessionResult
from mvs.core.roi import normalize_roi
from mvs.capture.triggering import TriggerPlan, build_trigger_plan
from mvs.session.time_mapping import (
    LinearTimeMapping,
    OnlineDevToHostMapper,
    collect_frame_pairs_from_metadata,
    fit_dev_to_host_ms,
    load_time_mappings_json,
    save_time_mappings_json,
)

__all__ = [
    "BandwidthEstimate",
    "CaptureSessionConfig",
    "CaptureSessionResult",
    "DeviceDesc",
    "FramePacket",
    "Grabber",
    "MvsEvent",
    "TriggerPlan",
    "build_trigger_plan",
    "estimate_camera_bandwidth",
    "format_bandwidth_report",
    "MvsError",
    "MvsBinding",
    "MvsCamera",
    "MvsDllNotFoundError",
    "MvsSdk",
    "QuadCapture",
    "SoftwareTriggerLoop",
    "TriggerGroupAssembler",
    "configure_exposure",
    "configure_pixel_format",
    "configure_resolution",
    "configure_trigger",
    "enumerate_devices",
    "load_mvs_binding",
    "normalize_roi",
    "open_quad_capture",
    "run_capture_session",
    "save_frame_as_bmp",
    "LinearTimeMapping",
    "collect_frame_pairs_from_metadata",
    "fit_dev_to_host_ms",
    "load_time_mappings_json",
    "save_time_mappings_json",
    "OnlineDevToHostMapper",
]
