# -*- coding: utf-8 -*-

"""MVS 采集封装包。

目标：把相机初始化、取流、保存/处理流程做成可复用的 API。

注意：
- 该包依赖海康 MVS 的 Python ctypes 示例绑定（MvImport 目录）。
- 运行机器需要能找到 MvCameraControl.dll（或通过参数/环境变量提供）。
"""

from mvs.binding import MvsBinding, MvsDllNotFoundError, load_mvs_binding
from mvs.camera import (
    MvsCamera,
    MvsError,
    MvsSdk,
    configure_exposure,
    configure_pixel_format,
    configure_resolution,
    configure_trigger,
)
from mvs.devices import DeviceDesc, enumerate_devices
from mvs.events import MvsEvent
from mvs.grab import FramePacket, Grabber
from mvs.grouping import TriggerGroupAssembler
from mvs.pipeline import QuadCapture, open_quad_capture
from mvs.save import save_frame_as_bmp
from mvs.soft_trigger import SoftwareTriggerLoop
from mvs.bandwidth import BandwidthEstimate, estimate_camera_bandwidth, format_bandwidth_report
from mvs.capture_session import (
    CaptureSessionConfig,
    CaptureSessionResult,
    TriggerPlan,
    build_trigger_plan,
    normalize_roi,
    run_capture_session,
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
]
