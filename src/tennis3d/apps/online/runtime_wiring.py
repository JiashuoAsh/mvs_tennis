"""在线模式：wiring（装配）工具。

职责：
- 加载 MVS binding；
- 加载标定并在需要时应用 ROI 标定变换；
- 创建 detector；
- 计算触发计划。

说明：
- 该模块属于 entry 层：可以依赖 mvs/tennis3d 的 core，但 core 不应反向依赖它。
- 尽量保持函数纯净（除了 load_mvs_binding/load_calibration 本质上的 IO）。
"""

from __future__ import annotations

from mvs import load_mvs_binding
from mvs.triggering import TriggerPlan, build_trigger_plan

from tennis3d.detectors import create_detector
from tennis3d.geometry.calibration import apply_sensor_roi_to_calibration, load_calibration

from .spec import OnlineRunSpec


def load_binding(spec: OnlineRunSpec):
    """加载 MVS binding（错误由调用方处理）。"""

    return load_mvs_binding(mvimport_dir=spec.mvimport_dir, dll_dir=spec.dll_dir)


def load_calibration_for_runtime(spec: OnlineRunSpec):
    """加载标定，并在需要时把“满幅标定”转换为“ROI 标定”。"""

    calib = load_calibration(spec.calib_path)

    # 关键点：如果相机输出启用了 ROI 裁剪（Width/Height + OffsetX/OffsetY），
    # detector 输出的 bbox/center 坐标是在 ROI 图像坐标系下。
    # 为了保持“检测像素坐标”与“标定内参”一致，这里把满幅标定转换为 ROI 标定（主点平移）。
    if (not bool(spec.camera_aoi_runtime)) and spec.image_width is not None and spec.image_height is not None:
        calib = apply_sensor_roi_to_calibration(
            calib,
            image_width=int(spec.image_width),
            image_height=int(spec.image_height),
            image_offset_x=int(spec.image_offset_x),
            image_offset_y=int(spec.image_offset_y),
        )

    return calib


def build_runtime_detector(spec: OnlineRunSpec):
    """创建 detector（错误由调用方处理）。"""

    return create_detector(
        name=spec.detector_name,
        model_path=spec.model_path,
        conf_thres=float(spec.min_score),
        pt_device=str(spec.pt_device),
    )


def build_runtime_trigger_plan(spec: OnlineRunSpec) -> TriggerPlan:
    """计算触发计划（纯配置计算）。"""

    return build_trigger_plan(
        serials=spec.serials,
        trigger_source=str(spec.trigger_source),
        master_serial=str(spec.master_serial),
        soft_trigger_fps=float(spec.soft_trigger_fps),
    )
