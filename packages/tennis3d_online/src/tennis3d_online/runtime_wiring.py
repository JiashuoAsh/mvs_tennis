"""在线模式：wiring（装配）工具。"""

from __future__ import annotations

from dataclasses import dataclass

from mvs import TriggerPlan, build_trigger_plan, load_mvs_binding

from tennis3d.geometry.calibration import apply_sensor_roi_to_calibration, load_calibration
from tennis3d_detectors import create_detector

from .spec import OnlineRunSpec


@dataclass(frozen=True, slots=True)
class SensorRoi:
    """传感器 ROI（相机侧裁剪）配置。"""

    width: int
    height: int
    offset_x: int
    offset_y: int


def load_binding(spec: OnlineRunSpec):
    """加载 MVS binding（错误由调用方处理）。"""

    return load_mvs_binding(mvimport_dir=spec.mvimport_dir, dll_dir=spec.dll_dir)


def load_calibration_for_runtime(spec: OnlineRunSpec, *, sensor_roi: SensorRoi | None = None):
    """加载标定，并在需要时把“满幅标定”转换为“ROI 标定”。"""

    calib = load_calibration(spec.calib_path)

    if (not bool(spec.camera_aoi_runtime)) and spec.image_width is not None and spec.image_height is not None:
        roi = sensor_roi or SensorRoi(
            width=int(spec.image_width),
            height=int(spec.image_height),
            offset_x=int(spec.image_offset_x),
            offset_y=int(spec.image_offset_y),
        )

        calib_sizes = {tuple(cam.image_size) for cam in calib.cameras.values()}
        if (
            len(calib_sizes) > 0
            and all((int(w) == int(roi.width) and int(h) == int(roi.height)) for (w, h) in calib_sizes)
            and (int(roi.offset_x) != 0 or int(roi.offset_y) != 0)
        ):
            print(
                "[warn] 标定文件的 image_size 已经等于相机 ROI 输出尺寸，且 offset 非零；"
                "将跳过 apply_sensor_roi_to_calibration() 以避免主点重复平移。"
            )
        else:
            calib = apply_sensor_roi_to_calibration(
                calib,
                image_width=int(roi.width),
                image_height=int(roi.height),
                image_offset_x=int(roi.offset_x),
                image_offset_y=int(roi.offset_y),
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
