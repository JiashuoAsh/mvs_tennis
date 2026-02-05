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

from dataclasses import dataclass

from mvs import TriggerPlan, build_trigger_plan, load_mvs_binding

from tennis3d.detectors import create_detector
from tennis3d.geometry.calibration import apply_sensor_roi_to_calibration, load_calibration

from .spec import OnlineRunSpec


@dataclass(frozen=True, slots=True)
class SensorRoi:
    """传感器 ROI（相机侧裁剪）配置。

    说明：
        - 本结构体表达的是“相机实际输出”的 ROI：Width/Height/OffsetX/OffsetY。
        - 在线模式下：
          - static ROI（camera_aoi_runtime=false）应把满幅标定转换为该 ROI 坐标系；
          - runtime AOI（camera_aoi_runtime=true）则不应静态平移标定主点。
    """

    width: int
    height: int
    offset_x: int
    offset_y: int


def load_binding(spec: OnlineRunSpec):
    """加载 MVS binding（错误由调用方处理）。"""

    return load_mvs_binding(mvimport_dir=spec.mvimport_dir, dll_dir=spec.dll_dir)


def load_calibration_for_runtime(spec: OnlineRunSpec, *, sensor_roi: SensorRoi | None = None):
    """加载标定，并在需要时把“满幅标定”转换为“ROI 标定”。

    Args:
        spec: 在线运行规格。
        sensor_roi: 可选的“相机实际 ROI”。用于处理 ROI 对齐/机型约束导致的
            (Width/Height/Offset) 与配置不完全一致的情况。
    """

    calib = load_calibration(spec.calib_path)

    # 关键点：如果相机输出启用了 ROI 裁剪（Width/Height + OffsetX/OffsetY），
    # detector 输出的 bbox/center 坐标是在 ROI 图像坐标系下。
    # 为了保持“检测像素坐标”与“标定内参”一致，这里把满幅标定转换为 ROI 标定（主点平移）。
    if (not bool(spec.camera_aoi_runtime)) and spec.image_width is not None and spec.image_height is not None:
        roi = sensor_roi or SensorRoi(
            width=int(spec.image_width),
            height=int(spec.image_height),
            offset_x=int(spec.image_offset_x),
            offset_y=int(spec.image_offset_y),
        )

        # 防呆：如果标定文件本身的 image_size 已经等于 ROI 输出尺寸，且 offset 非零，
        # 这通常意味着用户的标定已经是在 ROI 坐标系下完成的。
        # 此时再次平移主点会导致“二次 offset”，从而让 3D 定位直接崩掉。
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
