"""在线模式 ROI 控制器构建。"""

from __future__ import annotations

from typing import Any

from mvs.sdk.runtime_roi import get_int_node_info, try_set_int_node

from tennis3d.pipeline.roi import KinematicRoiConfig, KinematicRoiController
from tennis3d.pipeline.two_stage_roi import (
    CameraAoiRuntimeConfig,
    CameraAoiState,
    SoftwareCropConfig,
    TwoStageKinematicRoiController,
)

from .spec import OnlineRunSpec


def build_roi_controller(*, spec: OnlineRunSpec, cap: Any, binding: Any):
    """构建 ROI 控制器（Optional）。"""

    if bool(spec.camera_aoi_runtime):
        aoi_state_by_camera: dict[str, CameraAoiState] = {}
        for c in list(getattr(cap, "cameras", None) or []):
            serial = str(getattr(c, "serial", "") or "").strip()
            if not serial:
                continue

            w_info = get_int_node_info(binding=c.binding, cam=c.cam, key="Width")
            h_info = get_int_node_info(binding=c.binding, cam=c.cam, key="Height")
            ox_info = get_int_node_info(binding=c.binding, cam=c.cam, key="OffsetX")
            oy_info = get_int_node_info(binding=c.binding, cam=c.cam, key="OffsetY")

            aoi_w = int(w_info.cur) if w_info is not None else (
                int(spec.image_width) if spec.image_width is not None else 0
            )
            aoi_h = int(h_info.cur) if h_info is not None else (
                int(spec.image_height) if spec.image_height is not None else 0
            )

            if ox_info is None or oy_info is None:
                raise RuntimeError(
                    f"camera_aoi_runtime enabled but cannot read OffsetX/OffsetY node info for {serial}. "
                    "请确认该机型在取流中 OffsetX/OffsetY 可读，并且未被权限/状态锁定。"
                )
            if aoi_w <= 0 or aoi_h <= 0:
                raise RuntimeError(
                    f"camera_aoi_runtime enabled but cannot determine AOI size (Width/Height) for {serial}. "
                    "请设置 --image-width/--image-height，或确保 Width/Height 节点可读。"
                )

            aoi_state_by_camera[serial] = CameraAoiState(
                aoi_width=int(aoi_w),
                aoi_height=int(aoi_h),
                offset_x=int(ox_info.cur),
                offset_y=int(oy_info.cur),
                offset_x_info=ox_info,
                offset_y_info=oy_info,
                initial_offset_x=int(ox_info.cur),
                initial_offset_y=int(oy_info.cur),
            )

        cameras_by_serial = {
            str(c.serial): c
            for c in list(getattr(cap, "cameras", None) or [])
            if str(getattr(c, "serial", "") or "").strip()
        }

        class _Applier:
            def __init__(self, cams: dict[str, Any]):
                self._cams = cams

            def set_offsets(self, *, camera: str, offset_x: int, offset_y: int) -> bool:
                c = self._cams.get(str(camera))
                if c is None:
                    return False

                okx, _ = try_set_int_node(binding=c.binding, cam=c.cam, key="OffsetX", value=int(offset_x))
                oky, _ = try_set_int_node(binding=c.binding, cam=c.cam, key="OffsetY", value=int(offset_y))
                return bool(okx and oky)

        return TwoStageKinematicRoiController(
            crop_cfg=SoftwareCropConfig(
                crop_width=int(spec.detector_crop_size),
                crop_height=int(spec.detector_crop_size),
                smooth_alpha=float(spec.detector_crop_smooth_alpha),
                max_step_px=int(spec.detector_crop_max_step_px),
                reset_after_missed=int(spec.detector_crop_reset_after_missed),
            ),
            camera_cfg=CameraAoiRuntimeConfig(
                enabled=True,
                update_every_groups=int(spec.camera_aoi_update_every_groups),
                min_move_px=int(spec.camera_aoi_min_move_px),
                smooth_alpha=float(spec.camera_aoi_smooth_alpha),
                max_step_px=int(spec.camera_aoi_max_step_px),
                recenter_after_missed=int(spec.camera_aoi_recenter_after_missed),
            ),
            aoi_state_by_camera=aoi_state_by_camera,
            applier=_Applier(cameras_by_serial),
        )

    if int(spec.detector_crop_size) > 0:
        return KinematicRoiController(
            cfg=KinematicRoiConfig(
                crop_width=int(spec.detector_crop_size),
                crop_height=int(spec.detector_crop_size),
                smooth_alpha=float(spec.detector_crop_smooth_alpha),
                max_step_px=int(spec.detector_crop_max_step_px),
                reset_after_missed=int(spec.detector_crop_reset_after_missed),
            )
        )

    return None
