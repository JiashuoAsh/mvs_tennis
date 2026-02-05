"""在线模式运行循环（打开相机并持续输出）。

该模块承载在线模式的 wiring 与运行时循环：
- MVS binding 加载
- 打开多相机取流与组包
- 组装 ROI 控制器
- 调用核心 pipeline（detect -> triangulate）
- Optional写 JSONL 与终端打印

说明：
- 该模块属于 entry 层，可依赖 mvs/tennis3d 的 core，但 core 不应反向依赖它。
"""

from __future__ import annotations

from mvs import MvsDllNotFoundError
from mvs.sdk.runtime_roi import get_int_node_info

from tennis3d.pipeline import OnlineGroupWaitTimeout, iter_mvs_image_groups, run_localization_pipeline
from tennis3d.trajectory import apply_curve_stage

from .capture_factory import open_online_quad_capture
from .jsonl_output import open_optional_jsonl_writer
from .output_loop import run_output_loop
from .roi_controller_factory import build_roi_controller
from .runtime_wiring import (
    build_runtime_detector,
    build_runtime_trigger_plan,
    load_binding,
    load_calibration_for_runtime,
    SensorRoi,
)
from .spec import OnlineRunSpec


def _read_static_sensor_roi_from_capture(*, cap: object, spec: OnlineRunSpec) -> SensorRoi | None:
    """从已打开的相机读取当前传感器 ROI。

    目的：
        - MVS 的 Width/Height/OffsetX/OffsetY 常有步进/对齐约束；
          即使传入了某个 offset，最终“实际生效值”也可能被硬件向下对齐。
        - 若标定平移使用的是 spec 中的 offset，而相机实际输出使用了对齐后的 offset，
          将造成像素坐标系不一致，进而让 3D 定位质量显著下降甚至直接失败。

    Returns:
        若所有相机都能读到一致的 ROI，则返回 SensorRoi；否则返回 None 或抛异常。
    """

    cams = list(getattr(cap, "cameras", None) or [])
    if not cams:
        return None

    rois: list[tuple[str, SensorRoi]] = []
    unreadable: list[str] = []
    for c in cams:
        serial = str(getattr(c, "serial", "") or "").strip() or "<unknown>"
        w = get_int_node_info(binding=c.binding, cam=c.cam, key="Width")
        h = get_int_node_info(binding=c.binding, cam=c.cam, key="Height")
        ox = get_int_node_info(binding=c.binding, cam=c.cam, key="OffsetX")
        oy = get_int_node_info(binding=c.binding, cam=c.cam, key="OffsetY")
        if w is None or h is None or ox is None or oy is None:
            unreadable.append(serial)
            continue
        rois.append(
            (
                serial,
                SensorRoi(
                    width=int(w.cur),
                    height=int(h.cur),
                    offset_x=int(ox.cur),
                    offset_y=int(oy.cur),
                ),
            )
        )

    if unreadable:
        print(
            "[warn] 无法读取部分相机的 ROI 节点（Width/Height/OffsetX/OffsetY）："
            + ",".join(unreadable)
            + "。将退回使用配置中的 image_* 值进行标定平移（可能存在对齐误差）。"
        )

    if not rois:
        return None

    uniq = {(r.width, r.height, r.offset_x, r.offset_y) for _, r in rois}
    if len(uniq) != 1:
        details = "; ".join(
            f"{serial}: {roi.width}x{roi.height}, ox={roi.offset_x}, oy={roi.offset_y}" for serial, roi in rois
        )
        if unreadable:
            details = details + "; unreadable=" + ",".join(unreadable)
        raise RuntimeError(
            "多相机 ROI 配置不一致：这会导致不同相机像素坐标系不一致，无法用一套统一的 ROI 标定平移。"
            "请检查每台相机的 Width/Height/OffsetX/OffsetY 是否一致，或为每相机提供独立 ROI 参数。"
            f"（提示：可用 MVS Client 查看；本次读取摘要：{details}）"
        )

    actual = rois[0][1]
    if spec.image_width is not None and spec.image_height is not None:
        cfg = SensorRoi(
            width=int(spec.image_width),
            height=int(spec.image_height),
            offset_x=int(spec.image_offset_x),
            offset_y=int(spec.image_offset_y),
        )
        if (actual.width, actual.height, actual.offset_x, actual.offset_y) != (
            cfg.width,
            cfg.height,
            cfg.offset_x,
            cfg.offset_y,
        ):
            print(
                "[warn] 相机实际 ROI 与配置不一致："
                f" cfg=({cfg.width}x{cfg.height}, ox={cfg.offset_x}, oy={cfg.offset_y})"
                f" actual=({actual.width}x{actual.height}, ox={actual.offset_x}, oy={actual.offset_y})。"
                "将使用 actual 值进行标定平移。"
            )

    return actual


def run_online(spec: OnlineRunSpec) -> int:
    """运行在线模式。

    Args:
        spec: 运行规格（已完成参数校验）。

    Returns:
        进程退出码（0 表示成功）。
    """

    try:
        binding = load_binding(spec)
    except MvsDllNotFoundError as exc:
        print(str(exc))
        return 2

    detector = build_runtime_detector(spec)
    plan = build_runtime_trigger_plan(spec)

    groups_done = 0
    records_done = 0
    balls_done = 0

    try:
        with open_optional_jsonl_writer(spec) as jsonl_writer:
            with open_online_quad_capture(binding=binding, spec=spec, plan=plan) as cap:
                sensor_roi = None
                if (not bool(spec.camera_aoi_runtime)) and spec.image_width is not None and spec.image_height is not None:
                    sensor_roi = _read_static_sensor_roi_from_capture(cap=cap, spec=spec)

                calib = load_calibration_for_runtime(spec, sensor_roi=sensor_roi)
                roi_controller = build_roi_controller(spec=spec, cap=cap, binding=binding)

                base_groups_iter = iter_mvs_image_groups(
                    cap=cap,
                    binding=binding,
                    max_groups=spec.max_groups,
                    timeout_s=0.5,
                    max_wait_seconds=float(spec.max_wait_seconds),
                    time_sync_mode=str(spec.time_sync_mode),
                    time_mapping_warmup_groups=int(spec.time_mapping_warmup_groups),
                    time_mapping_window_groups=int(spec.time_mapping_window_groups),
                    time_mapping_update_every_groups=int(spec.time_mapping_update_every_groups),
                    time_mapping_min_points=int(spec.time_mapping_min_points),
                    time_mapping_hard_outlier_ms=float(spec.time_mapping_hard_outlier_ms),
                )

                def _counting_groups():
                    nonlocal groups_done
                    for meta, images in base_groups_iter:
                        groups_done += 1
                        yield meta, images

                records = run_localization_pipeline(
                    groups=_counting_groups(),
                    calib=calib,
                    detector=detector,
                    min_score=float(spec.min_score),
                    require_views=int(spec.require_views),
                    max_detections_per_camera=int(spec.max_detections_per_camera),
                    max_reproj_error_px=float(spec.max_reproj_error_px),
                    max_uv_match_dist_px=float(spec.max_uv_match_dist_px),
                    merge_dist_m=float(spec.merge_dist_m),
                    include_detection_details=True,
                    roi_controller=roi_controller,
                )

                # Optional：对 3D 输出做轨迹拟合增强（落点/落地时间/走廊）。
                records = apply_curve_stage(records, spec.curve_cfg)

                def _get_groups_done() -> int:
                    return int(groups_done)

                records_done, balls_done = run_output_loop(
                    records=records,
                    jsonl_writer=jsonl_writer,
                    spec=spec,
                    get_groups_done=_get_groups_done,
                )

    except OnlineGroupWaitTimeout as exc:
        print(str(exc))
        return 2
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except RuntimeError as exc:
        # 说明：该分支用于 ROI 控制器构建时的“可预期失败”。
        print(str(exc))
        return 2

    if str(spec.terminal_print_mode) != "none":
        if spec.out_path is not None:
            print(
                f"Done. groups={groups_done} records={records_done} balls={balls_done} out={spec.out_path}"
            )
        else:
            print(f"Done. groups={groups_done} records={records_done} balls={balls_done}")
    return 0

