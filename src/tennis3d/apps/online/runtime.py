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
)
from .spec import OnlineRunSpec


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

    calib = load_calibration_for_runtime(spec)
    detector = build_runtime_detector(spec)
    plan = build_runtime_trigger_plan(spec)

    groups_done = 0
    records_done = 0
    balls_done = 0

    try:
        with open_optional_jsonl_writer(spec) as jsonl_writer:
            with open_online_quad_capture(binding=binding, spec=spec, plan=plan) as cap:
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

