"""在线：使用 MVS 实时取流，检测并三角化输出 3D。

支持两种触发拓扑：
- 纯 Software：所有相机 trigger_source=Software，并对全部相机下发软触发。
- master(Software)-slave(硬触发)：指定 master 串口号；master 使用 Software 并下发软触发；
    其它相机使用 LineX 等硬触发输入，通常由 master 的输出线驱动。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Literal, Optional, Sequence, cast

from mvs import MvsDllNotFoundError, load_mvs_binding
from mvs.pipeline import open_quad_capture

from tennis3d.config import load_online_app_config
from tennis3d.detectors import create_detector
from tennis3d.geometry.calibration import load_calibration
from tennis3d.online.triggering import build_trigger_plan
from tennis3d.pipeline import iter_mvs_image_groups, run_localization_pipeline
from tennis3d.trajectory import CurveStageConfig, apply_curve_stage


def build_arg_parser() -> argparse.ArgumentParser:
    # 说明：Windows 终端编码差异较大，这里尽量使用 ASCII，避免 --help 乱码。
    p = argparse.ArgumentParser(description="Online: localize tennis ball 3D from MVS stream")
    p.add_argument(
        "--config",
        default="",
        help="Optional online config file (.json/.yaml/.yml). If set, other CLI args are ignored.",
    )
    p.add_argument(
        "--mvimport-dir",
        default=None,
        help=(
            "MVS official Python sample bindings directory (MvImport). "
            "Optional; or set env MVS_MVIMPORT_DIR"
        ),
    )
    p.add_argument(
        "--dll-dir",
        default=None,
        help="directory containing MvCameraControl.dll (optional); or set env MVS_DLL_DIR",
    )
    p.add_argument(
        "--serial",
        action="extend",
        nargs="+",
        default=[],
        help="camera serials (ordered) e.g. --serial A B C",
    )
    p.add_argument(
        "--trigger-source",
        default="Software",
        help=(
            "trigger source. If --master-serial is set, this is applied to slaves only "
            "(e.g. Line0/Line1/...). Master always uses Software."
        ),
    )
    p.add_argument(
        "--master-serial",
        default="",
        help=(
            "optional master camera serial. When set: master uses Software trigger; "
            "slaves use --trigger-source (typically Line0)."
        ),
    )
    p.add_argument(
        "--master-line-out",
        default="Line1",
        help=(
            "master output line selector (e.g. Line1). Empty means do not set via script."
        ),
    )
    p.add_argument(
        "--master-line-source",
        default="",
        help=(
            "master output line source (e.g. ExposureStartActive). Empty means do not set via script."
        ),
    )
    p.add_argument(
        "--master-line-mode",
        default="Output",
        help="master output line mode (usually Output)",
    )
    p.add_argument(
        "--soft-trigger-fps",
        type=float,
        default=5.0,
        help=(
            "soft trigger fps. In pure Software mode, sends to all cameras; "
            "in master/slave mode, sends to master only."
        ),
    )
    p.add_argument(
        "--trigger-activation",
        default="RisingEdge",
        help="trigger activation (RisingEdge/FallingEdge)",
    )
    p.add_argument(
        "--trigger-cache-enable",
        action="store_true",
        help="try to enable TriggerCacheEnable (some models may not support)",
    )
    p.add_argument(
        "--group-by",
        choices=["trigger_index", "frame_num", "sequence"],
        default="frame_num",
        help="grouping key",
    )
    p.add_argument("--timeout-ms", type=int, default=1000, help="single frame grab timeout")
    p.add_argument("--group-timeout-ms", type=int, default=1000, help="wait time for assembling a full group")
    p.add_argument("--max-pending-groups", type=int, default=256, help="max pending groups")
    p.add_argument("--max-groups", type=int, default=0, help="stop after N groups (0 = no limit)")
    p.add_argument(
        "--calib",
        default=str(Path(__file__).resolve().parents[3] / "data" / "calibration" / "example_triple_camera_calib.json"),
        help="Calibration path (.json/.yaml/.yml)",
    )
    p.add_argument(
        "--detector",
        choices=["fake", "color", "rknn", "pt"],
        default="fake",
        help="Detector backend",
    )
    p.add_argument("--model", default="", help="Model path (required when --detector rknn or pt)")
    p.add_argument("--min-score", type=float, default=0.25, help="Ignore detections below this confidence")
    p.add_argument("--require-views", type=int, default=2, help="Minimum camera views required")
    p.add_argument(
        "--max-detections-per-camera",
        type=int,
        default=10,
        help="TopK detections kept per camera (to limit combinations)",
    )
    p.add_argument(
        "--max-reproj-error-px",
        type=float,
        default=8.0,
        help="Max reprojection error in pixels for a ball candidate",
    )
    p.add_argument(
        "--max-uv-match-dist-px",
        type=float,
        default=25.0,
        help="Max pixel distance when matching projected 3D point to a detection center",
    )
    p.add_argument(
        "--merge-dist-m",
        type=float,
        default=0.08,
        help="3D merge distance in meters for deduplicating ball candidates",
    )
    p.add_argument(
        "--out-jsonl",
        default="",
        help="Optional output JSONL path (if empty, print only)",
    )

    # 在线时间轴（方案B）：实时滑窗映射 dev_timestamp -> host 时间轴。
    p.add_argument(
        "--time-sync-mode",
        choices=["frame_host_timestamp", "dev_timestamp_mapping"],
        default="frame_host_timestamp",
        help="time axis mode for capture_t_abs",
    )
    p.add_argument("--time-mapping-warmup-groups", type=int, default=20, help="warmup groups before first fit")
    p.add_argument("--time-mapping-window-groups", type=int, default=200, help="sliding window size in groups")
    p.add_argument(
        "--time-mapping-update-every-groups",
        type=int,
        default=5,
        help="refit mapping every N groups after warmup",
    )
    p.add_argument("--time-mapping-min-points", type=int, default=20, help="min pairs per camera to fit")
    p.add_argument("--time-mapping-hard-outlier-ms", type=float, default=50.0, help="hard outlier cutoff in ms")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    # 尽量固定 UTF-8 输出，避免在重定向到文件时出现乱码。
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)

    if str(getattr(args, "config", "") or "").strip():
        cfg = load_online_app_config(Path(str(args.config)).resolve())
        serials = list(cfg.serials)
        mvimport_dir = str(cfg.mvimport_dir) if cfg.mvimport_dir is not None else None
        dll_dir = str(cfg.dll_dir) if cfg.dll_dir is not None else None
        calib_path = Path(cfg.calib).resolve()
        detector_name = str(cfg.detector)
        model_path = Path(cfg.model).resolve() if cfg.model is not None else None
        min_score = float(cfg.min_score)
        require_views = int(cfg.require_views)
        max_detections_per_camera = int(cfg.max_detections_per_camera)
        max_reproj_error_px = float(cfg.max_reproj_error_px)
        max_uv_match_dist_px = float(cfg.max_uv_match_dist_px)
        merge_dist_m = float(cfg.merge_dist_m)
        group_by = cast(Literal["trigger_index", "frame_num", "sequence"], str(cfg.group_by))
        timeout_ms = int(cfg.timeout_ms)
        group_timeout_ms = int(cfg.group_timeout_ms)
        max_pending_groups = int(cfg.max_pending_groups)
        max_groups = int(cfg.max_groups)
        out_path = Path(cfg.out_jsonl).resolve() if cfg.out_jsonl is not None else None

        trigger_source = str(cfg.trigger.trigger_source)
        master_serial = str(cfg.trigger.master_serial)
        master_line_out = str(cfg.trigger.master_line_out)
        master_line_source = str(cfg.trigger.master_line_source)
        master_line_mode = str(cfg.trigger.master_line_mode)
        soft_trigger_fps = float(cfg.trigger.soft_trigger_fps)
        trigger_activation = str(cfg.trigger.trigger_activation)
        trigger_cache_enable = bool(cfg.trigger.trigger_cache_enable)
        curve_cfg = cfg.curve

        time_sync_mode = str(cfg.time_sync_mode)
        time_mapping_warmup_groups = int(cfg.time_mapping_warmup_groups)
        time_mapping_window_groups = int(cfg.time_mapping_window_groups)
        time_mapping_update_every_groups = int(cfg.time_mapping_update_every_groups)
        time_mapping_min_points = int(cfg.time_mapping_min_points)
        time_mapping_hard_outlier_ms = float(cfg.time_mapping_hard_outlier_ms)
    else:
        serials = [s.strip() for s in (args.serial or []) if s.strip()]
        if not serials:
            print("Please provide --serial (one or more).")
            return 2

        mvimport_dir = str(getattr(args, "mvimport_dir", None) or "").strip() or None
        dll_dir = args.dll_dir
        calib_path = Path(args.calib).resolve()
        detector_name = str(args.detector)
        model_path = (Path(args.model).resolve() if str(args.model).strip() else None)
        min_score = float(args.min_score)
        require_views = int(args.require_views)
        max_detections_per_camera = int(args.max_detections_per_camera)
        max_reproj_error_px = float(args.max_reproj_error_px)
        max_uv_match_dist_px = float(args.max_uv_match_dist_px)
        merge_dist_m = float(args.merge_dist_m)
        group_by = cast(Literal["trigger_index", "frame_num", "sequence"], str(args.group_by))
        timeout_ms = int(args.timeout_ms)
        group_timeout_ms = int(args.group_timeout_ms)
        max_pending_groups = int(args.max_pending_groups)
        max_groups = int(args.max_groups)
        out_path = Path(args.out_jsonl).resolve() if str(args.out_jsonl).strip() else None

        trigger_source = str(args.trigger_source)
        master_serial = str(args.master_serial or "").strip()
        master_line_out = str(args.master_line_out)
        master_line_source = str(args.master_line_source)
        master_line_mode = str(args.master_line_mode)
        soft_trigger_fps = float(args.soft_trigger_fps)
        trigger_activation = str(args.trigger_activation)
        trigger_cache_enable = bool(args.trigger_cache_enable)
        curve_cfg = CurveStageConfig()

        time_sync_mode = str(args.time_sync_mode)
        time_mapping_warmup_groups = int(args.time_mapping_warmup_groups)
        time_mapping_window_groups = int(args.time_mapping_window_groups)
        time_mapping_update_every_groups = int(args.time_mapping_update_every_groups)
        time_mapping_min_points = int(args.time_mapping_min_points)
        time_mapping_hard_outlier_ms = float(args.time_mapping_hard_outlier_ms)

    if master_serial and master_serial not in serials:
        print("--master-serial must be one of the provided --serial values.")
        return 2

    try:
        binding = load_mvs_binding(mvimport_dir=mvimport_dir, dll_dir=dll_dir)
    except MvsDllNotFoundError as exc:
        print(str(exc))
        return 2

    calib = load_calibration(calib_path)
    detector = create_detector(
        name=detector_name,
        model_path=model_path,
        conf_thres=float(min_score),
    )

    plan = build_trigger_plan(
        serials=serials,
        trigger_source=str(trigger_source),
        master_serial=str(master_serial),
        soft_trigger_fps=float(soft_trigger_fps),
    )
    f_out = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        f_out = out_path.open("w", encoding="utf-8")

    groups_done = 0
    records_done = 0
    balls_done = 0

    try:
        with open_quad_capture(
            binding=binding,
            serials=serials,
            trigger_sources=plan.trigger_sources,
            trigger_activation=str(args.trigger_activation),
            trigger_cache_enable=bool(trigger_cache_enable),
            timeout_ms=int(timeout_ms),
            group_timeout_ms=int(group_timeout_ms),
            max_pending_groups=int(max_pending_groups),
            group_by=group_by,
            enable_soft_trigger_fps=float(plan.enable_soft_trigger_fps),
            soft_trigger_serials=plan.soft_trigger_serials,
            camera_event_names=(),
            master_serial=str(master_serial),
            master_line_output=str(master_line_out) if master_serial else "",
            master_line_source=str(master_line_source) if master_serial else "",
            master_line_mode=str(master_line_mode) if master_serial else "",
            pixel_format="",
            image_width=None,
            image_height=None,
            image_offset_x=0,
            image_offset_y=0,
            exposure_auto="Off",
            exposure_time_us=10000.0,
            gain_auto="Off",
            gain=12.0,
        ) as cap:
            base_groups_iter = iter_mvs_image_groups(
                cap=cap,
                binding=binding,
                max_groups=max_groups,
                timeout_s=0.5,
                time_sync_mode=str(time_sync_mode),
                time_mapping_warmup_groups=int(time_mapping_warmup_groups),
                time_mapping_window_groups=int(time_mapping_window_groups),
                time_mapping_update_every_groups=int(time_mapping_update_every_groups),
                time_mapping_min_points=int(time_mapping_min_points),
                time_mapping_hard_outlier_ms=float(time_mapping_hard_outlier_ms),
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
                min_score=float(min_score),
                require_views=int(require_views),
                max_detections_per_camera=int(max_detections_per_camera),
                max_reproj_error_px=float(max_reproj_error_px),
                max_uv_match_dist_px=float(max_uv_match_dist_px),
                merge_dist_m=float(merge_dist_m),
                include_detection_details=True,
            )

            # 可选：对 3D 输出做轨迹拟合增强（落点/落地时间/走廊）。
            records = apply_curve_stage(records, curve_cfg)

            for out_rec in records:
                records_done += 1
                balls = out_rec.get("balls") or []
                if isinstance(balls, list):
                    balls_done += int(len(balls))

                if f_out is not None:
                    f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    f_out.flush()

                gi = out_rec.get("group_index")
                best = balls[0] if isinstance(balls, list) and balls else None
                best_xw = best.get("ball_3d_world") if isinstance(best, dict) else None
                best_used = best.get("used_cameras") if isinstance(best, dict) else None
                print(f"group={gi} balls={len(balls) if isinstance(balls, list) else 0} best_Xw={best_xw} used={best_used}")

    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    finally:
        if f_out is not None:
            f_out.close()

    if out_path is not None:
        print(f"Done. groups={groups_done} records={records_done} balls={balls_done} out={out_path}")
    else:
        print(f"Done. groups={groups_done} records={records_done} balls={balls_done}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
