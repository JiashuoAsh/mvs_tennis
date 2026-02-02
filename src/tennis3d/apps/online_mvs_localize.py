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
from typing import Any, Literal, Optional, Sequence, cast

from mvs import MvsDllNotFoundError, load_mvs_binding
from mvs.pipeline import open_quad_capture

from tennis3d.config import load_online_app_config
from tennis3d.detectors import create_detector
from tennis3d.geometry.calibration import apply_sensor_roi_to_calibration, load_calibration
from tennis3d.online.triggering import build_trigger_plan
from tennis3d.pipeline import OnlineGroupWaitTimeout, iter_mvs_image_groups, run_localization_pipeline
from tennis3d.trajectory import CurveStageConfig, apply_curve_stage


_TERMINAL_PRINT_MODE = Literal["best", "all", "none"]


class _JsonlBufferedWriter:
    """JSONL 写入器：支持按条数/按时间间隔 flush。

    说明：
        - 该类只负责写入与 flush 策略，不做 JSON 序列化。
        - 默认策略（flush_every_records=1 且 flush_interval_s=0）等价于“每条都 flush”。
          这是最安全但最慢的方式；若追求吞吐，可提高 flush_every_records 或设置 flush_interval_s。
    """

    def __init__(
        self,
        *,
        f,
        flush_every_records: int,
        flush_interval_s: float,
    ) -> None:
        self._f = f
        self._flush_every_records = int(flush_every_records)
        self._flush_interval_s = float(flush_interval_s)
        self._records_since_flush = 0
        self._last_flush_t = time.monotonic()

    def write_line(self, line: str) -> None:
        self._f.write(line)
        self._f.write("\n")
        self._records_since_flush += 1

        need_flush_by_count = (
            self._flush_every_records > 0
            and self._records_since_flush >= self._flush_every_records
        )
        need_flush_by_time = False
        if self._flush_interval_s > 0:
            now = time.monotonic()
            need_flush_by_time = (now - self._last_flush_t) >= self._flush_interval_s

        if need_flush_by_count or need_flush_by_time:
            self.flush()

    def flush(self) -> None:
        self._f.flush()
        self._records_since_flush = 0
        self._last_flush_t = time.monotonic()


def _normalize_roi_cli(
    *,
    image_width: int,
    image_height: int,
    image_offset_x: int,
    image_offset_y: int,
) -> tuple[int | None, int | None, int, int]:
    """把 ROI 参数从 CLI 风格（0 表示不设置）规整为 pipeline 使用的可选参数。"""

    w = int(image_width or 0)
    h = int(image_height or 0)
    ox = int(image_offset_x or 0)
    oy = int(image_offset_y or 0)

    # 约束：宽高必须同时设置，避免出现“只裁宽不裁高”的歧义。
    if (w > 0) ^ (h > 0):
        raise ValueError("ROI 参数错误：image_width 与 image_height 必须同时设置，或同时为 0（不设置）。")

    if w <= 0 and h <= 0:
        return None, None, ox, oy

    return w, h, ox, oy


def _format_float3(x: object) -> str:
    """把 3D 坐标格式化为稳定的人类可读字符串。"""

    if not isinstance(x, (list, tuple)) or len(x) != 3:
        return str(x)
    try:
        a, b, c = float(x[0]), float(x[1]), float(x[2])
    except Exception:
        return str(x)
    return f"({a:.4f}, {b:.4f}, {c:.4f})"


def _format_xyz(x: object) -> str:
    """把 3D 坐标格式化为带轴名的字符串，避免与符号/坐标轴概念混淆。"""

    if not isinstance(x, (list, tuple)) or len(x) != 3:
        return str(x)
    try:
        a, b, c = float(x[0]), float(x[1]), float(x[2])
    except Exception:
        return str(x)
    return f"(x={a:.4f}, y={b:.4f}, z={c:.4f})"


def _format_delta_map_ms(d: object) -> str:
    """把 {camera: delta_ms} 格式化成稳定字符串。

    说明：
        - delta_ms 为“相对组内中位数”的偏差（可能为正/负）。
        - 为了保证输出稳定可回归，这里按 camera key 排序。
    """

    if not isinstance(d, dict) or not d:
        return str(d)

    items: list[tuple[str, float]] = []
    for k, v in d.items():
        try:
            items.append((str(k), float(v)))
        except Exception:
            continue
    items.sort(key=lambda kv: kv[0])

    inner = ", ".join(f"{k}:{v:+.3f}" for k, v in items)
    return "{" + inner + "}"


def _format_best_ball_line(out_rec: dict[str, Any]) -> str | None:
    """从单条输出记录中生成“最佳球观测”的终端输出行。

    说明：
        - 若该组没有输出任何球（balls 为空），返回 None。
        - 该函数是纯格式化逻辑，便于单测；不做 IO。
    """

    balls = out_rec.get("balls") or []
    if not isinstance(balls, list) or not balls:
        return None

    gi = out_rec.get("group_index")
    t_abs = out_rec.get("capture_t_abs")

    best = balls[0] if isinstance(balls[0], dict) else None
    best_xw = best.get("ball_3d_world") if best is not None else None
    best_used = best.get("used_cameras") if best is not None else None
    best_q = best.get("quality") if best is not None else None
    best_nv = best.get("num_views") if best is not None else None

    # 说明：终端更适合用均值误差（mean），便于直观看整体拟合水平。
    # 若缺少逐相机误差字段，则回退到 median_reproj_error_px。
    best_err_mean = None
    if best is not None:
        reproj = best.get("reprojection_errors")
        if isinstance(reproj, list) and reproj:
            vals: list[float] = []
            for e in reproj:
                if not isinstance(e, dict):
                    continue
                v = e.get("error_px")
                if v is None:
                    continue
                try:
                    vals.append(float(v))
                except Exception:
                    continue
            if vals:
                best_err_mean = float(sum(vals) / len(vals))
        if best_err_mean is None:
            med = best.get("median_reproj_error_px")
            if med is not None:
                try:
                    best_err_mean = float(med)
                except Exception:
                    best_err_mean = None

    # 说明：t_abs 可能为 None（例如时间戳不可用）；此时仅打印 group。
    t_part = f"t={float(t_abs):.6f} " if t_abs is not None else ""
    q_part = f" q={float(best_q):.3f}" if best_q is not None else ""
    nv_part = f" views={int(best_nv)}" if best_nv is not None else ""
    err_part = f" err_mean={float(best_err_mean):.2f}px" if best_err_mean is not None else ""

    # 说明：时间映射后的“组内相机时间差”诊断。
    # - dt_raw：直接用 host_timestamp（归一化到 ms_epoch）统计的组内跨度。
    # - dt_map：用 dev_timestamp_mapping 映射后的 host_ms 统计的组内跨度。
    # - dt_map_by_cam：每台相机相对组内中位数的偏差（毫秒）。
    dt_part = ""
    dt_raw = out_rec.get("time_mapping_host_ms_spread_ms")
    dt_map = out_rec.get("time_mapping_mapped_host_ms_spread_ms")
    dt_map_by_cam = out_rec.get("time_mapping_mapped_host_ms_delta_to_median_by_camera")

    if dt_raw is not None:
        try:
            dt_part += f" dt_raw={float(dt_raw):.3f}ms"
        except Exception:
            pass
    if dt_map is not None:
        try:
            dt_part += f" dt_map={float(dt_map):.3f}ms"
        except Exception:
            pass
    if dt_map_by_cam is not None:
        dt_part += f" dt_map_by_cam={_format_delta_map_ms(dt_map_by_cam)}"

    return (
        f"{t_part}group={gi} xyz_w={_format_xyz(best_xw)}"
        f"{q_part}{nv_part}{err_part}{dt_part} used={best_used}"
    )


def _format_all_balls_lines(out_rec: dict[str, Any]) -> list[str]:
    """从单条输出记录中生成“所有球观测”的终端输出行列表。

    说明：
        - 若该组没有输出任何球（balls 为空），返回空列表。
        - 返回的第一行是 group 级摘要，后续每行对应一个 ball。
        - 该函数是纯格式化逻辑，便于单测；不做 IO。
    """

    balls = out_rec.get("balls") or []
    if not isinstance(balls, list) or not balls:
        return []

    gi = out_rec.get("group_index")
    t_abs = out_rec.get("capture_t_abs")

    t_part = f"t={float(t_abs):.6f} " if t_abs is not None else ""

    dt_part = ""
    dt_raw = out_rec.get("time_mapping_host_ms_spread_ms")
    dt_map = out_rec.get("time_mapping_mapped_host_ms_spread_ms")
    dt_map_by_cam = out_rec.get("time_mapping_mapped_host_ms_delta_to_median_by_camera")
    if dt_raw is not None:
        try:
            dt_part += f" dt_raw={float(dt_raw):.3f}ms"
        except Exception:
            pass
    if dt_map is not None:
        try:
            dt_part += f" dt_map={float(dt_map):.3f}ms"
        except Exception:
            pass
    if dt_map_by_cam is not None:
        dt_part += f" dt_map_by_cam={_format_delta_map_ms(dt_map_by_cam)}"

    header = f"{t_part}group={gi} balls={len(balls)}{dt_part}"
    lines: list[str] = [header]

    for i, b in enumerate(balls):
        if not isinstance(b, dict):
            lines.append(f"  - ball[{i}]=<invalid>")
            continue

        bid = b.get("ball_id", i)
        xw = b.get("ball_3d_world")
        used = b.get("used_cameras")
        q = b.get("quality")
        nv = b.get("num_views")
        tid = b.get("curve_track_id")

        err_mean = None
        reproj = b.get("reprojection_errors")
        if isinstance(reproj, list) and reproj:
            vals: list[float] = []
            for e in reproj:
                if not isinstance(e, dict):
                    continue
                v = e.get("error_px")
                if v is None:
                    continue
                try:
                    vals.append(float(v))
                except Exception:
                    continue
            if vals:
                err_mean = float(sum(vals) / len(vals))
        if err_mean is None:
            med = b.get("median_reproj_error_px")
            if med is not None:
                try:
                    err_mean = float(med)
                except Exception:
                    err_mean = None

        q_part = f" q={float(q):.3f}" if q is not None else ""
        nv_part = f" views={int(nv)}" if nv is not None else ""
        err_part = f" err_mean={float(err_mean):.2f}px" if err_mean is not None else ""
        tid_part = f" track={int(tid)}" if tid is not None else ""

        lines.append(
            f"  - id={bid}{tid_part} xyz_w={_format_xyz(xw)}{q_part}{nv_part}{err_part} used={used}"
        )

    return lines


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
        "--max-wait-seconds",
        type=float,
        default=0.0,
        help=(
            "exit if no full group is received for this many seconds (0 = no limit). "
            "Useful for debugging hardware trigger wiring."
        ),
    )

    # 相机图像参数（可选）。0/空字符串表示不设置，沿用相机当前配置。
    p.add_argument("--pixel-format", default="", help="PixelFormat (e.g. BayerRG8). Empty means do not set.")
    p.add_argument("--image-width", type=int, default=0, help="ROI width (0 = do not set)")
    p.add_argument("--image-height", type=int, default=0, help="ROI height (0 = do not set)")
    p.add_argument("--image-offset-x", type=int, default=0, help="ROI offset X")
    p.add_argument("--image-offset-y", type=int, default=0, help="ROI offset Y")
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
    p.add_argument(
        "--pt-device",
        default="cpu",
        help=(
            "Ultralytics device for --detector pt (default=cpu). "
            "CUDA examples: cuda:0 / 0 / cuda"
        ),
    )
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

    p.add_argument(
        "--out-jsonl-only-when-balls",
        action="store_true",
        help="if set: write JSONL only when balls is non-empty (reduces disk IO)",
    )
    p.add_argument(
        "--out-jsonl-flush-every-records",
        type=int,
        default=1,
        help=(
            "flush JSONL file every N records (1 = flush every record; 0 = disable count-based flush)"
        ),
    )
    p.add_argument(
        "--out-jsonl-flush-interval-s",
        type=float,
        default=0.0,
        help=(
            "flush JSONL file if time since last flush exceeds this seconds (0 = disable time-based flush)"
        ),
    )

    p.add_argument(
        "--terminal-print-mode",
        choices=["best", "all", "none"],
        default="best",
        help=(
            "terminal output mode: best prints only the top-1 ball per group; "
            "all prints all balls in the group; none prints nothing"
        ),
    )

    p.add_argument(
        "--terminal-status-interval-s",
        type=float,
        default=0.0,
        help=(
            "print a periodic status line every N seconds (0 = disable). Useful when no balls are detected."
        ),
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
        pt_device = str(getattr(cfg, "pt_device", "cpu") or "cpu").strip() or "cpu"
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
        max_wait_seconds = float(getattr(cfg, "max_wait_seconds", 0.0))
        out_path = Path(cfg.out_jsonl).resolve() if cfg.out_jsonl is not None else None

        out_jsonl_only_when_balls = bool(getattr(cfg, "out_jsonl_only_when_balls", False))
        out_jsonl_flush_every_records = int(getattr(cfg, "out_jsonl_flush_every_records", 1))
        out_jsonl_flush_interval_s = float(getattr(cfg, "out_jsonl_flush_interval_s", 0.0))

        terminal_print_mode = cast(_TERMINAL_PRINT_MODE, str(getattr(cfg, "terminal_print_mode", "best")))
        terminal_status_interval_s = float(getattr(cfg, "terminal_status_interval_s", 0.0))

        trigger_source = str(cfg.trigger.trigger_source)
        master_serial = str(cfg.trigger.master_serial)
        master_line_out = str(cfg.trigger.master_line_out)
        master_line_source = str(cfg.trigger.master_line_source)
        master_line_mode = str(cfg.trigger.master_line_mode)
        soft_trigger_fps = float(cfg.trigger.soft_trigger_fps)
        trigger_activation = str(cfg.trigger.trigger_activation)
        trigger_cache_enable = bool(cfg.trigger.trigger_cache_enable)
        curve_cfg = cfg.curve

        pixel_format = str(getattr(cfg, "pixel_format", "") or "").strip()
        image_width = getattr(cfg, "image_width", None)
        image_height = getattr(cfg, "image_height", None)
        image_offset_x = int(getattr(cfg, "image_offset_x", 0))
        image_offset_y = int(getattr(cfg, "image_offset_y", 0))

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
        pt_device = str(getattr(args, "pt_device", "cpu") or "cpu").strip() or "cpu"
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
        max_wait_seconds = float(getattr(args, "max_wait_seconds", 0.0))
        out_path = Path(args.out_jsonl).resolve() if str(args.out_jsonl).strip() else None

        out_jsonl_only_when_balls = bool(getattr(args, "out_jsonl_only_when_balls", False))
        out_jsonl_flush_every_records = int(getattr(args, "out_jsonl_flush_every_records", 1))
        out_jsonl_flush_interval_s = float(getattr(args, "out_jsonl_flush_interval_s", 0.0))

        terminal_print_mode = cast(_TERMINAL_PRINT_MODE, str(getattr(args, "terminal_print_mode", "best")))
        terminal_status_interval_s = float(getattr(args, "terminal_status_interval_s", 0.0))

        trigger_source = str(args.trigger_source)
        master_serial = str(args.master_serial or "").strip()
        master_line_out = str(args.master_line_out)
        master_line_source = str(args.master_line_source)
        master_line_mode = str(args.master_line_mode)
        soft_trigger_fps = float(args.soft_trigger_fps)
        trigger_activation = str(args.trigger_activation)
        trigger_cache_enable = bool(args.trigger_cache_enable)
        curve_cfg = CurveStageConfig()

        pixel_format = str(getattr(args, "pixel_format", "") or "").strip()
        try:
            image_width, image_height, image_offset_x, image_offset_y = _normalize_roi_cli(
                image_width=int(getattr(args, "image_width", 0)),
                image_height=int(getattr(args, "image_height", 0)),
                image_offset_x=int(getattr(args, "image_offset_x", 0)),
                image_offset_y=int(getattr(args, "image_offset_y", 0)),
            )
        except ValueError as exc:
            print(str(exc))
            return 2

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
    # 关键点：如果相机输出启用了 ROI 裁剪（Width/Height + OffsetX/OffsetY），
    # 那么 detector 输出的 bbox/center 坐标是在 ROI 图像坐标系下（原点为 ROI 左上角）。
    # 但标定通常是按满幅分辨率做的（原点为满幅左上角）。
    # 为了让“检测像素坐标”与“标定内参”一致，这里把满幅标定转换为 ROI 标定（主点平移）。
    if image_width is not None and image_height is not None:
        calib = apply_sensor_roi_to_calibration(
            calib,
            image_width=int(image_width),
            image_height=int(image_height),
            image_offset_x=int(image_offset_x),
            image_offset_y=int(image_offset_y),
        )
    detector = create_detector(
        name=detector_name,
        model_path=model_path,
        conf_thres=float(min_score),
        pt_device=str(pt_device),
    )

    plan = build_trigger_plan(
        serials=serials,
        trigger_source=str(trigger_source),
        master_serial=str(master_serial),
        soft_trigger_fps=float(soft_trigger_fps),
    )
    # 约束：flush 参数必须为非负（CLI 路径会走到这里；config 路径一般已在 loader 中校验）。
    if out_jsonl_flush_every_records < 0:
        print("--out-jsonl-flush-every-records must be >= 0")
        return 2
    if out_jsonl_flush_interval_s < 0:
        print("--out-jsonl-flush-interval-s must be >= 0")
        return 2

    if terminal_status_interval_s < 0:
        print("--terminal-status-interval-s must be >= 0")
        return 2

    f_out = None
    jsonl_writer: _JsonlBufferedWriter | None = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        f_out = out_path.open("w", encoding="utf-8")
        jsonl_writer = _JsonlBufferedWriter(
            f=f_out,
            flush_every_records=int(out_jsonl_flush_every_records),
            flush_interval_s=float(out_jsonl_flush_interval_s),
        )

    groups_done = 0
    records_done = 0
    balls_done = 0

    # 说明：用户更关心“有球时的位置”，无球时逐组打印会刷屏。
    # 这里采用：仅在观测到球的组打印位置。

    try:
        with open_quad_capture(
            binding=binding,
            serials=serials,
            trigger_sources=plan.trigger_sources,
            trigger_activation=str(trigger_activation),
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
            pixel_format=str(pixel_format),
            image_width=(int(image_width) if image_width is not None else None),
            image_height=(int(image_height) if image_height is not None else None),
            image_offset_x=int(image_offset_x),
            image_offset_y=int(image_offset_y),
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
                max_wait_seconds=float(max_wait_seconds),
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

            # 说明：无球阶段默认不逐组打印，但仍可用 terminal_status_interval_s 观察吞吐。
            if str(terminal_print_mode) != "none" or float(terminal_status_interval_s) > 0:
                print("Waiting for first ball observation...")

            # 周期性状态输出（心跳）。
            last_status_t = time.monotonic()
            last_status_records = 0
            last_status_groups = 0
            last_status_capture_host_ms: int | None = None
            last_iter_t: float | None = None
            last_iter_dt_ms: float | None = None
            sum_iter_dt_s = 0.0
            count_iter_dt = 0

            for out_rec in records:
                iter_now = time.monotonic()
                if last_iter_t is not None:
                    # 说明：这里统计的是“相邻两条输出记录的时间间隔”，等价于你体感的每轮循环耗时。
                    # 它包含：等待下一组数据 + 推理/几何 + 写盘/打印 等全部开销。
                    dt_s = iter_now - last_iter_t
                    if dt_s >= 0:
                        last_iter_dt_ms = dt_s * 1000.0
                        sum_iter_dt_s += dt_s
                        count_iter_dt += 1
                last_iter_t = iter_now

                records_done += 1
                balls = out_rec.get("balls") or []
                if isinstance(balls, list):
                    balls_done += int(len(balls))

                # 说明：即使没有球，也周期性打印“当前吞吐”，便于判断是否卡住。
                if float(terminal_status_interval_s) > 0:
                    now = time.monotonic()
                    if (now - last_status_t) >= float(terminal_status_interval_s):
                        dt_s = max(now - last_status_t, 1e-9)
                        rec_delta = records_done - last_status_records
                        grp_delta = groups_done - last_status_groups

                        # 捕获侧节拍（用 capture_host_timestamp 估算）。
                        cap_host_ms = out_rec.get("capture_host_timestamp")
                        cap_fps = None
                        if cap_host_ms is not None:
                            try:
                                cap_host_ms_i = int(cap_host_ms)
                                if last_status_capture_host_ms is not None:
                                    dms = cap_host_ms_i - last_status_capture_host_ms
                                    if dms > 0:
                                        cap_fps = 1000.0 * float(grp_delta) / float(dms)
                                last_status_capture_host_ms = cap_host_ms_i
                            except Exception:
                                pass

                        proc_fps = float(rec_delta) / dt_s

                        loop_ms_avg = None
                        if rec_delta > 0:
                            loop_ms_avg = 1000.0 * dt_s / float(rec_delta)

                        loop_ms_part = ""
                        if loop_ms_avg is not None:
                            loop_ms_part += f" loop_avg~{loop_ms_avg:.1f}ms"
                        if last_iter_dt_ms is not None:
                            loop_ms_part += f" loop_last~{last_iter_dt_ms:.1f}ms"

                        # 粗略处理延迟：created_at - capture_t_abs（两者通常同为 epoch 秒）。
                        lag_ms = None
                        try:
                            ca = out_rec.get("created_at")
                            ta = out_rec.get("capture_t_abs")
                            if ca is not None and ta is not None:
                                lag_ms = (float(ca) - float(ta)) * 1000.0
                        except Exception:
                            lag_ms = None

                        cap_part = f" cap_fps~{cap_fps:.2f}" if cap_fps is not None else ""
                        lag_part = f" lag~{lag_ms:.0f}ms" if lag_ms is not None else ""

                        print(
                            f"status: groups={groups_done} records={records_done} balls={balls_done} "
                            f"proc_fps~{proc_fps:.2f}{cap_part}{lag_part}{loop_ms_part}"
                        )

                        last_status_t = now
                        last_status_records = records_done
                        last_status_groups = groups_done

                        # 说明：重置窗口统计，便于观察最近一段时间的波动。
                        sum_iter_dt_s = 0.0
                        count_iter_dt = 0

                if jsonl_writer is not None:
                    # 说明：若用户只想记录“有球帧”，可以跳过 balls 为空的记录，显著减少写盘量。
                    if not (out_jsonl_only_when_balls and (not isinstance(balls, list) or not balls)):
                        jsonl_writer.write_line(json.dumps(out_rec, ensure_ascii=False))

                if str(terminal_print_mode) == "none":
                    continue
                if str(terminal_print_mode) == "all":
                    lines = _format_all_balls_lines(out_rec)
                    if not lines:
                        continue
                    for ln in lines:
                        print(ln)
                else:
                    line = _format_best_ball_line(out_rec)
                    if line is None:
                        continue
                    print(line)

    except OnlineGroupWaitTimeout as exc:
        print(str(exc))
        return 2
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    finally:
        if f_out is not None:
            # 说明：确保退出时落盘一次，避免最后一段缓冲丢失。
            try:
                if jsonl_writer is not None:
                    jsonl_writer.flush()
            except Exception:
                pass
            f_out.close()

    if str(terminal_print_mode) != "none":
        if out_path is not None:
            print(f"Done. groups={groups_done} records={records_done} balls={balls_done} out={out_path}")
        else:
            print(f"Done. groups={groups_done} records={records_done} balls={balls_done}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
