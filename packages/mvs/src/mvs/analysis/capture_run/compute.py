# -*- coding: utf-8 -*-

"""采集运行分析：统计计算。

说明：
- 该模块尽量只做“计算”，不负责长文本报告的排版。
- 报告渲染在 report.py；对外 API 在包 __init__.py。
"""

from __future__ import annotations

import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .event_tools import compute_exposure_stats, compute_soft_trigger_stats
from .io import read_jsonl_records
from .models import RunComputed, RunSummary
from .series_tools import (
    build_frame_num_continuity,
    build_per_camera_details,
    build_series_by_cam,
    extract_observed_cameras,
    frame_nums_by_cam,
)
from .stats_utils import safe_median



def compute_run_analysis(
    *,
    output_dir: Path,
    expected_cameras: Optional[int],
    expected_fps: Optional[float],
    fps_tolerance_ratio: float,
) -> Tuple[RunComputed, Dict[str, Any]]:
    """计算采集输出目录的统计结果。

    Returns:
        (computed, payload)

    说明：
    - 这里直接返回 payload（与历史实现保持一致），减少 report/api 层重复拼装。
    - report_text 由 report.py 负责渲染。
    """

    output_dir = Path(output_dir)
    meta_path = output_dir / "metadata.jsonl"

    all_records = read_jsonl_records(meta_path)
    group_records = [r for r in all_records if isinstance(r.get("frames"), list)]
    event_records = [r for r in all_records if str(r.get("type", "")).strip()]

    if not group_records:
        raise ValueError(
            "metadata.jsonl 中未找到任何包含 frames 的 group 记录。\n"
            "如果你只记录了事件（camera_event/soft_trigger_send），需要同时打开组包记录才能做完整分析。"
        )

    group_by_values = sorted(
        {
            str(r.get("group_by")).strip()
            for r in group_records
            if str(r.get("group_by", "")).strip()
        }
    )

    cams, serials = extract_observed_cameras(group_records)
    observed_num_cameras = len(cams)
    num_cameras = int(expected_cameras) if expected_cameras is not None else observed_num_cameras

    # 组完整性 & 文件缺失
    frames_per_group: List[int] = []
    groups_complete = 0
    groups_incomplete = 0
    missing_files = 0

    host_spreads_ms: List[int] = []
    dev_spreads_raw: List[int] = []
    dev_spreads_norm: List[int] = []

    arrival_by_cam: Dict[int, List[float]] = {}
    group_arrival_median: List[float] = []

    series_by_cam_map = build_series_by_cam(group_records)
    cam_index_to_serial: Dict[int, str] = {}
    for cam, series in series_by_cam_map.items():
        for it in series:
            s = str(it.get("serial", "")).strip()
            if s:
                cam_index_to_serial[cam] = s
                break

    lost_packet_total = 0
    lost_packet_max = 0
    groups_with_lost_packet = 0

    lost_packet_by_cam: Dict[int, int] = {}
    lost_packet_max_by_cam: Dict[int, int] = {}

    widths: set[int] = set()
    heights: set[int] = set()
    pixel_types: set[int] = set()

    base_dev_by_cam: Dict[int, int] = {}

    try:
        group_dirs = sum(1 for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("group_"))
    except Exception:
        group_dirs = 0

    base_frame_num_by_cam: Dict[int, int] = {}
    frame_num_norm_spreads: List[int] = []
    groups_with_frame_num_norm_mismatch = 0

    for r in group_records:
        frs = r.get("frames", []) or []
        frames_per_group.append(len(frs))

        cam_set = set()
        dev_ts_raw: List[int] = []
        dev_ts_norm: List[int] = []
        frame_num_norm: List[int] = []
        group_arrivals: List[float] = []

        for fr in frs:
            try:
                cam_set.add(int(fr.get("cam_index")))
            except Exception:
                pass

            try:
                widths.add(int(fr.get("width")))
                heights.add(int(fr.get("height")))
                pixel_types.add(int(fr.get("pixel_type")))
            except Exception:
                pass

            try:
                cam_idx = int(fr.get("cam_index"))
                ts = int(fr.get("dev_timestamp"))

                dev_ts_raw.append(ts)

                base = base_dev_by_cam.get(cam_idx)
                if base is None:
                    base_dev_by_cam[cam_idx] = ts
                    base = ts
                dev_ts_norm.append(ts - base)
            except Exception:
                pass

            lp = int(fr.get("lost_packet", 0) or 0)
            lost_packet_total += lp
            lost_packet_max = max(lost_packet_max, lp)
            try:
                cam_idx = int(fr.get("cam_index"))
                lost_packet_by_cam[cam_idx] = int(lost_packet_by_cam.get(cam_idx, 0)) + lp
                lost_packet_max_by_cam[cam_idx] = max(int(lost_packet_max_by_cam.get(cam_idx, 0)), lp)
            except Exception:
                pass

            try:
                cam_idx = int(fr.get("cam_index"))
                at = float(fr.get("arrival_monotonic"))
                arrival_by_cam.setdefault(cam_idx, []).append(at)
                group_arrivals.append(at)
            except Exception:
                pass

            try:
                cam_idx = int(fr.get("cam_index"))
                fn = int(fr.get("frame_num"))
                base_fn = base_frame_num_by_cam.get(cam_idx)
                if base_fn is None:
                    base_frame_num_by_cam[cam_idx] = fn
                    base_fn = fn
                frame_num_norm.append(fn - base_fn)
            except Exception:
                pass

            file_rel = fr.get("file")
            if file_rel:
                p = Path(str(file_rel))
                if not p.is_absolute():
                    if p.exists():
                        pass
                    else:
                        alt_candidates = []
                        if p.parts and output_dir.name and p.parts[0] != output_dir.name:
                            alt_candidates.append((output_dir / p).resolve())
                        alt_candidates.append((output_dir.parent / p).resolve())

                        if not any(x.exists() for x in alt_candidates):
                            missing_files += 1
                else:
                    if not p.exists():
                        missing_files += 1

        if any(int(fr.get("lost_packet", 0) or 0) > 0 for fr in frs):
            groups_with_lost_packet += 1

        if len(cam_set) == num_cameras and len(frs) == num_cameras:
            groups_complete += 1
        else:
            groups_incomplete += 1

        host_ts: List[int] = []
        for fr in frs:
            try:
                host_ts.append(int(fr.get("host_timestamp")))
            except Exception:
                pass
        if host_ts:
            host_spreads_ms.append(int(max(host_ts) - min(host_ts)))

        if dev_ts_raw:
            dev_spreads_raw.append(int(max(dev_ts_raw) - min(dev_ts_raw)))
        if dev_ts_norm:
            dev_spreads_norm.append(int(max(dev_ts_norm) - min(dev_ts_norm)))

        if frame_num_norm:
            spread = int(max(frame_num_norm) - min(frame_num_norm))
            frame_num_norm_spreads.append(spread)
            if spread != 0:
                groups_with_frame_num_norm_mismatch += 1

        if group_arrivals:
            group_arrival_median.append(float(statistics.median(group_arrivals)))

    frames_per_group_sorted = sorted(frames_per_group)

    created_ts: List[float] = []
    for r in group_records:
        try:
            created_at = r.get("created_at")
            if created_at is not None:
                created_ts.append(float(created_at))
        except Exception:
            pass
    created_dt = [b - a for a, b in zip(created_ts, created_ts[1:]) if (b - a) > 0]
    created_dt_med = safe_median(created_dt)
    approx_fps = (1.0 / created_dt_med) if (created_dt_med and created_dt_med > 0) else None

    group_arrival_dt = [b - a for a, b in zip(group_arrival_median, group_arrival_median[1:]) if (b - a) > 0]
    arrival_dt_med = safe_median(group_arrival_dt)
    arrival_fps = (1.0 / arrival_dt_med) if (arrival_dt_med and arrival_dt_med > 0) else None

    camera_arrival_fps_list: List[float] = []
    for _, ts_list in sorted(arrival_by_cam.items()):
        if len(ts_list) < 2:
            continue
        dur = float(ts_list[-1] - ts_list[0])
        if dur <= 0:
            continue
        camera_arrival_fps_list.append(float((len(ts_list) - 1) / dur))

    camera_arrival_fps_list_sorted = sorted(camera_arrival_fps_list)
    camera_arrival_fps_min = float(camera_arrival_fps_list_sorted[0]) if camera_arrival_fps_list_sorted else None
    camera_arrival_fps_median = (
        float(statistics.median(camera_arrival_fps_list_sorted)) if camera_arrival_fps_list_sorted else None
    )
    camera_arrival_fps_max = float(camera_arrival_fps_list_sorted[-1]) if camera_arrival_fps_list_sorted else None

    host_spreads_ms_sorted = sorted(host_spreads_ms) if host_spreads_ms else [0]
    dev_spreads_raw_sorted = sorted(dev_spreads_raw) if dev_spreads_raw else [0]
    dev_spreads_norm_sorted = sorted(dev_spreads_norm) if dev_spreads_norm else [0]
    frame_num_norm_spreads_sorted = sorted(frame_num_norm_spreads) if frame_num_norm_spreads else [0]

    soft_trigger_sends, soft_send_dt_med, soft_send_fps, soft_send_targets_count = compute_soft_trigger_stats(
        event_records
    )

    (
        exposure_event_name,
        exposure_events_count,
        exposure_dt_host_med,
        exposure_fps_host_med,
        cam_expo_fps_min,
        cam_expo_fps_median,
        cam_expo_fps_max,
        exposure_dt_ticks_med,
        per_serial_exposure,
    ) = compute_exposure_stats(event_records)

    per_camera = build_per_camera_details(
        series_by_cam=series_by_cam_map,
        cam_index_to_serial=cam_index_to_serial,
        lost_packet_by_cam=lost_packet_by_cam,
        lost_packet_max_by_cam=lost_packet_max_by_cam,
    )

    summary = RunSummary(
        jsonl_lines=len(all_records),
        records=len(group_records),
        num_cameras_observed=observed_num_cameras,
        cameras=cams,
        serials=serials,
        groups_complete=groups_complete,
        groups_incomplete=groups_incomplete,
        frames_per_group_min=min(frames_per_group_sorted),
        frames_per_group_median=float(statistics.median(frames_per_group_sorted)),
        frames_per_group_max=max(frames_per_group_sorted),
        group_dirs=int(group_dirs),
        width_unique=len(widths) if widths else 0,
        height_unique=len(heights) if heights else 0,
        pixel_type_unique=len(pixel_types) if pixel_types else 0,
        lost_packet_total=lost_packet_total,
        lost_packet_max=lost_packet_max,
        groups_with_lost_packet=groups_with_lost_packet,
        host_spread_ms_min=min(host_spreads_ms_sorted),
        host_spread_ms_median=float(statistics.median(host_spreads_ms_sorted)),
        host_spread_ms_max=max(host_spreads_ms_sorted),
        dev_spread_raw_min=min(dev_spreads_raw_sorted),
        dev_spread_raw_median=float(statistics.median(dev_spreads_raw_sorted)),
        dev_spread_raw_max=max(dev_spreads_raw_sorted),
        dev_spread_norm_min=min(dev_spreads_norm_sorted),
        dev_spread_norm_median=float(statistics.median(dev_spreads_norm_sorted)),
        dev_spread_norm_max=max(dev_spreads_norm_sorted),
        created_dt_s_median=created_dt_med,
        approx_fps_median=approx_fps,
        arrival_dt_s_median=arrival_dt_med,
        arrival_fps_median=arrival_fps,
        camera_arrival_fps_min=camera_arrival_fps_min,
        camera_arrival_fps_median=camera_arrival_fps_median,
        camera_arrival_fps_max=camera_arrival_fps_max,
        soft_trigger_sends=int(soft_trigger_sends),
        soft_trigger_dt_s_median=soft_send_dt_med,
        soft_trigger_fps_median=soft_send_fps,
        exposure_events=int(exposure_events_count),
        exposure_event_name=exposure_event_name,
        exposure_dt_s_median_host=exposure_dt_host_med,
        exposure_fps_median_host=exposure_fps_host_med,
        camera_exposure_fps_min=cam_expo_fps_min,
        camera_exposure_fps_median=cam_expo_fps_median,
        camera_exposure_fps_max=cam_expo_fps_max,
        exposure_dt_ticks_median=exposure_dt_ticks_med,
        missing_files=missing_files,
        frame_num_norm_spread_min=min(frame_num_norm_spreads_sorted),
        frame_num_norm_spread_median=float(statistics.median(frame_num_norm_spreads_sorted)),
        frame_num_norm_spread_max=max(frame_num_norm_spreads_sorted),
        groups_with_frame_num_norm_mismatch=groups_with_frame_num_norm_mismatch,
    )

    # frame_num 连续性（用于报告与 payload）
    by_cam = frame_nums_by_cam(group_records)
    cont_lines, cont_payload = build_frame_num_continuity(
        series_by_cam=series_by_cam_map,
        frame_nums_by_cam_map=by_cam,
    )

    # 检查项（用于报告与 payload）
    checks: List[Tuple[str, bool, str]] = []

    checks.append(
        (
            "组完整性（每组都凑齐所有相机）",
            summary.groups_incomplete == 0 and summary.groups_complete == summary.records,
            "同步采集最基础指标：每次触发必须每台相机都有一帧，否则下游对齐/推理会错位。",
        )
    )

    checks.append(
        (
            "网络丢包（lost_packet=0）",
            summary.lost_packet_total == 0,
            "GigE 丢包会导致图像损坏/延迟，严重时会触发超时丢组。",
        )
    )

    checks.append(
        (
            "图像格式一致（width/height/pixel_type）",
            (summary.width_unique in {0, 1})
            and (summary.height_unique in {0, 1})
            and (summary.pixel_type_unique in {0, 1}),
            "多相机同步通常要求分辨率与像素格式一致，否则保存/解码/推理与标定对齐都会更复杂。",
        )
    )

    if expected_fps is not None:
        fps_for_check = (
            summary.arrival_fps_median if summary.arrival_fps_median is not None else summary.approx_fps_median
        )
        if fps_for_check is None:
            fps_ok = False
        else:
            lo = float(expected_fps) * (1.0 - float(fps_tolerance_ratio))
            hi = float(expected_fps) * (1.0 + float(fps_tolerance_ratio))
            fps_ok = (lo <= float(fps_for_check) <= hi)
        checks.append(
            (
                "实际 FPS 接近期望",
                fps_ok,
                "优先使用 arrival_monotonic 估计触发/出图频率；若缺失才退化到 created_at 吞吐。"
                "如果 FPS 偏低，常见原因包括：触发频率未生效、带宽不足、或线程/队列/保存造成丢帧。",
            )
        )

    if summary.missing_files > 0:
        checks.append(
            (
                "保存文件完整（metadata 记录的文件都存在）",
                False,
                "文件缺失说明保存失败或路径拼接异常，会影响离线复现与标注/训练。",
            )
        )

    computed = RunComputed(
        output_dir=output_dir,
        meta_path=meta_path,
        group_by_values=group_by_values,
        expected_fps=expected_fps,
        fps_tolerance_ratio=float(fps_tolerance_ratio),
        summary=summary,
        checks=checks,
        frame_num_continuity_lines=cont_lines,
        frame_num_continuity_payload=cont_payload,
        per_camera=per_camera,
        per_serial_exposure=per_serial_exposure,
        soft_trigger_targets={k: int(v) for k, v in sorted(soft_send_targets_count.items())},
    )

    payload: Dict[str, Any] = {
        "summary": {**asdict(summary)},
        "frame_num_continuity": cont_payload,
        "per_camera": per_camera,
        "per_serial_exposure": per_serial_exposure,
        "soft_trigger_targets": {k: int(v) for k, v in sorted(soft_send_targets_count.items())},
        "checks": [{"name": name, "pass": ok, "why": why} for name, ok, why in checks],
    }

    return computed, payload
