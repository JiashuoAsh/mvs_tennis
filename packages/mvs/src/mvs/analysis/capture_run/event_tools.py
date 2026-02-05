# -*- coding: utf-8 -*-

"""采集运行分析：事件（event records）相关统计。

覆盖的事件类型：
- soft_trigger_send：上位机软触发下发记录（host_monotonic + targets）
- camera_event：相机事件记录（ExposureStart/ExposureEnd 等）

说明：
- 该模块只做纯计算，不读写文件；
- compute.py 会将结果写入 RunSummary 与 payload。
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .stats_utils import safe_median


def compute_soft_trigger_stats(event_records: Iterable[Dict[str, Any]]) -> Tuple[int, Optional[float], Optional[float], Dict[str, int]]:
    """统计 soft_trigger_send 事件。

    Returns:
        (sends, dt_median_s, fps_median, targets_count)
    """

    soft_send_times: List[float] = []
    soft_send_targets_count: Dict[str, int] = {}

    for r in event_records:
        if str(r.get("type")) != "soft_trigger_send":
            continue
        try:
            v = r.get("host_monotonic")
            if v is not None:
                soft_send_times.append(float(v))
        except Exception:
            pass

        try:
            for s in (r.get("targets") or []):
                ss = str(s).strip()
                if not ss:
                    continue
                soft_send_targets_count[ss] = int(soft_send_targets_count.get(ss, 0)) + 1
        except Exception:
            pass

    soft_send_times_sorted = sorted(soft_send_times)
    soft_send_dt = [b - a for a, b in zip(soft_send_times_sorted, soft_send_times_sorted[1:]) if (b - a) > 0]
    soft_send_dt_med = safe_median(soft_send_dt)
    soft_send_fps = (1.0 / soft_send_dt_med) if (soft_send_dt_med and soft_send_dt_med > 0) else None

    return int(len(soft_send_times_sorted)), soft_send_dt_med, soft_send_fps, soft_send_targets_count


def compute_exposure_stats(
    event_records: Iterable[Dict[str, Any]],
) -> Tuple[
    str,
    int,
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Dict[str, Any],
]:
    """统计 camera_event 中的曝光事件（优先 ExposureStart/ExposureEnd）。

    Returns:
        (
            exposure_event_name,
            exposure_events_count,
            exposure_dt_s_median_host,
            exposure_fps_median_host,
            camera_exposure_fps_min,
            camera_exposure_fps_median,
            camera_exposure_fps_max,
            exposure_dt_ticks_median,
            per_serial_exposure,
        )
    """

    cam_event_by_name: Dict[str, List[Dict[str, Any]]] = {}
    for r in event_records:
        if str(r.get("type")) != "camera_event":
            continue
        name = str(r.get("event_name") or r.get("requested_event_name") or "").strip()
        if not name:
            continue
        cam_event_by_name.setdefault(name, []).append(r)

    exposure_event_name = ""
    if "ExposureStart" in cam_event_by_name:
        exposure_event_name = "ExposureStart"
    elif "ExposureEnd" in cam_event_by_name:
        exposure_event_name = "ExposureEnd"
    elif cam_event_by_name:
        exposure_event_name = sorted(cam_event_by_name.keys())[0]

    exposure_events = cam_event_by_name.get(exposure_event_name, []) if exposure_event_name else []

    exposure_host_dt_medians: List[float] = []
    exposure_tick_dt_medians: List[float] = []
    exposure_fps_by_cam: List[float] = []

    exposure_by_serial: Dict[str, List[float]] = {}
    exposure_ticks_by_serial: Dict[str, List[int]] = {}

    for ev in exposure_events:
        serial = str(ev.get("serial", "")).strip()
        if not serial:
            continue
        try:
            v = ev.get("host_monotonic")
            if v is not None:
                exposure_by_serial.setdefault(serial, []).append(float(v))
        except Exception:
            pass
        try:
            v = ev.get("event_timestamp")
            if v is not None:
                exposure_ticks_by_serial.setdefault(serial, []).append(int(v))
        except Exception:
            pass

    for _, ts_list in sorted(exposure_by_serial.items()):
        ts = sorted(ts_list)
        if len(ts) >= 2:
            dts = [b - a for a, b in zip(ts, ts[1:]) if (b - a) > 0]
            med = safe_median(dts)
            if med is not None:
                exposure_host_dt_medians.append(float(med))
            dur = float(ts[-1] - ts[0])
            if dur > 0:
                exposure_fps_by_cam.append(float((len(ts) - 1) / dur))

    for _, tick_list in sorted(exposure_ticks_by_serial.items()):
        ticks = sorted(tick_list)
        if len(ticks) >= 2:
            dts = [float(b - a) for a, b in zip(ticks, ticks[1:]) if (b - a) > 0]
            med = safe_median(dts)
            if med is not None:
                exposure_tick_dt_medians.append(float(med))

    exposure_dt_host_med = safe_median(exposure_host_dt_medians)
    exposure_fps_host_med = (1.0 / exposure_dt_host_med) if (exposure_dt_host_med and exposure_dt_host_med > 0) else None

    exposure_fps_by_cam_sorted = sorted(exposure_fps_by_cam)
    cam_expo_fps_min = float(exposure_fps_by_cam_sorted[0]) if exposure_fps_by_cam_sorted else None
    cam_expo_fps_median = float(statistics.median(exposure_fps_by_cam_sorted)) if exposure_fps_by_cam_sorted else None
    cam_expo_fps_max = float(exposure_fps_by_cam_sorted[-1]) if exposure_fps_by_cam_sorted else None

    exposure_dt_ticks_med = safe_median(exposure_tick_dt_medians)

    # per-serial exposure 细节
    per_serial_exposure: Dict[str, Any] = {}
    for serial, ts_list in sorted(exposure_by_serial.items()):
        ts = sorted(ts_list)
        dts = [b - a for a, b in zip(ts, ts[1:]) if (b - a) > 0]
        dt_med = safe_median(dts)
        fps_med = (1.0 / dt_med) if (dt_med and dt_med > 0) else None

        fps_avg = None
        span_s = None
        if len(ts) >= 2:
            dur = float(ts[-1] - ts[0])
            span_s = dur
            if dur > 0:
                fps_avg = float((len(ts) - 1) / dur)

        ticks = sorted(exposure_ticks_by_serial.get(serial, []))
        tick_dts = [float(b - a) for a, b in zip(ticks, ticks[1:]) if (b - a) > 0]
        tick_dt_med = safe_median(tick_dts)

        per_serial_exposure[str(serial)] = {
            "serial": str(serial),
            "event_name": str(exposure_event_name),
            "events": int(len(ts)),
            "dt_s_median_host": dt_med,
            "fps_median_host": fps_med,
            "fps_avg_host": fps_avg,
            "span_s": span_s,
            "dt_ticks_median": tick_dt_med,
        }

    return (
        str(exposure_event_name),
        int(len(exposure_events)),
        exposure_dt_host_med,
        exposure_fps_host_med,
        cam_expo_fps_min,
        cam_expo_fps_median,
        cam_expo_fps_max,
        exposure_dt_ticks_med,
        per_serial_exposure,
    )
