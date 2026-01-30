# -*- coding: utf-8 -*-

"""分析 mvs.apps.quad_capture 的采集结果（metadata.jsonl + 输出目录）。

本模块定位：
- 纯分析/统计逻辑（尽量不依赖硬件/SDK）。
- CLI 入口放在 mvs.apps。

说明：
- host_timestamp / arrival_monotonic 是“主机侧观测”，受线程调度/网卡/驱动影响，只用于诊断。
- 严格同步建议以 trigger_index、以及相机侧 dev_timestamp（若已启用 PTP/同步）为主要依据。
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True, slots=True)
class RunSummary:
    """一次采集运行的核心统计结果（便于 JSON 输出）。"""

    jsonl_lines: int
    records: int
    num_cameras_observed: int
    cameras: List[int]
    serials: List[str]

    # 完整性
    groups_complete: int
    groups_incomplete: int
    frames_per_group_min: int
    frames_per_group_median: float
    frames_per_group_max: int

    # trigger_index
    trigger_index_unique: int
    trigger_index_all_same: bool
    trigger_index_min: int
    trigger_index_max: int

    # 输出目录结构
    trigger_dirs: int
    group_dirs: int

    # 图像格式一致性
    width_unique: int
    height_unique: int
    pixel_type_unique: int

    # 丢包
    lost_packet_total: int
    lost_packet_max: int
    groups_with_lost_packet: int

    # 时间（主机侧）
    host_spread_ms_min: int
    host_spread_ms_median: float
    host_spread_ms_max: int

    # 时间（相机侧，单位取决于机型/配置）
    dev_spread_raw_min: int
    dev_spread_raw_median: float
    dev_spread_raw_max: int
    dev_spread_norm_min: int
    dev_spread_norm_median: float
    dev_spread_norm_max: int

    # 频率
    created_dt_s_median: Optional[float]
    approx_fps_median: Optional[float]

    # 频率（更接近“相机实际出图/触发频率”：基于 Grabber 记录的 arrival_monotonic）
    arrival_dt_s_median: Optional[float]
    arrival_fps_median: Optional[float]
    camera_arrival_fps_min: Optional[float]
    camera_arrival_fps_median: Optional[float]
    camera_arrival_fps_max: Optional[float]

    # 频率：发命令（soft trigger）
    soft_trigger_sends: int
    soft_trigger_dt_s_median: Optional[float]
    soft_trigger_fps_median: Optional[float]

    # 频率：曝光（相机事件 ExposureStart/ExposureEnd）
    exposure_events: int
    exposure_event_name: str
    exposure_dt_s_median_host: Optional[float]
    exposure_fps_median_host: Optional[float]
    camera_exposure_fps_min: Optional[float]
    camera_exposure_fps_median: Optional[float]
    camera_exposure_fps_max: Optional[float]
    exposure_dt_ticks_median: Optional[float]

    # 文件
    missing_files: int

    # frame_num 对齐（当使用 frame_num/sequence 分组时更关键）
    frame_num_norm_spread_min: int
    frame_num_norm_spread_median: float
    frame_num_norm_spread_max: int
    groups_with_frame_num_norm_mismatch: int


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL 文件。"""

    if not path.exists():
        raise FileNotFoundError(str(path))

    records: List[Dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON decode failed at line {i}: {exc}") from exc

    if not records:
        raise ValueError(f"No records found in {path}")

    return records


def _safe_median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.median(values))


def _format_float(x: Optional[float], digits: int = 3) -> str:
    if x is None:
        return "-"
    if math.isnan(x) or math.isinf(x):
        return "-"
    return f"{x:.{digits}f}"


def _pct(part: int, total: int) -> str:
    if total <= 0:
        return "-"
    return f"{(100.0 * part / total):.1f}%"


def _extract_observed_cameras(records: Iterable[Dict[str, Any]]) -> Tuple[List[int], List[str]]:
    cams: set[int] = set()
    serials: set[str] = set()
    for r in records:
        for fr in r.get("frames", []) or []:
            try:
                cams.add(int(fr.get("cam_index")))
            except Exception:
                pass
            s = str(fr.get("serial", "")).strip()
            if s:
                serials.add(s)
    return sorted(cams), sorted(serials)


def _frame_nums_by_cam(records: Iterable[Dict[str, Any]]) -> Dict[int, List[int]]:
    by_cam: Dict[int, List[int]] = {}
    for r in records:
        for fr in r.get("frames", []) or []:
            try:
                cam = int(fr["cam_index"])
                fn = int(fr["frame_num"])
            except Exception:
                continue
            by_cam.setdefault(cam, []).append(fn)
    return by_cam


def _series_by_cam(records: Iterable[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """按 cam_index 聚合帧序列（尽量保留用于诊断的字段）。"""

    by_cam: Dict[int, List[Dict[str, Any]]] = {}
    for r in records:
        for fr in r.get("frames", []) or []:
            try:
                cam = int(fr.get("cam_index"))
            except Exception:
                continue

            item: Dict[str, Any] = {
                "cam_index": cam,
                "serial": str(fr.get("serial", "")).strip(),
                "frame_num": fr.get("frame_num"),
                "arrival_monotonic": fr.get("arrival_monotonic"),
                "lost_packet": fr.get("lost_packet", 0),
                "dev_timestamp": fr.get("dev_timestamp"),
            }
            by_cam.setdefault(cam, []).append(item)
    return by_cam


def _time_ordered_frame_nums(series: Sequence[Dict[str, Any]]) -> List[int]:
    """按 arrival_monotonic 排序后返回 frame_num 序列（用于分析回绕/乱序）。"""

    items: List[Tuple[float, int]] = []
    for it in series:
        try:
            t_raw = it.get("arrival_monotonic")
            fn_raw = it.get("frame_num")
            if t_raw is None or fn_raw is None:
                continue
            t = float(t_raw)
            fn = int(fn_raw)
        except Exception:
            continue
        items.append((t, fn))

    items.sort(key=lambda x: x[0])
    return [fn for _, fn in items]


def _count_resets(nums_in_time_order: Sequence[int]) -> int:
    """统计 frame_num 回绕/倒退次数（b < a 视为一次回绕）。"""

    if len(nums_in_time_order) <= 1:
        return 0
    resets = 0
    for a, b in zip(nums_in_time_order, nums_in_time_order[1:]):
        if b < a:
            resets += 1
    return resets


def _continuity_gaps_ignore_resets(nums_in_time_order: Sequence[int]) -> List[Tuple[int, int]]:
    """统计断档，但忽略回绕/重启。"""

    if len(nums_in_time_order) <= 1:
        return []

    gaps: List[Tuple[int, int]] = []
    for a, b in zip(nums_in_time_order, nums_in_time_order[1:]):
        if b < a:
            continue
        if b - a != 1:
            gaps.append((a, b))
    return gaps


def analyze_output_dir(
    *,
    output_dir: Path,
    expected_cameras: Optional[int],
    expected_fps: Optional[float],
    fps_tolerance_ratio: float,
    strict_trigger_index: bool,
) -> Tuple[RunSummary, str, Dict[str, Any]]:
    """分析采集输出目录并生成报告。

    Args:
        output_dir: mvs_quad_capture 的输出目录。
        expected_cameras: 期望相机数量；None 表示从数据自动推断。
        expected_fps: 期望 FPS；None 表示不做 FPS 合格判定。
        fps_tolerance_ratio: FPS 允许相对误差，例如 0.2 表示 ±20%。
        strict_trigger_index: 是否将 trigger_index 递增作为硬性检查。

    Returns:
        (summary, report_text, report_payload)
    """

    output_dir = Path(output_dir)
    meta_path = output_dir / "metadata.jsonl"

    all_records = _read_jsonl(meta_path)
    group_records = [r for r in all_records if isinstance(r.get("frames"), list)]
    event_records = [r for r in all_records if str(r.get("type", "")).strip()]

    if not group_records:
        raise ValueError(
            "metadata.jsonl 中未找到任何包含 frames 的 group 记录。\n"
            "如果你只记录了事件（camera_event/soft_trigger_send），需要同时打开组包记录才能做完整分析。"
        )

    group_by_values = sorted({str(r.get("group_by")).strip() for r in group_records if str(r.get("group_by", "")).strip()})

    cams, serials = _extract_observed_cameras(group_records)
    observed_num_cameras = len(cams)
    num_cameras = int(expected_cameras) if expected_cameras is not None else observed_num_cameras

    # 组完整性 & 文件缺失
    frames_per_group: List[int] = []
    groups_complete = 0
    groups_incomplete = 0
    missing_files = 0

    trigger_indices: List[int] = []
    host_spreads_ms: List[int] = []
    dev_spreads_raw: List[int] = []
    dev_spreads_norm: List[int] = []

    arrival_by_cam: Dict[int, List[float]] = {}
    group_arrival_median: List[float] = []

    series_by_cam = _series_by_cam(group_records)
    cam_index_to_serial: Dict[int, str] = {}
    for cam, series in series_by_cam.items():
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
        trigger_dirs = sum(1 for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("trigger_"))
    except Exception:
        trigger_dirs = 0
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

        # trigger_index
        try:
            trig = r.get("trigger_index")
            if trig is not None:
                trigger_indices.append(int(trig))
        except Exception:
            try:
                trigger_indices.append(int(frs[0].get("trigger_index")))  # type: ignore[union-attr]
            except Exception:
                pass

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
            else:
                pass

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

    trig_unique = len(set(trigger_indices)) if trigger_indices else 0
    trig_all_same = (trig_unique == 1) if trigger_indices else False
    trig_min = min(trigger_indices) if trigger_indices else 0
    trig_max = max(trigger_indices) if trigger_indices else 0

    created_ts: List[float] = []
    for r in group_records:
        try:
            created_at = r.get("created_at")
            if created_at is not None:
                created_ts.append(float(created_at))
        except Exception:
            pass
    created_dt = [b - a for a, b in zip(created_ts, created_ts[1:]) if (b - a) > 0]
    created_dt_med = _safe_median(created_dt)
    approx_fps = (1.0 / created_dt_med) if (created_dt_med and created_dt_med > 0) else None

    group_arrival_dt = [b - a for a, b in zip(group_arrival_median, group_arrival_median[1:]) if (b - a) > 0]
    arrival_dt_med = _safe_median(group_arrival_dt)
    arrival_fps = (1.0 / arrival_dt_med) if (arrival_dt_med and arrival_dt_med > 0) else None

    camera_arrival_fps_list: List[float] = []
    for cam, ts_list in sorted(arrival_by_cam.items()):
        if len(ts_list) < 2:
            continue
        dur = float(ts_list[-1] - ts_list[0])
        if dur <= 0:
            continue
        camera_arrival_fps_list.append(float((len(ts_list) - 1) / dur))

    camera_arrival_fps_list_sorted = sorted(camera_arrival_fps_list)
    camera_arrival_fps_min = float(camera_arrival_fps_list_sorted[0]) if camera_arrival_fps_list_sorted else None
    camera_arrival_fps_median = float(statistics.median(camera_arrival_fps_list_sorted)) if camera_arrival_fps_list_sorted else None
    camera_arrival_fps_max = float(camera_arrival_fps_list_sorted[-1]) if camera_arrival_fps_list_sorted else None

    host_spreads_ms_sorted = sorted(host_spreads_ms) if host_spreads_ms else [0]
    dev_spreads_raw_sorted = sorted(dev_spreads_raw) if dev_spreads_raw else [0]
    dev_spreads_norm_sorted = sorted(dev_spreads_norm) if dev_spreads_norm else [0]
    frame_num_norm_spreads_sorted = sorted(frame_num_norm_spreads) if frame_num_norm_spreads else [0]

    soft_send_times: List[float] = []
    soft_send_targets_count: Dict[str, int] = {}
    for r in event_records:
        if str(r.get("type")) != "soft_trigger_send":
            continue
        try:
            v = r.get("host_monotonic")
            if v is None:
                continue
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
    soft_send_dt_med = _safe_median(soft_send_dt)
    soft_send_fps = (1.0 / soft_send_dt_med) if (soft_send_dt_med and soft_send_dt_med > 0) else None

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

    for serial, ts_list in sorted(exposure_by_serial.items()):
        ts = sorted(ts_list)
        if len(ts) >= 2:
            dts = [b - a for a, b in zip(ts, ts[1:]) if (b - a) > 0]
            med = _safe_median(dts)
            if med is not None:
                exposure_host_dt_medians.append(float(med))
            dur = float(ts[-1] - ts[0])
            if dur > 0:
                exposure_fps_by_cam.append(float((len(ts) - 1) / dur))

    for serial, tick_list in sorted(exposure_ticks_by_serial.items()):
        ticks = sorted(tick_list)
        if len(ticks) >= 2:
            dts = [float(b - a) for a, b in zip(ticks, ticks[1:]) if (b - a) > 0]
            med = _safe_median(dts)
            if med is not None:
                exposure_tick_dt_medians.append(float(med))

    exposure_dt_host_med = _safe_median(exposure_host_dt_medians)
    exposure_fps_host_med = (1.0 / exposure_dt_host_med) if (exposure_dt_host_med and exposure_dt_host_med > 0) else None

    exposure_fps_by_cam_sorted = sorted(exposure_fps_by_cam)
    cam_expo_fps_min = float(exposure_fps_by_cam_sorted[0]) if exposure_fps_by_cam_sorted else None
    cam_expo_fps_median = float(statistics.median(exposure_fps_by_cam_sorted)) if exposure_fps_by_cam_sorted else None
    cam_expo_fps_max = float(exposure_fps_by_cam_sorted[-1]) if exposure_fps_by_cam_sorted else None
    exposure_dt_ticks_med = _safe_median(exposure_tick_dt_medians)

    per_camera: Dict[str, Any] = {}
    for cam_idx in sorted(series_by_cam.keys()):
        series = series_by_cam[cam_idx]

        arrival_ts = []
        for it in series:
            try:
                t_raw = it.get("arrival_monotonic")
                if t_raw is None:
                    continue
                arrival_ts.append(float(t_raw))
            except Exception:
                pass
        arrival_ts.sort()
        arrival_dts = [b - a for a, b in zip(arrival_ts, arrival_ts[1:]) if (b - a) > 0]
        arrival_dt_med = _safe_median(arrival_dts)
        arrival_fps_med = (1.0 / arrival_dt_med) if (arrival_dt_med and arrival_dt_med > 0) else None
        arrival_fps_avg = None
        arrival_span_s = None
        if len(arrival_ts) >= 2:
            dur = float(arrival_ts[-1] - arrival_ts[0])
            arrival_span_s = dur
            if dur > 0:
                arrival_fps_avg = float((len(arrival_ts) - 1) / dur)

        nums_time = _time_ordered_frame_nums(series)
        resets = _count_resets(nums_time)
        gaps_time = _continuity_gaps_ignore_resets(nums_time)

        lp_sum = int(lost_packet_by_cam.get(cam_idx, 0))
        lp_max = int(lost_packet_max_by_cam.get(cam_idx, 0))

        serial = str(cam_index_to_serial.get(cam_idx, "")).strip()
        per_camera[str(cam_idx)] = {
            "cam_index": cam_idx,
            "serial": serial,
            "frames": int(len(series)),
            "arrival_dt_s_median": arrival_dt_med,
            "arrival_fps_median": arrival_fps_med,
            "arrival_fps_avg": arrival_fps_avg,
            "arrival_span_s": arrival_span_s,
            "frame_num_first": (int(nums_time[0]) if nums_time else None),
            "frame_num_last": (int(nums_time[-1]) if nums_time else None),
            "frame_num_resets": int(resets),
            "frame_num_gap_samples": gaps_time[:5],
            "lost_packet_total": lp_sum,
            "lost_packet_max": lp_max,
        }

    per_serial_exposure: Dict[str, Any] = {}
    for serial, ts_list in sorted(exposure_by_serial.items()):
        ts = sorted(ts_list)
        dts = [b - a for a, b in zip(ts, ts[1:]) if (b - a) > 0]
        dt_med = _safe_median(dts)
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
        tick_dt_med = _safe_median(tick_dts)

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
        trigger_index_unique=trig_unique,
        trigger_index_all_same=trig_all_same,
        trigger_index_min=trig_min,
        trigger_index_max=trig_max,
        trigger_dirs=int(trigger_dirs),
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
        soft_trigger_sends=len(soft_send_times_sorted),
        soft_trigger_dt_s_median=soft_send_dt_med,
        soft_trigger_fps_median=soft_send_fps,
        exposure_events=len(exposure_events),
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

    by_cam = _frame_nums_by_cam(group_records)
    cont_lines: List[str] = []
    cont_payload: Dict[str, Any] = {}
    for cam in sorted(set(by_cam.keys()) | set(series_by_cam.keys())):
        nums_time = _time_ordered_frame_nums(series_by_cam.get(cam, []))
        if not nums_time and cam in by_cam:
            nums_time = by_cam[cam]

        if not nums_time:
            cont_lines.append(f"- cam{cam}: frame_num - 连续=否")
            cont_payload[f"cam{cam}"] = {"first": None, "last": None, "contiguous": False, "gaps": [], "resets": 0}
            continue

        resets = _count_resets(nums_time)
        gaps = _continuity_gaps_ignore_resets(nums_time)
        ok = (len(gaps) == 0)
        cont_lines.append(
            f"- cam{cam}: frame_num {nums_time[0]}..{nums_time[-1]} 连续={ '是' if ok else '否' }"
            + ("" if ok else f" 断档={gaps[:5]}" + ("..." if len(gaps) > 5 else ""))
            + ("" if resets <= 0 else f" 回绕/重启={resets}")
        )
        cont_payload[f"cam{cam}"] = {
            "first": int(nums_time[0]),
            "last": int(nums_time[-1]),
            "contiguous": ok,
            "gaps": gaps,
            "resets": int(resets),
        }

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
            (summary.width_unique in {0, 1}) and (summary.height_unique in {0, 1}) and (summary.pixel_type_unique in {0, 1}),
            "多相机同步通常要求分辨率与像素格式一致，否则保存/解码/推理与标定对齐都会更复杂。",
        )
    )

    if strict_trigger_index:
        checks.append(
            (
                "TriggerIndex 递增（用于严格证明同一次触发）",
                (not summary.trigger_index_all_same) and (summary.trigger_index_unique >= max(2, summary.records // 2)),
                "trigger_index 若正常递增，才能可靠用它做分组键；若恒为 0，会让“严格同步”缺少证据。",
            )
        )

    if expected_fps is not None:
        fps_for_check = summary.arrival_fps_median if summary.arrival_fps_median is not None else summary.approx_fps_median
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

    lines: List[str] = []
    lines.append("=== MVS 采集结果分析报告 ===")
    lines.append(f"output_dir: {output_dir}")
    lines.append(f"metadata: {meta_path}")
    lines.append("")

    lines.append("[概览]")
    lines.append(f"- JSONL 行数(jsonl_lines): {summary.jsonl_lines}")
    lines.append(f"- 记录组数(group records): {summary.records}")
    lines.append(f"- 观测到的相机(cam_index): {summary.cameras} (count={summary.num_cameras_observed})")
    if summary.serials:
        lines.append(f"- 观测到的序列号(serial): {', '.join(summary.serials)}")
    if group_by_values:
        lines.append(f"- 分组键(group_by): {', '.join(group_by_values)}")
    lines.append("")

    lines.append("[关键检查]")
    for name, ok, why in checks:
        lines.append(f"- {name}: {'PASS' if ok else 'FAIL'}")
        lines.append(f"  说明：{why}")
    lines.append("")

    lines.append("[组包完整性]")
    lines.append(
        f"- complete/incomplete: {summary.groups_complete}/{summary.groups_incomplete} "
        f"({ _pct(summary.groups_complete, summary.records) } complete)"
    )
    lines.append(
        f"- frames_per_group (min/median/max): "
        f"{summary.frames_per_group_min}/{_format_float(summary.frames_per_group_median, 1)}/{summary.frames_per_group_max}"
    )
    lines.append("")

    lines.append("[触发索引 trigger_index]")
    lines.append(
        f"- unique={summary.trigger_index_unique} range=[{summary.trigger_index_min}, {summary.trigger_index_max}] all_same={summary.trigger_index_all_same}"
    )
    lines.append(f"- trigger_* 目录数量: {summary.trigger_dirs}")
    if summary.group_dirs > 0:
        lines.append(f"- group_* 目录数量: {summary.group_dirs}")
    if group_by_values and any(x in {"frame_num", "sequence"} for x in group_by_values):
        lines.append(
            "- 说明：本次采集并非使用 trigger_index 作为分组键（metadata.jsonl 里记录了 group_by）。\n"
            "  trigger_index 在此更多用于诊断：若恒为 0，不一定意味着组包失败，但意味着它不能作为严格同步证据。"
        )
    else:
        lines.append(
            "- 含义：理想情况下每次触发 trigger_index 会递增；如果恒为 0，说明该字段未提供有效分组信息（需要排查触发/机型/节点支持）。"
        )
    lines.append("")

    lines.append("[图像格式]")
    lines.append(f"- width unique: {summary.width_unique}")
    lines.append(f"- height unique: {summary.height_unique}")
    lines.append(f"- pixel_type unique: {summary.pixel_type_unique}")
    lines.append("- 含义：unique=1 表示所有帧一致；>1 说明多相机配置不一致或某些帧元信息异常。")
    lines.append("")

    lines.append("[丢包 lost_packet]")
    lines.append(
        f"- total={summary.lost_packet_total} max={summary.lost_packet_max} groups_with_loss={summary.groups_with_lost_packet}/{summary.records}"
    )
    lines.append("- 含义：GigE 场景下丢包意味着链路不稳或带宽/包大小设置不佳，可能导致图像损坏或组包超时。")
    lines.append("")

    lines.append("[帧号连续性 frame_num]")
    lines.extend(cont_lines)
    lines.append("- 含义：frame_num 断档通常意味着取流/队列丢帧；即使组包完整，也可能出现错位或遗漏。")
    lines.append("")

    lines.append("[frame_num 归一化一致性（用于 frame_num/sequence 分组诊断）]")
    lines.append(
        f"- normalized spread (min/median/max): "
        f"{summary.frame_num_norm_spread_min}/{_format_float(summary.frame_num_norm_spread_median, 1)}/{summary.frame_num_norm_spread_max}"
    )
    lines.append(f"- groups_with_norm_mismatch: {summary.groups_with_frame_num_norm_mismatch}/{summary.records}")
    lines.append(
        "- 含义：对每台相机用 (frame_num - 首次frame_num) 做归一化，理想情况下同一组内应完全一致（spread=0）。\n"
        "  如果 spread 经常>0，说明存在丢帧/起始不同步/乱序等情况，frame_num/sequence 分组会变得不可靠。"
    )
    lines.append("")

    lines.append("[组内时间差（主机侧诊断）]")
    lines.append(
        f"- host_timestamp spread (ms) min/median/max: "
        f"{summary.host_spread_ms_min}/{_format_float(summary.host_spread_ms_median, 1)}/{summary.host_spread_ms_max}"
    )
    lines.append("- 含义：该指标反映“主机收到三张图的时间差”，受线程调度/网络/磁盘影响；只用于诊断，不等价于曝光不同步。")
    lines.append("")

    lines.append("[组内时间差（相机侧，诊断/同步参考）]")
    lines.append(
        f"- dev_timestamp spread RAW (units) min/median/max: "
        f"{summary.dev_spread_raw_min}/{_format_float(summary.dev_spread_raw_median, 1)}/{summary.dev_spread_raw_max}"
    )
    lines.append("- 含义：RAW spread 反映各相机时间戳的绝对偏移；若启用 PTP 且同域同步，RAW spread 通常应接近 0。")
    lines.append(
        f"- dev_timestamp spread NORMALIZED (units) min/median/max: "
        f"{summary.dev_spread_norm_min}/{_format_float(summary.dev_spread_norm_median, 1)}/{summary.dev_spread_norm_max}"
    )
    lines.append("- 含义：NORMALIZED 把每台相机的时间戳减去各自的首次值，关注相对变化；该 spread 更接近“同一次触发是否对齐”。")
    lines.append("")

    lines.append("[频率（send/exposure/arrival）]")
    lines.append(f"- send dt median (s): {_format_float(summary.soft_trigger_dt_s_median, 6)}")
    lines.append(f"- send fps (median): {_format_float(summary.soft_trigger_fps_median, 3)}")
    lines.append(f"- send events: {summary.soft_trigger_sends}")
    if soft_send_targets_count:
        targets_str = ", ".join([f"{k}={v}" for k, v in sorted(soft_send_targets_count.items())])
        lines.append(f"- send targets: {targets_str}")
    lines.append("")

    lines.append(f"- exposure event: {summary.exposure_event_name or '-'} events: {summary.exposure_events}")
    lines.append(f"- exposure dt median HOST (s): {_format_float(summary.exposure_dt_s_median_host, 6)}")
    lines.append(f"- exposure fps (median HOST): {_format_float(summary.exposure_fps_median_host, 3)}")
    lines.append(
        f"- per-camera exposure fps (min/median/max): "
        f"{_format_float(summary.camera_exposure_fps_min, 3)}/"
        f"{_format_float(summary.camera_exposure_fps_median, 3)}/"
        f"{_format_float(summary.camera_exposure_fps_max, 3)}"
    )
    lines.append(f"- exposure dt median DEVICE (ticks): {_format_float(summary.exposure_dt_ticks_median, 1)}")
    lines.append("")

    lines.append(f"- arrival dt median (s): {_format_float(summary.arrival_dt_s_median, 6)}")
    lines.append(f"- arrival fps (median): {_format_float(summary.arrival_fps_median, 3)}")
    lines.append(
        f"- per-camera arrival fps (min/median/max): "
        f"{_format_float(summary.camera_arrival_fps_min, 3)}/"
        f"{_format_float(summary.camera_arrival_fps_median, 3)}/"
        f"{_format_float(summary.camera_arrival_fps_max, 3)}"
    )
    lines.append(f"- created_at dt median (s): {_format_float(summary.created_dt_s_median, 6)}")
    lines.append(f"- approx fps (median): {_format_float(summary.approx_fps_median, 3)}")
    if expected_fps is not None:
        lines.append(
            f"- expected fps: {_format_float(float(expected_fps), 3)} tolerance: +/-{_format_float(100.0 * fps_tolerance_ratio, 1)}%"
        )
    lines.append(
        "- 含义：\n"
        "  send-fps：上位机下发 TriggerSoftware 的节拍（仅 soft trigger 场景）。\n"
        "  exposure-fps：相机端曝光事件节拍（建议订阅 ExposureStart；不支持时可用 ExposureEnd 近似）。\n"
        "  arrival-fps：主机侧实际拿到帧的节拍（受带宽/队列/丢帧影响，通常是最终有效吞吐）。\n"
        "  created_at：写 metadata 的时间点，反映端到端吞吐；保存较慢时 created_at 会显著低于 arrival。"
    )
    lines.append("")

    lines.append("[逐相机明细]")
    for cam_key in sorted(per_camera.keys(), key=lambda x: int(x)):
        c = per_camera[cam_key]
        serial = c.get("serial") or "-"
        lines.append(f"- cam{c['cam_index']} serial={serial}")
        lines.append(
            "  "
            f"frames={c['frames']} "
            f"arrival_fps_median={_format_float(c.get('arrival_fps_median'), 3)} "
            f"arrival_fps_avg_over_span={_format_float(c.get('arrival_fps_avg'), 3)} "
            f"arrival_span_s={_format_float(c.get('arrival_span_s'), 1)}"
        )
        gap_samples = c.get("frame_num_gap_samples") or []
        gaps_txt = (str(gap_samples) if gap_samples else "-")
        lines.append(
            "  "
            f"frame_num={c.get('frame_num_first')}..{c.get('frame_num_last')} "
            f"resets={c.get('frame_num_resets')} gap_samples={gaps_txt}"
        )
        lines.append(
            "  "
            f"lost_packet_total={c.get('lost_packet_total')} lost_packet_max={c.get('lost_packet_max')}"
        )

    lines.append("")
    lines.append("[逐相机曝光事件明细]")
    if not per_serial_exposure:
        lines.append("- 未观测到曝光事件（camera_event）。")
    else:
        for serial, srec in sorted(per_serial_exposure.items()):
            lines.append(
                f"- {serial} {srec.get('event_name')}: events={srec.get('events')} "
                f"fps_median={_format_float(srec.get('fps_median_host'), 3)} "
                f"fps_avg={_format_float(srec.get('fps_avg_host'), 3)} "
                f"span_s={_format_float(srec.get('span_s'), 1)} "
                f"dt_s_median={_format_float(srec.get('dt_s_median_host'), 6)} "
                f"dt_ticks_median={_format_float(srec.get('dt_ticks_median'), 1)}"
            )
    lines.append("")

    if summary.missing_files > 0:
        lines.append("[文件完整性]")
        lines.append(f"- missing files referenced by metadata: {summary.missing_files}")
        lines.append("")

    report_text = "\n".join(lines)

    payload: Dict[str, Any] = {
        "summary": {**asdict(summary)},
        "frame_num_continuity": cont_payload,
        "per_camera": per_camera,
        "per_serial_exposure": per_serial_exposure,
        "soft_trigger_targets": {k: int(v) for k, v in sorted(soft_send_targets_count.items())},
        "checks": [{"name": name, "pass": ok, "why": why} for name, ok, why in checks],
    }

    return summary, report_text, payload
