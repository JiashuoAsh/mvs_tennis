# -*- coding: utf-8 -*-

"""采集运行分析：帧序列相关的计算工具。

职责：
- 从 group records 提取“按相机聚合”的帧序列；
- 计算 frame_num 回绕/断档等连续性诊断；
- 生成 per-camera 细节与 frame_num 连续性 payload。

说明：
- 本模块只做纯计算，不做文件 IO。
- 该模块的函数被 compute.py 调用，以降低 compute.py 的体量与认知负担。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .stats_utils import safe_median


def extract_observed_cameras(records: Iterable[Dict[str, Any]]) -> Tuple[List[int], List[str]]:
    """从 records 中提取出现过的 cam_index 与 serial（去重+排序）。"""

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


def frame_nums_by_cam(records: Iterable[Dict[str, Any]]) -> Dict[int, List[int]]:
    """按 cam_index 聚合所有 frame_num（不排序、不去重）。"""

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


def build_series_by_cam(records: Iterable[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
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


def time_ordered_frame_nums(series: Sequence[Dict[str, Any]]) -> List[int]:
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


def count_resets(nums_in_time_order: Sequence[int]) -> int:
    """统计 frame_num 回绕/倒退次数（b < a 视为一次回绕）。"""

    if len(nums_in_time_order) <= 1:
        return 0
    resets = 0
    for a, b in zip(nums_in_time_order, nums_in_time_order[1:]):
        if b < a:
            resets += 1
    return resets


def continuity_gaps_ignore_resets(nums_in_time_order: Sequence[int]) -> List[Tuple[int, int]]:
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


def build_per_camera_details(
    *,
    series_by_cam: Dict[int, List[Dict[str, Any]]],
    cam_index_to_serial: Dict[int, str],
    lost_packet_by_cam: Dict[int, int],
    lost_packet_max_by_cam: Dict[int, int],
) -> Dict[str, Any]:
    """构建 per-camera 诊断 payload（与历史字段口径保持一致）。"""

    per_camera: Dict[str, Any] = {}
    for cam_idx in sorted(series_by_cam.keys()):
        series = series_by_cam[cam_idx]

        arrival_ts: List[float] = []
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
        arrival_dt_med = safe_median(arrival_dts)
        arrival_fps_med = (1.0 / arrival_dt_med) if (arrival_dt_med and arrival_dt_med > 0) else None

        arrival_fps_avg = None
        arrival_span_s = None
        if len(arrival_ts) >= 2:
            dur = float(arrival_ts[-1] - arrival_ts[0])
            arrival_span_s = dur
            if dur > 0:
                arrival_fps_avg = float((len(arrival_ts) - 1) / dur)

        nums_time = time_ordered_frame_nums(series)
        resets = count_resets(nums_time)
        gaps_time = continuity_gaps_ignore_resets(nums_time)

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

    return per_camera


def build_frame_num_continuity(
    *,
    series_by_cam: Dict[int, List[Dict[str, Any]]],
    frame_nums_by_cam_map: Dict[int, List[int]],
) -> Tuple[List[str], Dict[str, Any]]:
    """构建 frame_num 连续性诊断：文本行 + payload。"""

    cont_lines: List[str] = []
    cont_payload: Dict[str, Any] = {}

    for cam in sorted(set(frame_nums_by_cam_map.keys()) | set(series_by_cam.keys())):
        nums_time = time_ordered_frame_nums(series_by_cam.get(cam, []))
        if not nums_time and cam in frame_nums_by_cam_map:
            nums_time = frame_nums_by_cam_map[cam]

        if not nums_time:
            cont_lines.append(f"- cam{cam}: frame_num - 连续=否")
            cont_payload[f"cam{cam}"] = {
                "first": None,
                "last": None,
                "contiguous": False,
                "gaps": [],
                "resets": 0,
            }
            continue

        resets = count_resets(nums_time)
        gaps = continuity_gaps_ignore_resets(nums_time)
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

    return cont_lines, cont_payload
