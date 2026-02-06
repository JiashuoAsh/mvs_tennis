# -*- coding: utf-8 -*-

"""采集运行分析：文本报告渲染。"""

from __future__ import annotations

import math
from typing import List, Optional

from .models import RunComputed


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


def render_report_text(computed: RunComputed) -> str:
    """将统计结果渲染为人类可读的长文本报告。"""

    output_dir = computed.output_dir
    meta_path = computed.meta_path
    summary = computed.summary

    checks = computed.checks
    cont_lines = computed.frame_num_continuity_lines
    per_camera = computed.per_camera
    per_serial_exposure = computed.per_serial_exposure

    group_by_values = computed.group_by_values
    soft_send_targets = computed.soft_trigger_targets

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

    if summary.group_dirs > 0:
        lines.append("[输出目录结构]")
        lines.append(f"- group_* 目录数量: {summary.group_dirs}")
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
    if soft_send_targets:
        targets_str = ", ".join([f"{k}={v}" for k, v in sorted(soft_send_targets.items())])
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
    if computed.expected_fps is not None:
        lines.append(
            f"- expected fps: {_format_float(float(computed.expected_fps), 3)} tolerance: +/-{_format_float(100.0 * computed.fps_tolerance_ratio, 1)}%"
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

    return "\n".join(lines)
