# -*- coding: utf-8 -*-

"""采集会话：落盘录制实现。

职责：
 - 打开 `mvs.capture.pipeline.open_quad_capture` 并持续获取同步组包。
- 写 `metadata.jsonl`（组记录 + 相机事件记录）。
- 可选保存图像（SDK BMP）或原始帧数据（RAW）。
- 打印关键诊断信息（带宽估算、队列深度、dropped_groups 等）。

边界：
- 这是 I/O + 运行编排层，不属于纯 core。
- 若未来需要复用“采集循环”但不落盘/不打印，应在更上层新增对应 runner（而不是把逻辑继续塞进这里）。
"""

from __future__ import annotations

import json
import queue
import time
from pathlib import Path

from mvs.capture.bandwidth import estimate_camera_bandwidth, format_bandwidth_report
from mvs.capture.pipeline import open_quad_capture
from mvs.capture.save import save_frame_as_bmp
from mvs.core.events import MvsEvent

from .capture_session_types import CaptureSessionConfig, CaptureSessionResult


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _drain_event_queue(*, f_meta, event_queue: "queue.Queue[MvsEvent]") -> int:
    """把事件队列中的记录写入 metadata.jsonl。"""

    n = 0
    while True:
        try:
            ev = event_queue.get_nowait()
        except queue.Empty:
            break
        f_meta.write(json.dumps(ev, ensure_ascii=False) + "\n")
        n += 1
    if n:
        f_meta.flush()
    return n


def run_capture_session(*, binding, config: CaptureSessionConfig) -> CaptureSessionResult:
    """执行一次采集并把结果写入 output_dir。"""

    out_dir = Path(config.output_dir)
    _ensure_dir(out_dir)
    meta_path = out_dir / "metadata.jsonl"

    mapping = config.trigger_plan.mapping_str(config.serials)
    roi_str = "-"
    if config.image_width is not None and config.image_height is not None:
        roi_str = (
            f"{config.image_width}x{config.image_height} "
            f"offset=({config.image_offset_x},{config.image_offset_y})"
        )

    print(
        "采集配置：\n"
        f"- serials={config.serials}\n"
        f"- trigger_sources={mapping}\n"
        f"- master_serial={config.master_serial or '-'}\n"
        f"- master_line_out={config.master_line_out or '-'} master_line_source={config.master_line_source or '-'} master_line_mode={config.master_line_mode or '-'}\n"
        f"- soft_trigger_fps={float(config.soft_trigger_fps)} soft_trigger_serials={config.trigger_plan.soft_trigger_serials or '-'}\n"
        f"- group_by={config.group_by}\n"
        f"- group_timeout_ms={int(config.group_timeout_ms)} timeout_ms={int(config.timeout_ms)}\n"
        f"- pixel_format={config.pixel_format or '-'}\n"
        f"- roi={roi_str}\n"
        f"- save_mode={config.save_mode} output_dir={out_dir}"
    )

    groups_done = 0

    with open_quad_capture(
        binding=binding,
        serials=config.serials,
        trigger_sources=config.trigger_plan.trigger_sources,
        trigger_activation=str(config.trigger_activation),
        trigger_cache_enable=bool(config.trigger_cache_enable),
        timeout_ms=int(config.timeout_ms),
        group_timeout_ms=int(config.group_timeout_ms),
        max_pending_groups=int(config.max_pending_groups),
        group_by=config.group_by,
        enable_soft_trigger_fps=float(config.trigger_plan.enable_soft_trigger_fps),
        soft_trigger_serials=list(config.trigger_plan.soft_trigger_serials),
        camera_event_names=[str(x) for x in (config.camera_event_names or [])],
        master_serial=str(config.master_serial or ""),
        master_line_output=str(config.master_line_out or ""),
        master_line_source=str(config.master_line_source or ""),
        master_line_mode=str(config.master_line_mode or "Output"),
        pixel_format=str(config.pixel_format or ""),
        image_width=config.image_width,
        image_height=config.image_height,
        image_offset_x=int(config.image_offset_x),
        image_offset_y=int(config.image_offset_y),
        exposure_auto=str(config.exposure_auto or ""),
        exposure_time_us=float(config.exposure_time_us),
        gain_auto=str(config.gain_auto or ""),
        gain=float(config.gain),
    ) as cap:
        # 启动采集后立刻做一次带宽估算（便于快速判断是否“先天不可能跑满”。）
        overhead_factor = 1.10
        expected_fps = float(config.expected_fps or 0.0)
        soft_fps = float(config.soft_trigger_fps or 0.0)
        soft_targets = set(config.trigger_plan.soft_trigger_serials or [])

        global_fps_hint: float | None = None
        if expected_fps > 0:
            global_fps_hint = expected_fps
        elif config.master_serial and (soft_fps > 0) and (config.master_serial in soft_targets):
            global_fps_hint = soft_fps

        estimates = []
        for c in cap.cameras:
            fps_hint = None
            if global_fps_hint is not None:
                fps_hint = global_fps_hint
            elif (soft_fps > 0) and (c.serial in soft_targets):
                fps_hint = soft_fps

            estimates.append(
                estimate_camera_bandwidth(
                    binding=binding,
                    cam=c.cam,
                    serial=c.serial,
                    fps_hint=fps_hint,
                    overhead_factor=overhead_factor,
                )
            )

        print(format_bandwidth_report(estimates, overhead_factor=overhead_factor))

        last_log = time.monotonic()
        last_dropped = 0
        last_progress = time.monotonic()
        last_idle_log = 0.0

        if config.camera_event_names:
            requested = [str(x) for x in (config.camera_event_names or []) if str(x).strip()]
            print(f"已请求订阅相机事件: {requested}")
            for c in cap.cameras:
                enabled = getattr(c, "event_names_enabled", [])
                print(f"- {c.serial}: enabled={enabled or '-'}")

        with meta_path.open("a", encoding="utf-8") as f_meta:
            while True:
                _drain_event_queue(f_meta=f_meta, event_queue=cap.event_queue)

                if config.max_groups and groups_done >= int(config.max_groups):
                    break

                group = cap.get_next_group(timeout_s=0.5)
                if group is None:
                    _drain_event_queue(f_meta=f_meta, event_queue=cap.event_queue)
                    now = time.monotonic()
                    max_wait = float(config.max_wait_seconds)
                    if max_wait > 0 and (now - last_progress) > max_wait:
                        pending = getattr(cap.assembler, "pending_groups", 0)
                        try:
                            oldest_age = float(cap.assembler.pending_oldest_age_s())
                        except Exception:
                            oldest_age = 0.0
                        seen_by_cam = getattr(cap.assembler, "frames_seen_by_cam", {})
                        print(
                            "长时间未收到任何完整组包，已退出。\n"
                            f"- trigger_sources={mapping}\n"
                            f"- serials={config.serials}\n"
                            f"- output_dir={out_dir}\n"
                            f"- assembler: dropped_groups={cap.assembler.dropped_groups} pending_groups={pending} oldest_age_s={oldest_age:.3f} seen_by_cam={seen_by_cam}\n"
                            "如果你使用硬触发（Line0/Line1...），请确认外部触发脉冲已接到每台相机的对应输入口，且边沿/电平配置一致。\n"
                            "想先验证保存链路是否正常，可用：--trigger-source Software --soft-trigger-fps 5"
                        )
                        return CaptureSessionResult(
                            exit_code=2,
                            groups_done=groups_done,
                            output_dir=out_dir,
                            metadata_path=meta_path,
                        )

                    idle_log = float(config.idle_log_seconds)
                    if idle_log > 0 and (now - last_idle_log) > idle_log:
                        qsz = cap.frame_queue.qsize()
                        dropped = cap.assembler.dropped_groups
                        pending = getattr(cap.assembler, "pending_groups", 0)
                        try:
                            oldest_age = float(cap.assembler.pending_oldest_age_s())
                        except Exception:
                            oldest_age = 0.0
                        seen_by_cam = getattr(cap.assembler, "frames_seen_by_cam", {})
                        print(
                            "等待触发/组包中... "
                            f"qsize={qsz} dropped_groups={dropped} pending_groups={pending} oldest_age_s={oldest_age:.3f} "
                            f"seen_by_cam={seen_by_cam} "
                            f"trigger_sources={mapping} output_dir={out_dir}"
                        )
                        last_idle_log = now
                    continue

                # 走到这里说明拿到了一组完整的同步帧。
                group_seq = groups_done
                files: list[str | None] = [None] * len(group)

                if config.save_mode != "none":
                    group_dir = out_dir / f"group_{group_seq:010d}"
                    _ensure_dir(group_dir)

                    for fr in group:
                        if config.save_mode == "raw":
                            raw_path = group_dir / f"cam{fr.cam_index}_seq{group_seq:06d}_f{fr.frame_num}.bin"
                            raw_path.write_bytes(fr.data)
                            files[fr.cam_index] = str(raw_path)
                        elif config.save_mode == "sdk-bmp":
                            bmp_path = group_dir / f"cam{fr.cam_index}_seq{group_seq:06d}_f{fr.frame_num}.bmp"
                            try:
                                save_frame_as_bmp(
                                    binding=binding,
                                    cam=cap.cameras[fr.cam_index].cam,
                                    out_path=bmp_path,
                                    frame=fr,
                                    bayer_method=int(config.bayer_method),
                                )
                                files[fr.cam_index] = str(bmp_path)
                            except Exception as exc:
                                raw_path = (
                                    group_dir / f"cam{fr.cam_index}_seq{group_seq:06d}_f{fr.frame_num}.bin"
                                )
                                raw_path.write_bytes(fr.data)
                                files[fr.cam_index] = str(raw_path)
                                print(f"save bmp failed (cam{fr.cam_index}): {exc}; fallback to raw")

                record = {
                    "group_seq": group_seq,
                    "group_by": config.group_by,
                    "created_at": time.time(),
                    "frames": [
                        {
                            "cam_index": fr.cam_index,
                            "serial": fr.serial,
                            "frame_num": fr.frame_num,
                            "dev_timestamp": fr.dev_timestamp,
                            "host_timestamp": fr.host_timestamp,
                            "width": fr.width,
                            "height": fr.height,
                            "pixel_type": fr.pixel_type,
                            "frame_len": fr.frame_len,
                            "lost_packet": fr.lost_packet,
                            "arrival_monotonic": fr.arrival_monotonic,
                            "file": files[fr.cam_index],
                        }
                        for fr in group
                    ],
                }
                f_meta.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_meta.flush()

                _drain_event_queue(f_meta=f_meta, event_queue=cap.event_queue)

                groups_done += 1
                last_progress = time.monotonic()

                now = time.monotonic()
                if now - last_log > 2.0:
                    dropped = cap.assembler.dropped_groups
                    delta_dropped = dropped - last_dropped
                    print(
                        f"groups={groups_done} qsize={cap.frame_queue.qsize()} save_mode={config.save_mode} dropped_groups={dropped} (+{delta_dropped})"
                    )
                    last_dropped = dropped
                    last_log = now

    return CaptureSessionResult(exit_code=0, groups_done=groups_done, output_dir=out_dir, metadata_path=meta_path)
