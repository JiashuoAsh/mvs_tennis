# -*- coding: utf-8 -*-

"""采集会话（capture session）：把 open_quad_capture 的能力落到“可落盘复盘”的一次采集运行。

边界说明（很重要）：
- `mvs.pipeline.open_quad_capture`：更偏底层/可复用的采集管线（打开相机、抓流、分组）。
- 本模块：在不引入复杂抽象的前提下，提供“把采集结果写成 captures 目录”的薄封装能力：
  - 写 `metadata.jsonl`（组记录 + 事件记录）
  - 可选保存图像/RAW
  - 打印关键诊断信息（带宽估算、队列深度、dropped_groups 等）
- `mvs.apps.quad_capture`：命令行入口（参数解析、错误提示、调用本模块）。

注意：本模块仍然会执行文件 I/O 与打印日志，因此不属于纯 core；它主要用于复用 CLI 的“落盘采集”行为。
"""

from __future__ import annotations

import json
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Sequence

from mvs.bandwidth import estimate_camera_bandwidth, format_bandwidth_report
from mvs.events import MvsEvent
from mvs.pipeline import open_quad_capture
from mvs.save import save_frame_as_bmp


GroupBy = Literal["trigger_index", "frame_num", "sequence"]
SaveMode = Literal["none", "sdk-bmp", "raw"]


@dataclass(frozen=True)
class TriggerPlan:
    """根据 serial/master/trigger_source 推导出的触发计划。"""

    trigger_sources: List[str]
    soft_trigger_serials: List[str]

    def mapping_str(self, serials: Sequence[str]) -> str:
        return ", ".join([f"{s}->{src}" for s, src in zip(serials, self.trigger_sources)])


def build_trigger_plan(*, serials: Sequence[str], trigger_source: str, master_serial: str) -> TriggerPlan:
    """把 CLI 级别的参数（trigger_source + master_serial）规整成每相机的触发配置。

    规则（与旧脚本保持一致）：
    - 若指定 master_serial：master 强制 Software，其它相机使用 trigger_source；软触发仅对 master 下发。
    - 若未指定 master_serial 且 trigger_source=Software：软触发对所有相机下发。
    - 其它情况：不启用软触发。

    Raises:
        ValueError: master_serial 不在 serials 中。
    """

    master_serial = str(master_serial or "").strip()
    if master_serial and master_serial not in set(serials):
        raise ValueError(f"master_serial={master_serial} 不在 serials 中")

    if master_serial:
        trigger_sources = [("Software" if s == master_serial else str(trigger_source)) for s in serials]
        soft_trigger_serials = [master_serial]
        return TriggerPlan(trigger_sources=trigger_sources, soft_trigger_serials=soft_trigger_serials)

    trigger_sources = [str(trigger_source)] * len(list(serials))
    soft_trigger_serials = list(serials) if str(trigger_source).lower() == "software" else []
    return TriggerPlan(trigger_sources=list(trigger_sources), soft_trigger_serials=soft_trigger_serials)


def normalize_roi(
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


@dataclass(frozen=True)
class CaptureSessionConfig:
    """一次采集会话的配置（用于可复用调用）。"""

    serials: List[str]
    trigger_plan: TriggerPlan
    trigger_activation: str
    trigger_cache_enable: bool
    timeout_ms: int
    group_timeout_ms: int
    max_pending_groups: int
    group_by: GroupBy
    save_mode: SaveMode
    output_dir: Path
    max_groups: int
    bayer_method: int
    max_wait_seconds: float
    idle_log_seconds: float
    camera_event_names: List[str]

    # master/slave
    master_serial: str
    master_line_out: str
    master_line_source: str
    master_line_mode: str

    # 带宽估算
    expected_fps: float
    soft_trigger_fps: float

    # 图像参数
    pixel_format: str
    image_width: int | None
    image_height: int | None
    image_offset_x: int
    image_offset_y: int

    # 曝光/增益
    exposure_auto: str
    exposure_time_us: float
    gain_auto: str
    gain: float


@dataclass(frozen=True)
class CaptureSessionResult:
    """一次采集会话的结果（便于外部程序做后续处理/验收）。"""

    exit_code: int
    groups_done: int
    output_dir: Path
    metadata_path: Path


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
    """执行一次采集并把结果写入 output_dir。

    说明：
    - 本函数会打印诊断信息并持续运行直到：
      - 达到 max_groups；或
      - 用户 Ctrl+C；或
      - 在 max_wait_seconds 内始终没有拿到完整组包（返回 exit_code=2）。

    Args:
        binding: 已加载的 MVS 绑定。
        config: 会话配置。

    Returns:
        CaptureSessionResult。
    """

    out_dir = Path(config.output_dir)
    _ensure_dir(out_dir)
    meta_path = out_dir / "metadata.jsonl"

    mapping = config.trigger_plan.mapping_str(config.serials)
    roi_str = "-"
    if config.image_width is not None and config.image_height is not None:
        roi_str = f"{config.image_width}x{config.image_height} offset=({config.image_offset_x},{config.image_offset_y})"

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
        enable_soft_trigger_fps=float(config.soft_trigger_fps),
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
                trigger_index = group[0].trigger_index
                files: List[Optional[str]] = [None] * len(group)

                if config.save_mode != "none":
                    if config.group_by == "trigger_index":
                        group_dir = out_dir / f"trigger_{trigger_index:010d}"
                    else:
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
                                raw_path = group_dir / f"cam{fr.cam_index}_seq{group_seq:06d}_f{fr.frame_num}.bin"
                                raw_path.write_bytes(fr.data)
                                files[fr.cam_index] = str(raw_path)
                                print(f"save bmp failed (cam{fr.cam_index}): {exc}; fallback to raw")

                record = {
                    "group_seq": group_seq,
                    "group_by": config.group_by,
                    "trigger_index": trigger_index,
                    "created_at": time.time(),
                    "frames": [
                        {
                            "cam_index": fr.cam_index,
                            "serial": fr.serial,
                            "trigger_index": fr.trigger_index,
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
