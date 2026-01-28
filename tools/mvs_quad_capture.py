# -*- coding: utf-8 -*-

"""多相机同步采集（支持多种分组键）。

核心思路：
- 相机端使用同一外触发输入（推荐硬件触发/分配器），保证“同一时刻”。
- 上位机侧按 `--group-by` 选择分组键：
    - 设备支持且递增正常时：优先用 `trigger_index`（更强的“同一次触发”证据）；
    - 若你发现 `nTriggerIndex` 恒为 0：可用 `frame_num`（归一化帧号）或 `sequence`（进入分组器的顺序）。
- 时间戳建议以设备时间戳 `nDevTimeStampHigh/Low` 为主，主机时间戳 `nHostTimeStamp` 仅用于诊断。

注意：
- 4×2448×2048×30fps 在 1GbE 上通常不可行（带宽远超 1Gbps）。若你坚持 30fps+全分辨率，请考虑：ROI、降帧率、降低像素格式/位深、压缩、或多网卡/10GbE。
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Sequence, cast


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


from mvs import MvsDllNotFoundError, load_mvs_binding
from mvs.camera import MvsSdk
from mvs.devices import enumerate_devices
from mvs.pipeline import open_quad_capture
from mvs.save import save_frame_as_bmp
from mvs.bandwidth import estimate_camera_bandwidth, format_bandwidth_report
from mvs.events import MvsEvent


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _drain_event_queue(*, f_meta, event_queue: "queue.Queue[MvsEvent]") -> int:
    """把事件队列中的记录写入 metadata.jsonl。

    Returns:
        本次写入的事件条数。
    """

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


def main(argv: Optional[Sequence[str]] = None) -> int:
    # 尽量固定 UTF-8 输出，避免在重定向到文件时出现乱码。
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="MVS 多相机同步采集（支持按 group_by 分组）")
    parser.add_argument(
        "--dll-dir",
        default=None,
        help="包含 MvCameraControl.dll 的目录（可选）。也可用环境变量 MVS_DLL_DIR。",
    )
    parser.add_argument("--list", action="store_true", help="仅枚举并打印设备信息")
    parser.add_argument(
        "--num-cameras",
        type=int,
        default=None,
        help="参与同步的相机数量（默认按 --serial 的数量推导）",
    )
    parser.add_argument(
        "--serial",
        action="extend",
        nargs="+",
        default=[],
        help=(
            "按序列号选择相机（可一次性传多个，或重复多次）。"
            "顺序即 cam0..camN。示例：--serial A B C 或 --serial A --serial B --serial C"
        ),
    )
    parser.add_argument(
        "--trigger-source",
        default="Line0",
        help="触发源（例如 Line0/Line1/Line2/Line3）",
    )
    parser.add_argument(
        "--master-serial",
        default=None,
        help=(
            "指定一台相机作为 master（序列号）。"
            "master 使用 Software 触发；其余相机使用 --trigger-source（例如 Line0）。"
            "适用于：master 输出触发脉冲（如 Line1=ExposureStartActive）去硬触发其它相机。"
        ),
    )
    parser.add_argument(
        "--master-line-out",
        default="Line1",
        help=(
            "配置 master 的输出线，例如 Line1。"
            "如果留空，则不在脚本中主动配置输出线，依赖相机当前/持久化配置。"
        ),
    )
    parser.add_argument(
        "--master-line-source",
        default="",
        help=(
            "配置 master 输出线的信号源，例如 ExposureStartActive。"
            "（在 MVS Client UI 中可能显示为 'Exposure Start Active'）"
            "如果留空，则不在脚本中主动配置输出线。"
        ),
    )
    parser.add_argument(
        "--master-line-mode",
        default="Output",
        help="master 输出线模式，通常为 Output。",
    )
    parser.add_argument(
        "--soft-trigger-fps",
        type=float,
        default=20.0,
        help=(
            "软触发频率。"
            "当 --trigger-source Software 时，会对所有相机下发软触发；"
            "当指定 --master-serial 时，会只对 master 下发软触发。"
        ),
    )

    parser.add_argument(
        "--expected-fps",
        type=float,
        default=0.0,
        help=(
            "期望/外部触发帧率（Hz）。\n"
            "外触发场景下，SDK 未必能读取到帧率；此参数用于带宽估算。\n"
            "为 0 表示不强制，优先尝试从相机节点读取。"
        ),
    )
    parser.add_argument(
        "--trigger-activation",
        default="RisingEdge",
        help="触发沿（例如 RisingEdge/FallingEdge）",
    )

    parser.add_argument(
        "--pixel-format",
        default="",
        help=(
            "像素格式（PixelFormat），用于降低带宽/算力。"
            "留空表示不设置，保持相机当前配置。"
            "\n常见：Mono8（灰度/8bpp）、BayerRG8/BayerBG8/...（彩色 RAW/8bpp）、RGB8Packed（彩色/24bpp，带宽高）。"
            "\n支持候选列表：例如 'BayerRG8,BayerBG8' 会按序尝试。"
        ),
    )

    # ROI：用于降低带宽（注意是裁剪，不是缩放）。
    parser.add_argument(
        "--image-width",
        type=int,
        default=0,
        help=(
            "输出图像宽度（ROI 裁剪）。例如 1920。"
            "为 0 表示不设置，保持相机当前配置。"
        ),
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=0,
        help=(
            "输出图像高度（ROI 裁剪）。例如 1080。"
            "为 0 表示不设置，保持相机当前配置。"
        ),
    )
    parser.add_argument(
        "--image-offset-x",
        type=int,
        default=0,
        help="ROI 左上角 X 偏移（默认 0）。",
    )
    parser.add_argument(
        "--image-offset-y",
        type=int,
        default=0,
        help="ROI 左上角 Y 偏移（默认 0）。",
    )
    parser.add_argument(
        "--trigger-cache-enable",
        action="store_true",
        help="尝试开启 TriggerCacheEnable（部分机型不支持）",
    )
    parser.add_argument("--timeout-ms", type=int, default=1000, help="每次取流等待超时")
    parser.add_argument(
        "--group-timeout-ms",
        type=int,
        default=1000,
        help="同一 trigger_index 等待凑齐 N 台相机帧的超时",
    )
    parser.add_argument(
        "--group-by",
        choices=["trigger_index", "frame_num", "sequence"],
        default="frame_num",
        help=(
            "分组键：frame_num（默认；按帧号归一化分组，适用于 nTriggerIndex 恒为 0 的机型）；"
            "trigger_index（设备支持且递增正常时推荐；用于更严格的同步证据与分组）；"
            "sequence（按进入分组器的帧序号分组，要求不丢帧且触发节拍一致）。"
        ),
    )
    parser.add_argument(
        "--max-pending-groups",
        type=int,
        default=256,
        help="缓存的未凑齐 group 上限（防止内存增长）",
    )
    parser.add_argument(
        "--save-mode",
        choices=["none", "sdk-bmp", "raw"],
        default="none",
        help="保存模式：none 不保存；sdk-bmp 用 SDK 存 BMP；raw 保存原始数据 .bin",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_repo_root() / "data" / "captures"),
        help="输出目录（保存图片与 metadata.jsonl）",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=0,
        help="采集多少个同步组后退出（0 表示一直跑）",
    )
    parser.add_argument(
        "--bayer-method",
        type=int,
        default=2,
        help="SDK Bayer->RGB 插值方法：0-快速 1-均衡 2-最优 3-最优+",
    )
    parser.add_argument(
        "--max-wait-seconds",
        type=float,
        default=0.0,
        help=(
            "最长等待无组包的时间（秒）。0 表示一直等。"
            "用于排查硬触发未接入/无脉冲时脚本看起来“没反应”的情况。"
        ),
    )
    parser.add_argument(
        "--idle-log-seconds",
        type=float,
        default=5.0,
        help="长时间未收到帧时的提示打印间隔（秒，0 表示不打印）。",
    )
    # 默认开启
    parser.add_argument(
        "--camera-event",
        action="extend",
        nargs="+",
        default=["ExposureStart"],
        help=(
            "订阅并记录相机事件时间戳（写入 metadata.jsonl）。"
            "示例：--camera-event ExposureStart 或 --camera-event ExposureStart ExposureEnd。"
            "常用：ExposureStart/ExposureEnd/FrameStart。具体支持哪些事件取决于机型/固件。"
        ),
    )

    # 曝光/增益：默认给出“偏暗环境容错方案”的一套推荐值，便于直接开跑。
    # 你当前场景 20fps（周期 50ms），10ms 曝光有充足余量。
    parser.add_argument(
        "--exposure-auto",
        default="Off",
        choices=["Off", "Once", "Continuous"],
        help="自动曝光模式（默认 Off）。",
    )
    parser.add_argument(
        "--exposure-us",
        type=float,
        default=10000.0,
        help="曝光时间（微秒 us，默认 10000=10ms）。",
    )
    parser.add_argument(
        "--gain-auto",
        default="Off",
        choices=["Off", "Continuous"],
        help="自动增益模式（默认 Off）。",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=12.0,
        help="增益值（常见单位 dB，默认 12）。",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    group_by = cast(Literal["trigger_index", "frame_num", "sequence"], str(args.group_by))

    try:
        binding = load_mvs_binding(dll_dir=args.dll_dir)
    except MvsDllNotFoundError as exc:
        print(str(exc))
        return 2

    # 单独 list：只做 SDK init/枚举/反初始化
    if args.list:
        sdk = MvsSdk(binding)
        sdk.initialize()
        try:
            _, descs = enumerate_devices(binding)
            for d in descs:
                ip = d.ip or "-"
                print(
                    f"[{d.index}] model={d.model} serial={d.serial} name={d.user_name} ip={ip} tlayer=0x{d.tlayer_type:08X}"
                )
            return 0
        finally:
            sdk.finalize()

    serials = [s.strip() for s in args.serial if s.strip()]
    if len(serials) <= 0:
        print("请通过 --serial 指定要参与同步采集的相机序列号（按顺序对应 cam0..camN）。")
        return 2

    master_serial = (str(args.master_serial).strip() if args.master_serial is not None else "")
    if master_serial and master_serial not in set(serials):
        print(f"--master-serial={master_serial} 不在 --serial 列表中")
        return 2

    master_line_out = str(getattr(args, "master_line_out", "") or "").strip()
    master_line_source = str(getattr(args, "master_line_source", "") or "").strip()
    master_line_mode = str(getattr(args, "master_line_mode", "Output") or "Output").strip()

    trigger_sources = [
        ("Software" if (master_serial and s == master_serial) else str(args.trigger_source)) for s in serials
    ]
    soft_trigger_serials: List[str] = (
        [master_serial]
        if master_serial
        else (serials if str(args.trigger_source).lower() == "software" else [])
    )

    # 启动前先把关键配置打印出来，避免“我以为 master 也是 Line0”这类误会。
    mapping = ", ".join([f"{s}->{src}" for s, src in zip(serials, trigger_sources)])
    roi_str = "-"
    if int(getattr(args, "image_width", 0) or 0) > 0 or int(getattr(args, "image_height", 0) or 0) > 0:
        roi_str = (
            f"{int(getattr(args, 'image_width', 0) or 0)}x{int(getattr(args, 'image_height', 0) or 0)}"
            f" offset=({int(getattr(args, 'image_offset_x', 0) or 0)},{int(getattr(args, 'image_offset_y', 0) or 0)})"
        )
    print(
        "采集配置：\n"
        f"- serials={serials}\n"
        f"- trigger_sources={mapping}\n"
        f"- master_serial={master_serial or '-'}\n"
        f"- master_line_out={master_line_out or '-'} master_line_source={master_line_source or '-'} master_line_mode={master_line_mode or '-'}\n"
        f"- soft_trigger_fps={float(args.soft_trigger_fps)} soft_trigger_serials={soft_trigger_serials or '-'}\n"
        f"- group_by={group_by}\n"
        f"- group_timeout_ms={int(args.group_timeout_ms)} timeout_ms={int(args.timeout_ms)}\n"
        f"- pixel_format={str(getattr(args, 'pixel_format', '') or '-') }\n"
        f"- roi={roi_str}\n"
        f"- save_mode={args.save_mode} output_dir={args.output_dir}"
    )

    # master/slave 触发链路的常见“无图”原因：
    # - master 用 Software 触发，但没有把某个 Line 输出设置为 ExposureStartActive（或机型支持的等价信号）；
    # - 或者已设置但没有物理接线到 slave 的触发输入（例如 slave 的 Line0）。
    if master_serial:
        has_any_slave = any(
            (s != master_serial) and (str(src).lower() not in {"software", "triggersoftware"})
            for s, src in zip(serials, trigger_sources)
        )
        if has_any_slave and (not master_line_source):
            print(
                "注意：你启用了 --master-serial，但没有设置 --master-line-source。\n"
                "这意味着脚本不会主动把 master 的输出线配置为触发脉冲，slave 很可能一直收不到触发，因此只会看到 master 出图、无法凑齐组包。\n"
                "建议：\n"
                "- 增加参数：--master-line-source ExposureStartActive （与 MVS Client 中 'Exposure Start Active' 对应）；\n"
                "- 确认物理接线：master 的输出线（例如 Line1）-> slave 的触发输入（例如 Line0）；\n"
                "- 想先验证采集链路可用，可临时用：--trigger-source Software --soft-trigger-fps 5\n"
            )

    num_cameras = int(args.num_cameras) if args.num_cameras is not None else len(serials)
    if len(serials) != num_cameras:
        print(f"--serial 需要提供 {num_cameras} 个，当前为 {len(serials)}")
        return 2

    out_dir = Path(args.output_dir)
    _ensure_dir(out_dir)
    meta_path = out_dir / "metadata.jsonl"

    image_width = int(getattr(args, "image_width", 0) or 0)
    image_height = int(getattr(args, "image_height", 0) or 0)
    image_offset_x = int(getattr(args, "image_offset_x", 0) or 0)
    image_offset_y = int(getattr(args, "image_offset_y", 0) or 0)
    if (image_width > 0) ^ (image_height > 0):
        print("ROI 参数错误：--image-width 与 --image-height 必须同时设置，或同时为 0（不设置）。")
        return 2

    try:
        with open_quad_capture(
            binding=binding,
            serials=serials,
            trigger_sources=trigger_sources,
            trigger_activation=str(args.trigger_activation),
            trigger_cache_enable=bool(args.trigger_cache_enable),
            timeout_ms=int(args.timeout_ms),
            group_timeout_ms=int(args.group_timeout_ms),
            max_pending_groups=int(args.max_pending_groups),
            group_by=group_by,
            enable_soft_trigger_fps=float(args.soft_trigger_fps),
            soft_trigger_serials=soft_trigger_serials,
            camera_event_names=[str(x) for x in (args.camera_event or [])],
            master_serial=master_serial,
            master_line_output=master_line_out,
            master_line_source=master_line_source,
            master_line_mode=master_line_mode,
            pixel_format=str(getattr(args, "pixel_format", "") or ""),
            image_width=(image_width if image_width > 0 else None),
            image_height=(image_height if image_height > 0 else None),
            image_offset_x=int(image_offset_x),
            image_offset_y=int(image_offset_y),
            exposure_auto=str(args.exposure_auto),
            exposure_time_us=float(args.exposure_us),
            gain_auto=str(args.gain_auto),
            gain=float(args.gain),
        ) as cap:
            # 启动采集后立刻做一次带宽估算（便于快速判断是否“先天不可能跑满”。）
            overhead_factor = 1.10
            expected_fps = float(getattr(args, "expected_fps", 0.0) or 0.0)
            soft_fps = float(getattr(args, "soft_trigger_fps", 0.0) or 0.0)
            soft_targets = set(soft_trigger_serials or [])

            # 带宽估算的 fps 选择策略：
            # 1) 用户显式提供 expected_fps：对所有相机生效（外触发最可靠）。
            # 2) master_serial + 软触发 master：通常意味着 master 输出触发脉冲带动 slave，频率对所有相机一致。
            global_fps_hint = None
            if expected_fps > 0:
                global_fps_hint = expected_fps
            elif master_serial and (soft_fps > 0) and (master_serial in soft_targets):
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

            groups_done = 0
            last_log = time.monotonic()
            last_dropped = 0
            last_progress = time.monotonic()
            last_idle_log = 0.0

            if args.camera_event:
                requested = [str(x) for x in (args.camera_event or []) if str(x).strip()]
                print(f"已请求订阅相机事件: {requested}")
                for c in cap.cameras:
                    enabled = getattr(c, "event_names_enabled", [])
                    print(f"- {c.serial}: enabled={enabled or '-'}")

            with meta_path.open("a", encoding="utf-8") as f_meta:
                while True:
                    _drain_event_queue(f_meta=f_meta, event_queue=cap.event_queue)

                    if args.max_groups and groups_done >= int(args.max_groups):
                        break

                    group = cap.get_next_group(timeout_s=0.5)
                    if group is None:
                        _drain_event_queue(f_meta=f_meta, event_queue=cap.event_queue)
                        now = time.monotonic()
                        max_wait = float(args.max_wait_seconds)
                        if max_wait > 0 and (now - last_progress) > max_wait:
                            pending = getattr(cap.assembler, "pending_groups", 0)
                            oldest_age = 0.0
                            try:
                                oldest_age = float(cap.assembler.pending_oldest_age_s())
                            except Exception:
                                oldest_age = 0.0
                            seen_by_cam = getattr(cap.assembler, "frames_seen_by_cam", {})
                            print(
                                "长时间未收到任何完整组包，已退出。\n"
                                f"- trigger_source={args.trigger_source} trigger_activation={args.trigger_activation}\n"
                                f"- serials={serials}\n"
                                f"- output_dir={out_dir}\n"
                                f"- assembler: dropped_groups={cap.assembler.dropped_groups} pending_groups={pending} oldest_age_s={oldest_age:.3f} seen_by_cam={seen_by_cam}\n"
                                "如果你使用硬触发（Line0/Line1...），请确认外部触发脉冲已接到每台相机的对应输入口，且边沿/电平配置一致。\n"
                                "想先验证保存链路是否正常，可用：--trigger-source Software --soft-trigger-fps 5"
                            )
                            return 2

                        idle_log = float(args.idle_log_seconds)
                        if idle_log > 0 and (now - last_idle_log) > idle_log:
                            qsz = cap.frame_queue.qsize()
                            dropped = cap.assembler.dropped_groups
                            pending = getattr(cap.assembler, "pending_groups", 0)
                            oldest_age = 0.0
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

                    if args.save_mode != "none":
                        if group_by == "trigger_index":
                            group_dir = out_dir / f"trigger_{trigger_index:010d}"
                        else:
                            group_dir = out_dir / f"group_{group_seq:010d}"
                        _ensure_dir(group_dir)

                        for fr in group:
                            if args.save_mode == "raw":
                                raw_path = group_dir / f"cam{fr.cam_index}_seq{group_seq:06d}_f{fr.frame_num}.bin"
                                raw_path.write_bytes(fr.data)
                                files[fr.cam_index] = str(raw_path)
                            elif args.save_mode == "sdk-bmp":
                                bmp_path = group_dir / f"cam{fr.cam_index}_seq{group_seq:06d}_f{fr.frame_num}.bmp"
                                try:
                                    save_frame_as_bmp(
                                        binding=binding,
                                        cam=cap.cameras[fr.cam_index].cam,
                                        out_path=bmp_path,
                                        frame=fr,
                                        bayer_method=int(args.bayer_method),
                                    )
                                    files[fr.cam_index] = str(bmp_path)
                                except Exception as exc:
                                    raw_path = group_dir / f"cam{fr.cam_index}_seq{group_seq:06d}_f{fr.frame_num}.bin"
                                    raw_path.write_bytes(fr.data)
                                    files[fr.cam_index] = str(raw_path)
                                    print(f"save bmp failed (cam{fr.cam_index}): {exc}; fallback to raw")

                    record = {
                        "group_seq": group_seq,
                        "group_by": group_by,
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
                            f"groups={groups_done} qsize={cap.frame_queue.qsize()} save_mode={args.save_mode} dropped_groups={dropped} (+{delta_dropped})"
                        )
                        last_dropped = dropped
                        last_log = now

    except KeyboardInterrupt:
        # 用户主动中断时不打印堆栈，避免误解为程序异常。
        print("已中断（KeyboardInterrupt）。")
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
