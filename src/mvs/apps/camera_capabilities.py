# -*- coding: utf-8 -*-

"""相机能力/节点支持诊断。

用途：
- 你想知道“当前这台相机 + 当前固件 + 当前 SDK”到底支持哪些 GenICam 节点；
- 特别是：ROI(Width/Height/OffsetX/OffsetY) 的步进/范围、是否运行中可写；
- PixelFormat 当前值以及常见候选是否可设置；
- TriggerCacheEnable 是否存在/可写；
- 常见相机事件（ExposureStart/ExposureEnd/FrameStart）是否能开启通知。

设计原则：
- 默认尽量只读；
- 探测“可写性”时优先做“写回当前值”的探测（不改变状态）；
- 需要短暂改变状态的探测（例如 PixelFormat 候选、Offset 试写）需要显式开关。

运行示例：
- 仅查看：
  uv run python -m mvs.apps.camera_capabilities --serial A B C D
- 额外探测：
  uv run python -m mvs.apps.camera_capabilities --serial A B \
    --probe-pixel-formats --probe-offset-write --probe-offset-runtime --probe-events
"""

from __future__ import annotations

import argparse
import ctypes
import sys
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

from mvs import MvsDllNotFoundError, load_mvs_binding
from mvs.core._cleanup import best_effort
from mvs.sdk.binding import MvsBinding
from mvs.sdk.camera import MvsSdk
from mvs.sdk.devices import DeviceDesc, enumerate_devices


@dataclass(frozen=True, slots=True)
class IntNodeSnapshot:
    key: str
    ok: bool
    cur: int | None = None
    vmin: int | None = None
    vmax: int | None = None
    inc: int | None = None


@dataclass(frozen=True, slots=True)
class FloatNodeSnapshot:
    key: str
    ok: bool
    cur: float | None = None
    vmin: float | None = None
    vmax: float | None = None


@dataclass(frozen=True, slots=True)
class EnumNodeSnapshot:
    key: str
    ok: bool
    symbolic: str = ""
    cur_value: int | None = None


@dataclass(frozen=True, slots=True)
class BoolNodeSnapshot:
    key: str
    ok: bool
    cur: bool | None = None


@dataclass(frozen=True, slots=True)
class WriteProbe:
    key: str
    ok: bool
    ret: int
    note: str = ""


def _check_ret(binding: MvsBinding, ret: int) -> bool:
    try:
        return int(ret) == int(binding.MV_OK)
    except Exception:
        return False


def _try_get_int_info(*, binding: MvsBinding, cam: Any, key: str) -> IntNodeSnapshot:
    try:
        st = binding.params.MVCC_INTVALUE()
        ret = int(cam.MV_CC_GetIntValue(str(key), st))
        if not _check_ret(binding, ret):
            return IntNodeSnapshot(key=str(key), ok=False)
        return IntNodeSnapshot(
            key=str(key),
            ok=True,
            cur=int(getattr(st, "nCurValue")),
            vmin=int(getattr(st, "nMin")),
            vmax=int(getattr(st, "nMax")),
            inc=int(getattr(st, "nInc")),
        )
    except Exception:
        return IntNodeSnapshot(key=str(key), ok=False)


def _try_get_float_info(*, binding: MvsBinding, cam: Any, key: str) -> FloatNodeSnapshot:
    # 并非所有环境都暴露 MVCC_FLOATVALUE（不同版本 MvImport 可能差异）。
    try:
        st_type = getattr(binding.params, "MVCC_FLOATVALUE", None)
        if st_type is None:
            return FloatNodeSnapshot(key=str(key), ok=False)
        st = st_type()
        ret = int(cam.MV_CC_GetFloatValue(str(key), st))
        if not _check_ret(binding, ret):
            return FloatNodeSnapshot(key=str(key), ok=False)
        return FloatNodeSnapshot(
            key=str(key),
            ok=True,
            cur=float(getattr(st, "fCurValue")),
            vmin=float(getattr(st, "fMin")),
            vmax=float(getattr(st, "fMax")),
        )
    except Exception:
        return FloatNodeSnapshot(key=str(key), ok=False)


def _decode_c_string(buf: Any) -> str:
    # MVS 的 chSymbolic 通常是 char[64] 之类定长数组。
    try:
        raw = bytes(buf)
    except Exception:
        try:
            raw = buf  # type: ignore[assignment]
        except Exception:
            return ""

    try:
        s = raw.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")
    except Exception:
        try:
            s = raw.split(b"\x00", 1)[0].decode("gbk", errors="ignore")
        except Exception:
            s = ""
    return str(s).strip()


def _try_get_enum_info(*, binding: MvsBinding, cam: Any, key: str) -> EnumNodeSnapshot:
    sym = ""
    cur_val: int | None = None

    try:
        st = binding.params.MVCC_ENUMENTRY()
        ret = int(cam.MV_CC_GetEnumEntrySymbolic(str(key), st))
        if _check_ret(binding, ret):
            sym = _decode_c_string(getattr(st, "chSymbolic", b""))
    except Exception:
        sym = ""

    try:
        st_enum_type = getattr(binding.params, "MVCC_ENUMVALUE", None)
        if st_enum_type is not None:
            st_enum = st_enum_type()
            ret = int(cam.MV_CC_GetEnumValue(str(key), st_enum))
            if _check_ret(binding, ret):
                cur_val = int(getattr(st_enum, "nCurValue"))
    except Exception:
        cur_val = None

    ok = bool(sym) or (cur_val is not None)
    return EnumNodeSnapshot(key=str(key), ok=ok, symbolic=str(sym), cur_value=cur_val)


def _try_get_bool_info(*, binding: MvsBinding, cam: Any, key: str) -> BoolNodeSnapshot:
    # 部分 MvImport 版本可能没有 MVCC_BOOLVALUE 或没有 GetBoolValue。
    try:
        get_fn = getattr(cam, "MV_CC_GetBoolValue", None)
        st_type = getattr(binding.params, "MVCC_BOOLVALUE", None)
        if get_fn is None or st_type is None:
            return BoolNodeSnapshot(key=str(key), ok=False)
        st = st_type()
        ret = int(get_fn(str(key), st))
        if not _check_ret(binding, ret):
            return BoolNodeSnapshot(key=str(key), ok=False)
        cur = bool(getattr(st, "bCurValue"))
        return BoolNodeSnapshot(key=str(key), ok=True, cur=cur)
    except Exception:
        return BoolNodeSnapshot(key=str(key), ok=False)


def _probe_set_int_same(*, binding: MvsBinding, cam: Any, key: str, value: int) -> WriteProbe:
    try:
        ret = int(cam.MV_CC_SetIntValue(str(key), int(value)))
        return WriteProbe(key=str(key), ok=_check_ret(binding, ret), ret=int(ret), note="set_to_same")
    except Exception:
        return WriteProbe(key=str(key), ok=False, ret=-1, note="exception")


def _probe_set_int_nudge_and_restore(
    *,
    binding: MvsBinding,
    cam: Any,
    key: str,
    cur: int,
    vmin: int,
    vmax: int,
    inc: int,
) -> list[WriteProbe]:
    # 注意：该探测会短暂改变相机状态，仅用于诊断。
    inc = int(inc) if int(inc) > 0 else 1

    cand: int | None = None
    if int(cur) + inc <= int(vmax):
        cand = int(cur) + inc
    elif int(cur) - inc >= int(vmin):
        cand = int(cur) - inc

    out: list[WriteProbe] = []
    if cand is None or int(cand) == int(cur):
        return [WriteProbe(key=str(key), ok=False, ret=0, note="no_nudge_candidate")]

    # 1) 写入候选
    try:
        ret1 = int(cam.MV_CC_SetIntValue(str(key), int(cand)))
        out.append(WriteProbe(key=str(key), ok=_check_ret(binding, ret1), ret=int(ret1), note=f"nudge_to={cand}"))
    except Exception:
        out.append(WriteProbe(key=str(key), ok=False, ret=-1, note=f"nudge_exception_to={cand}"))
        return out

    # 2) 恢复原值
    try:
        ret2 = int(cam.MV_CC_SetIntValue(str(key), int(cur)))
        out.append(WriteProbe(key=str(key), ok=_check_ret(binding, ret2), ret=int(ret2), note=f"restore_to={cur}"))
    except Exception:
        out.append(WriteProbe(key=str(key), ok=False, ret=-1, note=f"restore_exception_to={cur}"))

    return out


def _probe_set_bool_toggle_and_restore(
    *,
    binding: MvsBinding,
    cam: Any,
    key: str,
) -> list[WriteProbe]:
    """通过 SetBoolValue 探测 bool 节点是否存在/可写。

    注意：
        - 该探测会短暂切换 True/False 并尝试恢复。
        - 因为部分 MvImport 版本缺少 GetBoolValue，这里无法可靠读取“当前值”，
          所以恢复策略是：先写 True，再写 False（或反过来）。
        - 对 TriggerCacheEnable 这类“非关键且本仓库本来也会 best-effort 设置”的节点，
          该探测风险可接受；其它节点默认不做。
    """

    set_fn = getattr(cam, "MV_CC_SetBoolValue", None)
    if set_fn is None:
        return [WriteProbe(key=str(key), ok=False, ret=-1, note="no_SetBoolValue")]

    out: list[WriteProbe] = []
    for v in [True, False]:
        try:
            ret = int(set_fn(str(key), bool(v)))
            out.append(
                WriteProbe(
                    key=str(key),
                    ok=_check_ret(binding, ret),
                    ret=int(ret),
                    note=f"set:{v}",
                )
            )
        except Exception:
            out.append(WriteProbe(key=str(key), ok=False, ret=-1, note=f"set_exception:{v}"))
            break
    return out


def _probe_set_enum_candidates(
    *,
    binding: MvsBinding,
    cam: Any,
    key: str,
    candidates: Sequence[str],
    restore_symbolic: str,
) -> tuple[list[str], list[WriteProbe]]:
    supported: list[str] = []
    probes: list[WriteProbe] = []

    set_fn = getattr(cam, "MV_CC_SetEnumValueByString", None)
    if set_fn is None:
        return supported, [WriteProbe(key=str(key), ok=False, ret=-1, note="no_SetEnumValueByString")]

    for cand in candidates:
        try:
            ret = int(set_fn(str(key), str(cand)))
        except Exception:
            probes.append(WriteProbe(key=str(key), ok=False, ret=-1, note=f"set_exception:{cand}"))
            continue

        ok = _check_ret(binding, ret)
        probes.append(WriteProbe(key=str(key), ok=ok, ret=int(ret), note=f"set_try:{cand}"))
        if ok:
            supported.append(str(cand))

    # 恢复原来的 symbolic（若能读到）
    if str(restore_symbolic).strip():
        try:
            ret = int(set_fn(str(key), str(restore_symbolic)))
            probes.append(WriteProbe(key=str(key), ok=_check_ret(binding, ret), ret=int(ret), note=f"restore:{restore_symbolic}"))
        except Exception:
            probes.append(WriteProbe(key=str(key), ok=False, ret=-1, note=f"restore_exception:{restore_symbolic}"))

    return supported, probes


def _probe_event_notification(*, binding: MvsBinding, cam: Any, event_name: str) -> WriteProbe:
    on_fn = getattr(cam, "MV_CC_EventNotificationOn", None)
    off_fn = getattr(cam, "MV_CC_EventNotificationOff", None)
    if on_fn is None or off_fn is None:
        return WriteProbe(key=f"EventNotification:{event_name}", ok=False, ret=-1, note="no_event_api")

    try:
        ret_on = int(on_fn(str(event_name)))
        ok_on = _check_ret(binding, ret_on)
    except Exception:
        return WriteProbe(key=f"EventNotification:{event_name}", ok=False, ret=-1, note="on_exception")

    # 尽力关闭（避免影响后续其它脚本）
    try:
        ret_off = int(off_fn(str(event_name)))
        ok_off = _check_ret(binding, ret_off)
    except Exception:
        ret_off = -1
        ok_off = False

    note = f"on=0x{int(ret_on):08X}, off=0x{int(ret_off):08X}, off_ok={ok_off}"
    return WriteProbe(key=f"EventNotification:{event_name}", ok=ok_on, ret=int(ret_on), note=note)


def _open_camera_by_desc(*, binding: MvsBinding, st_dev_list: Any, desc: DeviceDesc) -> Any:
    cam = binding.MvCamera()

    dev_info = ctypes.cast(
        st_dev_list.pDeviceInfo[int(desc.index)],
        ctypes.POINTER(binding.params.MV_CC_DEVICE_INFO),
    ).contents

    ret = int(cam.MV_CC_CreateHandle(dev_info))
    if not _check_ret(binding, ret):
        raise RuntimeError(f"CreateHandle({desc.serial}) failed, ret=0x{int(ret):08X}")

    try:
        # Control 权限打开（与仓库其它入口保持一致）
        ret = int(cam.MV_CC_OpenDevice(3, 0))
        if not _check_ret(binding, ret):
            raise RuntimeError(f"OpenDevice({desc.serial}) failed, ret=0x{int(ret):08X}")
        return cam
    except Exception:
        best_effort(cam.MV_CC_CloseDevice)
        best_effort(cam.MV_CC_DestroyHandle)
        raise


def _close_cam(cam: Any) -> None:
    best_effort(cam.MV_CC_StopGrabbing)
    best_effort(cam.MV_CC_CloseDevice)
    best_effort(cam.MV_CC_DestroyHandle)


def _print_kv(prefix: str, key: str, value: Any) -> None:
    print(f"{prefix}{key}: {value}")


def _best_effort_set_roi(
    *,
    binding: MvsBinding,
    cam: Any,
    width: int,
    height: int,
    offset_x: int,
    offset_y: int,
) -> None:
    """尽力设置 ROI（不做复杂对齐逻辑；用于诊断 staging）。

    说明：
        - 正式配置逻辑在 `mvs.sdk.camera.configure_resolution()`；
          这里为了避免引入过多依赖，仅做最小顺序：
          先清 offset -> 写 width/height -> 写 offset。
        - 如果某些节点不可写，本函数会尽力而为。
    """

    best_effort(cam.MV_CC_SetIntValue, "OffsetX", 0)
    best_effort(cam.MV_CC_SetIntValue, "OffsetY", 0)
    best_effort(cam.MV_CC_SetIntValue, "Width", int(width))
    best_effort(cam.MV_CC_SetIntValue, "Height", int(height))
    best_effort(cam.MV_CC_SetIntValue, "OffsetX", int(offset_x))
    best_effort(cam.MV_CC_SetIntValue, "OffsetY", int(offset_y))


def main(argv: Optional[Sequence[str]] = None) -> int:
    # 尽量固定 UTF-8 输出，避免在重定向到文件时出现乱码。
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    p = argparse.ArgumentParser(description="MVS 相机能力/节点支持诊断")
    p.add_argument(
        "--mvimport-dir",
        default=None,
        help=(
            "MVS 官方 Python 示例绑定目录（MvImport）。"
            "包含 MvCameraControl_class.py 等文件。"
            "可选；也可用环境变量 MVS_MVIMPORT_DIR。"
        ),
    )
    p.add_argument(
        "--dll-dir",
        default=None,
        help="包含 MvCameraControl.dll 的目录（可选）。也可用环境变量 MVS_DLL_DIR。",
    )
    p.add_argument("--list", action="store_true", help="仅枚举并打印设备信息")
    p.add_argument(
        "--serial",
        action="extend",
        nargs="+",
        default=[],
        help="按序列号选择相机（可一次性传多个，或重复多次）。为空时默认对所有枚举到的设备诊断。",
    )

    p.add_argument(
        "--probe-pixel-formats",
        action="store_true",
        help="探测常见 PixelFormat 候选是否可设置（会短暂切换再恢复，仅在未开始取流时执行）。",
    )
    p.add_argument(
        "--probe-offset-write",
        action="store_true",
        help="探测 OffsetX/OffsetY 是否可写（会短暂 nudge 再恢复，仅在未开始取流时执行）。",
    )
    p.add_argument(
        "--probe-offset-runtime",
        action="store_true",
        help="探测 StartGrabbing 后 OffsetX/OffsetY 是否仍可写（会短暂 nudge 再恢复）。",
    )
    p.add_argument(
        "--probe-events",
        action="store_true",
        help="探测常见事件（ExposureStart/ExposureEnd/FrameStart）是否能开启通知。",
    )

    p.add_argument(
        "--probe-trigger-cache-enable",
        action="store_true",
        help="探测 TriggerCacheEnable 是否可写（会短暂切换 True/False）。",
    )

    p.add_argument(
        "--stage-roi",
        nargs=4,
        metavar=("W", "H", "OX", "OY"),
        default=None,
        help=(
            "可选：诊断前先暂时把相机 ROI 设置为给定的 Width/Height/OffsetX/OffsetY，"
            "用于观察 offset 的范围/步进是否变化（诊断结束会尽力恢复原值）。"
        ),
    )

    args = p.parse_args(list(argv) if argv is not None else None)

    try:
        binding = load_mvs_binding(mvimport_dir=args.mvimport_dir, dll_dir=args.dll_dir)
    except MvsDllNotFoundError as exc:
        print(str(exc))
        return 2

    sdk = MvsSdk(binding)
    sdk.initialize()
    try:
        st_dev_list, descs = enumerate_devices(binding)
        if not descs:
            print("未枚举到相机设备。")
            return 2

        if bool(args.list):
            print("已枚举到设备：")
            for d in descs:
                typ = "GigE" if int(d.tlayer_type) == int(binding.params.MV_GIGE_DEVICE) else "USB"
                ip = d.ip or "-"
                print(f"- serial={d.serial} model={d.model} type={typ} ip={ip} user={d.user_name}")
            return 0

        requested = [str(x).strip() for x in (args.serial or []) if str(x).strip()]
        if not requested:
            requested = [d.serial for d in descs if str(d.serial).strip()]

        serial_to_desc = {d.serial: d for d in descs}

        # 常见 PixelFormat 候选：覆盖本仓库常用与常见替代。
        pf_candidates = [
            "Mono8",
            "BayerRG8",
            "BayerBG8",
            "BayerGR8",
            "BayerGB8",
            "RGB8Packed",
            "BGR8Packed",
        ]

        int_nodes = [
            "Width",
            "Height",
            "OffsetX",
            "OffsetY",
            "PayloadSize",
            "GevPayloadSize",
        ]
        float_nodes = [
            "ExposureTime",
            "Gain",
            "ResultingFrameRate",
            "AcquisitionFrameRate",
            "TriggerDelay",
        ]
        enum_nodes = [
            "PixelFormat",
            "TriggerMode",
            "TriggerSource",
            "TriggerActivation",
            "LineMode",
        ]
        bool_nodes = [
            "TriggerCacheEnable",
            "StrobeEnable",
            "LineInverter",
        ]

        for serial in requested:
            d = serial_to_desc.get(serial)
            if d is None:
                print(f"[error] 找不到序列号为 {serial} 的相机（请先 --list 确认）。")
                continue

            typ = "GigE" if int(d.tlayer_type) == int(binding.params.MV_GIGE_DEVICE) else "USB"
            ip = d.ip or "-"
            print("=" * 80)
            print(f"serial={d.serial} model={d.model} type={typ} ip={ip} user={d.user_name}")

            cam = _open_camera_by_desc(binding=binding, st_dev_list=st_dev_list, desc=d)
            try:
                prefix = "  "

                # 可选：先 staging 到某个 ROI，便于查看 Offset 的 max/步进。
                orig_w = _try_get_int_info(binding=binding, cam=cam, key="Width")
                orig_h = _try_get_int_info(binding=binding, cam=cam, key="Height")
                orig_ox = _try_get_int_info(binding=binding, cam=cam, key="OffsetX")
                orig_oy = _try_get_int_info(binding=binding, cam=cam, key="OffsetY")

                did_stage = False
                if args.stage_roi is not None:
                    try:
                        w2, h2, ox2, oy2 = [int(x) for x in list(args.stage_roi)]
                        _best_effort_set_roi(
                            binding=binding,
                            cam=cam,
                            width=int(w2),
                            height=int(h2),
                            offset_x=int(ox2),
                            offset_y=int(oy2),
                        )
                        did_stage = True
                        print(
                            "  [stage roi] "
                            f"requested=({w2}x{h2}, ox={ox2}, oy={oy2}) "
                            "(done best-effort; actual values see int nodes below)"
                        )
                    except Exception as exc:
                        print(f"  [stage roi] failed: {exc}")

                print("  [int nodes]")
                int_snaps = [_try_get_int_info(binding=binding, cam=cam, key=k) for k in int_nodes]
                for s in int_snaps:
                    if not s.ok:
                        _print_kv(prefix, s.key, "(unavailable)")
                    else:
                        _print_kv(prefix, s.key, f"cur={s.cur} min={s.vmin} max={s.vmax} inc={s.inc}")

                print("  [enum nodes]")
                enum_snaps = [_try_get_enum_info(binding=binding, cam=cam, key=k) for k in enum_nodes]
                for s in enum_snaps:
                    if not s.ok:
                        _print_kv(prefix, s.key, "(unavailable)")
                    else:
                        sym = s.symbolic or "-"
                        val = f"0x{int(s.cur_value):08X}" if s.cur_value is not None else "-"
                        _print_kv(prefix, s.key, f"symbolic={sym} value={val}")

                print("  [float nodes]")
                float_snaps = [_try_get_float_info(binding=binding, cam=cam, key=k) for k in float_nodes]
                for s in float_snaps:
                    if not s.ok:
                        _print_kv(prefix, s.key, "(unavailable)")
                    else:
                        _print_kv(prefix, s.key, f"cur={s.cur} min={s.vmin} max={s.vmax}")

                print("  [bool nodes]")
                bool_snaps = [_try_get_bool_info(binding=binding, cam=cam, key=k) for k in bool_nodes]
                for s in bool_snaps:
                    if not s.ok:
                        _print_kv(prefix, s.key, "(unavailable)")
                    else:
                        _print_kv(prefix, s.key, f"cur={s.cur}")

                # 写回当前值探测（不改变状态）
                print("  [write probe: set to same value]")
                for s in int_snaps:
                    if not s.ok or s.cur is None:
                        continue
                    if s.key in {"OffsetX", "OffsetY"}:
                        pr = _probe_set_int_same(binding=binding, cam=cam, key=s.key, value=int(s.cur))
                        _print_kv(prefix, s.key, f"ok={pr.ok} ret=0x{int(pr.ret):08X} note={pr.note}")

                # PixelFormat 候选探测（短暂切换再恢复）
                if bool(args.probe_pixel_formats):
                    pf_snap = next((x for x in enum_snaps if x.key == "PixelFormat"), None)
                    restore = pf_snap.symbolic if pf_snap is not None else ""
                    supported, probes = _probe_set_enum_candidates(
                        binding=binding,
                        cam=cam,
                        key="PixelFormat",
                        candidates=pf_candidates,
                        restore_symbolic=str(restore),
                    )
                    print("  [probe pixel formats]")
                    _print_kv(prefix, "supported_candidates", supported)
                    for pr in probes:
                        _print_kv(prefix, "PixelFormat", f"ok={pr.ok} ret=0x{int(pr.ret):08X} note={pr.note}")

                # Offset nudge 探测（短暂改变再恢复，仅用于诊断）
                if bool(args.probe_offset_write):
                    print("  [probe offset write (before grabbing)]")
                    for s in int_snaps:
                        if not s.ok or s.cur is None:
                            continue
                        if s.key not in {"OffsetX", "OffsetY"}:
                            continue
                        if s.vmin is None or s.vmax is None or s.inc is None:
                            continue
                        for pr in _probe_set_int_nudge_and_restore(
                            binding=binding,
                            cam=cam,
                            key=s.key,
                            cur=int(s.cur),
                            vmin=int(s.vmin),
                            vmax=int(s.vmax),
                            inc=int(s.inc),
                        ):
                            _print_kv(prefix, s.key, f"ok={pr.ok} ret=0x{int(pr.ret):08X} note={pr.note}")

                # 事件通知探测
                if bool(args.probe_events):
                    print("  [probe events]")
                    for ev in ["ExposureStart", "ExposureEnd", "FrameStart"]:
                        pr = _probe_event_notification(binding=binding, cam=cam, event_name=ev)
                        _print_kv(prefix, pr.key, f"ok={pr.ok} ret=0x{int(pr.ret):08X} note={pr.note}")

                # 运行中 Offset 可写探测
                if bool(args.probe_offset_runtime):
                    print("  [probe offset write (during grabbing)]")
                    try:
                        ret = int(cam.MV_CC_StartGrabbing())
                        _print_kv(prefix, "StartGrabbing", f"ok={_check_ret(binding, ret)} ret=0x{int(ret):08X}")

                        # 重新读一次 Offset 信息（有的机型 StartGrabbing 后范围/可写性会变化）
                        ox = _try_get_int_info(binding=binding, cam=cam, key="OffsetX")
                        oy = _try_get_int_info(binding=binding, cam=cam, key="OffsetY")
                        for s in [ox, oy]:
                            if not s.ok or s.cur is None or s.vmin is None or s.vmax is None or s.inc is None:
                                _print_kv(prefix, s.key, "(unavailable)")
                                continue
                            for pr in _probe_set_int_nudge_and_restore(
                                binding=binding,
                                cam=cam,
                                key=s.key,
                                cur=int(s.cur),
                                vmin=int(s.vmin),
                                vmax=int(s.vmax),
                                inc=int(s.inc),
                            ):
                                _print_kv(prefix, s.key, f"ok={pr.ok} ret=0x{int(pr.ret):08X} note={pr.note}")
                    finally:
                        best_effort(cam.MV_CC_StopGrabbing)

                # TriggerCacheEnable 探测（写 bool）
                if bool(args.probe_trigger_cache_enable):
                    print("  [probe TriggerCacheEnable]")
                    for pr in _probe_set_bool_toggle_and_restore(
                        binding=binding,
                        cam=cam,
                        key="TriggerCacheEnable",
                    ):
                        _print_kv(prefix, pr.key, f"ok={pr.ok} ret=0x{int(pr.ret):08X} note={pr.note}")

                # 输出一个“关键结论提示”
                ox = next((x for x in int_snaps if x.key == "OffsetX"), None)
                oy = next((x for x in int_snaps if x.key == "OffsetY"), None)
                if ox is not None and oy is not None and ox.ok and oy.ok:
                    incx = ox.inc or 1
                    incy = oy.inc or 1
                    print("  [hint]")
                    print(
                        "  - ROI 偏移的步进很关键：如果你的配置 offset 不是 inc 的倍数，相机可能会对齐到另一个值。"
                        f" 当前读取：OffsetXInc={incx}, OffsetYInc={incy}。"
                    )

                # 尽力恢复 staging 前的 ROI，避免影响用户的后续采集。
                if did_stage:
                    if (
                        orig_w.ok and orig_h.ok and orig_ox.ok and orig_oy.ok
                        and orig_w.cur is not None and orig_h.cur is not None
                        and orig_ox.cur is not None and orig_oy.cur is not None
                    ):
                        _best_effort_set_roi(
                            binding=binding,
                            cam=cam,
                            width=int(orig_w.cur),
                            height=int(orig_h.cur),
                            offset_x=int(orig_ox.cur),
                            offset_y=int(orig_oy.cur),
                        )
                        print(
                            "  [stage roi] restored best-effort to original "
                            f"({orig_w.cur}x{orig_h.cur}, ox={orig_ox.cur}, oy={orig_oy.cur})"
                        )
                    else:
                        print("  [stage roi] skip restore: cannot read original ROI reliably")

            finally:
                _close_cam(cam)

        print("=" * 80)
        print("Done.")
        return 0

    finally:
        best_effort(sdk.finalize)


if __name__ == "__main__":
    raise SystemExit(main())
