# -*- coding: utf-8 -*-

"""探测相机是否支持时间同步（重点：PTP/IEEE1588）。

说明：
- GigE Vision 常用 PTP/IEEE1588 实现跨相机“同一时间基准”。
- 本脚本通过尝试读取一组常见 GenICam 节点（如 GevIEEE1588 / PtpEnable 等）来判断“是否支持”。
- 若节点存在但状态未锁定（Locked/Slave 等），仍不能认为已经完成时间同步。

用法示例：
- 仅枚举设备：python tools/mvs_probe_time_sync.py --list
- 探测指定序列号：python tools/mvs_probe_time_sync.py --serial A B C

注意：
- 节点名称因机型/固件可能不同；本脚本做的是“快速判定”，并输出命中的节点便于你二次确认。
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mvs import DeviceDesc, MvsDllNotFoundError, MvsSdk, enumerate_devices, load_mvs_binding


@dataclass(frozen=True, slots=True)
class NodeProbeResult:
    key: str
    value_type: str
    value: object
    symbolic: Optional[str] = None


@dataclass(frozen=True, slots=True)
class NodeWriteResult:
    key: str
    ok: bool
    detail: str


def _decode_c_char_array(buf: Any) -> str:
    raw = bytes(buf)
    raw = raw.split(b"\x00", 1)[0]
    try:
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _try_get_bool(cam: Any, binding: Any, key: str) -> Optional[NodeProbeResult]:
    v = ctypes.c_bool(False)
    ret = cam.MV_CC_GetBoolValue(key, v)
    if int(ret) != int(binding.MV_OK):
        return None
    return NodeProbeResult(key=key, value_type="bool", value=bool(v.value))


def _try_get_int64(cam: Any, binding: Any, key: str) -> Optional[NodeProbeResult]:
    st = binding.params.MVCC_INTVALUE_EX()
    ret = cam.MV_CC_GetIntValueEx(key, st)
    if int(ret) != int(binding.MV_OK):
        return None
    return NodeProbeResult(key=key, value_type="int64", value=int(st.nCurValue))


def _try_get_int32(cam: Any, binding: Any, key: str) -> Optional[NodeProbeResult]:
    # 很多 GenICam Integer 节点是 32-bit（例如 DomainNumber、Priority 等）。
    st = binding.params.MVCC_INTVALUE()
    ret = cam.MV_CC_GetIntValue(key, st)
    if int(ret) != int(binding.MV_OK):
        return None
    return NodeProbeResult(key=key, value_type="int", value=int(st.nCurValue))


def _try_get_float(cam: Any, binding: Any, key: str) -> Optional[NodeProbeResult]:
    st = binding.params.MVCC_FLOATVALUE()
    ret = cam.MV_CC_GetFloatValue(key, st)
    if int(ret) != int(binding.MV_OK):
        return None
    return NodeProbeResult(key=key, value_type="float", value=float(st.fCurValue))


def _try_get_enum_cur_value(cam: Any, binding: Any, key: str) -> Optional[NodeProbeResult]:
    st = binding.params.MVCC_ENUMVALUE()
    ret = cam.MV_CC_GetEnumValue(key, st)
    if int(ret) != int(binding.MV_OK):
        return None

    cur = int(st.nCurValue)

    # 把枚举值映射成 symbolic（如果相机 XML 提供了）。
    sym: Optional[str] = None
    try:
        entry = binding.params.MVCC_ENUMENTRY()
        entry.nValue = cur
        ret2 = cam.MV_CC_GetEnumEntrySymbolic(key, entry)
        if int(ret2) == int(binding.MV_OK):
            sym = _decode_c_char_array(entry.chSymbolic)
    except Exception:
        sym = None

    return NodeProbeResult(key=key, value_type="enum", value=cur, symbolic=sym)


def _try_get_string(cam: Any, binding: Any, key: str) -> Optional[NodeProbeResult]:
    st = binding.params.MVCC_STRINGVALUE()
    ret = cam.MV_CC_GetStringValue(key, st)
    if int(ret) != int(binding.MV_OK):
        return None
    return NodeProbeResult(key=key, value_type="string", value=_decode_c_char_array(st.chCurValue))


def _try_set_bool(cam: Any, binding: Any, key: str, value: bool) -> NodeWriteResult:
    try:
        ret = cam.MV_CC_SetBoolValue(key, bool(value))
        if int(ret) == int(binding.MV_OK):
            return NodeWriteResult(key=key, ok=True, detail=f"bool={bool(value)}")
        return NodeWriteResult(key=key, ok=False, detail=f"ret=0x{int(ret):08X}")
    except Exception as exc:
        return NodeWriteResult(key=key, ok=False, detail=str(exc))


def _try_set_int(cam: Any, binding: Any, key: str, value: int) -> NodeWriteResult:
    try:
        ret = cam.MV_CC_SetIntValue(key, int(value))
        if int(ret) == int(binding.MV_OK):
            return NodeWriteResult(key=key, ok=True, detail=f"int={int(value)}")
        return NodeWriteResult(key=key, ok=False, detail=f"ret=0x{int(ret):08X}")
    except Exception as exc:
        return NodeWriteResult(key=key, ok=False, detail=str(exc))


def _try_set_enum_by_string(cam: Any, binding: Any, key: str, value: str) -> NodeWriteResult:
    try:
        ret = cam.MV_CC_SetEnumValueByString(key, str(value))
        if int(ret) == int(binding.MV_OK):
            return NodeWriteResult(key=key, ok=True, detail=f"enum={value}")
        return NodeWriteResult(key=key, ok=False, detail=f"ret=0x{int(ret):08X}")
    except Exception as exc:
        return NodeWriteResult(key=key, ok=False, detail=str(exc))


def _probe_node(cam: Any, binding: Any, key: str) -> Optional[NodeProbeResult]:
    # 许多节点在不同机型上类型不同（bool/enum/string/int），按常见顺序逐个尝试。
    return (
        _try_get_bool(cam, binding, key)
        or _try_get_enum_cur_value(cam, binding, key)
        or _try_get_int32(cam, binding, key)
        or _try_get_int64(cam, binding, key)
        or _try_get_float(cam, binding, key)
        or _try_get_string(cam, binding, key)
    )


def _open_device_for_probe(*, binding: Any, st_dev_list: Any, dev_index: int) -> Any:
    cam = binding.MvCamera()

    dev_info = ctypes.cast(
        st_dev_list.pDeviceInfo[dev_index],
        ctypes.POINTER(binding.params.MV_CC_DEVICE_INFO),
    ).contents

    ret = cam.MV_CC_CreateHandle(dev_info)
    if int(ret) != int(binding.MV_OK):
        raise RuntimeError(f"CreateHandle failed, ret=0x{int(ret):08X}")

    try:
        ret = cam.MV_CC_OpenDevice()
        if int(ret) != int(binding.MV_OK):
            raise RuntimeError(f"OpenDevice failed, ret=0x{int(ret):08X}")

        # 探测节点不需要取流，尽量把触发关掉，避免某些固件在触发模式下节点不可读/不可写。
        try:
            cam.MV_CC_SetEnumValueByString("TriggerMode", "Off")
        except Exception:
            pass

        return cam
    except Exception:
        try:
            cam.MV_CC_CloseDevice()
        except Exception:
            pass
        try:
            cam.MV_CC_DestroyHandle()
        except Exception:
            pass
        raise


def _close_cam(cam: Any) -> None:
    try:
        cam.MV_CC_CloseDevice()
    except Exception:
        pass
    try:
        cam.MV_CC_DestroyHandle()
    except Exception:
        pass


def _tlayer_to_str(binding: Any, tlayer: int) -> str:
    if int(tlayer) == int(binding.params.MV_GIGE_DEVICE):
        return "GigE"
    if int(tlayer) == int(binding.params.MV_USB_DEVICE):
        return "USB"
    return f"0x{int(tlayer):08X}"


# 常见时间同步（PTP/IEEE1588）相关节点名候选。
# 命名会因机型/固件差异而变化；这里尽量覆盖常见写法。
PTP_NODE_CANDIDATES: Tuple[str, ...] = (
    "GevIEEE1588",
    "GevIEEE1588DomainNumber",
    "GevIEEE1588Priority1",
    "GevIEEE1588Priority2",
    "GevIEEE1588ClockIdentity",
    "GevIEEE1588Status",
    "GevIEEE1588ClockAccuracy",
    "GevIEEE1588OffsetFromMaster",
    "GevIEEE1588ParentDSIdentity",
    "PtpEnable",
    "PtpDomainNumber",
    "PtpPriority1",
    "PtpPriority2",
    "PtpClockIdentity",
    "PtpStatus",
    "PtpPortState",
    "PtpOffsetFromMaster",
    "PtpClockAccuracy",
    "TimestampTickFrequency",
    "GevTimestampTickFrequency",
)


def _try_enable_ptp(cam: Any, binding: Any) -> List[str]:
    """尽力开启 PTP。

    返回值为已成功写入的节点列表（用于输出）。
    """

    enabled: List[str] = []

    # 常见开关节点名（不同机型可能不同）。
    for key in ("GevIEEE1588", "PtpEnable"):
        try:
            ret = cam.MV_CC_SetBoolValue(key, True)
            if int(ret) == int(binding.MV_OK):
                enabled.append(key)
                continue
        except Exception:
            pass

        # 某些固件把开关做成 enum（On/Off）。
        for s in ("On", "Enable", "True"):
            try:
                ret = cam.MV_CC_SetEnumValueByString(key, s)
                if int(ret) == int(binding.MV_OK):
                    enabled.append(key)
                    break
            except Exception:
                continue

    return enabled


def _try_apply_ptp_params(
    *,
    cam: Any,
    binding: Any,
    domain: Optional[int],
    priority1: Optional[int],
    priority2: Optional[int],
    force_slave_only: bool,
    force_master_only: bool,
) -> List[NodeWriteResult]:
    """尽力写入 PTP 相关参数。

    说明：不同机型/固件节点名不一致，本函数会对多组候选节点做 best-effort 写入。
    """

    results: List[NodeWriteResult] = []

    if domain is not None:
        for key in ("GevIEEE1588DomainNumber", "PtpDomainNumber"):
            results.append(_try_set_int(cam, binding, key, int(domain)))

    if priority1 is not None:
        for key in ("GevIEEE1588Priority1", "PtpPriority1"):
            results.append(_try_set_int(cam, binding, key, int(priority1)))

    if priority2 is not None:
        for key in ("GevIEEE1588Priority2", "PtpPriority2"):
            results.append(_try_set_int(cam, binding, key, int(priority2)))

    if bool(force_slave_only) and bool(force_master_only):
        # 参数解析阶段也会拦截，这里兜底。
        results.append(NodeWriteResult(key="(mode)", ok=False, detail="conflict: slave_only & master_only"))
        return results

    if bool(force_slave_only):
        # 常见 bool 节点写法
        for key in ("GevIEEE1588SlaveOnly", "PtpSlaveOnly"):
            results.append(_try_set_bool(cam, binding, key, True))

        # 常见 enum 节点写法
        for key in ("GevIEEE1588ClockMode", "GevIEEE1588Mode", "PtpClockMode", "PtpMode"):
            for v in ("SlaveOnly", "Slave", "OnlySlave"):
                r = _try_set_enum_by_string(cam, binding, key, v)
                results.append(r)
                if r.ok:
                    break

    if bool(force_master_only):
        for key in ("GevIEEE1588SlaveOnly", "PtpSlaveOnly"):
            results.append(_try_set_bool(cam, binding, key, False))

        for key in ("GevIEEE1588ClockMode", "GevIEEE1588Mode", "PtpClockMode", "PtpMode"):
            for v in ("MasterOnly", "Master", "OnlyMaster"):
                r = _try_set_enum_by_string(cam, binding, key, v)
                results.append(r)
                if r.ok:
                    break

    return results


def _looks_like_time_sync_supported(found: Dict[str, NodeProbeResult]) -> bool:
    # 快速判定：存在“启用开关”或“状态”节点，基本即可认为机型/固件支持 PTP 能力。
    keys = set(found.keys())
    return bool(keys.intersection({"GevIEEE1588", "PtpEnable", "GevIEEE1588Status", "PtpStatus"}))


def _is_master_status(found: Dict[str, NodeProbeResult]) -> Optional[bool]:
    r = found.get("GevIEEE1588Status") or found.get("PtpStatus")
    if r is None or r.value_type != "enum":
        return None
    if r.symbolic:
        return str(r.symbolic).lower() == "master"
    return None


def _get_int_value(found: Dict[str, NodeProbeResult], *keys: str) -> Optional[int]:
    for k in keys:
        r = found.get(k)
        if r is None:
            continue
        if r.value_type in {"int", "int64"}:
            try:
                return int(r.value)  # type: ignore[arg-type]
            except Exception:
                return None
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    # 尽量固定 UTF-8 输出，避免在重定向到文件时出现乱码。
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="探测相机是否支持时间同步（PTP/IEEE1588）")
    parser.add_argument(
        "--mvimport-dir",
        default=None,
        help=(
            "MVS 官方 Python 示例绑定目录（MvImport）。"
            "可选；也可用环境变量 MVS_MVIMPORT_DIR。"
        ),
    )
    parser.add_argument(
        "--dll-dir",
        default=None,
        help="包含 MvCameraControl.dll 的目录（可选）。也可用环境变量 MVS_DLL_DIR。",
    )
    parser.add_argument(
        "--enable-ptp",
        action="store_true",
        help="尝试开启相机 PTP（会写 GevIEEE1588/PtpEnable 等节点，失败会忽略并继续探测）。",
    )
    parser.add_argument(
        "--set-domain",
        type=int,
        default=None,
        help="尝试设置 PTP DomainNumber（best-effort，节点可能不存在/不可写）。",
    )
    parser.add_argument(
        "--set-priority1",
        type=int,
        default=None,
        help="尝试设置 PTP Priority1（数值越小优先级越高）。",
    )
    parser.add_argument(
        "--set-priority2",
        type=int,
        default=None,
        help="尝试设置 PTP Priority2（数值越小优先级越高）。",
    )
    parser.add_argument(
        "--force-slave-only",
        action="store_true",
        help="尝试把设备设置为 SlaveOnly（如果固件支持该节点）。",
    )
    parser.add_argument(
        "--force-master-only",
        action="store_true",
        help="尝试把设备设置为 MasterOnly（如果固件支持该节点）。",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=0.0,
        help="开启后轮询状态的时长（秒，0 表示不轮询）。",
    )
    parser.add_argument("--list", action="store_true", help="仅枚举并打印设备信息")
    parser.add_argument(
        "--serial",
        action="extend",
        nargs="+",
        default=[],
        help=(
            "按序列号选择相机（可一次性传多个，或重复多次）。"
            "不传则默认探测所有已枚举设备。"
        ),
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if bool(args.force_slave_only) and bool(args.force_master_only):
        print("参数错误：--force-slave-only 与 --force-master-only 不能同时使用。")
        return 2

    try:
        binding = load_mvs_binding(mvimport_dir=args.mvimport_dir, dll_dir=args.dll_dir)
    except MvsDllNotFoundError as exc:
        print(str(exc))
        return 2

    with MvsSdk(binding) as _sdk:
        st_dev_list, descs = enumerate_devices(binding)

        if args.list:
            for d in descs:
                ip = d.ip or "-"
                tl = _tlayer_to_str(binding, d.tlayer_type)
                print(f"[{d.index}] model={d.model} serial={d.serial} name={d.user_name} ip={ip} tlayer={tl}")
            return 0

        wanted = {s.strip() for s in args.serial if s.strip()}
        targets: List[DeviceDesc] = [d for d in descs if (not wanted or d.serial in wanted)]

        if not targets:
            if wanted:
                print(f"未找到指定序列号设备：{sorted(wanted)}")
            else:
                print("未枚举到任何设备")
            return 2

        all_found: List[Dict[str, NodeProbeResult]] = []

        for d in targets:
            ip = d.ip or "-"
            tl = _tlayer_to_str(binding, d.tlayer_type)
            print(f"\n=== serial={d.serial} model={d.model} ip={ip} tlayer={tl} ===")

            cam = None
            try:
                cam = _open_device_for_probe(binding=binding, st_dev_list=st_dev_list, dev_index=int(d.index))

                if bool(args.enable_ptp):
                    enabled = _try_enable_ptp(cam, binding)
                    if enabled:
                        print(f"已尝试开启 PTP：{enabled}")
                    else:
                        print("未能写入任何 PTP 开关节点（可能节点名不同/权限不足/机型不支持写入）。")

                if (
                    args.set_domain is not None
                    or args.set_priority1 is not None
                    or args.set_priority2 is not None
                    or bool(args.force_slave_only)
                    or bool(args.force_master_only)
                ):
                    wr = _try_apply_ptp_params(
                        cam=cam,
                        binding=binding,
                        domain=args.set_domain,
                        priority1=args.set_priority1,
                        priority2=args.set_priority2,
                        force_slave_only=bool(args.force_slave_only),
                        force_master_only=bool(args.force_master_only),
                    )
                    ok = [x for x in wr if x.ok]
                    if ok:
                        ok_items = ", ".join([f"{x.key}({x.detail})" for x in ok])
                        print(f"已尝试写入 PTP 参数：{ok_items}")
                    else:
                        print("未能成功写入任何 PTP 参数节点（可能节点不存在/不可写/权限不足）。")

                found: Dict[str, NodeProbeResult] = {}
                for key in PTP_NODE_CANDIDATES:
                    r = _probe_node(cam, binding, key)
                    if r is not None:
                        found[key] = r

                if not found:
                    print("未命中任何常见 PTP/IEEE1588 节点：大概率不支持（或节点名不同/被隐藏）。")
                    continue

                all_found.append(found)

                for key in sorted(found.keys()):
                    r = found[key]
                    if r.value_type == "enum" and r.symbolic:
                        print(f"- {r.key}: ({r.value_type}) {r.value} ({r.symbolic})")
                    else:
                        print(f"- {r.key}: ({r.value_type}) {r.value}")

                poll_s = float(args.poll_seconds)
                if poll_s > 0:
                    t0 = time.monotonic()
                    while (time.monotonic() - t0) < poll_s:
                        time.sleep(0.5)
                        snap: Dict[str, NodeProbeResult] = {}
                        for key in (
                            "GevIEEE1588",
                            "GevIEEE1588DomainNumber",
                            "GevIEEE1588Priority1",
                            "GevIEEE1588Priority2",
                            "GevIEEE1588Status",
                            "GevIEEE1588OffsetFromMaster",
                            "PtpEnable",
                            "PtpDomainNumber",
                            "PtpPriority1",
                            "PtpPriority2",
                            "PtpStatus",
                            "PtpOffsetFromMaster",
                        ):
                            r = _probe_node(cam, binding, key)
                            if r is not None:
                                snap[key] = r

                        if snap:
                            parts: List[str] = []
                            for k in sorted(snap.keys()):
                                r = snap[k]
                                if r.value_type == "enum" and r.symbolic:
                                    parts.append(f"{k}={r.value}({r.symbolic})")
                                else:
                                    parts.append(f"{k}={r.value}")
                            items = ", ".join(parts)
                            print(f"[poll] {items}")

                if _looks_like_time_sync_supported(found):
                    print("结论：检测到 PTP/IEEE1588 相关节点（机型/固件大概率支持时间同步）。")
                    print("下一步：需要进一步确认状态节点是否进入 Locked/Slave（或类似同步态）。")
                else:
                    print("结论：命中了一些疑似相关节点，但未命中明确的启用/状态节点；需要人工复核。")
            except Exception as exc:
                print(f"探测失败：{exc}")
            finally:
                if cam is not None:
                    _close_cam(cam)

        # 全局诊断：如果多台都处于 Master，通常说明 PTP 报文无法互通或 domain 不一致。
        masters = [x for x in ([_is_master_status(f) for f in all_found]) if x is True]
        domains = [
            d
            for d in (
                [_get_int_value(f, "GevIEEE1588DomainNumber", "PtpDomainNumber") for f in all_found]
            )
            if d is not None
        ]
        pri1 = [
            p
            for p in (
                [_get_int_value(f, "GevIEEE1588Priority1", "PtpPriority1") for f in all_found]
            )
            if p is not None
        ]

        if len(all_found) >= 2 and domains and len(set(domains)) > 1:
            print(
                "\n[diagnose] 检测到多台设备的 PTP DomainNumber 不一致。\n"
                "PTP 不同 Domain 之间不会互相选主/同步，表现可能是每台都显示 Master。\n"
                "建议：把所有相机 DomainNumber 统一（例如都设为 0），然后断电重启相机再复测。"
            )

        if len(all_found) >= 2 and len(masters) == len(all_found):
            print(
                "\n[diagnose] 多台设备均为 Master。常见原因：\n"
                "- 网络设备开启了 IGMP Snooping/组播抑制，导致 PTP 组播报文无法互通；\n"
                "- 设备处于不同的 PTP DomainNumber；\n"
                "- 设备不在同一二层交换域（看起来同网段，但二层被隔离/分段）。\n"
                "建议：把相机与主机全部改为同一台千兆交换机纯有线连接，再验证应出现 1 Master + N-1 Slave。"
            )

        if len(all_found) >= 2 and pri1 and len(set(pri1)) == 1:
            print(
                "\n[diagnose] 多台设备 Priority1 相同。\n"
                "这通常不会导致“完全无法选主”，但可能让主时钟在重启后不稳定。\n"
                "建议：指定一台作为优先主时钟（Priority1 更小），其余稍大。"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
