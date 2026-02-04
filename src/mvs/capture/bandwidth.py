# -*- coding: utf-8 -*-

"""带宽估算工具。

目的：在开始采集时，根据相机当前配置（分辨率、像素格式、PayloadSize、帧率等）
估算单路/多路所需的传输带宽，辅助排查“带宽不足导致丢包/花屏/掉帧”的问题。

说明：
- 对 GigE/USB3 等链路来说，真实占用带宽会包含协议开销、包头、重传等因素。
  因此本模块默认给出一个“含开销的保守估算”（raw * overhead_factor）。
- 外触发模式下，SDK 未必能直接给出帧率；此时可以提供 fps_hint（例如触发器频率）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from mvs.core.text import decode_c_string
from mvs.sdk.binding import MvsBinding


@dataclass(frozen=True, slots=True)
class BandwidthEstimate:
    """单台相机的带宽估算结果。"""

    serial: str
    width: int
    height: int
    payload_bytes: int
    pixel_format: str
    bits_per_pixel: Optional[float]
    fps: Optional[float]
    raw_mbps: Optional[float]
    mbps_with_overhead: Optional[float]


def _try_get_int_node(cam: Any, binding: MvsBinding, key: str) -> Optional[int]:
    """尽力读取 int 节点（返回当前值）。"""

    try:
        st = binding.params.MVCC_INTVALUE()
        ret = cam.MV_CC_GetIntValue(str(key), st)
        if int(ret) != int(binding.MV_OK):
            return None
        return int(getattr(st, "nCurValue"))
    except Exception:
        return None


def _try_get_float_node(cam: Any, binding: MvsBinding, key: str) -> Optional[float]:
    """尽力读取 float 节点（返回当前值）。"""

    try:
        st = binding.params.MVCC_FLOATVALUE()
        ret = cam.MV_CC_GetFloatValue(str(key), st)
        if int(ret) != int(binding.MV_OK):
            return None
        return float(getattr(st, "fCurValue"))
    except Exception:
        return None


def _try_get_enum_symbolic(cam: Any, binding: MvsBinding, key: str) -> str:
    """尽力读取 enum 节点的 symbolic（例如 PixelFormat -> Mono8）。"""

    try:
        st = binding.params.MVCC_ENUMENTRY()
        ret = cam.MV_CC_GetEnumEntrySymbolic(str(key), st)
        if int(ret) != int(binding.MV_OK):
            return ""
        sym = decode_c_string(getattr(st, "chSymbolic", b""))
        return str(sym).strip()
    except Exception:
        return ""


def _infer_bpp_from_symbolic(symbolic: str) -> Optional[int]:
    """从 PixelFormat 的 symbolic 名称推断位深/每像素比特数。

    只覆盖常见格式，推断失败返回 None。
    """

    s = str(symbolic or "").strip()
    if not s:
        return None

    # 常见 packed RGB/BGR
    if "RGB8" in s or "BGR8" in s:
        return 24
    if "RGBa8" in s or "RGBA8" in s or "BGRA8" in s:
        return 32

    # 常见 YUV/YCbCr
    if "422" in s and ("YUV" in s or "YCbCr" in s or "YCBCR" in s):
        return 16

    # Mono / Bayer / 其它：取名字里的数字（8/10/12/14/16...）
    m = re.search(r"(\d{1,2})", s)
    if not m:
        return None

    bits = int(m.group(1))
    if bits <= 0:
        return None

    # 例如 Mono10Packed 这种，数字仍表示每像素有效位深，按 bits 返回即可。
    return bits


def _infer_bpp_from_payload(*, payload_bytes: int, width: int, height: int) -> Optional[float]:
    if payload_bytes <= 0 or width <= 0 or height <= 0:
        return None
    pixels = int(width) * int(height)
    if pixels <= 0:
        return None
    return float(payload_bytes) * 8.0 / float(pixels)


def _first_non_none(values: Iterable[Optional[float]]) -> Optional[float]:
    for v in values:
        if v is not None:
            return v
    return None


def estimate_camera_bandwidth(
    *,
    binding: MvsBinding,
    cam: Any,
    serial: str,
    fps_hint: Optional[float] = None,
    overhead_factor: float = 1.10,
) -> BandwidthEstimate:
    """估算单台相机所需带宽。

    Args:
        binding: 已加载的 MVS 绑定。
        cam: SDK 相机句柄（MvCamera 实例）。
        serial: 相机序列号（用于展示）。
        fps_hint: 外部给定的帧率提示（例如软触发频率/外触发频率）。
        overhead_factor: 协议开销系数（默认 1.10，给一个偏保守的估算）。

    Returns:
        BandwidthEstimate。
    """

    width = _try_get_int_node(cam, binding, "Width") or 0
    height = _try_get_int_node(cam, binding, "Height") or 0

    payload = (
        _try_get_int_node(cam, binding, "PayloadSize")
        or _try_get_int_node(cam, binding, "GevPayloadSize")
        or 0
    )

    pixel_format = _try_get_enum_symbolic(cam, binding, "PixelFormat")
    if not pixel_format:
        # 读不到 symbolic 时退化为数值。
        enum_val = None
        try:
            st_enum = binding.params.MVCC_ENUMVALUE()
            ret = cam.MV_CC_GetEnumValue("PixelFormat", st_enum)
            if int(ret) == int(binding.MV_OK):
                enum_val = int(getattr(st_enum, "nCurValue"))
        except Exception:
            enum_val = None
        pixel_format = f"0x{enum_val:08X}" if enum_val is not None else "unknown"

    bpp_int = _infer_bpp_from_symbolic(pixel_format)
    bpp = float(bpp_int) if bpp_int is not None else _infer_bpp_from_payload(payload_bytes=payload, width=width, height=height)

    fps_from_nodes = _first_non_none(
        [
            _try_get_float_node(cam, binding, "ResultingFrameRate"),
            _try_get_float_node(cam, binding, "ResultingFrameRateAbs"),
            _try_get_float_node(cam, binding, "AcquisitionFrameRate"),
            _try_get_float_node(cam, binding, "AcquisitionFrameRateAbs"),
            _try_get_float_node(cam, binding, "FrameRate"),
            _try_get_float_node(cam, binding, "FrameRateAbs"),
        ]
    )

    fps: Optional[float] = None
    if fps_hint is not None and float(fps_hint) > 0:
        fps = float(fps_hint)
    elif fps_from_nodes is not None and float(fps_from_nodes) > 0:
        fps = float(fps_from_nodes)

    raw_mbps: Optional[float] = None
    mbps_with_overhead: Optional[float] = None
    if fps is not None and payload > 0:
        raw_mbps = float(payload) * float(fps) * 8.0 / 1_000_000.0
        mbps_with_overhead = float(raw_mbps) * float(overhead_factor)

    return BandwidthEstimate(
        serial=str(serial),
        width=int(width),
        height=int(height),
        payload_bytes=int(payload),
        pixel_format=str(pixel_format),
        bits_per_pixel=bpp,
        fps=fps,
        raw_mbps=raw_mbps,
        mbps_with_overhead=mbps_with_overhead,
    )


def format_bandwidth_report(
    estimates: list[BandwidthEstimate],
    *,
    overhead_factor: float,
    expected_link_mbps: Optional[float] = None,
) -> str:
    """把估算结果格式化为易读文本（适合直接 print）。

    Args:
        estimates: 多台相机的带宽估算结果。
        overhead_factor: 协议开销系数（用于展示说明；数值通常与估算时使用的一致）。
        expected_link_mbps: 可选的链路理论带宽（例如 1000 表示 1GbE）。

    Returns:
        可直接打印/写入日志的多行字符串。
    """

    lines: list[str] = []
    lines.append("带宽估算（越接近/超过链路上限越容易丢包）：")
    lines.append(f"- 协议开销系数（估算）：{overhead_factor:.2f}x")
    if expected_link_mbps is not None and expected_link_mbps > 0:
        lines.append(f"- 参考链路带宽：{expected_link_mbps:.0f} Mbps（理论值，不含有效吞吐折损）")

    total_raw = 0.0
    total_over = 0.0
    has_any = False
    unknown_fps_serials: list[str] = []

    for e in estimates:
        px = int(e.width) * int(e.height) if (e.width > 0 and e.height > 0) else 0
        bpp_str = f"{e.bits_per_pixel:.2f}" if e.bits_per_pixel is not None else "-"
        fps_str = f"{e.fps:.3f}" if e.fps is not None else "-"
        payload_kib = e.payload_bytes / 1024.0 if e.payload_bytes > 0 else 0.0

        lines.append(
            f"- {e.serial}: {e.width}x{e.height} ({px} px), PixelFormat={e.pixel_format}, Payload={payload_kib:.1f} KiB, bpp≈{bpp_str}, fps={fps_str}"
        )

        if e.raw_mbps is None or e.mbps_with_overhead is None:
            unknown_fps_serials.append(e.serial)
            continue

        has_any = True
        lines.append(
            f"  - 需要带宽：raw={e.raw_mbps:.1f} Mbps，含开销≈{e.mbps_with_overhead:.1f} Mbps"
        )
        total_raw += float(e.raw_mbps)
        total_over += float(e.mbps_with_overhead)

    if has_any:
        lines.append(f"- 合计：raw={total_raw:.1f} Mbps，含开销≈{total_over:.1f} Mbps")

        if expected_link_mbps is not None and expected_link_mbps > 0:
            util = total_over / float(expected_link_mbps)
            lines.append(f"- 含开销估算占用率：{util * 100.0:.1f}%")

    if unknown_fps_serials:
        lines.append(
            "提示：以下相机未能自动读到帧率（外触发模式很常见），因此无法给出 Mbps 数值："
            f"{unknown_fps_serials}。你可以提供 expected_fps（触发器频率）来补全计算。"
        )

    # 给一个常见参考：1GbE 的有效吞吐通常低于 1000Mbps。
    lines.append("参考：1GbE 理论 1000Mbps，但可用有效吞吐通常显著低于该值（还要看 MTU、网卡/驱动、CPU、重传等）。")

    return "\n".join(lines)
