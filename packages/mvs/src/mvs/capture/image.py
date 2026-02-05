# -*- coding: utf-8 -*-

"""把 MVS FramePacket 转成可用于 OpenCV 的图像数组。

说明：
- 在线推理/调试通常需要 BGR8 格式（cv2 默认）。
- MVS 的取流返回的是原始像素数据 + 像素格式枚举；这里优先走 SDK 的像素格式转换接口，
  这样可以兼容 Mono/Bayer/YUV 等多种输入（由 SDK 负责细节）。

注意：
- 该模块只做“帧数据 -> numpy”，不负责检测、不负责三角化。
"""

from __future__ import annotations

import ctypes
from typing import Any

import cv2
import numpy as np

from mvs.sdk.binding import MvsBinding

from .grab import FramePacket


def frame_to_bgr(
    *,
    binding: MvsBinding,
    cam: Any,
    frame: FramePacket,
) -> np.ndarray:
    """把一帧转换为 BGR8 numpy.ndarray。

    Args:
        binding: 已加载的 MVS 绑定。
        cam: SDK 相机句柄（MvCamera 实例）。
        frame: 抓取到的帧。

    Returns:
        (H, W, 3) 的 uint8 BGR 图像。

    Raises:
        RuntimeError: 转换失败或帧数据长度不合法。
    """

    w = int(frame.width)
    h = int(frame.height)
    if w <= 0 or h <= 0:
        raise RuntimeError(f"invalid frame size: {w}x{h}")

    # 快路径：已经是 BGR8/RGB8。
    if int(frame.pixel_type) == int(getattr(binding.params, "PixelType_Gvsp_BGR8_Packed", -999999)):
        expected = w * h * 3
        if int(frame.frame_len) < expected:
            raise RuntimeError(f"frame_len too small for BGR8: {frame.frame_len} < {expected}")
        arr = np.frombuffer(frame.data[:expected], dtype=np.uint8).reshape(h, w, 3)
        return arr.copy()

    if int(frame.pixel_type) == int(getattr(binding.params, "PixelType_Gvsp_RGB8_Packed", -999999)):
        expected = w * h * 3
        if int(frame.frame_len) < expected:
            raise RuntimeError(f"frame_len too small for RGB8: {frame.frame_len} < {expected}")
        rgb = np.frombuffer(frame.data[:expected], dtype=np.uint8).reshape(h, w, 3)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr

    # 通用路径：使用 SDK 像素格式转换到 BGR8。
    src_buf = ctypes.create_string_buffer(frame.data)
    dst_size = int(w * h * 3)
    dst_buf = ctypes.create_string_buffer(dst_size)

    cvt = binding.params.MV_CC_PIXEL_CONVERT_PARAM_EX()
    cvt.nWidth = int(w)
    cvt.nHeight = int(h)
    cvt.enSrcPixelType = int(frame.pixel_type)
    cvt.pSrcData = ctypes.cast(src_buf, ctypes.POINTER(ctypes.c_ubyte))
    cvt.nSrcDataLen = int(frame.frame_len)
    cvt.enDstPixelType = int(binding.params.PixelType_Gvsp_BGR8_Packed)
    cvt.pDstBuffer = ctypes.cast(dst_buf, ctypes.POINTER(ctypes.c_ubyte))
    cvt.nDstBufferSize = int(dst_size)
    cvt.nDstLen = 0

    ret = cam.MV_CC_ConvertPixelTypeEx(cvt)
    if int(ret) != int(binding.MV_OK):
        # 兜底：若转换失败但输入是常见 8bit Mono/Bayer，可以尝试用 OpenCV 自己转。
        try:
            return _fallback_bgr_from_raw_8bit(binding=binding, frame=frame)
        except Exception as exc:
            raise RuntimeError(
                f"MV_CC_ConvertPixelTypeEx failed, ret=0x{int(ret):08X}; fallback failed: {exc}"
            ) from exc

    n = int(cvt.nDstLen)
    if n <= 0 or n > dst_size:
        raise RuntimeError(f"invalid converted length: nDstLen={n}, dst_size={dst_size}")

    out = np.frombuffer(dst_buf.raw[:n], dtype=np.uint8)
    if out.size < (w * h * 3):
        raise RuntimeError(f"converted buffer too small: {out.size} < {w*h*3}")

    bgr = out[: w * h * 3].reshape(h, w, 3).copy()
    return bgr


def _fallback_bgr_from_raw_8bit(*, binding: MvsBinding, frame: FramePacket) -> np.ndarray:
    """不依赖 SDK 的兜底解码（仅覆盖最常见 8-bit Mono/Bayer）。"""

    w = int(frame.width)
    h = int(frame.height)
    pixel_type = int(frame.pixel_type)

    mono8 = int(getattr(binding.params, "PixelType_Gvsp_Mono8", -1))
    bayer_rg8 = int(getattr(binding.params, "PixelType_Gvsp_BayerRG8", -1))
    bayer_gr8 = int(getattr(binding.params, "PixelType_Gvsp_BayerGR8", -1))
    bayer_gb8 = int(getattr(binding.params, "PixelType_Gvsp_BayerGB8", -1))
    bayer_bg8 = int(getattr(binding.params, "PixelType_Gvsp_BayerBG8", -1))

    expected = w * h
    if int(frame.frame_len) < expected:
        raise RuntimeError(f"frame_len too small: {frame.frame_len} < {expected}")

    raw = np.frombuffer(frame.data[:expected], dtype=np.uint8).reshape(h, w)

    if pixel_type == mono8:
        return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    if pixel_type == bayer_rg8:
        return cv2.cvtColor(raw, cv2.COLOR_BayerRG2BGR)
    if pixel_type == bayer_gr8:
        return cv2.cvtColor(raw, cv2.COLOR_BayerGR2BGR)
    if pixel_type == bayer_gb8:
        return cv2.cvtColor(raw, cv2.COLOR_BayerGB2BGR)
    if pixel_type == bayer_bg8:
        return cv2.cvtColor(raw, cv2.COLOR_BayerBG2BGR)

    raise RuntimeError(f"unsupported raw 8-bit pixel_type={pixel_type}")
