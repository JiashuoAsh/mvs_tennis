# -*- coding: utf-8 -*-

"""保存图像（优先用 SDK 存 BMP，失败可由上层降级）。"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

from mvs.sdk.binding import MvsBinding

from .grab import FramePacket


def save_frame_as_bmp(
    *,
    binding: MvsBinding,
    cam: Any,
    out_path: Path,
    frame: FramePacket,
    bayer_method: int = 2,
) -> None:
    """使用 SDK 将一帧保存为 BMP。

    Notes:
        - 该函数依赖 MVS SDK 的 `MV_CC_SaveImageToFileEx`，通常能自动完成 Bayer 转换。
        - 输出路径编码：Windows 使用 MBCS（与官方示例一致），其它平台使用 UTF-8。

    Args:
        binding: 已加载的 MVS 绑定。
        cam: SDK 相机句柄（MvCamera 实例）。
        out_path: 输出 BMP 路径。
        frame: 待保存的帧。
        bayer_method: Bayer 插值算法。常见约定：0=快速, 1=均衡, 2=最优, 3=最优+。

    Raises:
        RuntimeError: SDK 保存失败。
    """

    buf = ctypes.create_string_buffer(frame.data)
    save = binding.params.MV_SAVE_IMAGE_TO_FILE_PARAM_EX()
    save.nWidth = int(frame.width)
    save.nHeight = int(frame.height)
    save.enPixelType = int(frame.pixel_type)
    save.pData = ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte))
    save.nDataLen = int(frame.frame_len)
    save.enImageType = int(binding.params.MV_Image_Bmp)

    if os.name == "nt":
        path_bytes = str(out_path).encode("mbcs", errors="replace")
    else:
        path_bytes = str(out_path).encode("utf-8", errors="replace")

    path_buf = ctypes.create_string_buffer(path_bytes)
    save.pcImagePath = ctypes.cast(path_buf, ctypes.POINTER(ctypes.c_char))

    save.nQuality = 0
    save.iMethodValue = int(bayer_method)

    ret = cam.MV_CC_SaveImageToFileEx(save)
    if int(ret) != binding.MV_OK:
        raise RuntimeError(f"MV_CC_SaveImageToFileEx failed, ret=0x{int(ret):08X}")
