# -*- coding: utf-8 -*-

"""文本/字符串相关的小工具。

这个模块的目标很简单：把 SDK 返回的各种“定长 C 字符数组/字节数组”稳定地解码成 Python 字符串。

说明：
- MVS 的示例绑定里，很多字段是 `c_char * N` 或 `uint8 * N`。
- 不同相机/固件可能会返回 ASCII/UTF-8/GBK 等编码；这里采用“先 UTF-8，失败再 GBK”的策略，
  并且始终在第一个 `\x00` 处截断（C 字符串语义）。
"""

from __future__ import annotations

from typing import Any


def decode_c_string(buf: Any) -> str:
    """把 C 风格的定长字符串缓冲区解码为 Python 字符串。

    Args:
        buf: 可以是 ctypes 的 `c_char * N`、`c_ubyte * N`，或任意可被 `bytes(buf)` 转换的对象。

    Returns:
        解码后的字符串；失败时返回空字符串。
    """

    try:
        raw = bytes(buf)
    except Exception:
        return ""

    raw = raw.split(b"\x00", 1)[0]

    try:
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        try:
            return raw.decode("gbk", errors="ignore")
        except Exception:
            return ""
