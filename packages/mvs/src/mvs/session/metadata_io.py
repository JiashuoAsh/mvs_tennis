# -*- coding: utf-8 -*-

"""metadata.jsonl 的稳健读取。

历史背景：
- 采集写盘通常是“每行一个 JSON 对象”（严格意义上的 JSONL）。
- 但在一些人工处理/拷贝/格式化过程中，metadata 可能被 pretty-print 成“多行一个对象”。

为避免下游（离线 pipeline、分析工具、重排工具）对文件格式过于敏感，
本模块提供一个统一的“流式 JSON 对象迭代器”，同时兼容：
- 逐行 JSONL
- 多行/缩进 JSON（多个对象串联，仅以空白分隔）

限制：
- 不支持“顶层是 JSON 数组”的格式（这不是本仓库约定）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def iter_metadata_records(path: Path) -> Iterator[dict[str, Any]]:
    """从 metadata.jsonl 迭代出每个 JSON 对象记录。"""

    p = Path(path).resolve()
    decoder = json.JSONDecoder()

    buf = ""
    with p.open("r", encoding="utf-8") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            buf += chunk

            while True:
                s = buf.lstrip()
                if not s:
                    buf = ""
                    break

                try:
                    obj, idx = decoder.raw_decode(s)
                except json.JSONDecodeError:
                    # 数据不足（对象尚未完整），继续读下一块。
                    buf = s
                    break

                buf = s[idx:]

                if isinstance(obj, dict):
                    yield obj

    # 文件读完后，若仍有残留且能解析出完整对象，则尽力解析。
    s = buf.lstrip()
    if not s:
        return
    try:
        obj, _idx = decoder.raw_decode(s)
    except json.JSONDecodeError:
        return
    if isinstance(obj, dict):
        yield obj
