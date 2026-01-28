"""时间戳解析（高内聚：只做文件名 -> 时间戳）。"""

from __future__ import annotations

import re
from datetime import datetime, timezone

_TIMESTAMP_RE = re.compile(r"Image_(\d{17})\.")


def parse_timestamp_from_name(file_name: str) -> tuple[str, int]:
    """从文件名解析 17 位时间戳，并返回 (ts_str, ts_epoch_ms)。

    Args:
        file_name: 例如 "Image_20260120191532433.bmp"。

    Returns:
        ts_str: 17 位字符串 YYYYMMDDHHMMSSmmm。
        ts_epoch_ms: 以毫秒为单位的 epoch（UTC）时间戳，便于差值比较与对齐。

    Raises:
        ValueError: 文件名不符合约定。
    """

    m = _TIMESTAMP_RE.search(file_name)
    if not m:
        raise ValueError(f"无法从文件名解析时间戳: {file_name}")

    ts_str = m.group(1)
    dt = datetime.strptime(ts_str[:-3], "%Y%m%d%H%M%S").replace(
        microsecond=int(ts_str[-3:]) * 1000,
        tzinfo=timezone.utc,
    )
    ts_epoch_ms = int(dt.timestamp() * 1000)
    return ts_str, ts_epoch_ms
