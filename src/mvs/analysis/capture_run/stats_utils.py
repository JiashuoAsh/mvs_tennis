# -*- coding: utf-8 -*-

"""采集运行分析：统计小工具（内部模块）。

说明：
- 该模块只放“纯计算”的小函数，避免 compute.py 堆满零碎细节。
- 注释与错误信息使用中文，方便与报告口径对齐。
"""

from __future__ import annotations

import statistics
from typing import Optional, Sequence


def safe_median(values: Sequence[float]) -> Optional[float]:
    """返回中位数；空序列返回 None。"""

    if not values:
        return None
    return float(statistics.median(values))
