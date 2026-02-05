"""时间戳与稳健聚合的小工具。

说明：
- 这些函数是纯计算（无 I/O、无 SDK 依赖），供在线/离线 source 复用。
- 主要用途：
  - 把同一组内多相机的 host_timestamp 做稳健聚合（中位数）。
  - 把 host_timestamp 归一化到秒/毫秒 epoch。
  - 评估组内各相机时间戳离散程度（spread、相对中位数偏差）。
"""

from __future__ import annotations

from typing import Any


def median_int(values: list[int]) -> int | None:
    """返回整型列表的中位数（不做插值）。"""

    if not values:
        return None
    xs = sorted(int(v) for v in values)
    return int(xs[len(xs) // 2])


def median_float(values: list[float]) -> float | None:
    """返回浮点列表的中位数（不做插值）。"""

    if not values:
        return None
    xs = sorted(float(v) for v in values)
    return float(xs[len(xs) // 2])


def host_timestamp_to_seconds(ts: Any) -> float | None:
    """把 host_timestamp（可能是 ms/ns/s）转换成 epoch 秒。

    说明：
    - 不同设备/配置下 host_timestamp 可能有不同单位。
    - 这里用数量级启发式做归一化，供下游曲线拟合等逻辑使用。
    """

    try:
        v = int(ts)
    except Exception:
        return None

    if v <= 0:
        return None

    # ns epoch
    if v >= 10**16:
        return float(v) / 1e9
    # ms epoch
    if v >= 10**11:
        return float(v) / 1e3
    # s epoch
    if v >= 10**8:
        return float(v)

    return None


def host_timestamp_to_ms_epoch(ts: Any) -> float | None:
    """把 host_timestamp（可能是 ms/ns/s）转换成 epoch 毫秒。"""

    try:
        v = int(ts)
    except Exception:
        return None

    if v <= 0:
        return None

    # ns epoch
    if v >= 10**16:
        return float(v) / 1e6
    # ms epoch
    if v >= 10**11:
        return float(v)
    # s epoch
    if v >= 10**8:
        return float(v) * 1000.0

    return None


def spread_ms(values_ms: list[float]) -> float | None:
    """返回一组时间戳（毫秒）的跨度：max - min。"""

    if len(values_ms) < 2:
        return None
    lo = min(float(x) for x in values_ms)
    hi = max(float(x) for x in values_ms)
    return float(hi - lo)


def delta_to_median_by_camera(values_by_camera: dict[str, float]) -> dict[str, float] | None:
    """把每相机时间戳转成“相对组内中位数”的偏差（毫秒）。"""

    if len(values_by_camera) < 2:
        return None

    med = median_float(list(values_by_camera.values()))
    if med is None:
        return None

    return {str(k): float(v) - float(med) for k, v in values_by_camera.items()}
