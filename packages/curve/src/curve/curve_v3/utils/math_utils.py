"""curve_v3 的轻量数学工具。

设计约束：
    - 仅依赖 NumPy（不引入额外第三方库）。
    - 以“足够用、易读”为优先，避免为了通用性过度设计。
"""

from __future__ import annotations

import numpy as np


def _sanitize_weighted_fit_inputs(
    t: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """清洗拟合输入，避免 NaN/Inf/负权重导致数值问题。

    说明：
        - 该工具函数只做最小化的防御性处理，避免线上/离线遇到极端数据时崩溃。
        - 权重按最大值归一化（乘以常数不改变加权最小二乘解），用于抑制溢出。
    """

    t = np.asarray(t, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=float).reshape(-1)

    n = int(min(t.size, y.size, w.size))
    if n <= 0:
        return (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
        )

    t = t[:n]
    y = y[:n]
    w = w[:n]

    w = np.maximum(w, 0.0)
    mask = np.isfinite(t) & np.isfinite(y) & np.isfinite(w)
    if not bool(np.all(mask)):
        t = t[mask]
        y = y[mask]
        w = w[mask]

    if w.size > 0:
        w_max = float(np.max(w))
        if np.isfinite(w_max) and w_max > 0.0:
            w = w / w_max

    return t, y, w


def _weighted_line_fit_closed_form(
    t: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
) -> tuple[float, float]:
    """加权最小二乘拟合 y = b*t + c 的闭式解。

    Returns:
        (b, c)
    """

    t, y, w = _sanitize_weighted_fit_inputs(t, y, w)
    if t.size < 2:
        c = float(np.mean(y)) if y.size > 0 else 0.0
        return 0.0, c

    sw = float(np.sum(w))
    if not np.isfinite(sw) or sw <= 0.0:
        # 权重异常时，退化为普通最小二乘。
        w = np.ones_like(t, dtype=float)
        sw = float(np.sum(w))

    st = float(np.sum(w * t))
    stt = float(np.sum(w * t * t))
    sy = float(np.sum(w * y))
    sty = float(np.sum(w * t * y))

    denom = float(sw * stt - st * st)
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        # 时间采样几乎相同，斜率不可辨识：取常数拟合。
        c = float(sy / sw) if sw > 0 else float(np.mean(y))
        return 0.0, c

    b = float((sw * sty - st * sy) / denom)
    c = float((stt * sy - st * sty) / denom)
    return b, c


def polyval(coeffs: np.ndarray, t: float) -> float:
    """在时刻 t 处计算多项式值。"""

    return float(np.polyval(np.asarray(coeffs, dtype=float), float(t)))


def polyder_val(coeffs: np.ndarray, t: float) -> float:
    """在时刻 t 处计算多项式的一阶导数值。"""

    return float(np.polyval(np.polyder(np.asarray(coeffs, dtype=float)), float(t)))


def constrained_quadratic_fit(
    t: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    fixed_a: float,
) -> tuple[np.ndarray, float]:
    """在固定二次项系数 a 的情况下，用加权最小二乘拟合二次多项式。

    Args:
        t: 时间采样点，shape=(N,)。
        y: 观测值，shape=(N,)。
        w: 每个样本的权重（非负），shape=(N,)。
        fixed_a: 固定的二次项系数 a。

    Returns:
        (coeffs, mse)：coeffs 形如 [a, b, c]。
    """

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    y_adj = y - float(fixed_a) * t**2

    # 优先使用闭式解，失败时回退到 np.polyfit（与旧实现一致）。
    try:
        b, c = _weighted_line_fit_closed_form(t, y_adj, w)
        t2, y2, _w2 = _sanitize_weighted_fit_inputs(t, y_adj, w)
        y_fit = b * t2 + c
        mse = float(np.mean((y_fit - y2) ** 2)) if t2.size > 0 else 0.0
    except Exception:
        t2, y2, w2 = _sanitize_weighted_fit_inputs(t, y_adj, w)
        bc = np.polyfit(t2, y2, deg=1, w=w2)
        b, c = float(bc[0]), float(bc[1])
        y_fit = np.polyval(np.asarray([b, c], dtype=float), t2)
        mse = float(np.mean((y_fit - y2) ** 2)) if t2.size > 0 else 0.0

    return np.array([float(fixed_a), float(b), float(c)], dtype=float), float(mse)


def weighted_linear_fit(t: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
    """用加权最小二乘拟合一阶多项式 y(t)=k t + b。"""

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    try:
        t2, y2, w2 = _sanitize_weighted_fit_inputs(t, y, w)
        coeffs = np.polyfit(t2, y2, deg=1, w=w2)
        y_fit = np.polyval(coeffs, t2)
        mse = float(np.mean((y_fit - y2) ** 2)) if t2.size > 0 else 0.0
        return np.asarray(coeffs, dtype=float), float(mse)
    except Exception:
        b, c = _weighted_line_fit_closed_form(t, y, w)
        t2, y2, _w2 = _sanitize_weighted_fit_inputs(t, y, w)
        y_fit = b * t2 + c
        mse = float(np.mean((y_fit - y2) ** 2)) if t2.size > 0 else 0.0
        return np.asarray([float(b), float(c)], dtype=float), float(mse)


def real_roots_of_quadratic(coeffs: np.ndarray) -> list[float]:
    """返回二次多项式的实根（升序），忽略虚部很小的复根。"""

    c = np.asarray(coeffs, dtype=float).reshape(-1)
    if c.size != 3:
        raise ValueError(f"Expected 3 coefficients [a,b,c], got size={int(c.size)}")
    if not bool(np.all(np.isfinite(c))):
        return []

    a, b, cc = float(c[0]), float(c[1]), float(c[2])

    # 退化：线性/常数。
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return []
        return [float(-cc / b)]

    d = b * b - 4.0 * a * cc
    if not np.isfinite(d) or d < 0.0:
        return []

    sqrt_d = float(np.sqrt(d))
    r1 = (-b - sqrt_d) / (2.0 * a)
    r2 = (-b + sqrt_d) / (2.0 * a)

    roots = [float(r1), float(r2)]
    roots.sort()
    return roots


def weighted_quantile_1d(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """计算一维样本的加权分位数。

    Args:
        values: 样本值，shape=(N,)。
        weights: 样本权重（非负），shape=(N,)。
        q: 分位数水平，取值范围 [0, 1]。

    Returns:
        加权分位数数值。

    Notes:
        - 当权重和为 0 时，会回退到普通（非加权）分位数。
        - 使用加权 CDF 上的分段线性插值。
    """

    v = np.asarray(values, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if v.size == 0:
        return float("nan")

    q = float(min(max(float(q), 0.0), 1.0))

    w = np.maximum(w, 0.0)
    sw = float(np.sum(w))
    if sw <= 0.0:
        # 权重和为 0 时，回退到普通（非加权）分位数。
        return float(np.quantile(v, q))

    idx = np.argsort(v)
    v_sorted = v[idx]
    w_sorted = w[idx]

    cdf = np.cumsum(w_sorted, dtype=float) / sw

    k = int(np.searchsorted(cdf, q, side="left"))
    if k <= 0:
        return float(v_sorted[0])
    if k >= v_sorted.size:
        return float(v_sorted[-1])

    c0 = float(cdf[k - 1])
    c1 = float(cdf[k])
    if abs(c1 - c0) < 1e-12:
        return float(v_sorted[k])

    alpha = (q - c0) / (c1 - c0)
    return float(v_sorted[k - 1] + alpha * (v_sorted[k] - v_sorted[k - 1]))


def weighted_quantiles_1d(values: np.ndarray, weights: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """对多个分位点 qs 的便捷封装（内部逐个计算）。"""

    q_arr = np.asarray(qs, dtype=float).reshape(-1)
    out = np.zeros((q_arr.size,), dtype=float)
    for i, q in enumerate(q_arr.tolist()):
        out[i] = weighted_quantile_1d(values, weights, float(q))
    return out
