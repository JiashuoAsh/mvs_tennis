"""低 SNR（低信噪比）策略：权重/噪声建模与退化判别。

本模块只做两件事：
    1) 根据 conf（置信度）把每个点的观测噪声尺度 σ 与权重 w 估出来。
    2) 根据窗口内的可辨识性指标（Δu 与 \bar{σ}）给出每轴 mode。

注意：
    - 这里的 conf 是“点级”信息（通常来自检测置信度/跟踪可信度）。
    - 轴向噪声这里采用对角近似（每轴独立），这与 docs/curve_low_snr_quick.md 的
      工程落地方向一致。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import AxisDecision, AxisMode, WindowDecisions


@dataclass(frozen=True)
class LowSnrPolicyParams:
    """低 SNR 判别的参数集合。

    说明：
        这组参数来自 docs/curve_low_snr_quick.md：
        - Δu < 4*σ 认为加速度不可辨识
        - Δu < 2*σ 或 N<3 认为速度也不可信
        - 更极端可用 Δu < 1*σ 直接忽略该轴

    为避免过度设计，这里只保留最少参数；上层可直接从 CurveV3Config 传入。
    """

    delta_k_freeze_a: float = 4.0
    delta_k_strong_v: float = 2.0
    delta_k_ignore: float = 1.0
    min_points_for_v: int = 3


def _conf_to_value(conf: float | None, *, c_min: float) -> float:
    """把可选 conf 归一成一个稳定的标量。"""

    if conf is None:
        return 1.0
    c = float(conf)
    if not np.isfinite(c):
        return 1.0
    return float(max(c, float(c_min)))


def sigma_from_conf(
    confs: list[float | None] | np.ndarray,
    *,
    sigma0: float,
    c_min: float,
) -> np.ndarray:
    """由 conf 构造观测噪声尺度 σ（米）。

    公式（最简版本）：
        σ_i = σ0 / sqrt(max(conf_i, c_min))

    Args:
        confs: 每个点的 conf（可为 None）。
        sigma0: 基础噪声尺度（米），相当于 conf=1 时的 σ。
        c_min: conf 的下限，避免 1/sqrt(conf) 发散。

    Returns:
        shape=(N,) 的 σ 数组。
    """

    sigma0 = float(max(float(sigma0), 1e-9))
    c_min = float(max(float(c_min), 1e-9))

    if isinstance(confs, np.ndarray):
        conf_list: list[float | None] = [float(c) if np.isfinite(c) else None for c in confs.reshape(-1).tolist()]
    else:
        conf_list = list(confs)

    cs = np.array([_conf_to_value(c, c_min=c_min) for c in conf_list], dtype=float)
    return sigma0 / np.sqrt(cs)


def weights_from_conf(
    confs: list[float | None] | np.ndarray,
    *,
    sigma0: float,
    c_min: float,
) -> np.ndarray:
    """由 conf 构造 WLS 权重 w=1/σ^2。"""

    sig = sigma_from_conf(confs, sigma0=sigma0, c_min=c_min)
    return 1.0 / np.maximum(sig * sig, 1e-12)


def decide_axis_mode(
    values: np.ndarray,
    sigmas: np.ndarray,
    *,
    params: LowSnrPolicyParams,
    disallow_ignore: bool,
) -> AxisDecision:
    """对单轴做退化判别并返回 mode。"""

    v = np.asarray(values, dtype=float).reshape(-1)
    s = np.asarray(sigmas, dtype=float).reshape(-1)
    n = int(v.size)

    if n <= 0:
        # 无数据：只能忽略（上游应当兜底用先验传播）。
        mode: AxisMode = "IGNORE_AXIS"
        if disallow_ignore:
            mode = "STRONG_PRIOR_V"
        return AxisDecision(mode=mode, delta=0.0, sigma_mean=float("inf"), n=0)

    delta = float(np.max(v) - np.min(v))
    sigma_mean = float(np.mean(s))

    # 按 docs/curve_low_snr_quick.md 的三段判别。
    if delta < float(params.delta_k_ignore) * sigma_mean:
        mode = "IGNORE_AXIS"
    elif delta < float(params.delta_k_strong_v) * sigma_mean or n < int(params.min_points_for_v):
        mode = "STRONG_PRIOR_V"
    elif delta < float(params.delta_k_freeze_a) * sigma_mean:
        mode = "FREEZE_A"
    else:
        mode = "FULL"

    if disallow_ignore and mode == "IGNORE_AXIS":
        mode = "STRONG_PRIOR_V"

    return AxisDecision(mode=mode, delta=delta, sigma_mean=sigma_mean, n=n)


def analyze_window(
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    confs: list[float | None] | np.ndarray,
    sigma_x0: float,
    sigma_y0: float,
    sigma_z0: float,
    c_min: float,
    params: LowSnrPolicyParams,
    disallow_ignore_y: bool = True,
) -> WindowDecisions:
    """对三轴同步做退化判别。

    说明：
        - 判别是“每轴独立”，但输出是一个同步的三轴结构。
        - y 轴默认不允许 IGNORE（符合文档建议：y 是时间基准更可靠）。
    """

    xs = np.asarray(xs, dtype=float).reshape(-1)
    ys = np.asarray(ys, dtype=float).reshape(-1)
    zs = np.asarray(zs, dtype=float).reshape(-1)

    sx = sigma_from_conf(confs, sigma0=float(sigma_x0), c_min=float(c_min))
    sy = sigma_from_conf(confs, sigma0=float(sigma_y0), c_min=float(c_min))
    sz = sigma_from_conf(confs, sigma0=float(sigma_z0), c_min=float(c_min))

    dx = decide_axis_mode(xs, sx, params=params, disallow_ignore=False)
    dy = decide_axis_mode(ys, sy, params=params, disallow_ignore=bool(disallow_ignore_y))
    dz = decide_axis_mode(zs, sz, params=params, disallow_ignore=False)

    return WindowDecisions(x=dx, y=dy, z=dz)
