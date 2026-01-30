"""从分段后的 shots 构建先验学习样本。

本模块用于构造“数据驱动先验”（curve.md §6）的轻量数据集：
学习在候选 (e, kt) 网格上的 w_m^0。

注意：这里不训练端到端的整段轨迹回归器。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from curve_v3.prior.models import PriorFeatures, features_from_v_minus
from curve_v3.types import BallObservation
from curve_v3.vl11.types import ShotTrajectory


@dataclass(frozen=True)
class PriorSample:
    """用于候选先验学习的一条带标签样本。"""

    feature: PriorFeatures
    weights: np.ndarray  # shape (M,)
    e_hat: float
    kt_hat: float


def _fit_velocity(points: Sequence[BallObservation]) -> np.ndarray:
    """用线性回归估计速度：p(t) = v*t + b。"""

    if len(points) < 2:
        return np.zeros((3,), dtype=float)

    ts = np.array([float(p.t) for p in points], dtype=float)
    xs = np.array([float(p.x) for p in points], dtype=float)
    ys = np.array([float(p.y) for p in points], dtype=float)
    zs = np.array([float(p.z) for p in points], dtype=float)

    # 只需要斜率（速度），用一阶 polyfit 即可。
    vx, _ = np.polyfit(ts, xs, deg=1)
    vy, _ = np.polyfit(ts, ys, deg=1)
    vz, _ = np.polyfit(ts, zs, deg=1)

    return np.array([float(vx), float(vy), float(vz)], dtype=float)


def estimate_v_minus_v_plus(
    shot: ShotTrajectory,
    *,
    window_points: int = 4,
) -> tuple[np.ndarray, np.ndarray] | None:
    """用 bounce 附近的局部窗口估计 v- 与 v+。"""

    if shot.bounce_index is None:
        return None

    b = int(shot.bounce_index)
    pts = list(shot.points)
    if b < 1 or b >= len(pts):
        return None

    w = max(int(window_points), 2)

    pre_start = max(b - w, 0)
    pre_end = b  # 注意：下面的 pre_pts 切片包含 bounce 点
    pre_pts = pts[pre_start : pre_end + 1]

    post_start = b + 1
    post_end = min(b + 1 + w, len(pts))
    post_pts = pts[post_start:post_end]

    if len(pre_pts) < 2 or len(post_pts) < 2:
        return None

    v_minus = _fit_velocity(pre_pts)
    v_plus = _fit_velocity(post_pts)
    return v_minus, v_plus


def estimate_e_kt(
    v_minus: np.ndarray,
    v_plus: np.ndarray,
    *,
    eps: float = 1e-6,
    e_clip: tuple[float, float] = (0.0, 1.2),
    kt_clip: tuple[float, float] = (0.0, 1.5),
) -> tuple[float, float] | None:
    """用最小触地模型从 v- 与 v+ 估计 (e, kt)。"""

    v_minus = np.asarray(v_minus, dtype=float).reshape(3)
    v_plus = np.asarray(v_plus, dtype=float).reshape(3)

    vy_minus = float(v_minus[1])
    vy_plus = float(v_plus[1])

    if vy_minus >= -eps:
        return None

    e_hat = float(vy_plus / max(-vy_minus, eps))

    v_t_minus = np.array([v_minus[0], 0.0, v_minus[2]], dtype=float)
    v_t_plus = np.array([v_plus[0], 0.0, v_plus[2]], dtype=float)

    denom = float(np.dot(v_t_minus, v_t_minus))
    if denom <= eps:
        return None

    # 沿入射切向方向做标量映射，得到 kt。
    kt_hat = float(np.dot(v_t_plus, v_t_minus) / denom)

    e_hat = float(min(max(e_hat, e_clip[0]), e_clip[1]))
    kt_hat = float(min(max(kt_hat, kt_clip[0]), kt_clip[1]))

    return e_hat, kt_hat


def _gaussian_weights(values: Sequence[float], center: float, sigma: float) -> np.ndarray:
    xs = np.array([float(v) for v in values], dtype=float)
    sigma = float(sigma)
    sigma2 = max(sigma * sigma, 1e-9)
    d = xs - float(center)
    w = np.exp(-0.5 * (d * d) / sigma2)
    s = float(np.sum(w))
    if s <= 0:
        return np.full_like(w, 1.0 / float(w.size))
    return w / s


def soft_label_candidate_weights(
    *,
    e_hat: float,
    kt_hat: float,
    e_bins: Sequence[float],
    kt_bins: Sequence[float],
    sigma_e: float = 0.10,
    sigma_kt: float = 0.10,
) -> np.ndarray:
    """把 (e_hat, kt_hat) 映射为候选网格上的 soft label。"""

    if not e_bins or not kt_bins:
        return np.zeros((0,), dtype=float)

    w_e = _gaussian_weights(e_bins, float(e_hat), float(sigma_e))
    w_k = _gaussian_weights(kt_bins, float(kt_hat), float(sigma_kt))

    # 需与 CurvePredictorV3._build_prior_candidates 的候选遍历顺序保持一致。
    out: list[float] = []
    for i in range(len(e_bins)):
        for j in range(len(kt_bins)):
            out.append(float(w_e[i] * w_k[j]))

    w = np.asarray(out, dtype=float)
    s = float(np.sum(w))
    if s <= 0:
        return np.full_like(w, 1.0 / float(w.size))
    return w / s


def build_prior_samples(
    shots: Sequence[ShotTrajectory],
    *,
    e_bins: Sequence[float],
    kt_bins: Sequence[float],
    window_points: int = 4,
    sigma_e: float = 0.10,
    sigma_kt: float = 0.10,
) -> list[PriorSample]:
    """从分段后的 shots 构建先验样本。

    仅使用：检测到 bounce 且 bounce 两侧各至少 2 个点的 shot。
    """

    samples: list[PriorSample] = []

    for shot in shots:
        vv = estimate_v_minus_v_plus(shot, window_points=int(window_points))
        if vv is None:
            continue
        v_minus, v_plus = vv

        est = estimate_e_kt(v_minus, v_plus)
        if est is None:
            continue
        e_hat, kt_hat = est

        weights = soft_label_candidate_weights(
            e_hat=float(e_hat),
            kt_hat=float(kt_hat),
            e_bins=e_bins,
            kt_bins=kt_bins,
            sigma_e=float(sigma_e),
            sigma_kt=float(sigma_kt),
        )

        feat = features_from_v_minus(v_minus)
        samples.append(
            PriorSample(
                feature=feat,
                weights=np.asarray(weights, dtype=float),
                e_hat=float(e_hat),
                kt_hat=float(kt_hat),
            )
        )

    return samples


__all__ = [
    "PriorSample",
    "build_prior_samples",
    "estimate_e_kt",
    "estimate_v_minus_v_plus",
    "soft_label_candidate_weights",
]
