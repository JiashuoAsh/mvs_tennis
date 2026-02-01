"""curve_v3 的动力学传播模块。

按 `docs/curve.md` 的主链路设计，这里只保留最小模型：

- y 方向：仅受重力影响（解析抛体）。
- x/z 方向：使用候选给定的 $v^+$ 与可选等效常加速度 $a_{xz}$。

说明：
    项目历史上曾尝试 RK4/drag/Magnus/自旋等扩展，但这会显著增加复杂度与耦合，
    且与 `docs/curve.md` 的“默认不在线建模气动/旋转”的工程取舍不一致，因此已移除。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .config import CurveV3Config

if TYPE_CHECKING:  # pragma: no cover
    from .types import BounceEvent, Candidate


def _propagate_post_bounce(
    *,
    bounce: "BounceEvent",
    v_plus: np.ndarray,
    tau: float,
    gravity: float,
    y0: float,
    ax: float = 0.0,
    az: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """用最小模型（常加速度）传播反弹后状态。

    该模型用于主链路传播：
    - x/z：一阶速度 + 可选水平等效常加速度（ax/az）
    - y：仅受重力（不估计 ay）

    Args:
        bounce: 反弹事件（相对时间、触地点坐标等）。
        v_plus: 反弹后初速度（tau=0），shape=(3,)。
        tau: 距反弹时刻的时间（秒）。
        gravity: 重力标量 g>0（沿 -y）。
        ax: x 方向等效常加速度。
        az: z 方向等效常加速度。

    Returns:
        (pos, vel): 位置/速度，shape=(3,)。
    """

    tau = float(max(tau, 0.0))
    g = float(gravity)

    vx0, vy0, vz0 = float(v_plus[0]), float(v_plus[1]), float(v_plus[2])
    ax = float(ax)
    az = float(az)

    x = float(bounce.x + vx0 * tau + 0.5 * ax * tau * tau)
    y = float(float(y0) + vy0 * tau - 0.5 * g * tau * tau)
    z = float(bounce.z + vz0 * tau + 0.5 * az * tau * tau)

    vx = float(vx0 + ax * tau)
    vy = float(vy0 - g * tau)
    vz = float(vz0 + az * tau)

    return np.array([x, y, z], dtype=float), np.array([vx, vy, vz], dtype=float)


def propagate_post_bounce_state(
    *,
    bounce: "BounceEvent",
    candidate: "Candidate",
    tau: float,
    cfg: CurveV3Config,
) -> tuple[np.ndarray, np.ndarray]:
    """反弹后传播入口（解析常加速度模型）。"""

    return _propagate_post_bounce(
        bounce=bounce,
        v_plus=np.asarray(candidate.v_plus, dtype=float),
        tau=tau,
        gravity=float(cfg.gravity),
        y0=float(cfg.bounce_contact_y()),
        ax=float(candidate.ax),
        az=float(candidate.az),
    )


def propagate_post_bounce_state_grid(
    *,
    bounce: "BounceEvent",
    candidate: "Candidate",
    taus: np.ndarray,
    cfg: CurveV3Config,
) -> tuple[np.ndarray, np.ndarray]:
    """反弹后网格传播入口（解析常加速度模型）。"""

    taus = np.asarray(taus, dtype=float)
    if taus.ndim != 1:
        raise ValueError("taus must be a 1D array")

    # 解析网格：向量化生成。
    tau = np.maximum(taus.astype(float), 0.0)
    g = float(cfg.gravity)
    y0 = float(cfg.bounce_contact_y())

    vx0 = float(candidate.v_plus[0])
    vy0 = float(candidate.v_plus[1])
    vz0 = float(candidate.v_plus[2])
    ax = float(candidate.ax)
    az = float(candidate.az)

    x = float(bounce.x) + vx0 * tau + 0.5 * ax * tau * tau
    y = y0 + vy0 * tau - 0.5 * g * tau * tau
    z = float(bounce.z) + vz0 * tau + 0.5 * az * tau * tau

    vx = vx0 + ax * tau
    vy = vy0 - g * tau
    vz = vz0 + az * tau

    pos = np.stack([x, y, z], axis=-1).astype(float)
    vel = np.stack([vx, vy, vz], axis=-1).astype(float)
    return pos, vel
