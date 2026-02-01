"""curve_v3 的 legacy 输出辅助函数。

说明：
    - v3 核心预测逻辑主要面向新 API（候选/走廊/后验状态）。
    - 仓库内仍有 `curve2.py` 时代的“接球点”输出格式，这里把相关实现从
      `curve_v3.core` 拆出，降低 core 的职责负担。

注意：
    - 这些函数不做任何缓存与状态管理，仅根据输入生成输出。
"""

from __future__ import annotations

import math

import numpy as np

from ..config import CurveV3Config
from ..utils import polyder_val, real_roots_of_quadratic
from ..types import PosteriorState


def build_legacy_receive_points(
    *,
    state: PosteriorState,
    time_base_abs: float,
    cfg: CurveV3Config,
    fit_samples: int,
    curve_1_samples: int,
) -> list[dict] | None:
    """生成 legacy 接球点（反弹后下降段）。

    说明：
        历史版本（`curve2.py`）对外输出的是一组“可接球点候选”。v3 的主 API
        更偏向“状态/走廊”，但为了不改动下游，这里保留 legacy 输出格式的生成逻辑。

    Args:
        state: 反弹后状态（优先使用 posterior；否则可用候选混合的名义状态）。
        time_base_abs: 绝对时间基准（用于把相对时间转换成绝对时间戳）。
        cfg: 配置。
        fit_samples: legacy 字段，总拟合点数。
        curve_1_samples: legacy 字段，反弹后点数。

    Returns:
        与 curve2 兼容的 list[dict]；若无法生成则返回 None。
    """

    # y(tau) = y0 + vy*tau - 0.5*g*tau^2；解 y=target 得到给定高度的到达时刻。
    y0 = float(cfg.bounce_contact_y())

    def tau_at_height(target_y: float) -> list[float]:
        a = -0.5 * float(cfg.gravity)
        b = float(state.vy)
        c = float(y0 - float(target_y))
        roots = real_roots_of_quadratic(np.array([a, b, c], dtype=float))
        return [r for r in roots if r >= 0.0]

    low_taus = tau_at_height(float(cfg.net_height_1))
    if not low_taus:
        return None

    high_taus = tau_at_height(float(cfg.net_height_2))
    if not high_taus:
        # 若从未达到上限高度，则回退到最高点（顶点）时刻。
        tau_apex = max(float(state.vy) / float(cfg.gravity), 0.0)
        high_taus = [tau_apex, tau_apex]

    tau_low_down = max(low_taus)
    tau_high_down = max(high_taus)

    t_start = float(state.t_b_rel) + float(tau_high_down)
    t_end = float(state.t_b_rel) + float(tau_low_down)

    dt = float(cfg.legacy_receive_dt)
    if dt <= 0:
        return None

    results: list[dict] = []
    t_rel = float(t_start)
    while t_rel <= t_end + 1e-9:
        tau = float(t_rel - float(state.t_b_rel))
        x = float(state.x_b + state.vx * tau + 0.5 * state.ax * tau * tau)
        y = float(y0 + state.vy * tau - 0.5 * float(cfg.gravity) * tau * tau)
        z = float(state.z_b + state.vz * tau + 0.5 * state.az * tau * tau)

        vx = float(state.vx + state.ax * tau)
        vz = float(state.vz + state.az * tau)

        results.append(
            {
                "point": [float(x), float(y), float(z), float(time_base_abs + t_rel)],
                "score": float(t_rel),
                "x_speed": float(vx),
                "z_speed": float(vz),
                "fit_samples": int(fit_samples),
                "curve_1_samples": int(curve_1_samples),
            }
        )

        t_rel += dt

    return results


def validate_z_speed_from_prefit(
    *,
    z_coeff: np.ndarray,
    t_rel_last: float,
    cfg: CurveV3Config,
    is_bot_fire: int,
) -> bool:
    """legacy 行为一致性：z 方向速度的基本合法性检查。

    说明：
        旧逻辑会对 z 速度做一个区间门限，避免拟合异常时输出“离谱”的曲线。
        v3 保持该检查，用于 legacy 适配层的快速降级。

    Args:
        z_coeff: prefit 的 z(t) 多项式系数。
        t_rel_last: 评估速度的相对时间（通常取最新一帧）。
        cfg: 配置。
        is_bot_fire: legacy 方向因子（通常为 +/-1）。

    Returns:
        若 signed z speed 在配置范围内则返回 True。
    """

    vz = float(polyder_val(np.asarray(z_coeff, dtype=float), float(t_rel_last)))
    zmin, zmax = cfg.z_speed_range
    vz_signed = vz * float(is_bot_fire)

    # 保持 legacy 行为偏保守：当出现 nan/inf 时不直接判失败（避免误伤）。
    if not math.isfinite(vz_signed):
        return True

    return float(zmin) <= vz_signed <= float(zmax)


__all__ = [
    "build_legacy_receive_points",
    "validate_z_speed_from_prefit",
]
