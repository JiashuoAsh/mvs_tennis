"""第一段（反弹前）状态估计工具。

本模块用于从第一段观测点中估计：
- 反弹事件时间 t_b（相对 time_base）
- 触地球心位置 p_b（至少 x/z；y 由接触高度给出）
- 触地前速度 v^-（用于第二段候选生成）

该实现从 `curve_v3.core` 中拆出，目的：降低文件体量与耦合。

坐标约定：
- x：向右为正
- z：向前为正
- y：向上为正
- 观测 y 为“球心高度”

因此触地/反弹事件建模为：
    y(t_b) = y_contact
其中 y_contact 默认为球半径（球心触地高度），也可通过配置加入偏置。
"""

from __future__ import annotations

import numpy as np

from ..config import CurveV3Config
from ..low_snr.types import WindowDecisions
from ..types import BounceEvent
from ..utils.math_utils import (
    constrained_quadratic_fit,
    polyder_val,
    polyval,
    real_roots_of_quadratic,
    weighted_linear_fit,
)


def _poly_coeffs_tau_to_t(coeffs_tau: np.ndarray, t_ref: float) -> np.ndarray:
    """把以 τ=t-t_ref 为自变量的多项式系数转换为以 t 为自变量的系数。

    仅支持 1 次或 2 次多项式（prefit 当前只会产出这两种）。
    """

    c = np.asarray(coeffs_tau, dtype=float).reshape(-1)
    t_ref = float(t_ref)

    if c.size == 2:
        # p(τ)=k*τ+b => p(t)=k*(t-t_ref)+b = k*t + (b-k*t_ref)
        k, b = float(c[0]), float(c[1])
        return np.array([k, b - k * t_ref], dtype=float)

    if c.size == 3:
        # p(τ)=a*τ^2 + b*τ + c
        # τ=t-t_ref
        # => a*t^2 + (b-2*a*t_ref)*t + (a*t_ref^2 - b*t_ref + c)
        a2, b1, c0 = float(c[0]), float(c[1]), float(c[2])
        return np.array(
            [
                a2,
                b1 - 2.0 * a2 * t_ref,
                a2 * t_ref * t_ref - b1 * t_ref + c0,
            ],
            dtype=float,
        )

    raise ValueError(f"Unsupported polynomial degree: coeffs size={int(c.size)}")


def _weighted_solve(H: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """用 sqrt(w) 做行缩放，求解加权最小二乘。"""

    H = np.asarray(H, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=float).reshape(-1)

    if H.ndim != 2 or y.ndim != 1:
        raise ValueError("H must be 2D and y must be 1D")
    if H.shape[0] != y.shape[0] or H.shape[0] != w.shape[0]:
        raise ValueError("H/y/w must have the same number of rows")

    # 清洗输入：避免 NaN/Inf 或极端权重导致底层 LAPACK/SVD 直接报错。
    row_ok = np.isfinite(w) & np.isfinite(y) & np.all(np.isfinite(H), axis=1)
    if not np.any(row_ok):
        return np.zeros((H.shape[1],), dtype=float)

    H = H[row_ok]
    y = y[row_ok]
    w = w[row_ok]

    w = np.maximum(w, 0.0)
    w_max = float(np.max(w))
    if w_max > 0.0:
        w = w / w_max

    # 对设计矩阵做列缩放，减小病态/溢出风险。
    col_scale = np.max(np.abs(H), axis=0)
    col_scale = np.where(col_scale > 0.0, col_scale, 1.0)
    Hs = H / col_scale

    # 优先沿用原实现（更接近直观的加权 least squares），失败则回退。
    sw = np.sqrt(np.maximum(w, 1e-12))
    Hw = Hs * sw[:, None]
    yw = y * sw
    try:
        theta_s, _, _, _ = np.linalg.lstsq(Hw, yw, rcond=None)
        theta = np.asarray(theta_s, dtype=float) / col_scale
        return theta
    except np.linalg.LinAlgError:
        # 退化回正规方程 + 轻微 Tikhonov 正则，避免 SVD 不收敛。
        A = Hs.T @ (w[:, None] * Hs)
        b = Hs.T @ (w * y)

        # 正则系数跟随 trace(A) 的量级，避免过度影响解。
        tr = float(np.trace(A))
        lam = max(1e-12, 1e-8 * (tr / max(1, int(A.shape[0]))))
        A = A + lam * np.eye(A.shape[0], dtype=float)

        try:
            theta_s = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            theta_s, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        theta = np.asarray(theta_s, dtype=float) / col_scale
        return theta


def _fit_xz_const_accel(
    *,
    t: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """用“水平常加速度”模型拟合 x/z。

    模型：
        x(t) = x0 + vx*t + 0.5*ax*t^2
        z(t) = z0 + vz*t + 0.5*az*t^2

    Returns:
        ((x0, vx, ax), (z0, vz, az))
    """

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    w = np.asarray(w, dtype=float)

    H = np.stack([np.ones_like(t), t, 0.5 * t * t], axis=1)
    thx = _weighted_solve(H, x, w)
    thz = _weighted_solve(H, z, w)

    x0, vx, ax = float(thx[0]), float(thx[1]), float(thx[2])
    z0, vz, az = float(thz[0]), float(thz[1]), float(thz[2])
    return (x0, vx, ax), (z0, vz, az)


def _fit_xz_coeffs_on_tau(
    *,
    tau: np.ndarray,
    xs: np.ndarray,
    zs: np.ndarray,
    w_all: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """在 τ 域拟合 x/z，多项式系数按 np.polyval 约定返回。

    说明：按 `docs/curve.md` 的工程规范，水平面强制使用等效常加速度模型。
    因此返回值恒为 2 次系数 [0.5*a, v, x0]。
    """

    tau = np.asarray(tau, dtype=float)
    xs = np.asarray(xs, dtype=float)
    zs = np.asarray(zs, dtype=float)
    w_all = np.asarray(w_all, dtype=float)

    # 常加速度模型：用组合权重 w_all 提升鲁棒性。
    (x0, vx0, ax), (z0, vz0, az) = _fit_xz_const_accel(t=tau, x=xs, z=zs, w=w_all)
    x_coeff = np.array([0.5 * float(ax), float(vx0), float(x0)], dtype=float)
    z_coeff = np.array([0.5 * float(az), float(vz0), float(z0)], dtype=float)
    return x_coeff, z_coeff


def _solve_linear_with_v_prior(
    *,
    t: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    v_prior: float,
    sigma_v: float,
) -> tuple[float, float]:
    """线性模型 y(t)=v*t+b 的加权 MAP 解（对 v 加强先验）。

    目标函数：
        sum_i w_i * (v*t_i + b - y_i)^2 + (v - v_prior)^2 / sigma_v^2

    说明：
        - 这是 STRONG_PRIOR_V 的最小落地版本：仅对速度项加先验。
        - b 不加先验，保证位置可以贴合窗口末端，避免整体漂移。
    """

    t = np.asarray(t, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=float).reshape(-1)

    if t.size == 0:
        return float(v_prior), float(0.0)

    w = np.maximum(w, 0.0)
    sw = float(np.sum(w))
    if sw <= 0.0:
        w = np.ones_like(w)

    s_tt = float(np.sum(w * t * t))
    s_t1 = float(np.sum(w * t))
    s_11 = float(np.sum(w))
    s_ty = float(np.sum(w * t * y))
    s_1y = float(np.sum(w * y))

    sigma_v2 = max(float(sigma_v) * float(sigma_v), 1e-12)
    q = 1.0 / sigma_v2

    # 正规方程（2x2）：[s_tt+q, s_t1; s_t1, s_11] [v;b] = [s_ty+q*v0; s_1y]
    A = np.array([[s_tt + q, s_t1], [s_t1, s_11]], dtype=float)
    bvec = np.array([s_ty + q * float(v_prior), s_1y], dtype=float)
    try:
        vb = np.linalg.solve(A, bvec)
    except np.linalg.LinAlgError:
        vb, _, _, _ = np.linalg.lstsq(A, bvec, rcond=None)

    v = float(vb[0])
    b0 = float(vb[1])
    return v, b0


def _axis_mode(low_snr: WindowDecisions | None, axis: str) -> str:
    """从 low_snr 决策中取出指定轴的 mode；未提供时回退 FULL。"""

    if low_snr is None:
        return "FULL"
    if axis == "x":
        return str(low_snr.x.mode)
    if axis == "y":
        return str(low_snr.y.mode)
    if axis == "z":
        return str(low_snr.z.mode)
    return "FULL"


def _fit_u_coeffs_on_tau(
    *,
    tau: np.ndarray,
    us: np.ndarray,
    w: np.ndarray,
    mode: str,
    v_prior: float | None,
    sigma_v_prior: float,
) -> np.ndarray:
    """在 τ 域拟合单轴多项式系数。

    返回值恒为 2 次系数 [0.5*a, v, u0]：
        - FULL：估计 a,v,u0
        - FREEZE_A：a=0，仅拟合线性 v,u0
        - STRONG_PRIOR_V：a=0，且 v 强贴 v_prior
        - IGNORE_AXIS：a=0，v=v_prior（或 0），u0 取最后点对齐
    """

    tau = np.asarray(tau, dtype=float).reshape(-1)
    us = np.asarray(us, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=float).reshape(-1)

    mode = str(mode).strip().upper()
    if mode == "FULL":
        # 常加速度：用与旧实现一致的参数化 u=u0+v*t+0.5*a*t^2
        H = np.stack([np.ones_like(tau), tau, 0.5 * tau * tau], axis=1)
        theta = _weighted_solve(H, us, w)
        u0, v, a = float(theta[0]), float(theta[1]), float(theta[2])
        return np.array([0.5 * a, v, u0], dtype=float)

    if mode == "FREEZE_A":
        # a=0：退化为线性 y=v*t+u0
        coeffs_1d, _ = weighted_linear_fit(tau, us, w)
        v, u0 = float(coeffs_1d[0]), float(coeffs_1d[1])
        return np.array([0.0, v, u0], dtype=float)

    if mode == "STRONG_PRIOR_V":
        # 先冻结加速度，再对速度做强先验。
        v0 = float(v_prior) if v_prior is not None else 0.0
        v, u0 = _solve_linear_with_v_prior(t=tau, y=us, w=w, v_prior=v0, sigma_v=float(sigma_v_prior))
        return np.array([0.0, float(v), float(u0)], dtype=float)

    if mode == "IGNORE_AXIS":
        # 该轴不参与拟合：只用先验传播。
        v0 = float(v_prior) if v_prior is not None else 0.0
        if tau.size <= 0:
            return np.array([0.0, v0, 0.0], dtype=float)
        # 用最后一个点对齐截距，避免输出跳变。
        u_last = float(us[-1])
        t_last = float(tau[-1])
        u0 = float(u_last - v0 * t_last)
        return np.array([0.0, v0, u0], dtype=float)

    # 未知 mode：按 FULL 兜底。
    H = np.stack([np.ones_like(tau), tau, 0.5 * tau * tau], axis=1)
    theta = _weighted_solve(H, us, w)
    u0, v, a = float(theta[0]), float(theta[1]), float(theta[2])
    return np.array([0.5 * a, v, u0], dtype=float)


def _robust_reweight_3d(
    *,
    t: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    x_coeff: np.ndarray,
    y_coeff: np.ndarray,
    z_coeff: np.ndarray,
    w: np.ndarray,
    delta_m: float,
) -> np.ndarray:
    """基于 3D 残差范数做一次鲁棒重加权（抑制离群点）。"""

    t = np.asarray(t, dtype=float)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    zs = np.asarray(zs, dtype=float)
    w = np.asarray(w, dtype=float)

    x_hat = np.polyval(np.asarray(x_coeff, dtype=float), t)
    y_hat = np.polyval(np.asarray(y_coeff, dtype=float), t)
    z_hat = np.polyval(np.asarray(z_coeff, dtype=float), t)

    r = np.sqrt((xs - x_hat) ** 2 + (ys - y_hat) ** 2 + (zs - z_hat) ** 2)
    delta = float(max(delta_m, 1e-6))
    scale = np.minimum(1.0, delta / np.maximum(r, 1e-12))
    return w * scale


def estimate_bounce_event_from_prefit(
    *,
    t_rel: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    xw: np.ndarray,
    yw: np.ndarray,
    zw: np.ndarray,
    low_snr: WindowDecisions | None = None,
    v_prior: np.ndarray | None = None,
    cfg: CurveV3Config,
) -> tuple[dict[str, np.ndarray], BounceEvent] | None:
    """估计第一段拟合与反弹事件（t_b, p_b, v^-）。

    当前实现遵循 `docs/curve.md` §2.4 / 附录B 的“低算力高稳定”建议：
    - 在短窗口内用 τ=t-t_ref 做时间归一（改善数值稳定性）。
    - y：固定重力项的二次拟合（a=-0.5*g 不拟合）。
        - x/z：默认使用等效常加速度（二次）模型；在低 SNR 的退化模式下，可退化为线性
            或仅按先验传播（见 `curve_v3.low_snr`）。
    - 触地时间：解二次方程 y(τ)=y_contact，取最新实根。

    Args:
        t_rel: 相对时间（秒），shape=(N,)。
        xs: x 观测，shape=(N,)。
        ys: y 观测（球心高度），shape=(N,)。
        zs: z 观测，shape=(N,)。
        xw: x 拟合权重，shape=(N,)。
        yw: y 拟合权重，shape=(N,)。
        zw: z 拟合权重，shape=(N,)。
        cfg: v3 配置。

    Returns:
        (pre_coeffs, bounce_event)，失败返回 None。

        pre_coeffs 用于 legacy 兼容（保持历史字段）：
        - "x": 二次多项式系数（等效常加速度；系数顺序遵循 np.polyval）
        - "y": 二次多项式系数（a 固定为 -0.5*g）
        - "z": 二次多项式系数（等效常加速度；系数顺序遵循 np.polyval）
        - "t_land": shape=(1,) 的预测触地相对时间
    """

    # region 输入清洗与基本检查
    t_rel = np.asarray(t_rel, dtype=float)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    zs = np.asarray(zs, dtype=float)
    xw = np.asarray(xw, dtype=float)
    yw = np.asarray(yw, dtype=float)
    zw = np.asarray(zw, dtype=float)

    if t_rel.ndim != 1 or t_rel.size < 5:
        return None
    # endregion

    # region 时间归一（τ）与 y 轴拟合：求触地时刻 t_land
    end = int(t_rel.size)
    xz_n = int(getattr(cfg, "prefit_xz_window_points", 12))
    xz_n = max(xz_n, 3)
    xz_start = max(end - xz_n, 0)
    # 按 docs/curve.md §2.4.2：用窗口参考时刻做 τ 归一。
    # 这里选用 x/z 窗口起点作为 t_ref（让 τ 的量级稳定在 ~0.0~0.3s）。
    t_ref = float(t_rel[xz_start])
    tau_all = np.asarray(t_rel - t_ref, dtype=float)

    a = -0.5 * float(cfg.gravity)
    y_coeff, _ = constrained_quadratic_fit(tau_all, ys, yw, fixed_a=a)
    # 解 y(t)=y_contact（y 为球心高度）以得到触地/反弹时刻。
    y_contact = float(cfg.bounce_contact_y())
    y_eq = np.array([float(y_coeff[0]), float(y_coeff[1]), float(y_coeff[2]) - y_contact], dtype=float)
    roots = real_roots_of_quadratic(y_eq)
    if not roots:
        return None
    tau_land = float(max(roots))
    if tau_land <= 0.0:
        return None
    t_land = float(t_ref + tau_land)
    # endregion

    # region 构造组合权重与选择 x/z 拟合窗口（含“只用触地前点”的防御）
    # 按轴权重：拟合时分别使用 x/y/z 各自权重。
    # 说明：
    # - 旧实现用 (xw+yw+zw)/3 得到一个“点级共享权重”，会把某一轴的低权重
    #   传染给其他轴（例如 x 很差时把 y/z 也一起压低）。
    # - 这里改为：拟合按轴使用 w_x / w_y / w_z，但鲁棒重加权仍使用 3D 残差
    #   计算一个点级 scale，并将该 scale 同步乘回三轴权重，以保持离群点抑制
    #   的一致性。
    w_x = np.asarray(xw, dtype=float).reshape(-1).copy()
    w_y = np.asarray(yw, dtype=float).reshape(-1).copy()
    w_z = np.asarray(zw, dtype=float).reshape(-1).copy()

    # 用当前触地时刻做一次 x/z 拟合（短窗口 + 可选常加速度）。
    # 额外防御：若上游误把少量 post 点喂进来，尽量只用 t<=t_land 的点。
    mask_pre = t_rel <= float(t_land + 1e-6)
    if int(np.sum(mask_pre)) < 5:
        mask_pre = np.ones_like(t_rel, dtype=bool)

    idx_pre = np.where(mask_pre)[0]
    if idx_pre.size <= 0:
        return None
    idx_xz = idx_pre[-xz_n:] if int(idx_pre.size) > xz_n else idx_pre

    # endregion

    # region 低 SNR 模式与先验速度（v_prior）准备

    # 低 SNR 退化策略：x/z 允许冻结加速度 / 强先验 / 忽略。
    # - y 轴用于时间基准（t_land），因此这里不实现 y 的 IGNORE（由 low_snr 层钳制）。
    mode_x = _axis_mode(low_snr, "x")
    mode_z = _axis_mode(low_snr, "z")

    vpx: float | None = None
    vpz: float | None = None
    if v_prior is not None:
        vp = np.asarray(v_prior, dtype=float).reshape(-1)
        if vp.size >= 3:
            vpx = float(vp[0])
            vpz = float(vp[2])

    sigma_v_strong = float(getattr(cfg, "low_snr_prefit_strong_sigma_v_mps", 0.5))

    # endregion

    # region x/z 拟合（按 low_snr mode 选择 FULL/FREEZE/STRONG/IGNORE）
    x_coeff = _fit_u_coeffs_on_tau(
        tau=tau_all[idx_xz],
        us=xs[idx_xz],
        w=w_x[idx_xz],
        mode=str(mode_x),
        v_prior=vpx,
        sigma_v_prior=float(sigma_v_strong),
    )
    z_coeff = _fit_u_coeffs_on_tau(
        tau=tau_all[idx_xz],
        us=zs[idx_xz],
        w=w_z[idx_xz],
        mode=str(mode_z),
        v_prior=vpz,
        sigma_v_prior=float(sigma_v_strong),
    )
    # endregion

    # region 鲁棒重加权迭代（抑制离群点）
    # 可选：做少量鲁棒重加权迭代，用于抑制离群点。
    robust_iters = int(getattr(cfg, "prefit_robust_iters", 1))
    if robust_iters > 0:
        delta = float(getattr(cfg, "prefit_robust_delta_m", 0.12))
        for _ in range(robust_iters):
            # 用更新后的组合权重重新拟合。
            mask_pre = t_rel <= float(t_land + 1e-6)
            if int(np.sum(mask_pre)) < int(getattr(cfg, "prefit_min_inlier_points", 5)):
                mask_pre = np.ones_like(t_rel, dtype=bool)

            idx_pre = np.where(mask_pre)[0]
            if idx_pre.size <= 0:
                return None
            idx_xz = idx_pre[-xz_n:] if int(idx_pre.size) > xz_n else idx_pre

            x_coeff = _fit_u_coeffs_on_tau(
                tau=tau_all[idx_xz],
                us=xs[idx_xz],
                w=w_x[idx_xz],
                mode=str(mode_x),
                v_prior=vpx,
                sigma_v_prior=float(sigma_v_strong),
            )
            z_coeff = _fit_u_coeffs_on_tau(
                tau=tau_all[idx_xz],
                us=zs[idx_xz],
                w=w_z[idx_xz],
                mode=str(mode_z),
                v_prior=vpz,
                sigma_v_prior=float(sigma_v_strong),
            )
            y_coeff, _ = constrained_quadratic_fit(tau_all, ys, w_y, fixed_a=a)

            # 点级鲁棒重加权：用 3D 残差计算同一个 scale，并同步乘回三轴权重。
            scale = _robust_reweight_3d(
                t=tau_all,
                xs=xs,
                ys=ys,
                zs=zs,
                x_coeff=x_coeff,
                y_coeff=y_coeff,
                z_coeff=z_coeff,
                w=np.ones_like(w_x, dtype=float),
                delta_m=delta,
            )
            w_x = w_x * scale
            w_y = w_y * scale
            w_z = w_z * scale

        # 用最终 y_coeff 重新计算触地/反弹时刻。
        y_eq = np.array([float(y_coeff[0]), float(y_coeff[1]), float(y_coeff[2]) - y_contact], dtype=float)
        roots = real_roots_of_quadratic(y_eq)
        if not roots:
            return None
        tau_land = float(max(roots))
        if tau_land <= 0.0:
            return None

        t_land = float(t_ref + tau_land)
    # endregion

    # region 由拟合结果导出触地点与触地前速度 v^-
    # 默认：用多项式导数给出 v^-（触地时刻的一阶导）。
    vx = float(polyder_val(x_coeff, tau_land))
    vz = float(polyder_val(z_coeff, tau_land))
    vy = float(polyder_val(y_coeff, tau_land))

    x_land = float(polyval(x_coeff, tau_land))
    z_land = float(polyval(z_coeff, tau_land))
    # endregion

    # region 轻量不确定度尺度（诊断/下游权重构造用；失败不影响主流程）
    # TODO : 更严谨的协方差与误差传播？
    # TODO : 权重是否要分轴？
    # 诊断用综合权重：三轴按轴权重的简单平均（仅用于统计尺度，不影响拟合）。
    w_all = (w_x + w_y + w_z) / 3.0

    # 轻量不确定度尺度（docs/curve.md 附录B.8）：用于后续构造权重矩阵 R。
    # 注意：这里不做完整协方差与严格误差传播，只给“尺度量”，并且失败时不影响主流程。
    sigma_t_rel: float | None = None
    sigma_v_minus: np.ndarray | None = None
    prefit_rms_m: float | None = None
    try:
        w = np.asarray(w_all, dtype=float).reshape(-1)
        w = np.maximum(w, 0.0)
        sw = float(np.sum(w))
        if sw <= 0.0:
            w = np.ones_like(w)
            sw = float(w.size)

        x_hat = np.polyval(np.asarray(x_coeff, dtype=float), tau_all)
        y_hat = np.polyval(np.asarray(y_coeff, dtype=float), tau_all)
        z_hat = np.polyval(np.asarray(z_coeff, dtype=float), tau_all)

        rx = xs - x_hat
        ry = ys - y_hat
        rz = zs - z_hat

        rms_x = float(np.sqrt(np.sum(w * rx * rx) / sw))
        rms_y = float(np.sqrt(np.sum(w * ry * ry) / sw))
        rms_z = float(np.sqrt(np.sum(w * rz * rz) / sw))

        prefit_rms_m = float(np.sqrt(np.sum(w * (rx * rx + ry * ry + rz * rz)) / sw))

        # sigma_t: 用 y 残差尺度除以 |dy/dt| 做一阶近似。
        dy_dt = float(polyder_val(y_coeff, tau_land))
        sigma_t_rel = float(rms_y / max(abs(dy_dt), 1e-3))

        # sigma_v: 用位置残差尺度除以窗口时间跨度做粗略近似。
        t_span = float(np.max(tau_all) - np.min(tau_all))
        t_span = float(max(t_span, 1e-3))
        sigma_v_minus = np.array([rms_x / t_span, rms_y / t_span, rms_z / t_span], dtype=float)
    except Exception:
        sigma_t_rel = None
        sigma_v_minus = None
        prefit_rms_m = None
    # endregion

    # region 组装输出（prefit 多项式系数 + bounce_event）
    # 对外输出保持以 t_rel 为自变量的多项式系数（兼容 core/legacy 的 point_at_time_rel）。
    pre_coeffs: dict[str, np.ndarray] = {
        "x": _poly_coeffs_tau_to_t(x_coeff, t_ref),
        "y": _poly_coeffs_tau_to_t(y_coeff, t_ref),
        "z": _poly_coeffs_tau_to_t(z_coeff, t_ref),
        "t_land": np.array([t_land], dtype=float),
    }

    bounce_event = BounceEvent(
        t_rel=float(t_land),
        x=float(x_land),
        z=float(z_land),
        v_minus=np.array([float(vx), float(vy), float(vz)], dtype=float),
        y=float(y_contact),
        sigma_t_rel=sigma_t_rel,
        sigma_v_minus=sigma_v_minus,
        prefit_rms_m=prefit_rms_m,
    )

    return pre_coeffs, bounce_event

    # endregion
