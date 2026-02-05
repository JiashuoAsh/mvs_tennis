"""第二阶段（posterior）拟合与融合。

`curve_v3.core` 负责流程编排；本模块承载“数学更重”的后验拟合逻辑，用于：
- 对每个候选做 MAP/正则最小二乘（docs/curve.md §5.2.1 / §5.3）
- 计算 J_post 作为候选打分
- 输出融合后的 PosteriorState

设计目标：让 `core.py` 更小、更专注于 orchestration。
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from ..config import CurveV3Config
from ..low_snr import weights_from_conf
from ..low_snr.types import WindowDecisions
from ..types import BallObservation, BounceEvent, Candidate, PosteriorState


def _safe_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """求解小规模线性方程组，并在病态时回退到最小二乘。

    说明：
        本工程里矩阵规模通常是 3x3 或 5x5。优先用 `np.linalg.solve`，当
        条件数差/奇异时回退到 `np.linalg.lstsq`，避免异常中断。
    """

    try:
        return np.asarray(np.linalg.solve(A, b), dtype=float)
    except np.linalg.LinAlgError:
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return np.asarray(x, dtype=float)


def _posterior_obs_sigma_m(cfg: CurveV3Config) -> float:
    """获取后验观测噪声尺度（米）。

    说明：
        - 若 cfg.posterior_obs_sigma_m 为 None，则回退到 cfg.weight_sigma_m。
        - 该尺度既用于 MAP 求解（决定数据项 vs 先验项的相对权重），也用于
          J_post 评分的归一化。
    """

    sigma = getattr(cfg, "posterior_obs_sigma_m", None)
    if sigma is None:
        sigma = cfg.weight_sigma_m
    return float(max(float(sigma), 1e-6))


def _solve_map_with_prior(
    *,
    H: np.ndarray,
    y_vec: np.ndarray,
    theta0: np.ndarray,
    Q: np.ndarray,
    sigma_m: float,
    fit_mode: str,
    rls_lambda: float,
) -> np.ndarray:
    """求解带高斯先验的 MAP（正则最小二乘）问题。

    目标函数（与 `docs/curve.md` 对齐）：
        (1/σ^2) * ||Hθ - y||^2 + (θ-θ0)^T Q (θ-θ0)

    Args:
        H: 设计矩阵，shape=(M,D)。
        y_vec: 观测向量，shape=(M,)。
        theta0: 先验均值，shape=(D,)。
        Q: 先验精度矩阵（对角阵），shape=(D,D)。
        sigma_m: 观测噪声尺度 σ。
        fit_mode: 拟合模式。
            - 当前实现只保留“信息形式递推”（RLS）这一条代码路径。
            - 若传入值不是 "rls"（例如历史遗留的 "ls"），会被当作等价情形："rls" 且 λ=1。
        rls_lambda: 遗忘因子 λ，(0,1]。

    Returns:
        MAP 解 theta，shape=(D,)。
    """

    inv_sigma2 = 1.0 / max(float(sigma_m) * float(sigma_m), 1e-12)

    mode = str(fit_mode).strip().lower()
    # 仅保留 RLS：历史上若传入 "ls"，这里等价为 "rls" 且 λ=1。
    lam = float(rls_lambda) if mode == "rls" else 1.0
    lam = float(min(max(lam, 1e-6), 1.0))

    # 信息形式递推：A <- λA + (1/σ^2) h^T h, b <- λb + (1/σ^2) h*y。
    # 说明：当 λ=1 时等价于批量正规方程；N<=5 时该实现足够快。
    A = np.asarray(Q, dtype=float).copy()
    b = (A @ np.asarray(theta0, dtype=float)).astype(float)

    H2 = np.asarray(H, dtype=float)
    y2 = np.asarray(y_vec, dtype=float).reshape(-1)

    for i in range(int(H2.shape[0])):
        hi = H2[i, :].reshape(-1, 1)
        yi = float(y2[i])
        A = lam * A + inv_sigma2 * (hi @ hi.T)
        b = lam * b + inv_sigma2 * (hi.reshape(-1) * yi)

    return _safe_solve(A, b)


def fit_posterior_map_for_candidate(
    *,
    bounce: BounceEvent,
    post_points: Sequence[BallObservation],
    candidate: Candidate,
    time_base_abs: float | None,
    low_snr: WindowDecisions | None = None,
    cfg: CurveV3Config,
) -> tuple[PosteriorState, float] | None:
    """对单条候选执行后验 MAP 拟合，并返回 J_post。

    与 `docs/curve.md §5.3` 对齐的流程：
        1) 候选给出先验均值 θ0（至少含 v^+，可选含 a_xz）。
        2) 求解带先验的正则 LS（MAP）。
        3) 计算 J_post（数据项 + 先验项），用于候选打分与权重更新。

    说明：
        - 观测权重这里用简化版 W=I，并用一个统一的 σ（米）做归一化。
        - σ 会同时影响“求解”与“评分”，避免出现只在评分里用 σ 的不一致。

    Returns:
        (posterior_state, J_post)；若无法构造线性系统则返回 None。
    """

    if time_base_abs is None:
        return None

    # Candidate prior (theta0) and its precision (Q).
    strength = float(cfg.posterior_prior_strength)
    if strength < 0.0:
        strength = 0.0

    mode: Literal["v_only", "v+axz"] = cfg.fit_params
    # 低 SNR：可对不同轴施加不同强度的速度先验（STRONG_PRIOR_V）。
    strong_scale = float(getattr(cfg, "low_snr_strong_prior_v_scale", 0.1))
    strong_scale = float(min(max(strong_scale, 1e-3), 1.0))

    if mode == "v_only":
        theta0 = np.array([candidate.v_plus[0], candidate.v_plus[1], candidate.v_plus[2]], dtype=float)
        if strength > 0.0:
            sigma_v = float(cfg.posterior_prior_sigma_v)
            sigs = np.array([sigma_v, sigma_v, sigma_v], dtype=float)
            if low_snr is not None:
                if str(low_snr.x.mode) == "STRONG_PRIOR_V":
                    sigs[0] = sigs[0] * strong_scale
                if str(low_snr.y.mode) == "STRONG_PRIOR_V":
                    sigs[1] = sigs[1] * strong_scale
                if str(low_snr.z.mode) == "STRONG_PRIOR_V":
                    sigs[2] = sigs[2] * strong_scale

            q = (1.0 / np.maximum(sigs * sigs, 1e-9)) * float(strength)
            Q = np.diag(q.astype(float))
        else:
            Q = np.zeros((3, 3), dtype=float)
    else:
        theta0 = np.array(
            [candidate.v_plus[0], candidate.v_plus[1], candidate.v_plus[2], candidate.ax, candidate.az],
            dtype=float,
        )
        if strength > 0.0:
            sigma_v = float(cfg.posterior_prior_sigma_v)
            sigma_a = float(cfg.posterior_prior_sigma_a)
            sigs_v = np.array([sigma_v, sigma_v, sigma_v], dtype=float)
            if low_snr is not None:
                if str(low_snr.x.mode) == "STRONG_PRIOR_V":
                    sigs_v[0] = sigs_v[0] * strong_scale
                if str(low_snr.y.mode) == "STRONG_PRIOR_V":
                    sigs_v[1] = sigs_v[1] * strong_scale
                if str(low_snr.z.mode) == "STRONG_PRIOR_V":
                    sigs_v[2] = sigs_v[2] * strong_scale

            inv_v2 = 1.0 / np.maximum(sigs_v * sigs_v, 1e-9)
            inv_a2 = 1.0 / max(sigma_a * sigma_a, 1e-9)
            q = np.array([inv_v2[0], inv_v2[1], inv_v2[2], inv_a2, inv_a2], dtype=float) * strength
            Q = np.diag(q.astype(float))
        else:
            Q = np.zeros((5, 5), dtype=float)

    # 重要约定：
    # - MAP“求解”与 J_post“评分”必须使用同一个 σ（观测噪声尺度）。
    # - 当 posterior_obs_sigma_m 为 None 时，σ 回退到 weight_sigma_m。
    #   这样可以避免出现“求解不受 σ 影响，但评分受 σ 影响”的不一致。
    sigma = _posterior_obs_sigma_m(cfg)
    fit_mode: Literal["ls", "rls"] = getattr(cfg, "fit_mode", "ls")
    if fit_mode not in ("ls", "rls"):
        fit_mode = "ls"

    min_tau = float(getattr(cfg, "posterior_min_tau_s", 1e-6))
    min_tau = float(max(min_tau, 0.0))

    def solve_for_tb(tb_rel: float) -> tuple[PosteriorState, float] | None:
        sys = _build_posterior_linear_system(
            bounce=bounce,
            post_points=post_points,
            time_base_abs=time_base_abs,
            cfg=cfg,
            t_b_rel=float(tb_rel),
            low_snr=low_snr,
        )
        if sys is None:
            return None

        H, y_vec, mode2, t_b2, x_b2, z_b2, bounce2 = sys

        theta = _solve_map_with_prior(
            H=H,
            y_vec=y_vec,
            theta0=theta0,
            Q=Q,
            sigma_m=float(sigma),
            fit_mode=fit_mode,
            rls_lambda=float(getattr(cfg, "posterior_rls_lambda", 1.0)),
        )

        if mode2 == "v_only":
            vx, vy, vz = float(theta[0]), float(theta[1]), float(theta[2])
            ax2, az2 = 0.0, 0.0
        else:
            vx, vy, vz, ax2, az2 = (
                float(theta[0]),
                float(theta[1]),
                float(theta[2]),
                float(theta[3]),
                float(theta[4]),
            )

        # Objective value J_post：保持与线性系统口径一致（docs/curve.md §5.3）。
        r = H @ theta - y_vec
        data_term = float(np.dot(r, r) / max(float(sigma) * float(sigma), 1e-12))

        d = theta - theta0
        prior_term = float(d.T @ Q @ d)
        j_post = float(data_term + prior_term)

        st = PosteriorState(t_b_rel=t_b2, x_b=x_b2, z_b=z_b2, vx=vx, vy=vy, vz=vz, ax=ax2, az=az2)
        return st, j_post

    # 方案3：可选联合估计 tb。
    if bool(getattr(cfg, "posterior_optimize_tb", False)):
        tb0 = float(bounce.t_rel)
        window = float(getattr(cfg, "posterior_tb_search_window_s", 0.05))
        step = float(getattr(cfg, "posterior_tb_search_step_s", 0.002))
        if step <= 0:
            step = 0.002

        # 上界：必须早于最早的 post 点，否则 tau<=0 会导致系统退化。
        t_rel_min = min(float(p.t - time_base_abs) for p in post_points)
        tb_lo = max(0.0, tb0 - max(window, 0.0))
        tb_hi = min(tb0 + max(window, 0.0), float(t_rel_min - min_tau))
        if tb_hi < tb_lo:
            tb_hi = tb_lo

        sigma_tb = float(getattr(cfg, "posterior_tb_prior_sigma_s", 0.03))
        sigma_tb2 = max(sigma_tb * sigma_tb, 1e-12)

        best_st: PosteriorState | None = None
        best_j = float("inf")

        # 一维网格搜索：对每个 tb 试算一次 MAP，并叠加 tb 先验惩罚项。
        tb = float(tb_lo)
        while tb <= tb_hi + 0.5 * step:
            out = solve_for_tb(tb)
            if out is not None:
                st, j = out
                tb_pen = (float(tb - tb0) ** 2) / sigma_tb2
                j2 = float(j + tb_pen)
                if j2 < best_j:
                    best_j = float(j2)
                    best_st = st
            tb += step

        if best_st is None:
            return None
        return best_st, float(best_j)

    # 默认：不优化 tb，按 bounce.t_rel 拟合。
    out0 = solve_for_tb(float(bounce.t_rel))
    return out0


def _build_posterior_linear_system(
    *,
    bounce: BounceEvent,
    post_points: Sequence[BallObservation],
    time_base_abs: float | None,
    cfg: CurveV3Config,
    t_b_rel: float | None = None,
    low_snr: WindowDecisions | None = None,
) -> tuple[np.ndarray, np.ndarray, Literal["v_only", "v+axz"], float, float, float, BounceEvent] | None:
    """构造后验拟合的线性系统 H*theta=y。

    Returns:
                (H, y, mode, t_b, x_b, z_b, bounce2)。若有效点不足则返回 None。

                说明：
                        - 当启用 tb 搜索（或显式传入 t_b_rel）时，需要将反弹事件在时间轴上
                            平移到新的 t_b：这里返回的 bounce2 就是“对齐到 t_b”的事件副本。
                        - bounce2 只调整 (t_rel, x, z)，并保留 v^- 不变：这是一个一阶近似，
                            用于吸收 prefit 对 t_b 的小偏差，避免 posterior 的 tau 被系统性放大。
    """

    if time_base_abs is None:
        return None

    pts = list(post_points)[-int(cfg.max_post_points) :]
    if not pts:
        return None

    # 低 SNR：按 conf 构造每点每轴的行权重；若未开启则 w=1。
    use_low_snr = bool(getattr(cfg, "low_snr_enabled", False)) and (low_snr is not None)
    if use_low_snr:
        confs = [getattr(p, "conf", None) for p in pts]
        wx = weights_from_conf(confs, sigma0=float(getattr(cfg, "low_snr_sigma_x0_m", 0.15)), c_min=float(getattr(cfg, "low_snr_conf_cmin", 0.05)))
        wy = weights_from_conf(confs, sigma0=float(getattr(cfg, "low_snr_sigma_y0_m", 0.15)), c_min=float(getattr(cfg, "low_snr_conf_cmin", 0.05)))
        wz = weights_from_conf(confs, sigma0=float(getattr(cfg, "low_snr_sigma_z0_m", 0.15)), c_min=float(getattr(cfg, "low_snr_conf_cmin", 0.05)))
    else:
        wx = np.ones((len(pts),), dtype=float)
        wy = np.ones((len(pts),), dtype=float)
        wz = np.ones((len(pts),), dtype=float)

    mode_x = str(low_snr.x.mode) if low_snr is not None else "FULL"
    mode_y = str(low_snr.y.mode) if low_snr is not None else "FULL"
    mode_z = str(low_snr.z.mode) if low_snr is not None else "FULL"

    min_tau = float(getattr(cfg, "posterior_min_tau_s", 1e-6))
    min_tau = float(max(min_tau, 0.0))

    bounce2 = _bounce_event_for_tb(bounce=bounce, t_b_rel=float(t_b_rel) if t_b_rel is not None else float(bounce.t_rel))

    t_b = float(bounce2.t_rel)
    x_b = float(bounce2.x)
    z_b = float(bounce2.z)

    rows: list[list[float]] = []
    ys: list[float] = []

    g = float(cfg.gravity)
    y0 = float(cfg.bounce_contact_y())
    mode: Literal["v_only", "v+axz"] = cfg.fit_params

    for i, p in enumerate(pts):
        t_rel = float(p.t - time_base_abs)
        tau = t_rel - t_b
        if tau <= min_tau:
            continue

        dx = float(p.x - x_b)
        dz = float(p.z - z_b)
        y_rhs = float(p.y + 0.5 * g * tau * tau - y0)

        sx = float(np.sqrt(max(float(wx[i]), 0.0)))
        sy = float(np.sqrt(max(float(wy[i]), 0.0)))
        sz = float(np.sqrt(max(float(wz[i]), 0.0)))

        # IGNORE_AXIS：直接不把该轴观测放进系统（等价 w=0）。
        if mode == "v_only":
            if mode_x != "IGNORE_AXIS":
                rows.append([sx * tau, 0.0, 0.0])
                ys.append(sx * dx)
            if mode_y != "IGNORE_AXIS":
                rows.append([0.0, sy * tau, 0.0])
                ys.append(sy * y_rhs)
            if mode_z != "IGNORE_AXIS":
                rows.append([0.0, 0.0, sz * tau])
                ys.append(sz * dz)
        else:
            # FREEZE_A/STRONG_PRIOR_V：冻结加速度（将对应列置 0）。
            ax_col = 0.5 * tau * tau if mode_x == "FULL" else 0.0
            az_col = 0.5 * tau * tau if mode_z == "FULL" else 0.0

            if mode_x != "IGNORE_AXIS":
                rows.append([sx * tau, 0.0, 0.0, sx * ax_col, 0.0])
                ys.append(sx * dx)
            if mode_y != "IGNORE_AXIS":
                rows.append([0.0, sy * tau, 0.0, 0.0, 0.0])
                ys.append(sy * y_rhs)
            if mode_z != "IGNORE_AXIS":
                rows.append([0.0, 0.0, sz * tau, 0.0, sz * az_col])
                ys.append(sz * dz)

    if not rows:
        return None

    H = np.asarray(rows, dtype=float)
    y_vec = np.asarray(ys, dtype=float)
    return H, y_vec, mode, t_b, x_b, z_b, bounce2


def fit_posterior_ls(
    *,
    bounce: BounceEvent,
    post_points: Sequence[BallObservation],
    time_base_abs: float | None,
    low_snr: WindowDecisions | None = None,
    cfg: CurveV3Config,
) -> PosteriorState | None:
    """在不使用候选先验时拟合后验参数。

    说明：
        当前实现只保留 RLS（信息形式递推）这一条代码路径；当 λ=1 时等价于批量 LS。
    """

    if time_base_abs is None:
        return None

    fit_mode = str(getattr(cfg, "fit_mode", "rls"))
    lam = float(getattr(cfg, "posterior_rls_lambda", 1.0)) if fit_mode == "rls" else 1.0

    sigma = _posterior_obs_sigma_m(cfg)
    min_tau = float(getattr(cfg, "posterior_min_tau_s", 1e-6))
    min_tau = float(max(min_tau, 0.0))

    def solve_for_tb(tb_rel: float) -> tuple[PosteriorState, float] | None:
        sys = _build_posterior_linear_system(
            bounce=bounce,
            post_points=post_points,
            time_base_abs=time_base_abs,
            cfg=cfg,
            t_b_rel=float(tb_rel),
            low_snr=low_snr,
        )
        if sys is None:
            return None

        H, y_vec, mode, t_b, x_b, z_b, bounce2 = sys

        # 无先验时：信息形式递推累积正规方程（λ=1 等价于批量 LS）。
        d = int(H.shape[1])
        theta0 = np.zeros((d,), dtype=float)
        Q = np.zeros((d, d), dtype=float)
        theta = _solve_map_with_prior(
            H=H,
            y_vec=y_vec,
            theta0=theta0,
            Q=Q,
            sigma_m=float(sigma),
            fit_mode="rls",
            rls_lambda=float(lam),
        )

        if mode == "v_only":
            vx, vy, vz = float(theta[0]), float(theta[1]), float(theta[2])
            ax2, az2 = 0.0, 0.0
        else:
            vx, vy, vz, ax2, az2 = (
                float(theta[0]),
                float(theta[1]),
                float(theta[2]),
                float(theta[3]),
                float(theta[4]),
            )

        # 评分仅用于 tb 搜索（无先验时），保持与线性系统口径一致。
        r = H @ theta - y_vec
        data_term = float(np.dot(r, r) / max(float(sigma) * float(sigma), 1e-12))

        st = PosteriorState(t_b_rel=t_b, x_b=x_b, z_b=z_b, vx=vx, vy=vy, vz=vz, ax=ax2, az=az2)
        return st, float(data_term)

    if bool(getattr(cfg, "posterior_optimize_tb", False)):
        tb0 = float(bounce.t_rel)
        window = float(getattr(cfg, "posterior_tb_search_window_s", 0.05))
        step = float(getattr(cfg, "posterior_tb_search_step_s", 0.002))
        if step <= 0:
            step = 0.002

        t_rel_min = min(float(p.t - time_base_abs) for p in post_points)
        tb_lo = max(0.0, tb0 - max(window, 0.0))
        tb_hi = min(tb0 + max(window, 0.0), float(t_rel_min - min_tau))
        if tb_hi < tb_lo:
            tb_hi = tb_lo

        sigma_tb = float(getattr(cfg, "posterior_tb_prior_sigma_s", 0.03))
        sigma_tb2 = max(sigma_tb * sigma_tb, 1e-12)

        best_st: PosteriorState | None = None
        best_j = float("inf")
        tb = float(tb_lo)
        while tb <= tb_hi + 0.5 * step:
            out = solve_for_tb(tb)
            if out is not None:
                st, data_term = out
                tb_pen = (float(tb - tb0) ** 2) / sigma_tb2
                j2 = float(data_term + tb_pen)
                if j2 < best_j:
                    best_j = float(j2)
                    best_st = st
            tb += step

        return best_st

    out0 = solve_for_tb(float(bounce.t_rel))
    if out0 is None:
        return None
    st0, _ = out0
    return st0


def _bounce_event_for_tb(*, bounce: BounceEvent, t_b_rel: float) -> BounceEvent:
    """根据 v^- 将 (t_b, x_b, z_b) 在时间轴上做一阶平移。

    说明：
        posterior 的观测方程以“反弹时刻”为时间零点（tau=t-t_b）。当 t_b 有偏时，
        直接用固定的 bounce.x/z 会把所有 post 点的 dx/dz 也引入系统偏差。

        这里使用 v^- 做一阶平移：
            x_b(tb) = x_b(tb0) + v^-_x * (tb - tb0)
            z_b(tb) = z_b(tb0) + v^-_z * (tb - tb0)

        这是工程化折中：
            - 好处：能显著降低 tb 小偏差对 posterior 的放大效应。
            - 局限：不调整 y（触地高度由 cfg.bounce_contact_y() 约束），也不建模
              v^- 的曲率/加速度误差。
    """

    tb = float(t_b_rel)
    dt = float(tb - float(bounce.t_rel))

    v_minus = np.asarray(bounce.v_minus, dtype=float).reshape(3)
    x = float(bounce.x + float(v_minus[0]) * dt)
    z = float(bounce.z + float(v_minus[2]) * dt)

    return BounceEvent(
        t_rel=tb,
        x=x,
        z=z,
        v_minus=np.asarray(bounce.v_minus, dtype=float),
        y=bounce.y,
        sigma_t_rel=bounce.sigma_t_rel,
        sigma_v_minus=bounce.sigma_v_minus,
        prefit_rms_m=bounce.prefit_rms_m,
    )


def fit_posterior_fused_map(
    *,
    bounce: BounceEvent,
    post_points: Sequence[BallObservation],
    best: Candidate | None,
    time_base_abs: float | None,
    low_snr: WindowDecisions | None = None,
    cfg: CurveV3Config,
) -> PosteriorState | None:
    """融合拟合后验状态（可选使用 best 候选作为高斯先验）。

    说明：
        - 当 best 存在且 posterior_prior_strength>0 时，执行 MAP（等价于带先验的 ridge 正则）。
        - 当 best 为 None 时，退化为普通 LS/RLS（由 cfg.fit_mode 控制）。
    """

    if best is None:
        return fit_posterior_ls(
            bounce=bounce,
            post_points=post_points,
            time_base_abs=time_base_abs,
            low_snr=low_snr,
            cfg=cfg,
        )

    out = fit_posterior_map_for_candidate(
        bounce=bounce,
        post_points=post_points,
        candidate=best,
        time_base_abs=time_base_abs,
        low_snr=low_snr,
        cfg=cfg,
    )
    if out is None:
        return None
    st, _ = out
    return st


def inject_posterior_anchor(
    *,
    candidates: Sequence[Candidate],
    posterior: PosteriorState,
    best: Candidate | None,
    cfg: CurveV3Config,
) -> list[Candidate]:
    """把 posterior 结果作为“锚点候选”注入混合，用于更新走廊。

    说明：
        这是一种工程化增强：当后验已经明显收敛时，用一个小权重把后验结果注入
        走廊混合，可以让 corridor 更快反映“已观测到的信息”，同时不完全抹掉 prior 多峰。
    """

    alpha = float(cfg.posterior_anchor_weight)
    alpha = min(max(alpha, 0.0), 1.0)
    if alpha <= 0.0:
        return list(candidates)

    scaled: list[Candidate] = []
    for c in candidates:
        scaled.append(
            Candidate(
                e=c.e,
                kt=c.kt,
                weight=float(c.weight) * (1.0 - alpha),
                v_plus=np.asarray(c.v_plus, dtype=float),
                kt_angle_rad=float(getattr(c, "kt_angle_rad", 0.0)),
                ax=float(c.ax),
                az=float(c.az),
            )
        )

    e = float(best.e) if best is not None else 0.0
    kt = float(best.kt) if best is not None else 0.0
    kt_ang = float(getattr(best, "kt_angle_rad", 0.0)) if best is not None else 0.0

    anchor = Candidate(
        e=e,
        kt=kt,
        weight=float(alpha),
        v_plus=np.array([posterior.vx, posterior.vy, posterior.vz], dtype=float),
        kt_angle_rad=float(kt_ang),
        ax=float(posterior.ax),
        az=float(posterior.az),
    )

    merged = scaled + [anchor]
    s = float(np.sum([float(c.weight) for c in merged]))
    if s <= 0.0:
        w = 1.0 / float(len(merged))
        return [
            Candidate(
                e=c.e,
                kt=c.kt,
                weight=float(w),
                v_plus=np.asarray(c.v_plus, dtype=float),
                kt_angle_rad=float(getattr(c, "kt_angle_rad", 0.0)),
                ax=float(c.ax),
                az=float(c.az),
            )
            for c in merged
        ]

    return [
        Candidate(
            e=c.e,
            kt=c.kt,
            weight=float(c.weight) / s,
            v_plus=np.asarray(c.v_plus, dtype=float),
            kt_angle_rad=float(getattr(c, "kt_angle_rad", 0.0)),
            ax=float(c.ax),
            az=float(c.az),
        )
        for c in merged
    ]


def prior_nominal_state(*, bounce: BounceEvent, candidates: Sequence[Candidate]) -> PosteriorState | None:
    """用候选混合的加权均值生成反弹后名义状态。"""

    if not candidates:
        return None

    weights = np.array([float(c.weight) for c in candidates], dtype=float)
    weights = weights / max(float(np.sum(weights)), 1e-12)

    v = np.sum(np.stack([np.asarray(c.v_plus, dtype=float) for c in candidates], axis=0) * weights[:, None], axis=0)
    ax = float(np.sum(weights * np.asarray([float(c.ax) for c in candidates], dtype=float)))
    az = float(np.sum(weights * np.asarray([float(c.az) for c in candidates], dtype=float)))

    return PosteriorState(
        t_b_rel=float(bounce.t_rel),
        x_b=float(bounce.x),
        z_b=float(bounce.z),
        vx=float(v[0]),
        vy=float(v[1]),
        vz=float(v[2]),
        ax=ax,
        az=az,
    )


__all__ = [
    "fit_posterior_fused_map",
    "fit_posterior_ls",
    "fit_posterior_map_for_candidate",
    "inject_posterior_anchor",
    "prior_nominal_state",
]
