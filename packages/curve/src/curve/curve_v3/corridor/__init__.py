"""走廊（corridor）统计计算。

本模块实现第一阶段/融合后的走廊输出：
- `corridor_by_time`：按时间输出 (x,z) 的均值/协方差（可选分位数包络）
- `corridor_on_plane`：按 y 平面输出穿越点与到达时间的统计（可选分位数/分量）

该模块用于把走廊计算从 `curve_v3.core` 中拆出，减少 core 的职责与体量。

说明：
    这里把 corridor 改为子包（目录），以减少 `curve_v3/` 根目录下文件数量。
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..config import CurveV3Config
from ..dynamics import propagate_post_bounce_state_grid
from ..types import BounceEvent, Candidate, CorridorByTime, CorridorComponent, CorridorOnPlane
from ..utils.math_utils import real_roots_of_quadratic, weighted_quantiles_1d, weighted_quantile_1d


def _split_components_k2(
    *,
    x_arr: np.ndarray,
    z_arr: np.ndarray,
    t_arr: np.ndarray,
    w: np.ndarray,
    crossing_prob: float,
) -> tuple[CorridorComponent, ...] | None:
    """把穿越样本简单分成最多 2 个走廊分量（K=2）。

    该函数用于给 `CorridorOnPlane.components` 提供一种“轻量多峰表达”。当
    穿越平面的落点分布明显双峰时，单个高斯（mu/cov）会把两个峰之间的
    “低概率区域”也覆盖进走廊里；分成 2 个分量更便于解释与可视化。

    算法（工程化近似）：
        1) 在穿越子集内，按权重 w 计算 (x,z) 的加权协方差；
        2) 取最大特征值对应的主轴方向 axis；
        3) 将样本投影到主轴 s 上，用 s 的 50% 加权分位数作为阈值做二分。

    权重域约定（非常关键）：
        - 入参 `w` 是“穿越子集内的条件权重”，已归一化：sum(w) == 1。
          它用于计算条件统计量（mu/cov/t_var）。
        - 入参 `crossing_prob` 是“无条件穿越概率”（或无条件总权重），即
          crossing_prob = sum(ws) / total_w。
        - 返回的每个分量 weight 都在“全局混合权重域”，满足：
          sum(component.weight) == crossing_prob。

    Args:
        x_arr: 穿越样本的 x（单位 m），shape (N,)。
        z_arr: 穿越样本的 z（单位 m），shape (N,)。
        t_arr: 穿越样本的 t_rel（相对 time_base 的秒），shape (N,)。
        w: 穿越子集内归一化权重，shape (N,)，sum(w)==1。
        crossing_prob: 无条件穿越概率/权重（0~1）。

    Returns:
        若样本数不足或无法稳定二分则返回 None；否则返回按权重降序排列的
        2 个 `CorridorComponent`。
    """

    n = int(x_arr.size)
    if n < 2:
        return None

    # 用加权协方差提取主轴方向（用于做一个稳定的二分）。
    mu_x = float(np.sum(w * x_arr))
    mu_z = float(np.sum(w * z_arr))
    dx = x_arr - mu_x
    dz = z_arr - mu_z
    cov_xx = float(np.sum(w * dx * dx))
    cov_zz = float(np.sum(w * dz * dz))
    cov_xz = float(np.sum(w * dx * dz))
    cov = np.array([[cov_xx, cov_xz], [cov_xz, cov_zz]], dtype=float)

    try:
        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, int(np.argmax(vals))]
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    except Exception:
        axis = np.array([1.0, 0.0], dtype=float)

    s = (dx * float(axis[0]) + dz * float(axis[1])).astype(float)
    # 用加权中位数做阈值：比用均值更稳健，且能保证两侧都有一定权重（通常）。
    thr = float(weighted_quantile_1d(s, w, 0.5))

    idx0 = np.where(s <= thr)[0]
    idx1 = np.where(s > thr)[0]
    if idx0.size == 0 or idx1.size == 0:
        return None

    def comp_for(idxs: np.ndarray) -> CorridorComponent:
        ww = w[idxs]
        sw = float(np.sum(ww))
        if sw <= 0.0:
            ww = np.full((idxs.size,), 1.0 / float(idxs.size), dtype=float)
            sw = 1.0
        # 分量内部再做一次归一化：得到“分量条件权重”，用于计算该分量的 mu/cov。
        w_norm = ww / sw

        xa = x_arr[idxs]
        za = z_arr[idxs]
        ta = t_arr[idxs]

        mx = float(np.sum(w_norm * xa))
        mz = float(np.sum(w_norm * za))
        mt = float(np.sum(w_norm * ta))

        ddx = xa - mx
        ddz = za - mz
        ddt = ta - mt

        cxx = float(np.sum(w_norm * ddx * ddx))
        czz = float(np.sum(w_norm * ddz * ddz))
        cxz = float(np.sum(w_norm * ddx * ddz))
        tv = float(np.sum(w_norm * ddt * ddt))

        # 分量权重使用“全局混合权重域”。
        # - sw 是该分量在“穿越条件下”的权重占比（因为 w.sum()==1）。
        # - 乘以 crossing_prob 后得到“无条件”的分量权重。
        weight = float(crossing_prob) * float(sw)

        return CorridorComponent(
            weight=weight,
            mu_xz=np.array([mx, mz], dtype=float),
            cov_xz=np.array([[cxx, cxz], [cxz, czz]], dtype=float),
            t_rel_mu=mt,
            t_rel_var=tv,
            num_candidates=int(idxs.size),
        )

    c0 = comp_for(idx0)
    c1 = comp_for(idx1)
    comps = sorted([c0, c1], key=lambda c: float(c.weight), reverse=True)
    return tuple(comps)


def _invalid_corridor_on_plane(target_y: float) -> CorridorOnPlane:
    """构造一个不可用的 `CorridorOnPlane` 占位结果。

    约定：
        - 数值统计项（mu/cov/t_rel）用 NaN，便于上层快速识别并避免误用。
        - valid_ratio 与 crossing_prob 置 0，并把 is_valid 置 False。

    Args:
        target_y: 平面 y==target_y。

    Returns:
        不可用走廊结果。
    """
    nan2 = np.array([math.nan, math.nan], dtype=float)
    nan22 = np.array([[math.nan, math.nan], [math.nan, math.nan]], dtype=float)
    return CorridorOnPlane(
        target_y=float(target_y),
        mu_xz=nan2,
        cov_xz=nan22,
        t_rel_mu=float(math.nan),
        t_rel_var=float(math.nan),
        valid_ratio=0.0,
        crossing_prob=0.0,
        is_valid=False,
    )


def build_corridor_by_time(
    *,
    bounce: BounceEvent,
    candidates: Sequence[Candidate],
    cfg: CurveV3Config,
) -> CorridorByTime | None:
    """按时间输出 (x,z) 走廊（均值/协方差 + 可选分位数包络）。

    输出解释：
        - 对每个时间点 t_rel[k]，统计所有候选轨迹在 (x,z) 的加权均值与协方差。
        - 权重来自 `Candidate.weight`，在本函数内会归一化为 sum(weights)==1。
        - 可选输出分位数包络（更适合多峰/非高斯分布）。

    坐标与时间：
        - (x,z) 为水平平面投影；坐标约定见 `curve_v3/types.py` 模块说明。
        - t_rel 是相对 `bounce.t_rel` / time_base 的相对时间（秒）。

    Args:
        bounce: 反弹锚点（包含 t_rel / x / z 等）。
        candidates: 反弹后阶段的候选集合。
        cfg: 配置，使用 corridor_dt / corridor_horizon_s / corridor_quantile_levels。

    Returns:
        若 candidates 为空或配置不合法则返回 None；否则返回按时间采样的走廊统计。
    """

    if not candidates:
        return None

    t0 = float(bounce.t_rel)
    dt = float(cfg.corridor_dt)
    horizon = float(cfg.corridor_horizon_s)
    if dt <= 0.0 or horizon <= 0.0:
        return None

    ts = np.arange(t0, t0 + horizon + 1e-9, dt, dtype=float)
    xs = np.zeros((len(candidates), ts.size), dtype=float)
    zs = np.zeros((len(candidates), ts.size), dtype=float)

    # 约定：候选权重在外部可能未严格归一化；这里归一化后用于统计。
    weights = np.array([float(c.weight) for c in candidates], dtype=float)
    weights = weights / max(float(np.sum(weights)), 1e-12)

    taus = (ts - t0).astype(float)
    for m, c in enumerate(candidates):
        pos, _ = propagate_post_bounce_state_grid(bounce=bounce, candidate=c, taus=taus, cfg=cfg)
        xs[m, :] = pos[:, 0]
        zs[m, :] = pos[:, 2]

    mu_x = np.sum(xs * weights[:, None], axis=0)
    mu_z = np.sum(zs * weights[:, None], axis=0)
    mu = np.stack([mu_x, mu_z], axis=-1)

    cov = np.zeros((ts.size, 2, 2), dtype=float)
    for k in range(ts.size):
        dx = xs[:, k] - mu_x[k]
        dz = zs[:, k] - mu_z[k]
        s00 = float(np.sum(weights * dx * dx))
        s11 = float(np.sum(weights * dz * dz))
        s01 = float(np.sum(weights * dx * dz))
        cov[k, :, :] = np.array([[s00, s01], [s01, s11]], dtype=float)

    # 可选：分位数包络输出（多峰更安全）。
    q_levels = np.asarray(getattr(cfg, "corridor_quantile_levels", ()), dtype=float).reshape(-1)
    q_levels = q_levels[np.isfinite(q_levels)]
    q_levels = q_levels[(q_levels >= 0.0) & (q_levels <= 1.0)]
    if q_levels.size > 0:
        qxz = np.zeros((ts.size, q_levels.size, 2), dtype=float)
        for k in range(ts.size):
            qxz[k, :, 0] = weighted_quantiles_1d(xs[:, k], weights, q_levels)
            qxz[k, :, 1] = weighted_quantiles_1d(zs[:, k], weights, q_levels)
        return CorridorByTime(t_rel=ts, mu_xz=mu, cov_xz=cov, quantile_levels=q_levels, quantiles_xz=qxz)

    return CorridorByTime(t_rel=ts, mu_xz=mu, cov_xz=cov)


def corridor_on_planes_y(
    *,
    bounce: BounceEvent | None,
    candidates: Sequence[Candidate],
    cfg: CurveV3Config,
    target_ys: Sequence[float],
) -> list[CorridorOnPlane]:
    """计算多个水平平面 y==const 的穿越走廊统计。

    目标：对每个 target_y，统计“候选轨迹穿越平面 y==target_y 时”的
    - 穿越点在 (x,z) 的均值/协方差
    - 穿越时刻 t_rel 的均值/方差
    并给出两个可解释的可靠性指标：valid_ratio 与 crossing_prob。

    关键定义（按本实现精确定义）：
        - total = len(candidates)
        - total_w = sum(candidate.weight)
        - 对固定平面 target_y，找到能穿越该平面的候选子集 S（能解出 tau>=0）。
        - valid_ratio = |S| / total
          （非加权比例，反映“有多少候选可用于统计”）
        - crossing_prob = sum_{i in S}(w_i) / total_w
          （加权比例，反映“无条件穿越概率/权重”）
        - 统计量 mu/cov/t_var 使用“条件权重” w'_i = w_i / sum_{j in S}(w_j)

    运动学求解：
        - y 方向使用解析二次方程：
          y_contact + vy * tau - 0.5 * g * tau^2 == target_y
        - 对 tau>=0 的实根，取 tau = max(roots)。直觉上：若轨迹上升后下降，
          会与同一平面相交两次；取较大的根对应“更靠后的一次穿越”。
        - x/z 方向使用常加速度解析：
          x = x_b + vx * tau + 0.5 * ax * tau^2
          z = z_b + vz * tau + 0.5 * az * tau^2

    Args:
        bounce: 反弹锚点。若为 None 则所有平面返回 invalid。
        candidates: 反弹后候选集合。若为空则返回 invalid。
        cfg: 配置（gravity / bounce_contact_y / corridor_components_k / corridor_quantile_levels）。
        target_ys: 需要统计的多个水平平面高度。

    Returns:
        每个 target_y 对应一个 `CorridorOnPlane`（顺序与输入一致）。
    """

    ys = [float(y) for y in target_ys]
    if not ys:
        return []

    if bounce is None or not candidates:
        return [_invalid_corridor_on_plane(y) for y in ys]

    g = float(cfg.gravity)
    if g <= 0.0:
        return [_invalid_corridor_on_plane(y) for y in ys]

    y_contact = float(cfg.bounce_contact_y())

    total = int(len(candidates))
    # total_w 用于把“穿越子集权重和”转成无条件比例 crossing_prob。
    total_w = float(np.sum([float(c.weight) for c in candidates]))
    if total <= 0 or total_w <= 0.0:
        return [_invalid_corridor_on_plane(y) for y in ys]

    xs_by_y: list[list[float]] = [[] for _ in ys]
    zs_by_y: list[list[float]] = [[] for _ in ys]
    ts_by_y: list[list[float]] = [[] for _ in ys]
    ws_by_y: list[list[float]] = [[] for _ in ys]

    # 按 docs/curve.md 的主链路：y 解析求根，x/z 常加速度解析。
    # 注意：这里先把每个平面的穿越样本收集出来，再按平面分别做统计。
    for c in candidates:
        vy = float(c.v_plus[1])
        a = -0.5 * g
        b = vy

        for j, target_y in enumerate(ys):
            cc = float(y_contact - float(target_y))
            # 求解 y_contact + vy*tau - 0.5*g*tau^2 = target_y 的实根。
            roots = real_roots_of_quadratic(np.array([a, b, cc], dtype=float))
            roots = [r for r in roots if r >= 0.0]
            if not roots:
                continue

            # 可能出现两个交点（上升/下降各一次）。取更大的根作为“后一次穿越”。
            tau = max(roots)
            t_rel = float(bounce.t_rel + tau)
            x = float(bounce.x + float(c.v_plus[0]) * tau + 0.5 * float(c.ax) * tau * tau)
            z = float(bounce.z + float(c.v_plus[2]) * tau + 0.5 * float(c.az) * tau * tau)

            xs_by_y[j].append(x)
            zs_by_y[j].append(z)
            ts_by_y[j].append(t_rel)
            ws_by_y[j].append(float(c.weight))

    results: list[CorridorOnPlane] = []
    for j, target_y in enumerate(ys):
        ws = ws_by_y[j]
        valid = int(len(ws))
        if valid <= 0:
            results.append(_invalid_corridor_on_plane(target_y))
            continue

        # 无条件穿越概率（或无条件权重占比）：子集权重和 / 全体权重和。
        crossing_prob = float(np.sum(np.asarray(ws, dtype=float)) / total_w)
        if crossing_prob <= 0.0:
            results.append(_invalid_corridor_on_plane(target_y))
            continue

        # w 是“穿越条件下”的权重，用于计算条件均值/协方差。
        w = np.asarray(ws, dtype=float)
        w = w / max(float(np.sum(w)), 1e-12)

        x_arr = np.asarray(xs_by_y[j], dtype=float)
        z_arr = np.asarray(zs_by_y[j], dtype=float)
        t_arr = np.asarray(ts_by_y[j], dtype=float)

        # 平均意义下的：“最可能的穿越点/穿越时刻”
        mu_x = float(np.sum(w * x_arr))
        mu_z = float(np.sum(w * z_arr))
        mu_t = float(np.sum(w * t_arr))

        dx = x_arr - mu_x
        dz = z_arr - mu_z
        dt_arr = t_arr - mu_t

        cov_xx = float(np.sum(w * dx * dx))
        cov_zz = float(np.sum(w * dz * dz))
        cov_xz = float(np.sum(w * dx * dz))
        t_var = float(np.sum(w * dt_arr * dt_arr))

        # 可选：用 2 分量表达多峰。注意 components 的 weight 仍在“无条件权重域”。
        components: tuple[CorridorComponent, ...] | None = None
        k = int(getattr(cfg, "corridor_components_k", 1))
        if k >= 2:
            components = _split_components_k2(
                x_arr=x_arr,
                z_arr=z_arr,
                t_arr=t_arr,
                w=w,
                crossing_prob=crossing_prob,
            )

        q_levels = np.asarray(getattr(cfg, "corridor_quantile_levels", ()), dtype=float).reshape(-1)
        q_levels = q_levels[np.isfinite(q_levels)]
        q_levels = q_levels[(q_levels >= 0.0) & (q_levels <= 1.0)]
        if q_levels.size > 0:
            qx = weighted_quantiles_1d(x_arr, w, q_levels)
            qz = weighted_quantiles_1d(z_arr, w, q_levels)
            qt = weighted_quantiles_1d(t_arr, w, q_levels)
            qxz = np.stack([qx, qz], axis=-1)
            results.append(
                CorridorOnPlane(
                    target_y=float(target_y),
                    mu_xz=np.array([mu_x, mu_z], dtype=float),
                    cov_xz=np.array([[cov_xx, cov_xz], [cov_xz, cov_zz]], dtype=float),
                    t_rel_mu=mu_t,
                    t_rel_var=t_var,
                    valid_ratio=float(valid) / float(total),
                    crossing_prob=crossing_prob,
                    is_valid=True,
                    quantile_levels=q_levels,
                    quantiles_xz=qxz,
                    quantiles_t_rel=qt,
                    components=components,
                )
            )
            continue

        results.append(
            CorridorOnPlane(
                target_y=float(target_y),
                mu_xz=np.array([mu_x, mu_z], dtype=float),
                cov_xz=np.array([[cov_xx, cov_xz], [cov_xz, cov_zz]], dtype=float),
                t_rel_mu=mu_t,
                t_rel_var=t_var,
                valid_ratio=float(valid) / float(total),
                crossing_prob=crossing_prob,
                is_valid=True,
                components=components,
            )
        )

    return results


__all__ = [
    "build_corridor_by_time",
    "corridor_on_planes_y",
]
