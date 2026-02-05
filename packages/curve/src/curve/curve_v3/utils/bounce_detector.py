"""反弹前/反弹后分段检测（prefit 冻结）。

`docs/curve.md` 的关键工程约束之一是：
    - 第一段 prefit 只能使用反弹前点，不能被 post 点污染。

现实数据里反弹点附近经常缺帧/遮挡，因此不能依赖“明确看到触地点”来分段。
本模块提供一个轻量、可复现的二态检测器：
    PRE_BOUNCE -> POST_BOUNCE（单向）。

检测思路（与 `docs/curve.md §2.4.4` 对齐，但保持实现尽量简单）：
    1) 竖直速度趋势反转：稳定下降 -> 稳定上升（去抖）。
    2) 近地约束：y 接近球心触地高度 y_contact，或最近窗口内存在近地局部最小。
    3) 触发后输出一个 cut_index：用于把输入序列切成 pre 段（只喂给 prefit）。

说明：
    - 这里的“去抖”使用时间而不是帧数，避免不同 FPS 下行为漂移。
    - 本模块只依赖 NumPy 与 `CurveV3Config` 的若干字段，不依赖 core/prefit，
      用于降低耦合。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import CurveV3Config


def _estimate_vy_end(ts: np.ndarray, ys: np.ndarray) -> float:
    """估计末端竖直速度（dy/dt）。

    说明：
        - 这里刻意不用“单帧差分”，而用一个短差分窗口抑制抖动。
        - 这是工程上的速度估计启发式：足够稳定且很便宜。

    Args:
        ts: 相对时间序列，shape=(N,)。
        ys: 高度序列，shape=(N,)。

    Returns:
        vy_end（m/s）。
    """

    ts = np.asarray(ts, dtype=float).reshape(-1)
    ys = np.asarray(ys, dtype=float).reshape(-1)

    n = int(ts.size)
    if n < 2:
        return 0.0

    # 优先用 3 点跨距做差分，避免单点噪声。
    if n >= 3:
        i0 = n - 3
        i1 = n - 1
    else:
        i0 = n - 2
        i1 = n - 1

    dt = float(ts[i1] - ts[i0])
    if dt <= 1e-9:
        return 0.0
    return float((ys[i1] - ys[i0]) / dt)


def _find_last_gap_index(ts: np.ndarray, *, gap_mult: float) -> int | None:
    """在时间序列中寻找“最后一个明显缺口”的左端索引。

    说明：
        - 若返回 k，表示在 ts[k] 与 ts[k+1] 之间存在 gap。
        - gap 的判定基于 dts 与其中位数的倍数阈值，适配不同 FPS。

    Args:
        ts: 相对时间序列，shape=(N,)。
        gap_mult: gap 阈值倍数；thr = gap_mult * median(diff(ts)).

    Returns:
        k（gap 左端索引），或 None。
    """

    ts = np.asarray(ts, dtype=float).reshape(-1)
    if ts.size < 4:
        return None

    dts = np.diff(ts)
    if dts.size < 3:
        return None

    # 仅用正的、有限的 dt 估计 nominal FPS。
    dts_ok = dts[np.isfinite(dts) & (dts > 0.0)]
    if dts_ok.size < 3:
        return None

    med = float(np.median(dts_ok))
    if not np.isfinite(med) or med <= 1e-6:
        return None

    thr = float(gap_mult) * med
    idx = np.where(dts > thr)[0]
    if idx.size == 0:
        return None
    return int(idx[-1])


def _predict_contact_time_from_vertical_fit(
    ts: np.ndarray,
    ys: np.ndarray,
    *,
    y_contact: float,
    g: float,
) -> float | None:
    """用固定重力的竖直模型拟合并预测触地时刻。

    拟合形式（取 t_ref=ts[-1], tau=t-t_ref<=0）：

        y(tau) = y0 + vy*tau - 0.5*g*tau^2

    移项得到线性最小二乘：

        y(tau) + 0.5*g*tau^2 = y0 + vy*tau

    然后解触地根（t>0）：

        y0 + vy*t - 0.5*g*t^2 = y_contact

    说明：
        - 这里的目标不是“精确 bounce 时刻”，而是用于判断 bounce 是否大概率
          落在一个可见性缺口（gap）内，从而做“安全切分/冻结”。

    Args:
        ts: 时间序列（秒），shape=(N,)。应当是 gap 左侧的观测点。
        ys: 高度序列（米），shape=(N,)。
        y_contact: 触地球心高度（米）。
        g: 重力加速度标量（m/s^2），应为正。

    Returns:
        预测触地时刻 tb_pred（秒），或 None。
    """

    ts = np.asarray(ts, dtype=float).reshape(-1)
    ys = np.asarray(ys, dtype=float).reshape(-1)
    if ts.size < 4 or ys.size != ts.size:
        return None

    g = float(g)
    if not np.isfinite(g) or g <= 1e-6:
        return None

    t_ref = float(ts[-1])
    tau = ts - t_ref

    y_lin = ys + 0.5 * g * tau * tau
    a = np.stack([np.ones_like(tau), tau], axis=1)

    try:
        sol, *_ = np.linalg.lstsq(a, y_lin, rcond=None)
    except np.linalg.LinAlgError:
        return None

    y0 = float(sol[0])
    vy = float(sol[1])

    # 解：y0 + vy*t - 0.5*g*t^2 = y_contact
    # <=> 0.5*g*t^2 - vy*t + (y_contact - y0) = 0
    qa = 0.5 * g
    qb = -vy
    qc = float(y_contact) - y0

    disc = qb * qb - 4.0 * qa * qc
    if not np.isfinite(disc) or disc < 0.0:
        return None

    s = float(np.sqrt(disc))
    t1 = (-qb - s) / (2.0 * qa)
    t2 = (-qb + s) / (2.0 * qa)
    cand = [float(t) for t in (t1, t2) if np.isfinite(t) and t > 0.0]
    if not cand:
        return None

    dt_contact = float(min(cand))
    return float(t_ref + dt_contact)


@dataclass
class BounceTransitionDetector:
    """检测 PRE->POST 的一阶状态机。"""

    cfg: CurveV3Config

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # 单向触发：一旦产出 cut_index，就冻结并保持返回稳定值。
        self._frozen_cut_index: int | None = None
        self._frozen_reason: str | None = None

        # 是否已经确认出现“稳定下降”。只有确认下降后才开始统计“稳定上升”。
        self._down_confirmed = False

        # 连续趋势持续的时间（秒）：用于去抖。
        self._down_time_s = 0.0
        self._up_time_s = 0.0

        # 上一次更新时间戳。
        self._last_t: float | None = None

    def find_cut_index(
        self,
        *,
        ts: np.ndarray,
        ys: np.ndarray,
        y_contact: float,
    ) -> tuple[int | None, str | None]:
        """尝试检测分段点并返回 cut_index。

        Args:
            ts: 相对时间序列（秒），shape=(N,)。
            ys: 高度序列（米，球心高度），shape=(N,)。
            y_contact: 触地球心高度（米）。

        Returns:
            (cut_index, reason)
            - cut_index: 若触发返回分割索引（pre 段末端索引，包含该点）；否则 None。
            - reason: 触发原因码（用于日志/诊断）；未触发时为 None。
        """

        ts = np.asarray(ts, dtype=float).reshape(-1)
        ys = np.asarray(ys, dtype=float).reshape(-1)

        if self._frozen_cut_index is not None:
            return int(self._frozen_cut_index), self._frozen_reason

        n = int(ts.size)
        if n < 3:
            return None, None

        # 配置项：要求 cfg 字段完整；若缺字段应当直接报错（避免悄悄走默认值）。
        v_down = float(self.cfg.bounce_detector_v_down_mps)
        v_up = float(self.cfg.bounce_detector_v_up_mps)
        eps_y = float(self.cfg.bounce_detector_eps_y_m)

        down_need = float(self.cfg.bounce_detector_down_debounce_s)
        up_need = float(self.cfg.bounce_detector_up_debounce_s)

        l = int(self.cfg.bounce_detector_local_min_window)
        l = int(max(l, 3))

        min_points = int(self.cfg.bounce_detector_min_points)
        if n < max(min_points, 3):
            return None, None

        # region 网球近地点太高了, y_obs_min>0.2m, 导致 near_ground 永不成立，
        # 额外的“安全切分”路径：当反弹附近不可见（例如 y<0.2m 不输出）时，
        # near_ground 往往永远不成立，此时不能等待 vy_flip_and_near_ground。
        # 这里用“最后一个可见性缺口（gap）+ 竖直模型预测 tb 落在 gap 内”来
        # 触发冻结，并把 cut 放在 gap 左侧最后一点，从工程角度阻断 prefit 污染。
        enable_gap_freeze = bool(self.cfg.bounce_detector_gap_freeze_enabled)
        if enable_gap_freeze:
            gap_mult = float(self.cfg.bounce_detector_gap_mult)
            gap_tb_margin_s = float(self.cfg.bounce_detector_gap_tb_margin_s)
            gap_fit_points = int(self.cfg.bounce_detector_gap_fit_points)
            gap_fit_points = int(max(gap_fit_points, 4))

            k = _find_last_gap_index(ts, gap_mult=gap_mult)
            if k is not None and 0 <= int(k) <= (n - 2):
                k = int(k)
                # 用 gap 左侧的末端窗口拟合，减少早期点对拟合的干扰。
                start = int(max(0, (k + 1) - gap_fit_points))
                g = float(self.cfg.gravity)
                tb_pred = _predict_contact_time_from_vertical_fit(
                    ts[start : k + 1],
                    ys[start : k + 1],
                    y_contact=float(y_contact),
                    g=g,
                )

                if tb_pred is not None:
                    left = float(ts[k]) - gap_tb_margin_s
                    right = float(ts[k + 1]) + gap_tb_margin_s
                    if left <= float(tb_pred) <= right:
                        self._frozen_cut_index = int(k)
                        self._frozen_reason = "visibility_gap_freeze"
                        return int(self._frozen_cut_index), self._frozen_reason
        # endregion

        # region 计算时间增量 dt
        t_now = float(ts[-1])
        if self._last_t is None:
            self._last_t = t_now
            return None, None
        dt = float(t_now - float(self._last_t))
        self._last_t = t_now
        if not np.isfinite(dt) or dt <= 0.0:
            return None, None
        # endregion

        # region 近地判定：当前高度接近触地高度，或最近窗口内存在近地局部最小。
        vy = _estimate_vy_end(ts, ys)
        y_last = float(ys[-1])
        tail = ys[-l:] if n >= l else ys
        y_min = float(np.min(tail))
        near_ground = (y_last <= float(y_contact) + eps_y) or (y_min <= float(y_contact) + eps_y)

        if vy < -abs(v_down):
            self._down_time_s += dt
            self._up_time_s = 0.0
        elif self._down_confirmed and vy > abs(v_up):
            self._up_time_s += dt
        else:
            # 趋势不明确时，清空累计；避免噪声下误触发。
            self._down_time_s = 0.0
            self._up_time_s = 0.0

        if (not self._down_confirmed) and self._down_time_s >= down_need:
            self._down_confirmed = True
            # 转入“等待上升”时把 up_time 重置。
            self._up_time_s = 0.0

        if self._down_confirmed and self._up_time_s >= up_need and near_ground:
            # 触发：用最近窗口的局部最小作为切分点，尽量接近真实反弹点。
            # 返回的是“pre 段末端索引”，因此包含该最小点。
            if n >= l:
                local = ys[-l:]
                local_arg = int(np.argmin(local))
                cut = int(n - l + local_arg)
            else:
                cut = int(np.argmin(ys))

            cut = int(min(max(cut, 0), n - 1))
            self._frozen_cut_index = int(cut)
            self._frozen_reason = "vy_flip_and_near_ground"
            return int(self._frozen_cut_index), self._frozen_reason
        # endregion

        return None, None
