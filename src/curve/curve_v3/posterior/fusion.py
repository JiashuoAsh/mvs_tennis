"""curve_v3 v1.1 融合相关的纯逻辑。

说明：
    `CurvePredictorV3` 内部既要维护观测/状态，又要实现 v1.1 的
    “每候选做 MAP 校正再打分”的融合步骤。

    为降低 `core.py` 体量并提升可测试性，这里把“纯计算”的部分抽出：
    - 计算候选的原始轨迹残差（诊断用途）。
    - 对每个候选做 MAP 后验拟合，得到 J_post，并用 log-sum-exp + beta 退火
      更新权重、选择最佳分支。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from curve_v3.config import CurveV3Config
from curve_v3.dynamics import propagate_post_bounce_state
from curve_v3.posterior import fit_posterior_map_for_candidate
from curve_v3.low_snr.types import WindowDecisions
from curve_v3.types import BallObservation, BounceEvent, Candidate, PosteriorState


def candidate_costs(
    *,
    bounce: BounceEvent,
    candidates: Sequence[Candidate],
    post_points: Sequence[BallObservation],
    time_base_abs: float | None,
    low_snr: WindowDecisions | None = None,
    cfg: CurveV3Config,
) -> np.ndarray:
    """计算每个候选的归一化 SSE（诊断用途）。

    说明：
        v1.1 的融合流程使用“每候选 posterior MAP”再评分；这里保留一个
        诊断函数：直接用 prior 轨迹残差评分（不做每候选校正）。

    Returns:
        shape=(M,) 的 costs 数组。
    """

    if time_base_abs is None:
        return np.zeros(len(candidates), dtype=float)

    sigma = float(cfg.weight_sigma_m)
    sigma2 = max(sigma * sigma, 1e-9)

    costs: list[float] = []
    for c in candidates:
        sse = 0.0
        for p in post_points:
            t_rel = float(p.t - time_base_abs)
            tau = t_rel - float(bounce.t_rel)
            if tau <= 0:
                continue

            pos, _ = propagate_post_bounce_state(
                bounce=bounce,
                candidate=c,
                tau=float(tau),
                cfg=cfg,
            )

            dx = float(p.x - float(pos[0]))
            dy = float(p.y - float(pos[1]))
            dz = float(p.z - float(pos[2]))

            # 低 SNR：诊断评分也尊重 IGNORE_AXIS（其余模式不影响 SSE 口径）。
            if low_snr is not None and str(low_snr.x.mode) == "IGNORE_AXIS":
                dx = 0.0
            if low_snr is not None and str(low_snr.y.mode) == "IGNORE_AXIS":
                dy = 0.0
            if low_snr is not None and str(low_snr.z.mode) == "IGNORE_AXIS":
                dz = 0.0

            sse += dx * dx + dy * dy + dz * dz

        costs.append(float(sse / sigma2))

    return np.asarray(costs, dtype=float)


def reweight_candidates_and_select_best(
    *,
    bounce: BounceEvent,
    candidates: Sequence[Candidate],
    post_points: Sequence[BallObservation],
    time_base_abs: float | None,
    low_snr: WindowDecisions | None = None,
    cfg: CurveV3Config,
) -> tuple[list[Candidate], Candidate | None, int | None, PosteriorState | None]:
    """v1.1：逐候选 MAP 校正 -> 打分 -> 重赋权 -> 选最佳。"""

    if time_base_abs is None:
        return list(candidates), None, None, None

    post_states: list[PosteriorState | None] = []
    costs: list[float] = []
    for c in candidates:
        out = fit_posterior_map_for_candidate(
            bounce=bounce,
            post_points=post_points,
            candidate=c,
            time_base_abs=time_base_abs,
            low_snr=low_snr,
            cfg=cfg,
        )
        if out is None:
            post_states.append(None)
            costs.append(float("inf"))
        else:
            st, j = out
            post_states.append(st)
            costs.append(float(j))

    costs_arr = np.asarray(costs, dtype=float)
    if costs_arr.size == 0:
        return list(candidates), None, None, None

    # 结合 prior 权重与 likelihood 权重（log-sum-exp 保持数值稳定）。
    prior_w = np.array([float(c.weight) for c in candidates], dtype=float)
    prior_w = np.maximum(prior_w, 1e-12)

    beta = float(cfg.candidate_likelihood_beta(len(post_points)))
    logw = np.log(prior_w) + (-0.5 * beta) * costs_arr

    m = float(np.max(logw))
    if not np.isfinite(m):
        w = np.full_like(prior_w, 1.0 / float(len(candidates)))
    else:
        expw = np.exp(logw - m)
        s = float(np.sum(expw))
        if s <= 0.0 or not np.isfinite(s):
            w = np.full_like(expw, 1.0 / float(len(candidates)))
        else:
            w = expw / s

    updated: list[Candidate] = []
    for c, wi in zip(candidates, w):
        updated.append(
            Candidate(
                e=c.e,
                kt=c.kt,
                weight=float(wi),
                v_plus=np.asarray(c.v_plus, dtype=float),
                kt_angle_rad=float(getattr(c, "kt_angle_rad", 0.0)),
                ax=float(c.ax),
                az=float(c.az),
            )
        )

    # 选 best：先按最小 J_post，平局再按更新后权重最大。
    min_cost = float(np.min(costs_arr))
    tie = np.where(np.isclose(costs_arr, min_cost, atol=1e-9, rtol=0.0))[0]
    if tie.size <= 1:
        best_idx = int(np.argmin(costs_arr))
    else:
        best_idx = int(tie[int(np.argmax(w[tie]))])

    best = updated[best_idx]
    best_post = post_states[best_idx]
    return updated, best, best_idx, best_post


__all__ = [
    "candidate_costs",
    "reweight_candidates_and_select_best",
]
