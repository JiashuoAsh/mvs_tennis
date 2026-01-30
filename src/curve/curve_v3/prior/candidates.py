"""curve_v3 第一阶段（prior）候选生成。

说明：
        - `CurvePredictorV3` 是高层编排器；候选生成属于可独立测试的纯逻辑，
            因此从 `core.py` 拆出以降低文件体量与耦合。
        - 这里保持实现“最小可用”：只负责把 (e, k_t, φ) 网格展开成候选并赋权。
            其中 φ 在实现中用 `kt_angle_bins_rad` 表示（绕地面法向旋转切向速度）。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from curve_v3.config import CurveV3Config
from curve_v3.prior.models import PriorModel, features_from_v_minus
from curve_v3.types import BounceEvent, Candidate


class _OnlinePriorProtocol(Protocol):
    """OnlinePriorWeights 的最小依赖接口。

    说明：
        这样 `prior_candidates` 就不需要在运行时 import `curve_v3.prior.online_prior`，
        避免不必要的依赖链与潜在循环引用。
    """

    def apply_to_base_weights(self, *, base_weights_ekt: np.ndarray) -> np.ndarray:  # noqa: D401
        """把在线沉淀权重作用到 (e*kt) 的 base 权重上，并展开到全维。"""

        ...


def build_prior_candidates(
    *,
    bounce: BounceEvent,
    cfg: CurveV3Config,
    prior_model: PriorModel | None,
    online_prior: _OnlinePriorProtocol | None,
    e_bins_override: list[float] | None = None,
    kt_bins_override: list[float] | None = None,
) -> list[Candidate]:
    """生成反弹后的离散候选集合。

    Args:
        bounce: 第一段 prefit 估计得到的反弹事件（含 v^-）。
        cfg: 配置。
        prior_model: 可选的先验模型，仅输出 (e*kt) 的 base 权重。
        online_prior: 可选的在线沉淀权重池（把 base 权重扩展到 e*kt*φ）。
        e_bins_override: 可选覆盖 e_bins（用于二阶段局部细化）。
            - 为 None 时使用 cfg.e_bins。
        kt_bins_override: 可选覆盖 kt_bins（用于二阶段局部细化）。
            - 为 None 时使用 cfg.kt_bins。
    Returns:
        覆盖 (e, kt, φ) 网格的一组候选。

    说明：
        - 当提供 e/kt override 时，通常意味着“局部细化”阶段；该阶段的 bins
          与在线权重池的维度未必一致，因此本函数会保守地跳过 online_prior，
          以避免把不匹配的权重错误地乘到候选上。
    """

    v_minus = np.asarray(bounce.v_minus, dtype=float)

    # 法向/切向分解：默认地面法向为 (0,1,0)，但允许通过配置覆盖。
    # 说明：这里使用单位法向 n_hat，把速度投影到法向/切向（切平面）上。
    n_hat = np.asarray(getattr(cfg, "ground_normal", (0.0, 1.0, 0.0)), dtype=float).reshape(3)
    n_norm = float(np.linalg.norm(n_hat))
    n_hat = n_hat / n_norm if n_norm > 1e-9 else np.array((0.0, 1.0, 0.0), dtype=float)

    v_n = float(np.dot(v_minus, n_hat)) * n_hat
    v_t = v_minus - v_n

    def rotate_in_tangent_plane(v: np.ndarray, axis_hat: np.ndarray, angle_rad: float) -> np.ndarray:
        """绕法向轴旋转切向向量（Rodrigues 公式）。

        Args:
            v: 待旋转的向量（通常是切向速度），shape=(3,)。
            axis_hat: 单位旋转轴（法向），shape=(3,)。
            angle_rad: 旋转角（弧度）。

        Returns:
            旋转后的向量，shape=(3,)。
        """

        ang = float(angle_rad)
        c = float(np.cos(ang))
        s = float(np.sin(ang))
        v = np.asarray(v, dtype=float)
        k = np.asarray(axis_hat, dtype=float)
        return v * c + np.cross(k, v) * s + k * float(np.dot(k, v)) * (1.0 - c)

    e_bins = [float(x) for x in (e_bins_override if e_bins_override is not None else cfg.e_bins)]
    kt_bins = [float(x) for x in (kt_bins_override if kt_bins_override is not None else cfg.kt_bins)]
    angle_bins_raw = getattr(cfg, "kt_angle_bins_rad", None)
    angle_bins = [float(a) for a in angle_bins_raw] if angle_bins_raw else [0.0]
    if not e_bins or not kt_bins:
        return []

    # prior_model 只输出 (e,kt) 的权重；若存在 angle_bins，则将其均匀分摊到角度维度。
    m = int(len(e_bins) * len(kt_bins))
    weights: np.ndarray | None = None
    if prior_model is not None:
        try:
            features = features_from_v_minus(v_minus)
            weights = np.asarray(
                prior_model.predict_candidate_weights(
                    features,
                    e_bins=e_bins,
                    kt_bins=kt_bins,
                ),
                dtype=float,
            ).reshape(-1)
        except Exception:
            weights = None

    if weights is None or weights.size != m or float(np.sum(weights)) <= 0:
        weights = np.full((m,), 1.0 / float(m), dtype=float)
    else:
        weights = np.maximum(weights, 0.0)
        s = float(np.sum(weights))
        if s <= 0.0:
            weights = np.full((m,), 1.0 / float(m), dtype=float)
        else:
            weights = weights / s

    # 在线沉淀：把 (e*kt) base prior 乘上沉淀权重并展开到 (e*kt*angle)。
    weights_full: np.ndarray | None = None
    # 说明：当使用 override bins 做局部细化时，在线沉淀权重通常与 bins 不匹配；
    # 这里做一个保守防御：有 override 时不应用在线沉淀。
    if online_prior is not None and e_bins_override is None and kt_bins_override is None:
        try:
            weights_full = online_prior.apply_to_base_weights(base_weights_ekt=weights)
        except Exception:
            weights_full = None

    candidates: list[Candidate] = []

    e_min, e_max = getattr(cfg, "e_range", (1e-3, 1.05))
    e_min = float(e_min)
    e_max = float(e_max)
    if e_max < e_min:
        e_min, e_max = e_max, e_min

    kt_min, kt_max = getattr(cfg, "kt_range", (-1.2, 1.2))
    kt_min = float(kt_min)
    kt_max = float(kt_max)
    if kt_max < kt_min:
        kt_min, kt_max = kt_max, kt_min

    w_idx = 0
    w_full_idx = 0
    for e in e_bins:
        for kt in kt_bins:
            e_clamped = float(min(max(float(e), e_min), e_max))
            kt_clamped = float(min(max(float(kt), kt_min), kt_max))

            # 默认：把 (e*kt) 的 base weight 均匀分摊到 angle 维度；
            # 若启用在线沉淀，则直接使用沉淀后的全维权重。
            base_w = float(weights[w_idx])
            per_w = base_w / float(len(angle_bins))
            for ang in angle_bins:
                w_use = float(weights_full[w_full_idx]) if weights_full is not None else float(per_w)
                w_full_idx += 1

                # 切向偏转角：绕法向轴在切平面内旋转（对应 `docs/curve.md` 的 φ）。
                v_t_rot = rotate_in_tangent_plane(v_t, n_hat, float(ang))

                v_plus = -e_clamped * v_n + kt_clamped * v_t_rot
                candidates.append(
                    Candidate(
                        e=e_clamped,
                        kt=kt_clamped,
                        weight=float(w_use),
                        v_plus=np.asarray(v_plus, dtype=float),
                        kt_angle_rad=float(ang),
                        ax=0.0,
                        az=0.0,
                    )
                )
            w_idx += 1

    return candidates
