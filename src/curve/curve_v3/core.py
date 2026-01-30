"""curve_v3 的核心编排实现。

设计目标（来自 `curve.md`）：
    - 第一阶段（prior）：基于反弹前信息生成多候选，并输出不确定性走廊（corridor）。
    - 第二阶段（posterior）：当出现反弹后少量观测点（N<=5）时，快速校正并融合输出。

坐标约定（与 `legacy/curve2.py` 保持一致）：
    - x：向右为正。
    - z：向前为正。
    - y：向上为正。
    - 重力方向沿 -y。

实现原则：
    - 依赖尽量少（NumPy + 标准库）。
    - API 尽量稳定，legacy 兼容由 `curve_v3.legacy` 提供。
"""

from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np

from curve_v3.utils import BounceTransitionDetector
from curve_v3.low_snr import LowSnrPolicyParams, analyze_window, weights_from_conf
from curve_v3.corridor import build_corridor_by_time, corridor_on_planes_y as compute_corridor_on_planes_y
from curve_v3.config import CurveV3Config
from curve_v3.utils import polyval
from curve_v3.posterior import (
    fit_posterior_fused_map,
    inject_posterior_anchor,
    prior_nominal_state,
)
from curve_v3.posterior.fusion import candidate_costs as compute_candidate_costs
from curve_v3.posterior.fusion import reweight_candidates_and_select_best
from curve_v3.prior import PriorModel, build_prior_candidates, estimate_bounce_event_from_prefit
from curve_v3.prior import maybe_init_online_prior, maybe_update_online_prior
from curve_v3.utils import default_logger
from curve_v3.types import (
    BallObservation,
    BounceEvent,
    Candidate,
    CorridorByTime,
    CorridorOnPlane,
    FusionInfo,
    LowSnrAxisModes,
    LowSnrInfo,
    PrefitFreezeInfo,
    PosteriorState,
)


class CurvePredictorV3:
    """两阶段反弹预测器（prior + posterior）。

    说明：
        - 这是 v3 的新 API；legacy 兼容接口由 `curve_v3.legacy.Curve` 提供。
        - 内部会维护反弹前拟合（用于推断反弹时刻/位置/入射速度），以及反弹后的候选/后验。
    """

    # 时间递增权重：越新的点权重越大（指数增长）。
    # 说明：这是历史版本的工程启发式，配合低 SNR 的 conf 权重能更稳地抑制早期噪声。
    _FIT_X_WEIGHT = 1.01
    _FIT_Y_WEIGHT = 1.01
    _FIT_Z_WEIGHT = 1.01

    def __init__(
        self,
        config: CurveV3Config | None = None,
        logger: logging.Logger | None = None,
        prior_model: PriorModel | None = None,
    ) -> None:
        self._cfg = config or CurveV3Config()
        self._logger = logger or default_logger()
        self._prior_model = prior_model

        # 在线权重池属于“跨回合”的经验状态：不应被 reset() 清空。
        self._online_prior = maybe_init_online_prior(cfg=self._cfg, logger=self._logger)
        self.reset()

    def reset(self) -> None:
        self._time_base_abs: float | None = None

        self._xs: list[float] = []
        self._ys: list[float] = []
        self._zs: list[float] = []
        self._ts_rel: list[float] = []

        # 可选的点级置信度（用于低 SNR 权重建模）；None 表示未知。
        self._confs: list[float | None] = []

        self._x_ws: list[float] = []
        self._y_ws: list[float] = []
        self._z_ws: list[float] = []

        # 时间递增权重（仅随点序号递推，不包含 conf 权重）
        self._w_time_x: float = 1.0
        self._w_time_y: float = 1.0
        self._w_time_z: float = 1.0

        self._pre_coeffs: dict[str, np.ndarray] | None = None
        self._bounce_event: BounceEvent | None = None

        self._candidates: list[Candidate] = []
        self._best_candidate: Candidate | None = None
        self._nominal_candidate_id: int | None = None
        self._posterior_anchor_used: bool = False
        self._corridor_by_time: CorridorByTime | None = None

        self._post_points: list[BallObservation] = []
        self._posterior_state: PosteriorState | None = None

        # 一旦确认进入反弹后段，就冻结 prefit/bounce_event，避免后续 post 点
        # 参与 prefit 造成 t_land 漂移。
        self._prefit_frozen: bool = False
        self._prefit_cut_index: int | None = None

        # 分段检测器：用于决定何时冻结 prefit。
        self._bounce_detector = BounceTransitionDetector(cfg=self._cfg)
        self._prefit_freeze_reason: str | None = None
        self._prefit_freeze_t_rel: float | None = None

        # 低 SNR 诊断信息：分别记录 prefit 与 posterior 最近一次的 mode 标签。
        self._low_snr_prefit: LowSnrAxisModes | None = None
        self._low_snr_posterior: LowSnrAxisModes | None = None

    @property
    def time_base_abs(self) -> float | None:
        return self._time_base_abs

    def add_observation(self, obs: BallObservation) -> None:
        """添加一帧观测并更新内部模型。

        Args:
            obs: 带绝对时间戳的观测。
        """

        if self._time_base_abs is None:
            self._time_base_abs = obs.t

        t_rel = obs.t - self._time_base_abs

        self._xs.append(float(obs.x))
        self._ys.append(float(obs.y))
        self._zs.append(float(obs.z))
        self._ts_rel.append(float(t_rel))
        self._confs.append(getattr(obs, "conf", None))

        # 说明：
        # - 旧逻辑只做“时间递增权重”（越新越重要）。
        # - 低 SNR 方案引入 conf 后，将其作为额外的“观测可信度权重”乘子。
        # - 为避免在未显式开启 low_snr 时改变既有行为，这里仅在 cfg.low_snr_enabled
        #   且 obs.conf 提供时才使用 conf。
        # 重要：时间权重必须与 conf 权重解耦。
        # 否则在 conf 存在时，若用上一帧“总权重”递推，会把 conf 权重重复乘入，
        # 导致权重以 (conf/σ0^2)^n 级联爆炸并最终溢出为 inf。
        if self._x_ws:
            self._w_time_x *= self._FIT_X_WEIGHT
            self._w_time_y *= self._FIT_Y_WEIGHT
            self._w_time_z *= self._FIT_Z_WEIGHT

        w_time_x = self._w_time_x
        w_time_y = self._w_time_y
        w_time_z = self._w_time_z

        if bool(getattr(self._cfg, "low_snr_enabled", False)) and obs.conf is not None:
            cmin = float(getattr(self._cfg, "low_snr_conf_cmin", 0.05))
            sx0 = float(getattr(self._cfg, "low_snr_sigma_x0_m", 0.15))
            sy0 = float(getattr(self._cfg, "low_snr_sigma_y0_m", 0.15))
            sz0 = float(getattr(self._cfg, "low_snr_sigma_z0_m", 0.15))

            # w = 1/σ^2，σ = σ0/sqrt(conf) => w = conf/σ0^2
            # 复用 low_snr 包的实现，保证 prefit/posterior 的权重口径一致。
            conf_val = float(obs.conf)
            wx = float(weights_from_conf([conf_val], sigma0=sx0, c_min=cmin)[0])
            wy = float(weights_from_conf([conf_val], sigma0=sy0, c_min=cmin)[0])
            wz = float(weights_from_conf([conf_val], sigma0=sz0, c_min=cmin)[0])
        else:
            wx, wy, wz = 1.0, 1.0, 1.0

        self._x_ws.append(float(w_time_x) * float(wx))
        self._y_ws.append(float(w_time_y) * float(wy))
        self._z_ws.append(float(w_time_z) * float(wz))

        self._update_models()

    def _update_models(self) -> None:
        """用累计观测更新 prefit / prior / posterior / corridor。

        设计说明：
            这是在线入口 add_observation() 的主更新点。为了保持“高内聚、低耦合”，
            这里仅做流程编排，具体子步骤拆分到若干私有方法中。
        """

        if len(self._ts_rel) < 5:
            return

        t = np.asarray(self._ts_rel, dtype=float)
        xs = np.asarray(self._xs, dtype=float)
        ys = np.asarray(self._ys, dtype=float)
        zs = np.asarray(self._zs, dtype=float)

        xw = np.asarray(self._x_ws, dtype=float)
        yw = np.asarray(self._y_ws, dtype=float)
        zw = np.asarray(self._z_ws, dtype=float)

        self._maybe_update_prefit_and_bounce_event(t=t, xs=xs, ys=ys, zs=zs, xw=xw, yw=yw, zw=zw)

        if self._pre_coeffs is None or self._bounce_event is None:
            return

        t_land = float(self._bounce_event.t_rel)
        if t_land <= 0.0:
            self._clear_prefit_state()
            return

        self._post_points = self._extract_post_points_after_land_time(t_land)
        self._update_post_models_and_corridor()

    def _low_snr_params(self) -> LowSnrPolicyParams:
        """从配置构造低 SNR 策略参数。

        说明：
            prefit 与 posterior 使用同一组阈值口径；集中在这里构造，避免重复
            代码散落导致的“改一处忘一处”。
        """

        return LowSnrPolicyParams(
            delta_k_freeze_a=float(getattr(self._cfg, "low_snr_delta_k_freeze_a", 4.0)),
            delta_k_strong_v=float(getattr(self._cfg, "low_snr_delta_k_strong_v", 2.0)),
            delta_k_ignore=float(getattr(self._cfg, "low_snr_delta_k_ignore", 1.0)),
            min_points_for_v=int(getattr(self._cfg, "low_snr_min_points_for_v", 3)),
        )

    def _clear_prefit_state(self) -> None:
        """清理 prefit / bounce_event 相关状态。"""

        self._pre_coeffs = None
        self._bounce_event = None

    def _maybe_update_prefit_and_bounce_event(
        self,
        *,
        t: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        zs: np.ndarray,
        xw: np.ndarray,
        yw: np.ndarray,
        zw: np.ndarray,
    ) -> None:
        """更新反弹前拟合（prefit）与反弹事件（bounce_event）。

        目前设计：prefit 只应该由“反弹前点”驱动。

        真实数据里反弹附近可能缺失，如果把 post 点也喂进 prefit，触地时刻
        会被错误推迟（max root 漂移），进而导致 posterior 的 tau 变大、拟合变差。

        因此这里使用一个可复现的二态检测器（见 `curve_v3.utils.bounce_detector`）来检测
        PRE_BOUNCE -> POST_BOUNCE 的切换；一旦触发就设置 cut_index，并在成功 prefit 后冻结
        prefit/bounce_event，避免后续 post 点污染 t_land。
        """

        if self._prefit_frozen and self._pre_coeffs is not None and self._bounce_event is not None:
            return

        t_fit = t
        xs_fit = xs
        ys_fit = ys
        zs_fit = zs
        xw_fit = xw
        yw_fit = yw
        zw_fit = zw

        # region 检测 prefit/post 切分点    
        if self._prefit_cut_index is None:
            cut, reason = self._bounce_detector.find_cut_index(
                ts=t,
                ys=ys,
                y_contact=float(self._cfg.bounce_contact_y()),
            )
            if cut is not None and (int(cut) + 1) >= 5:
                self._prefit_cut_index = int(cut)
                self._prefit_freeze_reason = str(reason) if reason is not None else None
                # 记录“触发冻结”的时刻（而不是 cut 点时刻）。
                # 这样更贴近 docs/curve.md §2.4.4 的契约：t_freeze=首次满足触发条件的那一帧。
                self._prefit_freeze_t_rel = float(t[-1])
        # endregion

        # region 根据 cut_index 划分 prefit 段 [t_fit, xs_fit, ys_fit, zs_fit, xw_fit, yw_fit, zw_fit] 重新划分
        if self._prefit_cut_index is not None:
            k = int(self._prefit_cut_index) + 1
            if k >= 5:
                t_fit = t[:k]
                xs_fit = xs[:k]
                ys_fit = ys[:k]
                zs_fit = zs[:k]
                xw_fit = xw[:k]
                yw_fit = yw[:k]
                zw_fit = zw[:k]
        # endregion

        prefit_low_snr = None
        if bool(getattr(self._cfg, "low_snr_enabled", False)):
            tail_n = int(self._cfg.low_snr_prefit_window_points)
            tail_n = int(max(tail_n, 3))
            tail_start = max(int(t_fit.size) - tail_n, 0)
            prefit_low_snr = analyze_window(
                xs=xs_fit[tail_start:],
                ys=ys_fit[tail_start:],
                zs=zs_fit[tail_start:],
                confs=self._confs[: int(t_fit.size)][tail_start:],
                sigma_x0=float(getattr(self._cfg, "low_snr_sigma_x0_m", 0.15)),
                sigma_y0=float(getattr(self._cfg, "low_snr_sigma_y0_m", 0.15)),
                sigma_z0=float(getattr(self._cfg, "low_snr_sigma_z0_m", 0.15)),
                c_min=float(getattr(self._cfg, "low_snr_conf_cmin", 0.05)),
                params=self._low_snr_params(),
                disallow_ignore_y=bool(getattr(self._cfg, "low_snr_disallow_ignore_y", True)),
            )

        pre = estimate_bounce_event_from_prefit(
            t_rel=t_fit,
            xs=xs_fit,
            ys=ys_fit,
            zs=zs_fit,
            xw=xw_fit,
            yw=yw_fit,
            zw=zw_fit,
            # 低 SNR：在与 x/z 拟合同一窗口上做判别与退化动作。
            low_snr=prefit_low_snr,
            v_prior=(np.asarray(self._bounce_event.v_minus, dtype=float) if self._bounce_event is not None else None),
            cfg=self._cfg,
        )
        if pre is None:
            self._clear_prefit_state()
            return

        self._pre_coeffs, self._bounce_event = pre

        # 记录 prefit 阶段的低 SNR mode（用于上游诊断）。
        if prefit_low_snr is not None:
            self._low_snr_prefit = LowSnrAxisModes(
                mode_x=prefit_low_snr.x.mode,
                mode_y=prefit_low_snr.y.mode,
                mode_z=prefit_low_snr.z.mode,
            )
        else:
            self._low_snr_prefit = None

        t_land = float(self._bounce_event.t_rel)
        if t_land <= 0.0:
            self._clear_prefit_state()
            return

        # 若分段检测器给出了 cut_index，则冻结 prefit/bounce_event。
        # 说明：冻结后不会再更新 prefit，以避免 post 点污染时间基准。
        if self._prefit_cut_index is not None:
            self._prefit_frozen = True

    def _extract_post_points_after_land_time(self, t_land: float) -> list[BallObservation]:
        """按预测触地时刻把观测点划分出 post 段。"""

        post: list[BallObservation] = []
        for i, ti in enumerate(self._ts_rel):
            if float(ti) > float(t_land):
                post.append(
                    BallObservation(
                        x=float(self._xs[i]),
                        y=float(self._ys[i]),
                        z=float(self._zs[i]),
                        t=float((self._time_base_abs or 0.0) + float(ti)),
                        conf=self._confs[i] if i < len(self._confs) else None,
                    )
                )
        return post

    def _update_post_models_and_corridor(self) -> None:
        """更新 prior 候选、posterior 状态与 corridor 输出。"""

        if self._bounce_event is None:
            return

        self._candidates = self._build_prior_candidates(self._bounce_event)
        self._best_candidate = None
        self._nominal_candidate_id = None
        self._posterior_anchor_used = False
        self._posterior_state = None

        # posterior 低 SNR：用反弹后点做一次退化判别，决定是否忽略/冻结/强先验。
        post_low_snr = None
        if bool(getattr(self._cfg, "low_snr_enabled", False)) and self._post_points:
            pts2 = list(self._post_points)[-int(self._cfg.max_post_points) :]
            post_low_snr = analyze_window(
                xs=np.array([float(p.x) for p in pts2], dtype=float),
                ys=np.array([float(p.y) for p in pts2], dtype=float),
                zs=np.array([float(p.z) for p in pts2], dtype=float),
                confs=[getattr(p, "conf", None) for p in pts2],
                sigma_x0=float(getattr(self._cfg, "low_snr_sigma_x0_m", 0.15)),
                sigma_y0=float(getattr(self._cfg, "low_snr_sigma_y0_m", 0.15)),
                sigma_z0=float(getattr(self._cfg, "low_snr_sigma_z0_m", 0.15)),
                c_min=float(getattr(self._cfg, "low_snr_conf_cmin", 0.05)),
                params=self._low_snr_params(),
                disallow_ignore_y=bool(getattr(self._cfg, "low_snr_disallow_ignore_y", True)),
            )
            self._low_snr_posterior = LowSnrAxisModes(
                mode_x=post_low_snr.x.mode,
                mode_y=post_low_snr.y.mode,
                mode_z=post_low_snr.z.mode,
            )
        else:
            self._low_snr_posterior = None

        if self._post_points and self._candidates:
            # 第一阶段：粗网格融合。
            (
                candidates_1,
                best_1,
                nominal_id_1,
                posterior_1,
            ) = reweight_candidates_and_select_best(
                bounce=self._bounce_event,
                candidates=self._candidates,
                post_points=self._post_points,
                time_base_abs=self._time_base_abs,
                low_snr=post_low_snr,
                cfg=self._cfg,
            )

            self._candidates = candidates_1
            self._best_candidate = best_1
            self._nominal_candidate_id = nominal_id_1
            self._posterior_state = posterior_1

            # 在线沉淀：用融合后的候选权重回灌 prior（docs/curve.md §7）。
            maybe_update_online_prior(
                online_prior=self._online_prior,
                cfg=self._cfg,
                candidates=self._candidates,
                logger=self._logger,
            )

        # 兜底：如果已经有反弹后点，但未能得到“逐候选 posterior”，则回退为
        # 不带候选先验锚定的 LS/MAP 后验（用于输出/走廊更新）。
        if self._post_points and self._posterior_state is None:
            self._posterior_state = fit_posterior_fused_map(
                bounce=self._bounce_event,
                post_points=self._post_points,
                best=None,
                time_base_abs=self._time_base_abs,
                low_snr=post_low_snr,
                cfg=self._cfg,
            )

        # 走廊更新：可选将 posterior “锚点候选”注入混合，提升短期走廊一致性。
        candidates_for_corridor = list(self._candidates)
        if self._posterior_state is not None and self._cfg.posterior_anchor_weight > 0:
            self._posterior_anchor_used = True
            candidates_for_corridor = inject_posterior_anchor(
                candidates=self._candidates,
                best=self._best_candidate,
                posterior=self._posterior_state,
                cfg=self._cfg,
            )

        self._corridor_by_time = build_corridor_by_time(
            bounce=self._bounce_event,
            candidates=candidates_for_corridor,
            cfg=self._cfg,
        )

    def _candidate_costs(
        self,
        bounce: BounceEvent,
        candidates: Sequence[Candidate],
        post_points: Sequence[BallObservation],
    ) -> np.ndarray:
        """诊断用途：计算每个候选的归一化 SSE（不做每候选后验校正）。"""

        return compute_candidate_costs(
            bounce=bounce,
            candidates=candidates,
            post_points=post_points,
            time_base_abs=self._time_base_abs,
            cfg=self._cfg,
        )

    def get_bounce_event(self) -> BounceEvent | None:
        return self._bounce_event

    def get_pre_fit_coeffs(self) -> dict[str, np.ndarray] | None:
        """获取反弹前拟合的多项式系数（用于解耦 legacy 访问私有字段）。

        该接口用于替代对私有字段 `_pre_coeffs` 的直接访问，以降低模块间耦合。

        Returns:
            若反弹前拟合尚不可用，返回 None；否则返回一个 dict：
            - "x": 反弹前 x(t_rel) 多项式系数（2 次：等效常加速度模型）
            - "y": 2 次多项式系数（a 固定为 -0.5*g）
            - "z": 反弹前 z(t_rel) 多项式系数（2 次：等效常加速度模型）
            - "t_land": shape (1,) 的数组，表示预测落地相对时间
        """

        if self._pre_coeffs is None:
            return None

        # 返回 copy，避免外部误改内部状态。
        return {k: np.asarray(v, dtype=float).copy() for k, v in self._pre_coeffs.items()}

    def get_prior_candidates(self) -> list[Candidate]:
        return list(self._candidates)

    def get_best_candidate(self) -> Candidate | None:
        """获取融合后得分最佳的候选（已使用反弹后点重赋权）。"""

        return self._best_candidate

    def get_fusion_info(self) -> FusionInfo:
        """获取 prior/posterior 融合流程的诊断信息。"""

        return FusionInfo(
            nominal_candidate_id=self._nominal_candidate_id,
            posterior_anchor_used=bool(self._posterior_anchor_used),
        )

    def get_prefit_freeze_info(self) -> PrefitFreezeInfo:
        """获取 prefit 冻结（分段）诊断信息。

        说明：
            - 该接口用于对齐 `docs/curve.md` 的“接口契约”与回放复现需求。
            - 不返回内部 detector 的全部状态，仅返回对下游排障有用且稳定的字段。
        """

        return PrefitFreezeInfo(
            is_frozen=bool(self._prefit_frozen),
            cut_index=self._prefit_cut_index,
            freeze_t_rel=self._prefit_freeze_t_rel,
            freeze_reason=self._prefit_freeze_reason,
        )

    def get_corridor_by_time(self) -> CorridorByTime | None:
        return self._corridor_by_time

    def corridor_on_plane_y(self, target_y: float) -> CorridorOnPlane | None:
        """计算水平平面 y==target_y 的穿越走廊。

        说明：
            - 本仓库坐标系中 y 为高度（y-up）。
            - 若候选轨迹未穿越该高度，会返回 None。

        Args:
            target_y: 目标平面高度（米）。

        Returns:
            走廊统计；不可用时返回 None。
        """

        r = self.corridor_on_planes_y([float(target_y)])[0]
        if not r.is_valid:
            return None
        return r

    def corridor_on_planes_y(self, target_ys: Sequence[float]) -> list[CorridorOnPlane]:
        """批量计算多个水平平面 y==const 的穿越走廊统计。"""

        ys = [float(y) for y in target_ys]
        if not ys:
            return []

        candidates = list(self._candidates)
        if self._posterior_state is not None and self._cfg.posterior_anchor_weight > 0:
            candidates = inject_posterior_anchor(
                candidates=self._candidates,
                best=self._best_candidate,
                posterior=self._posterior_state,
                cfg=self._cfg,
            )

        return compute_corridor_on_planes_y(
            bounce=self._bounce_event,
            candidates=candidates,
            cfg=self._cfg,
            target_ys=ys,
        )

    def corridor_on_plane_y_range(
        self,
        y_min: float,
        y_max: float,
        step: float,
    ) -> list[CorridorOnPlane]:
        """在 [y_min, y_max] 的固定网格上批量计算穿越走廊。

        说明：
            这是对 :meth:`corridor_on_planes_y` 的便捷封装。

        Args:
            y_min: 最小高度（米）。
            y_max: 最大高度（米）。
            step: 网格步长（米），必须 > 0。

        Returns:
            与 y 网格对齐的一组走廊统计。
        """

        y_min = float(y_min)
        y_max = float(y_max)
        step = float(step)
        if step <= 0:
            return []

        if y_max < y_min:
            y_min, y_max = y_max, y_min

        ys = list(np.arange(y_min, y_max + 1e-9, step, dtype=float))
        return self.corridor_on_planes_y(ys)

    def get_posterior_state(self) -> PosteriorState | None:
        return self._posterior_state

    def get_low_snr_info(self) -> LowSnrInfo:
        """获取低 SNR 的诊断信息（mode 标签）。"""

        return LowSnrInfo(prefit=self._low_snr_prefit, posterior=self._low_snr_posterior)

    def _fit_posterior_fused(
        self,
        bounce: BounceEvent,
        post_points: Sequence[BallObservation],
        *,
        best: Candidate | None,
    ) -> PosteriorState | None:
        """后验融合拟合（legacy/private helper）。

        说明：该方法保留是为了单测与内部调试便利；具体实现放在
        `curve_v3.posterior.fit_posterior_fused_map`，避免 core.py 继续膨胀。
        """

        return fit_posterior_fused_map(
            bounce=bounce,
            post_points=post_points,
            best=best,
            time_base_abs=self._time_base_abs,
            cfg=self._cfg,
        )

    def _build_prior_candidates(self, bounce: BounceEvent) -> list[Candidate]:
        return build_prior_candidates(
            bounce=bounce,
            cfg=self._cfg,
            prior_model=self._prior_model,
            online_prior=self._online_prior,
        )

    def point_at_time_rel(self, t_rel: float) -> list[float] | None:
        """在给定相对时间 t_rel 处查询轨迹点 [x,y,z]。

        说明：
            - t_rel 的时间基准为 `time_base_abs`（第一次观测的时间戳）。
            - 反弹前：使用 prefit 的多项式。
            - 反弹后：优先使用 posterior；否则使用候选混合的加权均值状态。
        """

        if self._pre_coeffs is None or self._bounce_event is None:
            return None

        t_land = float(self._bounce_event.t_rel)
        if t_rel < t_land:
            return [
                polyval(self._pre_coeffs["x"], t_rel),
                polyval(self._pre_coeffs["y"], t_rel),
                polyval(self._pre_coeffs["z"], t_rel),
            ]

        # 反弹后：优先使用 posterior；否则回退为候选混合的加权均值状态。
        state = self._posterior_state or prior_nominal_state(
            bounce=self._bounce_event,
            candidates=self._candidates,
        )
        if state is None:
            return None

        tau = float(t_rel - state.t_b_rel)
        if tau < 0:
            tau = 0.0

        x = state.x_b + state.vx * tau + 0.5 * state.ax * tau * tau
        y0 = float(self._cfg.bounce_contact_y())
        y = y0 + state.vy * tau - 0.5 * float(self._cfg.gravity) * tau * tau
        z = state.z_b + state.vz * tau + 0.5 * state.az * tau * tau
        return [float(x), float(y), float(z)]

    def predicted_land_time_rel(self) -> float | None:
        if self._bounce_event is None:
            return None
        return float(self._bounce_event.t_rel)

    def predicted_land_speed(self) -> list[float] | None:
        if self._bounce_event is None:
            return None
        v = np.asarray(self._bounce_event.v_minus, dtype=float)
        vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
        return [vx, vy, vz, float(math.sqrt(vx * vx + vz * vz))]

    def predicted_land_point(self) -> list[float] | None:
        if self._bounce_event is None:
            return None
        y0 = float(self._cfg.bounce_contact_y())
        return [float(self._bounce_event.x), y0, float(self._bounce_event.z), float(self._bounce_event.t_rel)]
