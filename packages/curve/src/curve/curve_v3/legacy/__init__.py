"""curve_v3 的 legacy 适配层。

该包提供一个名为 `Curve` 的类，用于尽可能兼容 `curve2.Curve` 的对外接口，
以便不改动下游业务代码即可切换到 v3 实现。

注意（LEGACY）：
    - 该适配层仅用于保留外部集成，不建议新代码依赖。
    - 新代码请优先使用 `curve_v3.core.CurvePredictorV3`。

legacy 返回值沿用旧约定：
    - None：数据不足 / 在反弹附近被抑制输出
    - -1：基本合法性校验失败
    - list[dict]：接球点候选列表
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np

from ..config import CurveV3Config
from ..core import CurvePredictorV3
from .outputs import build_legacy_receive_points, validate_z_speed_from_prefit
from ..posterior import prior_nominal_state
from ..types import BallObservation


def _try_get_hit_logger() -> logging.Logger | None:
    try:
        from hit.utils import get_logger  # type: ignore

        return get_logger(
            "ball_curve_v3",
            console_output=True,
            file_output=False,
            console_level="DEBUG",
            file_level="DEBUG",
            time_interval=-0.1,
            use_buffering=True,
            buffer_capacity=100,
            flush_interval=2.0,
        )
    except Exception:
        return None


class Curve:
    """兼容 `curve2.Curve` 的 legacy 接口。

    方法:
        reset: 重置内部状态。
        add_frame: 添加观测并返回 legacy 接球点候选。
        get_point_at_time: 查询相对时间的点位。
        get_current_curve_land_speed: 返回当前曲线段 id 与落地速度。
        get_bounce_speed: 返回反弹速度估计（legacy 约定）。
        calc_net_clearance: 返回 z==12 处的过网高度（legacy 行为）。

    说明:
        - `get_point_at_time(t)` 的 t 是相对 time_base 的时间。
        - `ball_loc[-1]` 预期为绝对时间戳。
    """

    def __init__(self, config: CurveV3Config | None = None) -> None:
        self.is_debug = False

        # legacy 默认配置：尽量与 curve2 的坐标/阈值约定保持一致。
        # - curve2 里触地点 y 约定为 0（而不是球半径）。
        # - curve2 的 z 速度下限为 0.5。
        self._cfg = config or CurveV3Config(
            bounce_contact_y_m=0.0,
            z_speed_range=(0.5, 27.0),
        )

        logger = _try_get_hit_logger() or logging.getLogger("curve_v3.legacy")
        self.logger = logger

        self._engine = CurvePredictorV3(config=self._cfg, logger=logger)
        self.reset()

    def reset(self, is_bot_serve: bool = True) -> None:  # noqa: ARG002
        self._engine.reset()

        # legacy 对外暴露/依赖的状态字段。
        # legacy 行为：reset() 后 time_base 即有值；第一次 add_frame 会覆盖为首帧绝对时间。
        self.time_base: float | None = float(time.perf_counter())

        self.ball_start_cnt: list[int] = []
        self.curve_samples_cnt: list[int] = [0, 0, 0]

        self.land_point: list[list[float] | None] = [None, None, None]
        self.land_speed: list[list[float] | None] = [None, None, None]

        self.loc_results: list[dict[str, Any]] = []

        # 仅保留少量历史，满足部分 legacy 方法的需求。
        self._last_obs: BallObservation | None = None
        self._n_obs: int = 0

    def add_frame(self, ball_loc, is_bot_fire: int = -1):
        """legacy 的 add_frame。

        Args:
            ball_loc: [x, y, z, timestamp_abs]
            is_bot_fire: -1 用户来球（球向机器人方向飞行），+1 机器人发球。

        Returns:
            None | -1 | list[dict]
        """

        self.loc_results = []

        obs = BallObservation(
            x=float(ball_loc[0]),
            y=float(ball_loc[1]),
            z=float(ball_loc[2]),
            t=float(ball_loc[3]),
            conf=1.0,
        )

        if self._n_obs == 0:
            self.time_base = obs.t
            self.ball_start_cnt = [0]

        self._n_obs += 1
        self._last_obs = obs

        self._engine.add_observation(obs)

        if self._engine.time_base_abs is None:
            return None

        # 更新 legacy 的落地信息缓存。
        # legacy：land_point 的 y 约定为 0（curve2 的输出）。
        lp0 = self._engine.predicted_land_point()
        if lp0 is None:
            self.land_point[0] = None
        else:
            self.land_point[0] = [float(lp0[0]), 0.0, float(lp0[2]), float(lp0[3])]
        self.land_speed[0] = self._engine.predicted_land_speed()

        t_land = self._engine.predicted_land_time_rel()
        t_rel_now = obs.t - self._engine.time_base_abs

        # 在反弹附近抑制输出，提升稳定性（保持 legacy 行为）。
        #
        # 重要约定：仅在“已经进入反弹后段（t_rel_now > t_land）”且非常接近反弹时刻时才抑制。
        # 否则在 pre 段（尤其是恰好 t_rel_now==t_land 的那一帧）会把输出永远压成 None，
        # 导致 legacy 下游无法获取接球点。
        if t_land is not None:
            dt_land = float(t_rel_now - float(t_land))
            # 浮点误差下可能出现 t_rel_now≈t_land 但 dt_land 为一个极小正数，
            # 这会把“恰好触地那一帧”的输出错误压成 None。
            if dt_land > 1e-6 and dt_land < float(self._cfg.legacy_too_close_to_land_s):
                return None

        # 曲线段 id：0 表示预测反弹前，1 表示预测反弹后。
        seg_id = 0
        if t_land is not None and t_rel_now > t_land:
            seg_id = 1
            if len(self.ball_start_cnt) == 1:
                self.ball_start_cnt.append(self._n_obs - 1)

        self.curve_samples_cnt[seg_id] += 1

        # 点数不足时不输出。
        if self._n_obs < 5:
            return None

        # 基本合法性检查（保持 legacy 行为）。
        pre = self._engine.get_pre_fit_coeffs()
        if pre is not None and "z" in pre:
            ok = validate_z_speed_from_prefit(
                z_coeff=np.asarray(pre["z"], dtype=float),
                t_rel_last=float(t_rel_now),
                cfg=self._cfg,
                is_bot_fire=int(is_bot_fire),
            )
        else:
            ok = True

        if not ok:
            self.logger.error("z 方向速度检查失败，标记曲线为无效。")
            return -1

        # Legacy 行为：始终生成 curve 1（反弹后）上的接球点。
        fit_samples = int(self.curve_samples_cnt[0] + self.curve_samples_cnt[1])
        curve_1_samples = int(self.curve_samples_cnt[1])

        bounce = self._engine.get_bounce_event()
        if bounce is None or self._engine.time_base_abs is None:
            return None

        state = self._engine.get_posterior_state() or prior_nominal_state(
            bounce=bounce,
            candidates=self._engine.get_prior_candidates(),
        )
        if state is None:
            return None

        # 补全 legacy 的“反弹后下一次落地”缓存（curve2 的 land_point[1]/land_speed[1]）。
        # 说明：v3 核心只显式建模一次反弹；这里用 state 的解析式把下一次触地推出来。
        g = float(self._cfg.gravity)
        vy0 = float(state.vy)
        if g > 1e-9 and vy0 > 0.0:
            tau2 = float(2.0 * vy0 / g)
            x2 = float(state.x_b + state.vx * tau2 + 0.5 * state.ax * tau2 * tau2)
            z2 = float(state.z_b + state.vz * tau2 + 0.5 * state.az * tau2 * tau2)

            vx2 = float(state.vx + state.ax * tau2)
            vz2 = float(state.vz + state.az * tau2)
            vy2 = float(vy0 - g * tau2)  # = -vy0
            self.land_point[1] = [x2, 0.0, z2, float(state.t_b_rel + tau2)]
            self.land_speed[1] = [vx2, vy2, vz2, float(math.sqrt(vx2 * vx2 + vz2 * vz2))]
        else:
            self.land_point[1] = None
            self.land_speed[1] = None

        points = build_legacy_receive_points(
            state=state,
            time_base_abs=float(self._engine.time_base_abs),
            cfg=self._cfg,
            fit_samples=int(fit_samples),
            curve_1_samples=int(curve_1_samples),
        )
        if points is None:
            # 与 legacy curve2 对齐：数据足够时，即使没有接球点也返回空列表。
            self.loc_results = []
            return self.loc_results

        self.loc_results = points
        return self.loc_results

    def get_point_at_time(self, t: float):
        return self._engine.point_at_time_rel(float(t))

    def corridor_on_plane_y(self, target_y: float):
        """legacy 包装：按 y 平面计算穿越走廊。"""

        return self._engine.corridor_on_plane_y(float(target_y))

    def corridor_on_plane_y_range(self, y_min: float, y_max: float, step: float):
        """legacy 包装：按固定 y 网格批量计算穿越走廊。"""

        return self._engine.corridor_on_plane_y_range(
            y_min=float(y_min),
            y_max=float(y_max),
            step=float(step),
        )

    def get_fusion_info(self):
        """返回 v3 融合诊断信息（名义候选/是否使用后验锚点）。"""

        return self._engine.get_fusion_info()

    def get_current_curve_land_speed(self):
        seg_id = max(len(self.ball_start_cnt) - 1, 0)
        return seg_id, self.land_speed[0], self.land_speed[1]

    def get_current_curve_y_view_fit(self):
        # v3 未实现该接口，保留 legacy 签名以兼容下游。
        return None, None, None, None

    def get_bounce_speed(self):
        seg_id = max(len(self.ball_start_cnt) - 1, 0)
        if self.land_point[0] is None or self._last_obs is None or self.time_base is None:
            return seg_id, self.land_speed[0], None

        land_t_rel = float(self.land_point[0][-1])
        land_x = float(self.land_point[0][0])
        land_z = float(self.land_point[0][2])

        t_rel_now = float(self._last_obs.t - self.time_base)
        dt = float(t_rel_now - land_t_rel)
        if dt <= 1e-6:
            return seg_id, self.land_speed[0], None

        dx = float(self._last_obs.x - land_x)
        dz = float(self._last_obs.z - land_z)

        bounce_vxz = float(math.sqrt(dx * dx + dz * dz) / dt)
        bounce_vx = float(dx / dt)
        bounce_vz = float(dz / dt)

        # legacy 约定：若可用，则从落地速度里取 vy 作为反弹时刻的 vy。
        bounce_vy = 0.0
        if self.land_speed[0] is not None:
            bounce_vy = float(self.land_speed[0][1])

        return seg_id, self.land_speed[0], [bounce_vx, -bounce_vy, bounce_vz, bounce_vxz]

    def predicted_second_land_time_rel(self) -> float | None:
        """预测第二段（反弹后）的落地相对时刻（相对 time_base）。

        说明：
            legacy 适配层在 add_frame 中会尝试补全 land_point[1]（对齐 curve2 的约定），
            其中 land_point[1][-1] 即为第二段落地的相对时刻。
        """

        lp = self.land_point[1]
        if lp is None:
            return None
        try:
            return float(lp[-1])
        except Exception:
            return None

    def predicted_second_land_time_abs(self) -> float | None:
        """预测第二段（反弹后）的落地绝对时刻。"""

        if self.time_base is None:
            return None
        t_rel = self.predicted_second_land_time_rel()
        if t_rel is None:
            return None
        try:
            return float(self.time_base + float(t_rel))
        except Exception:
            return None

    def calc_net_clearance(self):
        # Legacy：使用反弹前的线性 z(t) 求解 z==12 的时刻，并在该时刻计算 y。
        # 这要求 pre-bounce 拟合已存在。
        bounce = self._engine.get_bounce_event()
        pre = self._engine.get_pre_fit_coeffs()
        if bounce is None or pre is None:
            return None

        z_coeff = pre.get("z")
        y_coeff = pre.get("y")
        if z_coeff is None or y_coeff is None:
            return None

        # z(t) = v*t + c。
        v = float(z_coeff[0])
        c = float(z_coeff[1])
        if abs(v) < 1e-9:
            return None

        t_at_net = float((12.0 - c) / v)
        return float(np.polyval(y_coeff, t_at_net))


__all__ = ["Curve"]
