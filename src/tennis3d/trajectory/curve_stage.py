"""将 3D 定位输出流转换为轨迹拟合（curve）输出的后处理 stage。

设计目标：
- 组合而非侵入：不修改 `tennis3d.pipeline.core.run_localization_pipeline` 的内部逻辑。
- 支持多球：把每帧 0..N 个 3D 点关联成多个 track，每个 track 维护一套拟合器状态。
- 下游友好：输出“落点 + 落地时间 + 置信走廊”的轻量 JSON 结构（可直接写入 jsonl）。

注意：
- 当前多球关联使用简单的最近邻 + gating（工程够用、易解释）。当两球非常接近/交叉时，
  仍可能发生 track 交换；如果你的场景确实高频出现，需要升级为更强的关联（匈牙利/卡尔曼等）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from collections import deque

import math


@dataclass(frozen=True)
class CurveStageConfig:
    """Curve stage 配置。

    说明：
        该配置刻意保持“少而关键”。curve_v3 内部的细粒度参数仍使用其默认值，
        后续若要暴露再逐步增加。
    """

    enabled: bool = False

    # 主输出算法：
    # - v3：输出落点/落地时间/走廊（默认）
    # - v2：输出 legacy 的接球点候选（curve2）
    # - v3_legacy：输出 v3 的 legacy 适配层接球点候选
    primary: str = "v3"  # v3|v2|v3_legacy

    # 对比开关：用于离线评测 v2 vs v3。
    compare_v2: bool = False
    compare_v3_legacy: bool = False

    # 多球关联/track 管理
    max_tracks: int = 5
    association_dist_m: float = 0.6
    max_missed_s: float = 0.15
    min_dt_s: float = 1e-6

    # 走廊输出：按 y 平面穿越输出一组统计（用于规划挑选击球高度范围）
    corridor_y_min: float = 0.6
    corridor_y_max: float = 1.6
    corridor_y_step: float = 0.1

    # 点级置信度来源
    conf_from: str = "quality"  # quality|constant
    constant_conf: float = 1.0

    # 可选：对输入到 curve 的坐标做轻量变换。
    # 说明：
    # - 仅影响“传给 curve_v3/v2/legacy 的观测值”，不会修改上游定位输出 balls[*].ball_3d_world。
    # - 默认不变换（保持旧行为）。
    # - 若你的坐标系需要把 y 轴翻转并做零点校正，可用：y' = -(y - y_offset_m)。
    y_offset_m: float = 0.0
    y_negate: bool = False

    # legacy 需要的字段（沿用旧接口约定）
    is_bot_fire: int = -1

    # ----------------------------
    # 观测过滤（在进入关联/拟合之前）
    # ----------------------------
    # 说明：
    # - 这些过滤使用 run_localization_pipeline 的输出诊断字段（若存在）。
    # - 默认全部关闭，保持旧行为。
    obs_min_views: int = 0
    obs_min_quality: float = 0.0
    obs_max_median_reproj_error_px: float | None = None
    obs_max_ball_3d_std_m: float | None = None

    # ----------------------------
    # 轨迹有效性/片段（episode）判定
    # ----------------------------
    # 目标：当同时存在很多 track 时，明确回答：
    # - “哪条 track 当前处于有效飞行（击球）阶段？”
    # - “该阶段何时开始/结束？结束原因是什么？”
    #
    # 设计取舍：
    # - 该逻辑在 curve_stage 层实现，不依赖具体 detector。
    # - 默认关闭，不影响既有曲线拟合与输出字段。
    episode_enabled: bool = False

    # episode 起始判定（唯一方案）：分轴(z方向) + y轴重力一致性。
    #
    # 说明：
    # - 该判定更贴近“真实飞行轨迹”（而不是准备阶段晃动/误检）。
    # - 要求：z 方向满足“朝机器人”运动（可配置方向）且 y(t) 的二次拟合加速度接近 g。
    # - 该逻辑一旦启用，会比简单位移/速度门限更严格；必要时请调整阈值与容差。
    episode_buffer_s: float = 0.6
    episode_min_obs: int = 5

    # z 方向门控。
    # 说明：不同标定坐标系中“朝机器人”的 z 正负可能不同，所以做成可配置。
    # - 0：不限制方向，仅用 |dz| / |vz|
    # - +1：要求 dz 为正（z 增大）
    # - -1：要求 dz 为负（z 减小）
    episode_z_dir: int = 0
    episode_min_abs_dz_m: float = 0.25
    episode_min_abs_vz_mps: float = 1.0

    # y 轴重力一致性检查。
    # 做法：对窗口内 (t, y) 做二次拟合 y = a t^2 + b t + c，估计加速度 ay = 2a。
    # - 默认用“加速度大小接近 g”的判据（不强制符号），避免不同 y 轴朝向导致误判。
    episode_gravity_mps2: float = 9.8
    episode_gravity_tol_mps2: float = 3.0

    # episode 结束判定：
    # - 若速度持续过低（准备阶段/手持晃动），认为本段结束。
    episode_stationary_speed_mps: float = 0.25
    episode_end_if_stationary_s: float = 0.35
    # - 若 v3 给出 predicted_land_time_abs，且当前时间超过落地一段缓冲，则认为本段结束。
    episode_end_after_predicted_land_s: float = 0.2

    # episode 行为：
    # - reset_predictor_on_episode_start：episode 确认后，把窗口内观测重放给拟合器，
    #   让 curve 的 time_base_abs/早期拟合更贴近“真实开始”。
    # - reset_predictor_on_episode_end：episode 结束后清空拟合器，避免下一段被污染。
    reset_predictor_on_episode_start: bool = True
    reset_predictor_on_episode_end: bool = True
    # - 若开启：仅在 episode_active 时才把观测喂给 curve（可抑制准备阶段污染）。
    feed_curve_only_when_episode_active: bool = False

    # - 若开启：当任意 track 进入 episode_active 后，立即丢弃其它 track，只保留该 track。
    #   目的：在“单球击打”为主的在线场景下避免多 track 互相抢占，让下游始终有唯一目标。
    #
    #   注意：
    #   - 同一时刻出现的其它球/误检将被忽略（不会继续创建/维护新的 track）。
    #   - 当该 track 的 episode 结束、或该 track 因 max_missed_s 被丢弃后，会自动解除锁定。
    #   - 仅在 episode_enabled=True 时生效。
    episode_lock_single_track: bool = False


@dataclass
class _BallMeas:
    ball_id: int
    x: float
    y: float
    z: float
    quality: float
    num_views: int | None = None
    median_reproj_error_px: float | None = None
    ball_3d_std_m_max: float | None = None


@dataclass
class _RecentObs:
    """track 内部缓存的最近观测（已应用 y 变换）。

    说明：
        - 该缓存用于 episode 起始判定与（可选）重放给 curve 拟合器。
        - t_abs 采用 stage 选出的绝对时间轴（capture_t_abs/created_at）。
    """

    t_abs: float
    x: float
    y: float
    z: float
    conf: float | None

    @property
    def pos(self) -> tuple[float, float, float]:
        return (float(self.x), float(self.y), float(self.z))


def _solve_3x3(A: list[list[float]], b: list[float]) -> tuple[float, float, float] | None:
    """解 3x3 线性方程组 A x = b（高斯消元）。

    说明：
        - 不依赖 numpy，避免引入额外依赖。
        - 对数值病态情况返回 None。
    """

    # 复制，避免原地修改。
    M = [[float(A[i][j]) for j in range(3)] + [float(b[i])] for i in range(3)]

    for col in range(3):
        # 选主元
        piv = col
        piv_abs = abs(M[col][col])
        for r in range(col + 1, 3):
            v = abs(M[r][col])
            if v > piv_abs:
                piv_abs = v
                piv = r
        if piv_abs < 1e-12:
            return None
        if piv != col:
            M[col], M[piv] = M[piv], M[col]

        # 归一化
        div = float(M[col][col])
        for j in range(col, 4):
            M[col][j] = float(M[col][j]) / div

        # 消元
        for r in range(3):
            if r == col:
                continue
            factor = float(M[r][col])
            if abs(factor) < 1e-12:
                continue
            for j in range(col, 4):
                M[r][j] = float(M[r][j]) - factor * float(M[col][j])

    return (float(M[0][3]), float(M[1][3]), float(M[2][3]))


def _estimate_const_accel_y(window: list[_RecentObs]) -> float | None:
    """用二次拟合估计窗口内 y 轴常加速度。

    返回：ay（m/s^2）或 None。

    说明：
        - 拟合模型：y = a t^2 + b t + c（t 相对窗口起点）。
        - 常加速度 ay = 2a。
    """

    if len(window) < 3:
        return None

    t0 = float(window[0].t_abs)
    ts: list[float] = []
    ys: list[float] = []
    for o in window:
        ts.append(float(o.t_abs) - t0)
        ys.append(float(o.y))

    # 构造正规方程 (X^T X) beta = X^T y，其中 X=[t^2, t, 1]
    s_t4 = 0.0
    s_t3 = 0.0
    s_t2 = 0.0
    s_t1 = 0.0
    s_1 = float(len(ts))
    s_y_t2 = 0.0
    s_y_t1 = 0.0
    s_y = 0.0
    for t, y in zip(ts, ys):
        t2 = float(t * t)
        t3 = float(t2 * t)
        t4 = float(t2 * t2)
        s_t1 += float(t)
        s_t2 += float(t2)
        s_t3 += float(t3)
        s_t4 += float(t4)
        s_y += float(y)
        s_y_t1 += float(y) * float(t)
        s_y_t2 += float(y) * float(t2)

    A = [
        [s_t4, s_t3, s_t2],
        [s_t3, s_t2, s_t1],
        [s_t2, s_t1, s_1],
    ]
    bb = [s_y_t2, s_y_t1, s_y]
    sol = _solve_3x3(A, bb)
    if sol is None:
        return None
    a, _b, _c = sol
    return float(2.0 * float(a))


@dataclass
class _Track:
    track_id: int
    created_t_abs: float | None = None
    last_t_abs: float | None = None
    last_pos: tuple[float, float, float] | None = None
    prev_t_abs: float | None = None
    prev_pos: tuple[float, float, float] | None = None
    n_obs: int = 0

    # 运动学诊断（便于下游做可解释筛选）
    last_speed_mps: float | None = None
    speed_ewma_mps: float | None = None
    last_motion_t_abs: float | None = None

    # episode 状态
    episode_id: int = 0
    episode_active: bool = False
    episode_start_t_abs: float | None = None
    episode_end_t_abs: float | None = None
    episode_end_reason: str | None = None

    # v3 可用的“落地时间”缓存（用于 episode 结束判定）
    predicted_land_time_abs: float | None = None

    # 最近观测缓存（用于 episode 判定/重放）
    recent: deque[_RecentObs] = field(default_factory=deque)

    # 拟合器实例（按需创建）
    v3: Any | None = None
    v3_legacy: Any | None = None
    v2: Any | None = None


def _dist3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    dz = float(a[2] - b[2])
    return float(math.sqrt(dx * dx + dy * dy + dz * dz))


def _predict_track_pos(tr: _Track, *, t_abs: float, min_dt_s: float) -> tuple[float, float, float] | None:
    """预测当前时刻 track 的位置（用于数据关联）。

    设计说明：
        - 纯距离最近邻在多球接近/交叉时容易 swap。
        - 这里使用“最近两次观测”估计常速度，并外推到当前时刻做 gating。
        - 若历史不足或 dt 异常，则回退到 last_pos，保持行为可解释。
    """

    if tr.last_pos is None or tr.last_t_abs is None:
        return None

    # 历史不足：无法估速度，回退到 last_pos。
    if tr.prev_pos is None or tr.prev_t_abs is None:
        return tr.last_pos

    dt_hist = float(tr.last_t_abs - tr.prev_t_abs)
    if dt_hist < float(min_dt_s):
        return tr.last_pos

    vx = float(tr.last_pos[0] - tr.prev_pos[0]) / dt_hist
    vy = float(tr.last_pos[1] - tr.prev_pos[1]) / dt_hist
    vz = float(tr.last_pos[2] - tr.prev_pos[2]) / dt_hist

    dt = float(t_abs - tr.last_t_abs)
    return (
        float(tr.last_pos[0] + vx * dt),
        float(tr.last_pos[1] + vy * dt),
        float(tr.last_pos[2] + vz * dt),
    )


def _as_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _extract_meas_list(out_rec: dict[str, Any], cfg: CurveStageConfig) -> list[_BallMeas]:
    balls = out_rec.get("balls")
    if not isinstance(balls, list) or not balls:
        return []

    out: list[_BallMeas] = []
    for b in balls:
        if not isinstance(b, dict):
            continue
        bid = int(b.get("ball_id", len(out)))
        p = b.get("ball_3d_world")
        if not (isinstance(p, list) and len(p) == 3):
            continue
        x = _as_float(p[0])
        y = _as_float(p[1])
        z = _as_float(p[2])
        if x is None or y is None or z is None:
            continue
        q = _as_float(b.get("quality"))
        if q is None:
            q = 0.0

        # 可选诊断字段（可能不存在）
        nv = None
        nv_raw = b.get("num_views")
        if nv_raw is not None:
            try:
                # 允许 str/int/float 形式；其它类型忽略。
                nv = int(nv_raw)
            except Exception:
                nv = None

        med_err = _as_float(b.get("median_reproj_error_px"))

        std_max = None
        std = b.get("ball_3d_std_m")
        if isinstance(std, list) and len(std) == 3:
            try:
                vals = [float(std[0]), float(std[1]), float(std[2])]
                std_max = float(max(vals))
            except Exception:
                std_max = None

        m = _BallMeas(
            ball_id=bid,
            x=float(x),
            y=float(y),
            z=float(z),
            quality=float(q),
            num_views=nv,
            median_reproj_error_px=med_err,
            ball_3d_std_m_max=std_max,
        )

        # 观测过滤：默认关闭；开启后会抑制几何不稳/质量过低的点。
        if int(cfg.obs_min_views) > 0:
            if m.num_views is None or int(m.num_views) < int(cfg.obs_min_views):
                continue
        if float(m.quality) < float(cfg.obs_min_quality):
            continue
        if cfg.obs_max_median_reproj_error_px is not None:
            if m.median_reproj_error_px is None or float(m.median_reproj_error_px) > float(cfg.obs_max_median_reproj_error_px):
                continue
        if cfg.obs_max_ball_3d_std_m is not None:
            if m.ball_3d_std_m_max is None or float(m.ball_3d_std_m_max) > float(cfg.obs_max_ball_3d_std_m):
                continue

        out.append(m)

    # 质量高的先分配，更稳（避免低质量点抢占轨迹）。
    out.sort(key=lambda m: float(m.quality), reverse=True)
    return out


def _choose_t_abs(out_rec: dict[str, Any]) -> tuple[float, str]:
    # 优先用 source 注入的 capture_t_abs（来自 host_timestamp）。
    t = out_rec.get("capture_t_abs")
    t_abs = _as_float(t)
    if t_abs is not None and math.isfinite(t_abs):
        return float(t_abs), "capture_t_abs"

    # 回退：处理时间（不是曝光时刻，但至少单调）。
    t2 = _as_float(out_rec.get("created_at"))
    if t2 is not None and math.isfinite(t2):
        return float(t2), "created_at"

    # 最差兜底：0
    return 0.0, "fallback"


def _obs_conf(cfg: CurveStageConfig, quality: float) -> float | None:
    if cfg.conf_from == "quality":
        # 质量通常在 [0,1] 左右；保持原样即可。
        return float(max(0.0, float(quality)))
    if cfg.conf_from == "constant":
        return float(max(0.0, float(cfg.constant_conf)))
    return None


def _corridor_to_json(r: Any, time_base_abs: float | None) -> dict[str, Any]:
    # r 预期为 curve.curve_v3.types.CorridorOnPlane
    mu_xz = getattr(r, "mu_xz", None)
    cov_xz = getattr(r, "cov_xz", None)

    def _vec2(x: Any) -> list[float] | None:
        try:
            return [float(x[0]), float(x[1])]
        except Exception:
            return None

    def _mat22(x: Any) -> list[list[float]] | None:
        try:
            return [[float(x[0][0]), float(x[0][1])], [float(x[1][0]), float(x[1][1])]]
        except Exception:
            return None

    t_rel_mu = _as_float(getattr(r, "t_rel_mu", None))
    t_rel_var = _as_float(getattr(r, "t_rel_var", None))

    t_abs_mu = None
    if time_base_abs is not None and t_rel_mu is not None:
        t_abs_mu = float(time_base_abs + float(t_rel_mu))

    return {
        "target_y": float(getattr(r, "target_y")),
        "mu_xz": _vec2(mu_xz),
        "cov_xz": _mat22(cov_xz),
        "t_rel_mu": float(t_rel_mu) if t_rel_mu is not None else None,
        "t_rel_var": float(t_rel_var) if t_rel_var is not None else None,
        "t_abs_mu": t_abs_mu,
        "valid_ratio": float(getattr(r, "valid_ratio")),
        "crossing_prob": float(getattr(r, "crossing_prob")),
        "is_valid": bool(getattr(r, "is_valid")),
    }


def _track_snapshot(track: _Track, cfg: CurveStageConfig) -> dict[str, Any]:
    out: dict[str, Any] = {
        "track_id": int(track.track_id),
        "n_obs": int(track.n_obs),
        "last_t_abs": float(track.last_t_abs) if track.last_t_abs is not None else None,
        "last_pos": [float(x) for x in track.last_pos] if track.last_pos is not None else None,
    }

    # 仅在 episode 功能启用时输出额外状态，避免改变旧行为的输出体积。
    if bool(cfg.episode_enabled):
        out["kinematics"] = {
            "last_speed_mps": float(track.last_speed_mps) if track.last_speed_mps is not None else None,
            "speed_ewma_mps": float(track.speed_ewma_mps) if track.speed_ewma_mps is not None else None,
        }
        out["episode"] = {
            "episode_id": int(track.episode_id) if int(track.episode_id) > 0 else None,
            "active": bool(track.episode_active),
            "start_t_abs": float(track.episode_start_t_abs) if track.episode_start_t_abs is not None else None,
            "end_t_abs": float(track.episode_end_t_abs) if track.episode_end_t_abs is not None else None,
            "end_reason": str(track.episode_end_reason) if track.episode_end_reason is not None else None,
            "predicted_land_time_abs": float(track.predicted_land_time_abs) if track.predicted_land_time_abs is not None else None,
        }

    if track.v3 is not None:
        v3 = track.v3
        time_base_abs = getattr(v3, "time_base_abs", None)
        out_v3: dict[str, Any] = {
            "time_base_abs": float(time_base_abs) if time_base_abs is not None else None,
            "predicted_land_time_rel": None,
            "predicted_land_time_abs": None,
            "predicted_land_point": None,
            "predicted_land_speed": None,
            "corridor_on_planes_y": None,
            "diagnostics": None,
        }

        t_rel = None
        try:
            t_rel = v3.predicted_land_time_rel()
        except Exception:
            t_rel = None

        if t_rel is not None and time_base_abs is not None:
            out_v3["predicted_land_time_rel"] = float(t_rel)
            out_v3["predicted_land_time_abs"] = float(time_base_abs + float(t_rel))

        try:
            lp = v3.predicted_land_point()
        except Exception:
            lp = None
        if lp is not None:
            # [x, y_contact, z, t_rel]
            try:
                out_v3["predicted_land_point"] = [float(lp[0]), float(lp[1]), float(lp[2])]
            except Exception:
                out_v3["predicted_land_point"] = None

        try:
            sp = v3.predicted_land_speed()
        except Exception:
            sp = None
        if sp is not None:
            try:
                out_v3["predicted_land_speed"] = [float(sp[0]), float(sp[1]), float(sp[2]), float(sp[3])]
            except Exception:
                out_v3["predicted_land_speed"] = None

        # corridor：只有在 bounce_event 成立后才有意义（predicted_land_time_rel != None）。
        if t_rel is not None:
            try:
                cors = v3.corridor_on_plane_y_range(cfg.corridor_y_min, cfg.corridor_y_max, cfg.corridor_y_step)
            except Exception:
                cors = []
            out_v3["corridor_on_planes_y"] = [
                _corridor_to_json(r, time_base_abs if isinstance(time_base_abs, (int, float)) else None)
                for r in (cors or [])
            ]

        # 诊断信息（对齐排障需求，字段相对稳定）。
        try:
            freeze = v3.get_prefit_freeze_info()
        except Exception:
            freeze = None
        try:
            low_snr = v3.get_low_snr_info()
        except Exception:
            low_snr = None
        try:
            fusion = v3.get_fusion_info()
        except Exception:
            fusion = None

        out_v3["diagnostics"] = {
            "prefit_freeze": {
                "is_frozen": bool(getattr(freeze, "is_frozen")) if freeze is not None else None,
                "cut_index": int(getattr(freeze, "cut_index")) if freeze is not None and getattr(freeze, "cut_index") is not None else None,
                "freeze_t_rel": float(getattr(freeze, "freeze_t_rel")) if freeze is not None and getattr(freeze, "freeze_t_rel") is not None else None,
                "freeze_reason": str(getattr(freeze, "freeze_reason")) if freeze is not None and getattr(freeze, "freeze_reason") is not None else None,
            },
            "low_snr": {
                "prefit": getattr(low_snr, "prefit", None) is not None,
                "posterior": getattr(low_snr, "posterior", None) is not None,
                # mode 标签展开（如果可用）
                "prefit_modes": {
                    "x": str(getattr(getattr(low_snr, "prefit", None), "mode_x", "")) if low_snr is not None else "",
                    "y": str(getattr(getattr(low_snr, "prefit", None), "mode_y", "")) if low_snr is not None else "",
                    "z": str(getattr(getattr(low_snr, "prefit", None), "mode_z", "")) if low_snr is not None else "",
                },
                "posterior_modes": {
                    "x": str(getattr(getattr(low_snr, "posterior", None), "mode_x", "")) if low_snr is not None else "",
                    "y": str(getattr(getattr(low_snr, "posterior", None), "mode_y", "")) if low_snr is not None else "",
                    "z": str(getattr(getattr(low_snr, "posterior", None), "mode_z", "")) if low_snr is not None else "",
                },
            },
            "fusion": {
                "nominal_candidate_id": int(getattr(fusion, "nominal_candidate_id")) if fusion is not None and getattr(fusion, "nominal_candidate_id") is not None else None,
                "posterior_anchor_used": bool(getattr(fusion, "posterior_anchor_used")) if fusion is not None else None,
            },
        }

        out["v3"] = out_v3

    def _num_list(x: Any) -> list[float | None] | None:
        if not isinstance(x, (list, tuple)):
            return None
        out2: list[float | None] = []
        for v in x:
            if v is None:
                out2.append(None)
                continue
            fv = _as_float(v)
            if fv is None or not math.isfinite(float(fv)):
                out2.append(None)
            else:
                out2.append(float(fv))
        return out2

    if (cfg.primary == "v3_legacy" or cfg.compare_v3_legacy) and track.v3_legacy is not None:
        # legacy 的输出就是“接球点候选列表”；可用于单独验证 v3 legacy 行为。
        out["v3_legacy"] = {
            "last_receive_points": getattr(track, "_last_v3_legacy_points", None),
        }

    if (cfg.primary == "v2" or cfg.compare_v2) and track.v2 is not None:
        v2 = track.v2

        cur_land_speed = None
        try:
            seg_id, ls0, ls1 = v2.get_current_curve_land_speed()
            cur_land_speed = {
                "seg_id": int(seg_id),
                "curve0": _num_list(ls0),
                "curve1": _num_list(ls1),
            }
        except Exception:
            cur_land_speed = None

        bounce_speed = None
        try:
            seg_id2, land0, bounce = v2.get_bounce_speed()
            bounce_speed = {
                "seg_id": int(seg_id2),
                "land_speed_curve0": _num_list(land0),
                "bounce_speed": _num_list(bounce),
            }
        except Exception:
            bounce_speed = None

        out["v2"] = {
            "last_receive_points": getattr(track, "_last_v2_points", None),
            "current_curve_land_speed": cur_land_speed,
            "bounce_speed": bounce_speed,
        }

    return out


class CurveStage:
    """对 out_rec 流做曲线拟合增强。"""

    def __init__(self, cfg: CurveStageConfig) -> None:
        self._cfg = cfg
        self._next_track_id = 1
        self._tracks: list[_Track] = []

        # episode 锁定：当某条 track 进入 episode 后，可选地丢弃其它 track，只保留该条。
        self._episode_locked_track_id: int | None = None

        # 延迟 import：避免在未启用时引入 curve 相关依赖与启动开销。
        self._imports_ready = False
        self._CurvePredictorV3 = None
        self._BallObservation = None
        self._CurveV2 = None
        self._CurveV3Legacy = None

    def _ensure_curve_imports(self) -> None:
        if bool(self._imports_ready):
            return

        # 仅按需引入，避免未使用算法的启动开销。
        need_v3 = str(self._cfg.primary) == "v3"
        need_v2 = str(self._cfg.primary) == "v2" or bool(self._cfg.compare_v2)
        need_v3_legacy = str(self._cfg.primary) == "v3_legacy" or bool(self._cfg.compare_v3_legacy)

        if need_v3:
            from curve.curve_v3.core import CurvePredictorV3
            from curve.curve_v3.types import BallObservation

            self._CurvePredictorV3 = CurvePredictorV3
            self._BallObservation = BallObservation

        if need_v2:
            from curve.curve_v2 import Curve as CurveV2

            self._CurveV2 = CurveV2

        if need_v3_legacy:
            from curve.curve_v3.legacy import Curve as CurveV3Legacy

            self._CurveV3Legacy = CurveV3Legacy

        self._imports_ready = True

    def reset(self) -> None:
        """清空所有 track 状态。"""

        self._tracks.clear()
        self._next_track_id = 1
        self._episode_locked_track_id = None

    def _episode_lock_to_track(
        self,
        *,
        keep_track_id: int,
        now_t_abs: float,
        events: list[dict[str, Any]],
        ended_tracks: list[dict[str, Any]],
    ) -> None:
        """进入 episode 后可选地把系统锁定到单条 track。

        说明：
            - 该逻辑用于解决“多 track 同时存在时，下游不知道跟哪条”的工程问题。
            - 这里选择最小可解释方案：一旦 episode_start 触发，就把其它 track 全部丢弃。
        """

        if not bool(self._cfg.episode_enabled):
            return
        if not bool(self._cfg.episode_lock_single_track):
            return
        if self._episode_locked_track_id is not None:
            return

        keep: _Track | None = None
        dropped: list[_Track] = []
        for tr in self._tracks:
            if int(tr.track_id) == int(keep_track_id):
                keep = tr
            else:
                dropped.append(tr)
        if keep is None:
            return

        # 丢弃其它 track，并记录诊断事件（仅在 episode_enabled 时会输出）。
        for tr in dropped:
            if bool(tr.episode_active):
                tr.episode_active = False
                tr.episode_end_t_abs = float(tr.last_t_abs) if tr.last_t_abs is not None else float(now_t_abs)
                tr.episode_end_reason = "track_drop_episode_lock"
                events.append(
                    {
                        "event": "episode_end",
                        "track_id": int(tr.track_id),
                        "episode_id": int(tr.episode_id),
                        "t_abs": float(tr.episode_end_t_abs),
                        "reason": str(tr.episode_end_reason),
                    }
                )
            events.append(
                {
                    "event": "track_drop",
                    "track_id": int(tr.track_id),
                    "t_abs": float(now_t_abs),
                    "reason": "episode_lock_single_track",
                }
            )
            ended_tracks.append(_track_snapshot(tr, self._cfg))

        self._tracks = [keep]
        self._episode_locked_track_id = int(keep.track_id)

    def _episode_maybe_release_lock(self, tr: _Track) -> None:
        """当锁定 track 的 episode 结束时，自动解除锁定。"""

        if self._episode_locked_track_id is None:
            return
        if int(tr.track_id) != int(self._episode_locked_track_id):
            return
        if bool(tr.episode_active):
            return

        # 说明：只要 episode 不再活跃，就解除锁定，允许下一段重新建 track。
        self._episode_locked_track_id = None

    def _new_v3(self) -> Any | None:
        if self._CurvePredictorV3 is None:
            return None
        try:
            return self._CurvePredictorV3()  # type: ignore[operator]
        except Exception:
            return None

    def _new_v2(self) -> Any | None:
        if self._CurveV2 is None:
            return None
        try:
            return self._CurveV2()  # type: ignore[operator]
        except Exception:
            return None

    def _new_v3_legacy(self) -> Any | None:
        if self._CurveV3Legacy is None:
            return None
        try:
            return self._CurveV3Legacy()  # type: ignore[operator]
        except Exception:
            return None

    def _reset_track_predictors(self, tr: _Track) -> None:
        """重置某条 track 的拟合器状态。

        说明：
            - 保留 track_id 与关联的 last_pos/last_t_abs，用于连续关联。
            - 清空 predictor 内部状态，避免跨 episode 污染。
        """

        if self._cfg.primary == "v3" and self._CurvePredictorV3 is not None:
            tr.v3 = self._new_v3()
        if (self._cfg.primary == "v2" or self._cfg.compare_v2) and self._CurveV2 is not None:
            tr.v2 = self._new_v2()
        if (self._cfg.primary == "v3_legacy" or self._cfg.compare_v3_legacy) and self._CurveV3Legacy is not None:
            tr.v3_legacy = self._new_v3_legacy()
        tr.predicted_land_time_abs = None
        setattr(tr, "_last_v2_points", None)
        setattr(tr, "_last_v3_legacy_points", None)

    def _maybe_update_predicted_land_time_abs(self, tr: _Track) -> None:
        """从 v3 predictor 读取 predicted_land_time_abs 并缓存到 track。

        说明：
            - 这是 episode 结束判定所需的关键信号之一。
            - 若 predictor 尚无法预测，则保持为 None。
        """

        if tr.v3 is None:
            return

        v3 = tr.v3
        time_base_abs = getattr(v3, "time_base_abs", None)
        if not isinstance(time_base_abs, (int, float)):
            return
        try:
            t_rel = v3.predicted_land_time_rel()
        except Exception:
            t_rel = None
        if t_rel is None:
            return
        try:
            tr.predicted_land_time_abs = float(time_base_abs + float(t_rel))
        except Exception:
            pass

    def _speed_mps(self, *, prev_pos: tuple[float, float, float] | None, prev_t: float | None, pos: tuple[float, float, float], t: float) -> float | None:
        if prev_pos is None or prev_t is None:
            return None
        dt = float(t - float(prev_t))
        if dt < float(self._cfg.min_dt_s):
            return None
        d = _dist3(pos, prev_pos)
        return float(d / dt) if dt > 0 else None

    def _episode_trim_recent(self, tr: _Track, *, now_t_abs: float) -> None:
        if not tr.recent:
            return
        if float(self._cfg.episode_buffer_s) <= 0:
            tr.recent.clear()
            return
        t_min = float(now_t_abs) - float(self._cfg.episode_buffer_s)
        while tr.recent and float(tr.recent[0].t_abs) < t_min:
            tr.recent.popleft()

    def _episode_try_start(self, tr: _Track, *, events: list[dict[str, Any]]) -> None:
        """尝试从 recent 缓冲中判定 episode 开始。

        说明：
            - 起始判定固定为：z 方向位移/速度门控 + y 轴重力一致性（ay≈g）。
            - 该规则用于抑制准备阶段的小幅抖动与误检导致的“伪轨迹”。
        """

        if not bool(self._cfg.episode_enabled):
            return
        if bool(tr.episode_active):
            return
        # y 轴二次拟合需要至少 3 个点。
        if int(self._cfg.episode_min_obs) < 3:
            return
        if len(tr.recent) < int(self._cfg.episode_min_obs):
            return

        # 在 recent 中找一个长度为 episode_min_obs 的窗口（取最后一个窗口即可，够用且稳定）。
        k = int(self._cfg.episode_min_obs)
        window = list(tr.recent)[-k:]
        t0 = float(window[0].t_abs)
        t1 = float(window[-1].t_abs)
        if t1 <= t0:
            return

        # 基础统计（用于诊断输出）。
        p0 = window[0].pos
        p1 = window[-1].pos
        dx = float(p1[0] - p0[0])
        dy = float(p1[1] - p0[1])
        dz = float(p1[2] - p0[2])
        dt = float(t1 - t0)
        if dt <= 0:
            return

        disp3 = float(math.sqrt(dx * dx + dy * dy + dz * dz))
        avg_speed3 = float(disp3 / dt)

        abs_dz = float(abs(dz))
        abs_vz = float(abs_dz / dt)

        # z 方向：必须有足够的前后位移/速度。
        if float(abs_dz) < float(self._cfg.episode_min_abs_dz_m):
            return
        if float(abs_vz) < float(self._cfg.episode_min_abs_vz_mps):
            return
        z_dir = int(self._cfg.episode_z_dir)
        if z_dir not in {-1, 0, 1}:
            z_dir = 0
        if z_dir != 0:
            # 方向要求：例如 z_dir=-1 表示 z 必须减小（朝机器人）。
            if float(dz) * float(z_dir) <= 0:
                return

        # y 轴重力一致性：ay≈g。
        ay = _estimate_const_accel_y(window)
        if ay is None or not math.isfinite(float(ay)):
            return
        g = float(self._cfg.episode_gravity_mps2)
        tol = float(self._cfg.episode_gravity_tol_mps2)
        # 默认用“大小接近 g”判据，避免 y 轴方向差异。
        if abs(abs(float(ay)) - abs(g)) > float(tol):
            return

        # 确认 episode 开始：将 start 追溯到窗口起点。
        tr.episode_id += 1
        tr.episode_active = True
        tr.episode_start_t_abs = float(t0)
        tr.episode_end_t_abs = None
        tr.episode_end_reason = None
        tr.last_motion_t_abs = float(t1)

        events.append(
            {
                "event": "episode_start",
                "track_id": int(tr.track_id),
                "episode_id": int(tr.episode_id),
                "t_abs": float(t0),
                "avg_speed3_mps": float(avg_speed3),
                "displacement3_m": float(disp3),
                "dz_m": float(dz),
                "abs_vz_mps": float(abs_vz),
                "ay_mps2": float(ay),
            }
        )

        # 可选：把窗口内观测重放给 predictor，让拟合从“episode 开始”一致。
        if bool(self._cfg.reset_predictor_on_episode_start):
            self._reset_track_predictors(tr)
            if bool(self._cfg.feed_curve_only_when_episode_active):
                # 若之前没有喂给 curve，则此处回放 recent 窗口即可。
                for o in window:
                    self._feed_track_predictors(tr, o)

    def _episode_try_end(self, tr: _Track, *, now_t_abs: float, events: list[dict[str, Any]]) -> None:
        """尝试判定 episode 结束。"""

        if not bool(self._cfg.episode_enabled):
            return
        if not bool(tr.episode_active):
            return

        # 结束条件1：超过预测落地时间。
        if tr.predicted_land_time_abs is not None:
            if float(now_t_abs) >= float(tr.predicted_land_time_abs) + float(self._cfg.episode_end_after_predicted_land_s):
                tr.episode_active = False
                tr.episode_end_t_abs = float(tr.predicted_land_time_abs)
                tr.episode_end_reason = "after_predicted_land"
                events.append(
                    {
                        "event": "episode_end",
                        "track_id": int(tr.track_id),
                        "episode_id": int(tr.episode_id),
                        "t_abs": float(tr.episode_end_t_abs),
                        "reason": str(tr.episode_end_reason),
                    }
                )
                if bool(self._cfg.reset_predictor_on_episode_end):
                    self._reset_track_predictors(tr)
                return

        # 结束条件2：长时间低速（准备阶段/手持晃动）。
        if tr.last_motion_t_abs is not None:
            if float(now_t_abs - float(tr.last_motion_t_abs)) >= float(self._cfg.episode_end_if_stationary_s):
                tr.episode_active = False
                tr.episode_end_t_abs = float(now_t_abs)
                tr.episode_end_reason = "stationary"
                events.append(
                    {
                        "event": "episode_end",
                        "track_id": int(tr.track_id),
                        "episode_id": int(tr.episode_id),
                        "t_abs": float(tr.episode_end_t_abs),
                        "reason": str(tr.episode_end_reason),
                    }
                )
                if bool(self._cfg.reset_predictor_on_episode_end):
                    self._reset_track_predictors(tr)

    def _feed_track_predictors(self, tr: _Track, o: _RecentObs) -> None:
        """把单条观测喂给各 predictor（按 primary/compare 配置）。"""

        # v3：新 API（落点/走廊）。
        if tr.v3 is not None and self._BallObservation is not None:
            try:
                obs = self._BallObservation(  # type: ignore[misc]
                    x=float(o.x),
                    y=float(o.y),
                    z=float(o.z),
                    t=float(o.t_abs),
                    conf=o.conf,
                )
                tr.v3.add_observation(obs)
            except Exception:
                pass

        # v2/v3_legacy：legacy API（接球点候选），记录“最后一次输出”。
        if tr.v2 is not None:
            try:
                pts = tr.v2.add_frame(
                    [float(o.x), float(o.y), float(o.z), float(o.t_abs)],
                    is_bot_fire=int(self._cfg.is_bot_fire),
                )
            except Exception:
                pts = None
            setattr(tr, "_last_v2_points", pts)

        if tr.v3_legacy is not None:
            try:
                pts = tr.v3_legacy.add_frame(
                    [float(o.x), float(o.y), float(o.z), float(o.t_abs)],
                    is_bot_fire=int(self._cfg.is_bot_fire),
                )
            except Exception:
                pts = None
            setattr(tr, "_last_v3_legacy_points", pts)

    def _transform_y(self, y: float) -> float:
        """对输入到 curve 的 y 做可选变换。

        公式：y' = sign * (y - offset)
        其中 sign = -1 当 cfg.y_negate=True，否则为 +1。
        """

        y2 = float(y) - float(self._cfg.y_offset_m)
        if bool(self._cfg.y_negate):
            y2 = -float(y2)
        return float(y2)

    def process_record(self, out_rec: dict[str, Any]) -> dict[str, Any]:
        """处理单条定位输出记录，返回增强后的记录。"""

        if not bool(self._cfg.enabled):
            return out_rec

        self._ensure_curve_imports()
        t_abs, t_source = _choose_t_abs(out_rec)

        track_events: list[dict[str, Any]] = []
        ended_tracks: list[dict[str, Any]] = []

        # 先清理长期未更新的 track。
        if t_abs > 0:
            alive: list[_Track] = []
            for tr in self._tracks:
                if tr.last_t_abs is None:
                    alive.append(tr)
                    continue
                if float(t_abs - tr.last_t_abs) <= float(self._cfg.max_missed_s):
                    alive.append(tr)
                    continue

                # track 被丢弃：若 episode 正在进行，输出一次结束事件与最终快照。
                if bool(self._cfg.episode_enabled):
                    if bool(tr.episode_active):
                        tr.episode_active = False
                        tr.episode_end_t_abs = float(tr.last_t_abs)
                        tr.episode_end_reason = "track_drop_max_missed"
                        track_events.append(
                            {
                                "event": "episode_end",
                                "track_id": int(tr.track_id),
                                "episode_id": int(tr.episode_id),
                                "t_abs": float(tr.episode_end_t_abs),
                                "reason": str(tr.episode_end_reason),
                            }
                        )
                    track_events.append(
                        {
                            "event": "track_drop",
                            "track_id": int(tr.track_id),
                            "t_abs": float(t_abs),
                            "reason": "max_missed_s",
                        }
                    )
                    ended_tracks.append(_track_snapshot(tr, self._cfg))

                # 若锁定 track 被丢弃，则解除锁定。
                if self._episode_locked_track_id is not None and int(tr.track_id) == int(self._episode_locked_track_id):
                    self._episode_locked_track_id = None
            self._tracks = alive

        meas = _extract_meas_list(out_rec, self._cfg)
        assignments: list[dict[str, Any]] = []
        updated_tracks: list[_Track] = []

        used_track_ids: set[int] = set()

        # 逐球贪心分配：质量高的先处理。
        for m in meas:
            y_in = self._transform_y(float(m.y))
            pos = (float(m.x), float(y_in), float(m.z))

            best_tr: _Track | None = None
            best_d = float("inf")
            for tr in self._tracks:
                if tr.track_id in used_track_ids:
                    continue
                if tr.last_pos is None or tr.last_t_abs is None:
                    continue
                dt = float(t_abs - float(tr.last_t_abs))
                if dt < float(self._cfg.min_dt_s):
                    continue
                pred = _predict_track_pos(tr, t_abs=float(t_abs), min_dt_s=float(self._cfg.min_dt_s))
                if pred is None:
                    continue
                d = _dist3(pos, pred)
                if d <= float(self._cfg.association_dist_m) and d < best_d:
                    best_d = d
                    best_tr = tr

            if best_tr is None:
                # 若处于 episode 单 track 锁定状态，则不再创建新 track。
                if self._episode_locked_track_id is not None:
                    continue
                if len(self._tracks) >= int(self._cfg.max_tracks):
                    # track 已满：不创建新 track，跳过该点。
                    continue

                tr = _Track(track_id=int(self._next_track_id), created_t_abs=float(t_abs))
                self._next_track_id += 1

                # 创建拟合器（primary + 可选 compare）。
                if self._cfg.primary == "v3" and self._CurvePredictorV3 is not None:
                    tr.v3 = self._new_v3()

                if (self._cfg.primary == "v2" or self._cfg.compare_v2) and self._CurveV2 is not None:
                    tr.v2 = self._new_v2()

                if (self._cfg.primary == "v3_legacy" or self._cfg.compare_v3_legacy) and self._CurveV3Legacy is not None:
                    tr.v3_legacy = self._new_v3_legacy()

                self._tracks.append(tr)
                best_tr = tr

            # 更新 track
            if best_tr.last_t_abs is not None:
                dt = float(t_abs - float(best_tr.last_t_abs))
                if dt < float(self._cfg.min_dt_s):
                    continue

            conf = _obs_conf(self._cfg, m.quality)
            obs = _RecentObs(t_abs=float(t_abs), x=float(m.x), y=float(y_in), z=float(m.z), conf=conf)

            # 运动学诊断：更新速度与 EWMA。
            sp = self._speed_mps(prev_pos=best_tr.last_pos, prev_t=best_tr.last_t_abs, pos=pos, t=float(t_abs))
            if sp is not None and math.isfinite(float(sp)):
                best_tr.last_speed_mps = float(sp)
                if best_tr.speed_ewma_mps is None:
                    best_tr.speed_ewma_mps = float(sp)
                else:
                    # 说明：EWMA 系数固定为 0.2，足够平滑且无需额外配置。
                    best_tr.speed_ewma_mps = 0.8 * float(best_tr.speed_ewma_mps) + 0.2 * float(sp)

                # “运动”定义：速度超过 stationary 阈值才认为真的在动。
                if (not bool(self._cfg.episode_enabled)) or float(sp) >= float(self._cfg.episode_stationary_speed_mps):
                    best_tr.last_motion_t_abs = float(t_abs)

            # 维护 recent 缓冲（用于 episode 判定/回放）。
            best_tr.recent.append(obs)
            self._episode_trim_recent(best_tr, now_t_abs=float(t_abs))

            # episode 起始判定（可能触发 predictor reset + 回放）。
            was_episode_active = bool(best_tr.episode_active)
            self._episode_try_start(best_tr, events=track_events)

            # 若本次触发了 episode_start，且启用了“单 track 锁定”，则丢弃其它 track。
            if (not was_episode_active) and bool(best_tr.episode_active):
                self._episode_lock_to_track(
                    keep_track_id=int(best_tr.track_id),
                    now_t_abs=float(t_abs),
                    events=track_events,
                    ended_tracks=ended_tracks,
                )

            # 是否喂给 curve：
            # - 默认与旧行为一致（始终喂）。
            # - 若启用 feed_curve_only_when_episode_active，则仅在 episode_active 时喂。
            should_feed = True
            if bool(self._cfg.episode_enabled) and bool(self._cfg.feed_curve_only_when_episode_active):
                should_feed = bool(best_tr.episode_active)

            if should_feed:
                self._feed_track_predictors(best_tr, obs)
                self._maybe_update_predicted_land_time_abs(best_tr)

            # episode 结束判定（可能 reset predictor）。
            self._episode_try_end(best_tr, now_t_abs=float(t_abs), events=track_events)

            # 若锁定的 episode 已结束，则解除锁定。
            if bool(self._cfg.episode_enabled) and bool(self._cfg.episode_lock_single_track):
                self._episode_maybe_release_lock(best_tr)

            # 维护 prev/last，用于下一帧的速度预测。
            best_tr.prev_t_abs = best_tr.last_t_abs
            best_tr.prev_pos = best_tr.last_pos
            best_tr.last_t_abs = float(t_abs)
            best_tr.last_pos = pos
            best_tr.n_obs += 1

            used_track_ids.add(int(best_tr.track_id))
            updated_tracks.append(best_tr)
            assignments.append({"ball_id": int(m.ball_id), "track_id": int(best_tr.track_id), "dist_m": float(best_d) if best_d < float("inf") else None})

        # 若发生了 episode 单 track 锁定，确保输出与回写只包含当前仍存活的 track。
        active_ids = {int(tr.track_id) for tr in self._tracks}
        if active_ids:
            updated_tracks = [tr for tr in updated_tracks if int(tr.track_id) in active_ids]
            assignments = [a for a in assignments if int(a.get("track_id", -1)) in active_ids]

        # 把 track_id 回写到 balls（便于下游按球筛选/回放）。
        balls = out_rec.get("balls")
        if isinstance(balls, list):
            bid_to_tid = {int(a["ball_id"]): int(a["track_id"]) for a in assignments}
            for b in balls:
                if isinstance(b, dict):
                    bid = int(b.get("ball_id", -1))
                    if bid in bid_to_tid:
                        b["curve_track_id"] = int(bid_to_tid[bid])

        # 输出增强字段
        out_rec["curve"] = {
            "primary": str(self._cfg.primary),
            "t_abs": float(t_abs),
            "t_source": str(t_source),
            "assignments": assignments,
            "track_updates": [_track_snapshot(tr, self._cfg) for tr in updated_tracks],
            "num_active_tracks": int(len(self._tracks)),
        }

        if bool(self._cfg.episode_enabled):
            out_rec["curve"]["track_events"] = track_events
            out_rec["curve"]["ended_tracks"] = ended_tracks

        return out_rec


def apply_curve_stage(records: Iterable[dict[str, Any]], cfg: CurveStageConfig) -> Iterable[dict[str, Any]]:
    """对记录流应用 curve stage（保持 generator 风格）。"""

    stage = CurveStage(cfg)
    for r in records:
        if not isinstance(r, dict):
            continue
        yield stage.process_record(r)
