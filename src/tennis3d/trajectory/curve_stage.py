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

from dataclasses import dataclass
from typing import Any, Iterable

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


@dataclass
class _BallMeas:
    ball_id: int
    x: float
    y: float
    z: float
    quality: float


@dataclass
class _Track:
    track_id: int
    last_t_abs: float | None = None
    last_pos: tuple[float, float, float] | None = None
    prev_t_abs: float | None = None
    prev_pos: tuple[float, float, float] | None = None
    n_obs: int = 0

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


def _extract_meas_list(out_rec: dict[str, Any]) -> list[_BallMeas]:
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
        out.append(_BallMeas(ball_id=bid, x=float(x), y=float(y), z=float(z), quality=float(q)))

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

        # 先清理长期未更新的 track。
        if t_abs > 0:
            alive: list[_Track] = []
            for tr in self._tracks:
                if tr.last_t_abs is None:
                    alive.append(tr)
                    continue
                if float(t_abs - tr.last_t_abs) <= float(self._cfg.max_missed_s):
                    alive.append(tr)
            self._tracks = alive

        meas = _extract_meas_list(out_rec)
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
                if len(self._tracks) >= int(self._cfg.max_tracks):
                    # track 已满：不创建新 track，跳过该点。
                    continue

                tr = _Track(track_id=int(self._next_track_id))
                self._next_track_id += 1

                # 创建拟合器（primary + 可选 compare）。
                if self._cfg.primary == "v3" and self._CurvePredictorV3 is not None:
                    tr.v3 = self._CurvePredictorV3()  # type: ignore[operator]

                if (self._cfg.primary == "v2" or self._cfg.compare_v2) and self._CurveV2 is not None:
                    tr.v2 = self._CurveV2()  # type: ignore[operator]

                if (self._cfg.primary == "v3_legacy" or self._cfg.compare_v3_legacy) and self._CurveV3Legacy is not None:
                    tr.v3_legacy = self._CurveV3Legacy()  # type: ignore[operator]

                self._tracks.append(tr)
                best_tr = tr

            # 更新 track
            if best_tr.last_t_abs is not None:
                dt = float(t_abs - float(best_tr.last_t_abs))
                if dt < float(self._cfg.min_dt_s):
                    continue

            conf = _obs_conf(self._cfg, m.quality)

            # v3：新 API（落点/走廊）。
            if best_tr.v3 is not None and self._BallObservation is not None:
                try:
                    obs = self._BallObservation(  # type: ignore[misc]
                        x=float(m.x),
                        y=float(y_in),
                        z=float(m.z),
                        t=float(t_abs),
                        conf=conf,
                    )
                    best_tr.v3.add_observation(obs)
                except Exception:
                    pass

            # v2/v3_legacy：legacy API（接球点候选），记录“最后一次输出”。
            if best_tr.v2 is not None:
                try:
                    pts = best_tr.v2.add_frame(
                        [float(m.x), float(y_in), float(m.z), float(t_abs)],
                        is_bot_fire=int(self._cfg.is_bot_fire),
                    )
                except Exception:
                    pts = None
                setattr(best_tr, "_last_v2_points", pts)

            if best_tr.v3_legacy is not None:
                try:
                    pts = best_tr.v3_legacy.add_frame(
                        [float(m.x), float(y_in), float(m.z), float(t_abs)],
                        is_bot_fire=int(self._cfg.is_bot_fire),
                    )
                except Exception:
                    pts = None
                setattr(best_tr, "_last_v3_legacy_points", pts)

            # 维护 prev/last，用于下一帧的速度预测。
            best_tr.prev_t_abs = best_tr.last_t_abs
            best_tr.prev_pos = best_tr.last_pos
            best_tr.last_t_abs = float(t_abs)
            best_tr.last_pos = pos
            best_tr.n_obs += 1

            used_track_ids.add(int(best_tr.track_id))
            updated_tracks.append(best_tr)
            assignments.append({"ball_id": int(m.ball_id), "track_id": int(best_tr.track_id), "dist_m": float(best_d) if best_d < float("inf") else None})

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

        return out_rec


def apply_curve_stage(records: Iterable[dict[str, Any]], cfg: CurveStageConfig) -> Iterable[dict[str, Any]]:
    """对记录流应用 curve stage（保持 generator 风格）。"""

    stage = CurveStage(cfg)
    for r in records:
        if not isinstance(r, dict):
            continue
        yield stage.process_record(r)
