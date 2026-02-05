from __future__ import annotations

import math
from typing import Any

from tennis3d.trajectory.curve_stage.config import CurveStageConfig
from tennis3d.trajectory.curve_stage.models import _BallMeas


def _as_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _choose_t_abs(out_rec: dict[str, Any]) -> tuple[float, str]:
    """选择 curve stage 使用的绝对时间轴。"""

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
