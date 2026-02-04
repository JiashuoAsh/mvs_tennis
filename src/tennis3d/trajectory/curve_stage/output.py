from __future__ import annotations

import math
from typing import Any

from tennis3d.trajectory.curve_stage.config import CurveStageConfig
from tennis3d.trajectory.curve_stage.obs_extract import _as_float
from tennis3d.trajectory.curve_stage.models import _Track


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
