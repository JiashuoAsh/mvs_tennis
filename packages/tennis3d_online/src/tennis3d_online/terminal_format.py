"""在线模式的终端输出格式化（纯函数，便于单测）。"""

from __future__ import annotations

from typing import Any


def _format_float3(x: object) -> str:
    if not isinstance(x, (list, tuple)) or len(x) != 3:
        return str(x)
    try:
        a, b, c = float(x[0]), float(x[1]), float(x[2])
    except Exception:
        return str(x)
    return f"({a:.4f}, {b:.4f}, {c:.4f})"


def _format_xyz(x: object) -> str:
    if not isinstance(x, (list, tuple)) or len(x) != 3:
        return str(x)
    try:
        a, b, c = float(x[0]), float(x[1]), float(x[2])
    except Exception:
        return str(x)
    return f"(x={a:.4f}, y={b:.4f}, z={c:.4f})"


def _format_delta_map_ms(d: object) -> str:
    if not isinstance(d, dict) or not d:
        return str(d)

    items: list[tuple[str, float]] = []
    for k, v in d.items():
        try:
            items.append((str(k), float(v)))
        except Exception:
            continue
    items.sort(key=lambda kv: kv[0])

    inner = ", ".join(f"{k}:{v:+.3f}" for k, v in items)
    return "{" + inner + "}"


def _format_best_ball_line(out_rec: dict[str, Any]) -> str | None:
    balls = out_rec.get("balls") or []
    if not isinstance(balls, list) or not balls:
        return None

    gi = out_rec.get("group_index")
    t_abs = out_rec.get("capture_t_abs")

    best = balls[0] if isinstance(balls[0], dict) else None
    best_xw = best.get("ball_3d_world") if best is not None else None
    best_used = best.get("used_cameras") if best is not None else None
    best_q = best.get("quality") if best is not None else None
    best_nv = best.get("num_views") if best is not None else None

    best_err_mean = None
    if best is not None:
        reproj = best.get("reprojection_errors")
        if isinstance(reproj, list) and reproj:
            vals: list[float] = []
            for e in reproj:
                if not isinstance(e, dict):
                    continue
                v = e.get("error_px")
                if v is None:
                    continue
                try:
                    vals.append(float(v))
                except Exception:
                    continue
            if vals:
                best_err_mean = float(sum(vals) / len(vals))
        if best_err_mean is None:
            med = best.get("median_reproj_error_px")
            if med is not None:
                try:
                    best_err_mean = float(med)
                except Exception:
                    best_err_mean = None

    t_part = f"t={float(t_abs):.6f} " if t_abs is not None else ""
    q_part = f" q={float(best_q):.3f}" if best_q is not None else ""
    nv_part = f" views={int(best_nv)}" if best_nv is not None else ""
    err_part = f" err_mean={float(best_err_mean):.2f}px" if best_err_mean is not None else ""

    dt_part = ""
    dt_raw = out_rec.get("time_mapping_host_ms_spread_ms")
    dt_map = out_rec.get("time_mapping_mapped_host_ms_spread_ms")
    dt_map_by_cam = out_rec.get("time_mapping_mapped_host_ms_delta_to_median_by_camera")

    if dt_raw is not None:
        try:
            dt_part += f" dt_raw={float(dt_raw):.3f}ms"
        except Exception:
            pass
    if dt_map is not None:
        try:
            dt_part += f" dt_map={float(dt_map):.3f}ms"
        except Exception:
            pass
    if dt_map_by_cam is not None:
        dt_part += f" dt_map_by_cam={_format_delta_map_ms(dt_map_by_cam)}"

    return (
        f"{t_part}group={gi} xyz_w={_format_xyz(best_xw)}"
        f"{q_part}{nv_part}{err_part}{dt_part} used={best_used}"
    )


def _format_all_balls_lines(out_rec: dict[str, Any]) -> list[str]:
    balls = out_rec.get("balls") or []
    if not isinstance(balls, list) or not balls:
        return []

    gi = out_rec.get("group_index")
    t_abs = out_rec.get("capture_t_abs")

    t_part = f"t={float(t_abs):.6f} " if t_abs is not None else ""

    dt_part = ""
    dt_raw = out_rec.get("time_mapping_host_ms_spread_ms")
    dt_map = out_rec.get("time_mapping_mapped_host_ms_spread_ms")
    dt_map_by_cam = out_rec.get("time_mapping_mapped_host_ms_delta_to_median_by_camera")
    if dt_raw is not None:
        try:
            dt_part += f" dt_raw={float(dt_raw):.3f}ms"
        except Exception:
            pass
    if dt_map is not None:
        try:
            dt_part += f" dt_map={float(dt_map):.3f}ms"
        except Exception:
            pass
    if dt_map_by_cam is not None:
        dt_part += f" dt_map_by_cam={_format_delta_map_ms(dt_map_by_cam)}"

    header = f"{t_part}group={gi} balls={len(balls)}{dt_part}"
    lines: list[str] = [header]

    for i, b in enumerate(balls):
        if not isinstance(b, dict):
            lines.append(f"  - ball[{i}]=<invalid>")
            continue

        bid = b.get("ball_id", i)
        xw = b.get("ball_3d_world")
        used = b.get("used_cameras")
        q = b.get("quality")
        nv = b.get("num_views")
        tid = b.get("curve_track_id")

        err_mean = None
        reproj = b.get("reprojection_errors")
        if isinstance(reproj, list) and reproj:
            vals: list[float] = []
            for e in reproj:
                if not isinstance(e, dict):
                    continue
                v = e.get("error_px")
                if v is None:
                    continue
                try:
                    vals.append(float(v))
                except Exception:
                    continue
            if vals:
                err_mean = float(sum(vals) / len(vals))
        if err_mean is None:
            med = b.get("median_reproj_error_px")
            if med is not None:
                try:
                    err_mean = float(med)
                except Exception:
                    err_mean = None

        q_part = f" q={float(q):.3f}" if q is not None else ""
        nv_part = f" views={int(nv)}" if nv is not None else ""
        err_part = f" err_mean={float(err_mean):.2f}px" if err_mean is not None else ""
        tid_part = f" track={int(tid)}" if tid is not None else ""

        lines.append(
            f"  - id={bid}{tid_part} xyz_w={_format_xyz(xw)}{q_part}{nv_part}{err_part} used={used}"
        )

    return lines
