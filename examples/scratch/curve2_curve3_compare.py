"""离线对比：curve2 vs curve3（基于现有 jsonl 输出）。

目标：
- 读取在线/离线输出 jsonl（例如 `data/tools_output/online_positions_3d.master_slave.jsonl`）。
- 按 track_id 抽取时序 3D 点序列（优先使用 `curve.track_updates[*].last_pos` + `t_abs`）。
- 参考 `curve.curve_v2.ball_tracer.BallTracer.is_user_return_ball` 的同口径规则，
  过滤掉“不是用户回球曲线段”的早期点。
- 用 curve2 与 curve3 分别拟合，并输出：
  1) 对比 JSON 报告（落地点、y=target_y 的击球点/走廊等）。
  2) 单文件 HTML 可视化（点云 + 拟合曲线；以及收敛过程曲线）。

说明：
- 本脚本不新增第三方依赖（仅用标准库 + NumPy；NumPy 已是项目依赖）。
- 这里的“落地点”定义为第一次触地/反弹点（y=0）。
- 这里的“击球点”定义为 y=target_y 平面（默认 0.7m）处轨迹穿越位置。

运行示例：
    # 1) 直接把整个 jsonl 当作“一球轨迹”来拟合（不按 track_id 过滤，推荐）
    uv run python examples/scratch/curve2_curve3_compare.py --jsonl "data/tools_output/online points.jsonl"

    # 2) 如果你的 jsonl 确实包含多条轨迹，也可以指定 track_id
    uv run python examples/scratch/curve2_curve3_compare.py --jsonl data/tools_output/online_positions_3d.master_slave.jsonl --track-id 1
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from curve.curve_v2.bot_motion_config import BotMotionConfig
from curve.curve_v2.curve2 import Curve as CurveV2
from curve.curve_v3.config import CurveV3Config
from curve.curve_v3.core import CurvePredictorV3
from curve.curve_v3.types import BallObservation
from mvs.session.metadata_io import iter_metadata_records


@dataclass(frozen=True)
class ObsPoint:
    """单个观测点（世界坐标系，绝对时间）。

    Args:
        x: 横向坐标（m）。
        y: 高度（m）。
        z: 纵向坐标（m）。
        t_abs: 绝对时间戳（s）。
        n_obs: 在线侧 track 的累计观测数（若能从日志里取到；否则为 None）。
    """

    x: float
    y: float
    z: float
    t_abs: float
    n_obs: int | None = None


def _estimate_observed_hit_on_plane_y(
    points: list[ObsPoint], *, target_y: float, t_min_abs: float | None = None, pick: str = "first"
) -> dict[str, Any] | None:
    """从离散观测点中估计 y=target_y 平面处的交点（观测击球点）。

    说明：离线数据里通常只有有限帧观测点，且不保证严格覆盖到 y=target_y。
    这里采用“优先线性插值，否则取最近点”的保守策略：

    - 若存在相邻两点 (p0,p1) 满足 y0 与 y1 跨过 target_y，则用时间轴做线性插值，
      得到 (x,z,t_abs)；并把 y 固定为 target_y。
    - 若不存在穿越，则选择 |y-target_y| 最小的点作为近似，并标记 method=closest。

    Args:
        points: 观测点序列（按时间递增）。
        target_y: 目标高度（米）。
        t_min_abs: 仅使用 t_abs >= t_min_abs 的点来估计（用于只取反弹后/第二段）。
        pick: 在存在多个穿越时选择哪个：first=最早穿越（通常对应上升段），last=最晚穿越（通常对应下降段）。

    Returns:
        估计结果字典（含 t_abs/x/y/z/method 等），若无点则返回 None。
    """

    if not points:
        return None

    if t_min_abs is not None and math.isfinite(float(t_min_abs)):
        t0 = float(t_min_abs)
        pts = [p for p in points if float(p.t_abs) >= t0]
    else:
        pts = list(points)

    if len(pts) < 2:
        return None

    ty = float(target_y)

    # 先收集所有穿越点，再按 pick 选择。
    crosses: list[dict[str, Any]] = []
    for i in range(len(pts) - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        y0 = float(p0.y)
        y1 = float(p1.y)
        if not (math.isfinite(y0) and math.isfinite(y1)):
            continue
        if y0 == y1:
            continue

        # 覆盖两种方向的穿越。
        if (y0 - ty) * (y1 - ty) > 0.0:
            continue

        a = (ty - y0) / (y1 - y0)
        if not math.isfinite(float(a)):
            continue
        a = float(min(max(float(a), 0.0), 1.0))
        t_abs = float(p0.t_abs + a * (p1.t_abs - p0.t_abs))
        x = float(p0.x + a * (p1.x - p0.x))
        z = float(p0.z + a * (p1.z - p0.z))
        crosses.append(
            {
                "target_y": ty,
                "method": "linear_interp",
                "i0": int(i),
                "i1": int(i + 1),
                "t_abs": t_abs,
                "x": x,
                "y": ty,
                "z": z,
            }
        )

    pick = str(pick).strip().lower() or "first"
    if crosses:
        return crosses[0] if pick == "first" else crosses[-1]

    # 找不到穿越：取 |y-target_y| 最小的点。
    best_i = 0
    best_d = float("inf")
    for i, p in enumerate(pts):
        d = abs(float(p.y) - ty)
        if d < best_d:
            best_d = d
            best_i = i
    p = pts[best_i]
    return {
        "target_y": ty,
        "method": "closest",
        "i": int(best_i),
        "t_abs": float(p.t_abs),
        "x": float(p.x),
        "y": float(p.y),
        "z": float(p.z),
        "abs_dy": float(best_d),
    }


def _nan(v: Any) -> float:
    """把可能为 None 的数值转换为 float/NaN。"""

    if v is None:
        return float("nan")
    try:
        x = float(v)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def _as_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _as_vec3(x: Any) -> tuple[float, float, float] | None:
    if not isinstance(x, list) or len(x) != 3:
        return None
    a = _as_float(x[0])
    b = _as_float(x[1])
    c = _as_float(x[2])
    if a is None or b is None or c is None:
        return None
    return float(a), float(b), float(c)


def _safe_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def _extract_track_update(curve: dict[str, Any], track_id: int) -> dict[str, Any] | None:
    tu = curve.get("track_updates")
    if not isinstance(tu, list):
        return None
    for u in tu:
        if not isinstance(u, dict):
            continue
        tid = _safe_int(u.get("track_id"))
        if tid is None or tid != int(track_id):
            continue
        return u
    return None


def _extract_best_track_update(curve: dict[str, Any]) -> dict[str, Any] | None:
    """在不指定 track_id 时，选择“最像主轨迹”的 track_update。

    经验规则：优先选择 n_obs 最大的那条（通常是持续时间最长、最完整的轨迹）。
    """

    tu = curve.get("track_updates")
    if not isinstance(tu, list) or not tu:
        return None

    best: dict[str, Any] | None = None
    best_n = -1
    for u in tu:
        if not isinstance(u, dict):
            continue
        n = _safe_int(u.get("n_obs"))
        n_val = int(n) if n is not None else 0
        if best is None or n_val > best_n:
            best = u
            best_n = n_val
    return best


def _extract_record_t_abs(rec: dict[str, Any], curve: dict[str, Any] | None) -> float | None:
    """从一条记录中提取绝对时间戳（秒）。

    优先级：
    1) curve.t_abs（在线侧通常把这个作为统一时间基准）
    2) capture_t_abs（捕获时刻）
    3) rec.t_abs
    4) created_at（兜底：写入时间，可能与捕获时间不同）
    """

    if isinstance(curve, dict):
        t = _as_float(curve.get("t_abs"))
        if t is not None:
            return float(t)
    t = _as_float(rec.get("capture_t_abs"))
    if t is not None:
        return float(t)
    t = _as_float(rec.get("t_abs"))
    if t is not None:
        return float(t)
    t = _as_float(rec.get("created_at"))
    if t is not None:
        return float(t)
    return None


def _iter_track_points(
    *,
    jsonl_path: Path,
    track_id: int | None,
    max_records: int,
    session_gap_s: float = 2.0,
    session_pick: str = "longest",
    point_source: str = "auto",
) -> list[ObsPoint]:
    """从 jsonl 中抽取一条轨迹的时序点序列。

    - 若指定 track_id：只抽取该 track。
    - 若不指定 track_id：把整个 jsonl 当作“一球轨迹”，不按 track 过滤。

    约定：优先使用 `curve.track_updates[*].last_pos`（该字段通常已经包含 y 轴修正）。
    如无法取得，则退化为 `balls[0].ball_3d_world` 并做保守的 y 轴符号修正。

    Args:
        jsonl_path: 输入 jsonl 路径。
        track_id: 目标 track id；None 表示不按 track 过滤。
        max_records: 最多处理 N 条记录（0 表示不限制）。
        point_source: "auto" | "track_updates" | "balls"。

    Notes:
        同一个 jsonl 可能包含多次运行/多个回合的输出；同一 track_id 会在时间上出现
        明显断层（例如几分钟/几小时后再次从 n_obs=1 开始）。

        为避免把多个回合混在一起导致拟合/过滤异常，这里按相邻点时间间隔切分成多个
        session 段：当 dt > session_gap_s 时开启新段。然后按 session_pick 选择：
        - "longest"：选择点数最多的一段（默认，通常对应“完整的一球”）
        - "last"：选择最后一段（通常对应“最新的一球”）

    Returns:
        选中的点序列（内部已按时间递增）。
    """

    session_gap_s = float(session_gap_s)
    if session_gap_s <= 0:
        session_gap_s = 2.0
    session_pick = str(session_pick).strip().lower() or "longest"

    point_source = str(point_source).strip().lower() or "auto"
    segments: list[list[ObsPoint]] = []
    cur: list[ObsPoint] = []
    last_t: float | None = None

    seen = 0
    for rec in iter_metadata_records(Path(jsonl_path)):
        if not isinstance(rec, dict):
            continue
        seen += 1
        if max_records > 0 and seen > max_records:
            break

        curve = rec.get("curve")
        if not isinstance(curve, dict):
            continue

        t_abs_rec = _extract_record_t_abs(rec, curve)
        if t_abs_rec is None:
            continue

        u: dict[str, Any] | None = None
        if point_source != "balls":
            if track_id is not None:
                u = _extract_track_update(curve, int(track_id))
            else:
                u = _extract_best_track_update(curve)

        p3: tuple[float, float, float] | None = None
        n_obs: int | None = None
        t_abs: float | None = None

        if u is not None and point_source in {"auto", "track_updates"}:
            # 注意：部分日志里 track_update 没有 t_abs，只有 last_t_abs。
            t_abs = _as_float(u.get("last_t_abs")) or _as_float(u.get("t_abs")) or float(t_abs_rec)
            p3 = _as_vec3(u.get("last_pos"))
            n_obs = _safe_int(u.get("n_obs"))

        if p3 is None and point_source in {"auto", "balls"}:
            balls = rec.get("balls")
            if isinstance(balls, list) and balls:
                b0 = balls[0]
                if isinstance(b0, dict):
                    p3 = _as_vec3(b0.get("ball_3d_world"))
                    if p3 is not None:
                        # 说明：部分输出的 world_y 方向与拟合所需的 y（高度）相反。
                        # 这里用“若为负则取反”的保守规则，避免把高度当成负数。
                        y = float(-p3[1]) if float(p3[1]) < 0.0 else float(p3[1])
                        p3 = (float(p3[0]), y, float(p3[2]))
            t_abs = float(t_abs_rec)

        if t_abs is None or p3 is None:
            continue

        if last_t is not None:
            if t_abs <= last_t:
                # 保守去重/去乱序：在线日志里可能出现重复时间戳。
                continue
            if float(t_abs - last_t) > float(session_gap_s) and cur:
                # 时间断层：开启新 session 段。
                segments.append(cur)
                cur = []

        cur.append(ObsPoint(x=p3[0], y=p3[1], z=p3[2], t_abs=t_abs, n_obs=n_obs))
        last_t = t_abs

    if cur:
        segments.append(cur)

    if not segments:
        return []
    if session_pick == "last":
        return list(segments[-1])
    # 默认：取最长段。
    return list(max(segments, key=lambda s: len(s)))


def _find_user_return_start_index(points: list[ObsPoint], *, bot_z: float = 0.0) -> int | None:
    """用同口径启发式寻找“用户回球段”的起始索引。

    该逻辑与 `BallTracer.is_user_return_ball()` 保持一致：
    - 最近 RETURN_BALL_LEN 帧 z 必须连续变近（z 递减）
    - 过程中高度 y 必须一直 >= 0.6m
    - 球到车的 z 距离必须 >= RETURN_BALL_MINIMUS_DIS
    - 最近窗口内 max_dis 必须 > RETURN_BALL_BEGIN_DIS
    - 检测窗口时长 < 0.5s
    - 首帧 z 间隔 > RETURN_BALL_FIRST_GAP

    Args:
        points: 点序列（按时间递增）。
        bot_z: 机器人 z 位置（离线默认 0）。

    Returns:
        起始索引；找不到则返回 None。
    """

    l = 4  # BallTracer.RETURN_BALL_LEN
    if len(points) < l:
        return None

    for i in range(l - 1, len(points)):
        window = points[i - l + 1 : i + 1]

        ok = True
        for j in range(l - 1):
            z_curr = float(window[j].z)
            z_next = float(window[j + 1].z)
            y_curr = float(window[j].y)
            if z_curr < z_next:
                ok = False
                break
            if y_curr < 0.6:
                ok = False
                break
        if not ok:
            continue

        ball_bot_dis = float(window[-1].z) - float(bot_z)
        if ball_bot_dis < float(BotMotionConfig.RETURN_BALL_MINIMUS_DIS):
            continue

        # 最近 12 帧窗口中的最大飞行距离（沿 z）。
        start_index = max(i - 11, 0)
        z_last = float(points[i].z)
        max_dis = 0.0
        for p in points[start_index : i + 1]:
            max_dis = max(max_dis, float(p.z) - z_last)
        if max_dis <= float(BotMotionConfig.RETURN_BALL_BEGIN_DIS):
            continue

        time_gap = float(window[-1].t_abs) - float(window[0].t_abs)
        if time_gap >= 0.5:
            continue

        first_gap = float(window[0].z) - float(window[1].z)
        if first_gap <= float(BotMotionConfig.RETURN_BALL_FIRST_GAP):
            continue

        return i - l + 1

    return None


def _filter_user_return_points(points: list[ObsPoint]) -> tuple[list[ObsPoint], dict[str, Any]]:
    """过滤掉回球段之前的点，并返回诊断信息。"""

    if not points:
        return [], {"start_index": None, "reason": "no_points"}

    idx = _find_user_return_start_index(points, bot_z=0.0)
    if idx is None:
        return points, {"start_index": None, "reason": "no_return_detected"}

    return points[idx:], {"start_index": int(idx), "reason": "detected"}


def _finite_stats(values: np.ndarray) -> dict[str, float | int]:
    """计算一组数值的有限值统计。

    Args:
        values: 1D 数组。

    Returns:
        统计字典：n、rmse、mae、max_abs。
    """

    v = np.asarray(values, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "rmse": float("nan"), "mae": float("nan"), "max_abs": float("nan")}

    rmse = float(np.sqrt(np.mean(v * v)))
    mae = float(np.mean(np.abs(v)))
    max_abs = float(np.max(np.abs(v)))
    return {"n": int(v.size), "rmse": rmse, "mae": mae, "max_abs": max_abs}


def _jsonify_float_list(arr: Any) -> list[float] | None:
    """把可能包含 numpy 类型的数组转成 JSON 友好的 float list。"""

    if arr is None:
        return None
    try:
        a = np.asarray(arr, dtype=float).reshape(-1)
    except Exception:
        return None
    return [float(x) for x in a.tolist()]


def _curve2_fit_analysis(curve: CurveV2, points: list[ObsPoint]) -> dict[str, Any]:
    """生成 curve2 的拟合质量分析报告（残差统计 + 关键内部状态）。

    说明：这里把“拟合好不好”量化为：在观测时刻 t_abs 上，curve2 预测的
    (x,y,z) 与观测值的差（残差）的统计。

    Args:
        curve: 已完成 add_frame 的 curve2 对象。
        points: 用于拟合的点（时间递增）。

    Returns:
        JSON 可序列化的分析字典。
    """

    if not points:
        return {"n_points": 0, "status": "no_points"}

    if curve.time_base is None:
        return {"n_points": int(len(points)), "status": "no_time_base"}

    obs_x = np.asarray([p.x for p in points], dtype=float)
    obs_y = np.asarray([p.y for p in points], dtype=float)
    obs_z = np.asarray([p.z for p in points], dtype=float)

    pred_x: list[float] = []
    pred_y: list[float] = []
    pred_z: list[float] = []
    ok = 0
    for p in points:
        q = _curve2_point_at_abs_time(curve, float(p.t_abs))
        if q is None:
            pred_x.append(float("nan"))
            pred_y.append(float("nan"))
            pred_z.append(float("nan"))
            continue
        pred_x.append(float(q[0]))
        pred_y.append(float(q[1]))
        pred_z.append(float(q[2]))
        ok += 1

    px = np.asarray(pred_x, dtype=float)
    py = np.asarray(pred_y, dtype=float)
    pz = np.asarray(pred_z, dtype=float)

    dx = px - obs_x
    dy = py - obs_y
    dz = pz - obs_z
    d3 = np.sqrt(dx * dx + dy * dy + dz * dz)

    # 额外：给出最后一个时刻的 z 方向速度（便于快速判断是否被速度门限打断）。
    vz_last = float("nan")
    try:
        if curve.time_base is not None and curve.z_coeff[0] is not None and curve.ts:
            vz_last = float(np.polyval(np.polyder(curve.z_coeff[0]), float(curve.ts[-1])))
    except Exception:
        vz_last = float("nan")

    return {
        "n_points": int(len(points)),
        "n_pred_ok": int(ok),
        "residual": {
            "dx": _finite_stats(dx),
            "dy": _finite_stats(dy),
            "dz": _finite_stats(dz),
            "d3": _finite_stats(d3),
        },
        "internal": {
            "is_curve_valid": bool(getattr(curve, "is_curve_valid", True)),
            "curve_samples_cnt": [int(x) for x in getattr(curve, "curve_samples_cnt", [])],
            "ball_start_cnt": [int(x) for x in getattr(curve, "ball_start_cnt", [])],
            "land_point": [
                None if lp is None else [float(x) for x in np.asarray(lp, dtype=float).tolist()]
                for lp in getattr(curve, "land_point", [])
            ],
            "land_speed": [
                None if sp is None else [float(x) for x in np.asarray(sp, dtype=float).tolist()]
                for sp in getattr(curve, "land_speed", [])
            ],
            "vz_last": vz_last,
            "error_rate": {
                "x": [float(x) for x in np.asarray(getattr(curve, "x_error_rate", []), dtype=float).tolist()],
                "y": [float(x) for x in np.asarray(getattr(curve, "y_error_rate", []), dtype=float).tolist()],
                "z": [float(x) for x in np.asarray(getattr(curve, "z_error_rate", []), dtype=float).tolist()],
            },
        },
    }


def _run_curve3(points: list[ObsPoint], *, cfg: CurveV3Config) -> CurvePredictorV3:
    pred = CurvePredictorV3(config=cfg)
    for p in points:
        pred.add_observation(BallObservation(x=p.x, y=p.y, z=p.z, t=p.t_abs))
    return pred


def _curve3_predicted_land(pred: CurvePredictorV3) -> dict[str, Any] | None:
    base = pred.time_base_abs
    if base is None:
        return None

    p = pred.predicted_land_point()
    if p is None or len(p) != 4:
        return None

    t_rel = _as_float(p[3])
    if t_rel is None:
        return None

    return {
        "x": float(p[0]),
        "y": float(p[1]),
        "z": float(p[2]),
        "t_abs": float(base + float(t_rel)),
        "t_rel": float(t_rel),
    }


def _curve3_hit_on_plane(pred: CurvePredictorV3, *, target_y: float) -> dict[str, Any] | None:
    base = pred.time_base_abs
    if base is None:
        return None

    r = pred.corridor_on_plane_y(float(target_y))
    if r is None:
        return None

    return {
        "target_y": float(target_y),
        "x": float(r.mu_xz[0]),
        "z": float(r.mu_xz[1]),
        "t_abs": float(base + float(r.t_rel_mu)),
        "t_rel": float(r.t_rel_mu),
        "valid_ratio": float(r.valid_ratio),
        "crossing_prob": float(r.crossing_prob),
        "is_valid": bool(r.is_valid),
    }


def _curve3_point_at_abs_time(pred: CurvePredictorV3, t_abs: float) -> tuple[float, float, float] | None:
    base = pred.time_base_abs
    if base is None:
        return None

    p = pred.point_at_time_rel(float(t_abs - base))
    if p is None or len(p) != 3:
        return None
    return float(p[0]), float(p[1]), float(p[2])


def _run_curve2(points: list[ObsPoint]) -> CurveV2:
    curve = CurveV2()
    # 说明：收敛曲线会对每个 prefix 重复拟合一次。curve2 的 legacy logger
    # 默认会输出大量 INFO，容易淹没真正的错误信息；离线工具中直接禁用。
    curve.logger.disabled = True
    # 离线对比：默认按“用户回球”方向（is_bot_fire=-1）。
    for p in points:
        r = curve.add_frame([p.x, p.y, p.z, p.t_abs], is_bot_fire=-1)
        if r == -1:
            # 说明：curve2 内部做了速度门限等质检；失败时会返回 -1。
            # 离线工具不直接抛异常，而是在报告里体现 is_curve_valid=false。
            break
    return curve


def _curve2_predicted_land(curve: CurveV2) -> dict[str, Any] | None:
    lp = curve.land_point[0]
    if lp is None or len(lp) != 4:
        return None
    if curve.time_base is None:
        return None

    return {
        "x": float(lp[0]),
        "y": 0.0,
        "z": float(lp[2]),
        "t_abs": float(curve.time_base + float(lp[3])),
        "t_rel": float(lp[3]),
    }


def _curve2_hit_on_plane(curve: CurveV2, *, target_y: float, pick: str = "down") -> dict[str, Any] | None:
    if curve.time_base is None:
        return None

    # 这里把击球点定义在反弹后第一段（id=1）轨迹上。
    x_coeff = curve.x_coeff[1]
    y_coeff = curve.y_coeff[1]
    z_coeff = curve.z_coeff[1]
    if x_coeff is None or y_coeff is None or z_coeff is None:
        return None

    ts = curve.calc_t_at_height(np.asarray(y_coeff, dtype=float), float(target_y))
    if ts is None or len(ts) != 2:
        return None

    # 说明：反弹后第二段通常会“先上升再下降”，因此在 y=target_y 平面
    # 上会出现两个交点：上升段（t_up）与下降段（t_down）。
    t_up = float(min(ts))
    t_down = float(max(ts))
    pick = str(pick).strip().lower() or "down"
    t = t_up if pick == "up" else t_down

    x = float(np.polyval(x_coeff, t))
    y = float(np.polyval(y_coeff, t))
    z = float(np.polyval(z_coeff, t))

    return {
        "target_y": float(target_y),
        "x": x,
        "y": y,
        "z": z,
        "t_abs": float(curve.time_base + t),
        "t_rel": float(t),
        "pick": str(pick),
        "t_up_rel": float(t_up),
        "t_down_rel": float(t_down),
    }


def _curve2_point_at_abs_time(curve: CurveV2, t_abs: float) -> tuple[float, float, float] | None:
    if curve.time_base is None:
        return None
    t_rel = float(t_abs - float(curve.time_base))
    p = curve.get_point_at_time(t_rel)
    if p is None or len(p) != 3:
        return None
    return float(p[0]), float(p[1]), float(p[2])


def _sample_abs_times(*, t0_abs: float, t1_abs: float, n: int) -> list[float]:
    n = int(n)
    if n <= 2:
        return [float(t0_abs), float(t1_abs)]
    return [float(x) for x in np.linspace(float(t0_abs), float(t1_abs), n, dtype=float)]


def _sample_trajectory(
    *,
    t_grid_abs: list[float],
    point_at_abs_time,
) -> dict[str, list[float | None]]:
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []

    for t_abs in t_grid_abs:
        p = point_at_abs_time(t_abs)
        if p is None:
            xs.append(None)
            ys.append(None)
            zs.append(None)
            continue
        xs.append(float(p[0]))
        ys.append(float(p[1]))
        zs.append(float(p[2]))

    return {"t_abs": [float(t) for t in t_grid_abs], "x": xs, "y": ys, "z": zs}


def _build_html(*, title: str, payload: dict[str, Any]) -> str:
    """构建单文件 HTML（不依赖第三方前端库）。"""

    data_json = json.dumps(payload, ensure_ascii=False)
    # 避免 </script> 提前闭合。
    data_json = data_json.replace("</", "<\\/")

    return (
        "<!doctype html>\n"
        "<html lang=\"zh-CN\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\"/>\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>\n"
        f"  <title>{title}</title>\n"
        "  <style>\n"
        "    body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial; margin:16px; color:#111;}\n"
        "    .row{display:flex; gap:12px; flex-wrap:wrap;}\n"
        "    .card{border:1px solid #ddd; border-radius:10px; padding:12px; background:#fff;}\n"
        "    .card h3{margin:0 0 8px 0; font-size:14px;}\n"
        "    canvas{border:1px solid #eee; border-radius:8px; background:#fafafa;}\n"
        "    .meta{font-family:ui-monospace,SFMono-Regular,Consolas,Menlo,monospace; font-size:12px; white-space:pre-wrap;}\n"
        "    label{font-size:12px;}\n"
        "    select{font-size:12px;}\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        f"<h2>{title}</h2>\n"
        "<div class=\"card\">\n"
        "  <div class=\"row\">\n"
        "    <div>\n"
        "      <label>post 点数 N：</label>\n"
        "      <select id=\"postN\"></select>\n"
        "    </div>\n"
        "    <div>\n"
        "      <label>坐标范围：</label>\n"
        "      <select id=\"axisMode\">\n"
        "        <option value=\"robust\" selected>稳健(2%~98%)</option>\n"
        "        <option value=\"full\">全量(min/max)</option>\n"
        "      </select>\n"
        "    </div>\n"
        "    <div>\n"
        "      <label><input id=\"hidePrefitExtrap\" type=\"checkbox\" checked/> 隐藏 curve3 prefit 外推</label>\n"
        "    </div>\n"
        "  </div>\n"
        "  <div class=\"meta\" id=\"meta\"></div>\n"
        "</div>\n"
        "\n"
        "<div class=\"row\">\n"
        "  <div class=\"card\">\n"
        "    <h3>x-z 俯视图（点 + 两算法曲线）</h3>\n"
        "    <canvas id=\"xz\" width=\"640\" height=\"360\"></canvas>\n"
        "  </div>\n"
        "  <div class=\"card\">\n"
        "    <h3>y-t（点 + 两算法曲线）</h3>\n"
        "    <canvas id=\"yt\" width=\"640\" height=\"360\"></canvas>\n"
        "  </div>\n"
        "</div>\n"
        "\n"
        "<div class=\"row\">\n"
        "  <div class=\"card\">\n"
        "    <h3>击球点收敛（距最终值 |Δxz|，越低越好）</h3>\n"
        "    <canvas id=\"hit\" width=\"640\" height=\"360\"></canvas>\n"
        "  </div>\n"
        "  <div class=\"card\">\n"
        "    <h3>落地点收敛（距最终值 |Δxz|，越低越好）</h3>\n"
        "    <canvas id=\"land\" width=\"640\" height=\"360\"></canvas>\n"
        "  </div>\n"
        "</div>\n"
        "\n"
        f"<script id=\"payload\" type=\"application/json\">{data_json}</script>\n"
        "<script>\n"
        "const payload = JSON.parse(document.getElementById('payload').textContent);\n"
        "const showCurve3 = !(payload.meta && payload.meta.plot && payload.meta.plot.curve3 === false);\n"
        "\n"
        "function fmt(v){\n"
        "  if(v===null||v===undefined) return 'null';\n"
        "  if(typeof v==='number') return Number.isFinite(v) ? v.toFixed(4) : String(v);\n"
        "  return String(v);\n"
        "}\n"
        "\n"
        "function collectFinite(points, getter){\n"
        "  const arr=[];\n"
        "  for(const p of points){\n"
        "    const v=getter(p);\n"
        "    if(v===null||v===undefined) continue;\n"
        "    if(!Number.isFinite(v)) continue;\n"
        "    arr.push(v);\n"
        "  }\n"
        "  return arr;\n"
        "}\n"
        "\n"
        "function quantileSorted(sorted, q){\n"
        "  if(sorted.length===0) return null;\n"
        "  const qq=Math.min(1, Math.max(0, q));\n"
        "  const idx=(sorted.length-1)*qq;\n"
        "  const lo=Math.floor(idx);\n"
        "  const hi=Math.ceil(idx);\n"
        "  if(lo===hi) return sorted[lo];\n"
        "  const t=idx-lo;\n"
        "  return sorted[lo]*(1-t) + sorted[hi]*t;\n"
        "}\n"
        "\n"
        "function getExtentsFull(points, getX, getY){\n"
        "  let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;\n"
        "  for(const p of points){\n"
        "    const x=getX(p); const y=getY(p);\n"
        "    if(x===null||y===null||x===undefined||y===undefined) continue;\n"
        "    if(!Number.isFinite(x)||!Number.isFinite(y)) continue;\n"
        "    xmin=Math.min(xmin,x); xmax=Math.max(xmax,x);\n"
        "    ymin=Math.min(ymin,y); ymax=Math.max(ymax,y);\n"
        "  }\n"
        "  if(xmin===Infinity){ return {xmin:0,xmax:1,ymin:0,ymax:1}; }\n"
        "  const padX=(xmax-xmin)*0.10+1e-6;\n"
        "  const padY=(ymax-ymin)*0.10+1e-6;\n"
        "  return {xmin:xmin-padX,xmax:xmax+padX,ymin:ymin-padY,ymax:ymax+padY};\n"
        "}\n"
        "\n"
        "function getExtentsRobust(points, getX, getY, qLo, qHi){\n"
        "  const xs=collectFinite(points, getX);\n"
        "  const ys=collectFinite(points, getY);\n"
        "  if(xs.length===0||ys.length===0){ return {xmin:0,xmax:1,ymin:0,ymax:1}; }\n"
        "  xs.sort((a,b)=>a-b);\n"
        "  ys.sort((a,b)=>a-b);\n"
        "  const xmin=quantileSorted(xs, qLo);\n"
        "  const xmax=quantileSorted(xs, qHi);\n"
        "  const ymin=quantileSorted(ys, qLo);\n"
        "  const ymax=quantileSorted(ys, qHi);\n"
        "  if(xmin===null||xmax===null||ymin===null||ymax===null){ return {xmin:0,xmax:1,ymin:0,ymax:1}; }\n"
        "  const dx=Math.max(1e-6, (xmax-xmin));\n"
        "  const dy=Math.max(1e-6, (ymax-ymin));\n"
        "  const padX=dx*0.12;\n"
        "  const padY=dy*0.12;\n"
        "  return {xmin:xmin-padX,xmax:xmax+padX,ymin:ymin-padY,ymax:ymax+padY};\n"
        "}\n"
        "\n"
        "function getExtents(points, getX, getY){\n"
        "  const mode = document.getElementById('axisMode') ? document.getElementById('axisMode').value : 'robust';\n"
        "  if(mode==='full'){\n"
        "    return getExtentsFull(points, getX, getY);\n"
        "  }\n"
        "  return getExtentsRobust(points, getX, getY, 0.02, 0.98);\n"
        "}\n"
        "\n"
        "function drawAxes(ctx, w, h, ext, xlabel, ylabel){\n"
        "  ctx.save();\n"
        "  ctx.strokeStyle='#ddd';\n"
        "  ctx.lineWidth=1;\n"
        "  ctx.strokeRect(40, 10, w-55, h-45);\n"
        "  ctx.fillStyle='#555';\n"
        "  ctx.font='12px ui-monospace,Consolas,monospace';\n"
        "  ctx.fillText(xlabel, w/2-20, h-10);\n"
        "  ctx.save();\n"
        "  ctx.translate(12, h/2+20);\n"
        "  ctx.rotate(-Math.PI/2);\n"
        "  ctx.fillText(ylabel, 0, 0);\n"
        "  ctx.restore();\n"
        "  ctx.fillText(`x:[${fmt(ext.xmin)}, ${fmt(ext.xmax)}]`, 46, 22);\n"
        "  ctx.fillText(`y:[${fmt(ext.ymin)}, ${fmt(ext.ymax)}]`, 46, 38);\n"
        "  ctx.restore();\n"
        "}\n"
        "\n"
        "function mapToCanvas(x, y, w, h, ext){\n"
        "  const px=40 + (x-ext.xmin)/(ext.xmax-ext.xmin) * (w-55);\n"
        "  const py=10 + (1-(y-ext.ymin)/(ext.ymax-ext.ymin)) * (h-45);\n"
        "  return [px, py];\n"
        "}\n"
        "\n"
        "function drawScatter(ctx, w, h, ext, pts, getX, getY, color){\n"
        "  ctx.save();\n"
        "  ctx.fillStyle=color;\n"
        "  for(const p of pts){\n"
        "    const x=getX(p); const y=getY(p);\n"
        "    if(x===null||y===null||x===undefined||y===undefined) continue;\n"
        "    if(!Number.isFinite(x)||!Number.isFinite(y)) continue;\n"
        "    const [px,py]=mapToCanvas(x,y,w,h,ext);\n"
        "    ctx.beginPath();\n"
        "    ctx.arc(px,py,2.2,0,Math.PI*2);\n"
        "    ctx.fill();\n"
        "  }\n"
        "  ctx.restore();\n"
        "}\n"
        "\n"
        "function drawPolyline(ctx, w, h, ext, xs, ys, color){\n"
        "  ctx.save();\n"
        "  ctx.strokeStyle=color;\n"
        "  ctx.lineWidth=2;\n"
        "  let started=false;\n"
        "  for(let i=0;i<xs.length;i++){\n"
        "    const x=xs[i], y=ys[i];\n"
        "    if(x===null||y===null||x===undefined||y===undefined) { started=false; continue; }\n"
        "    if(!Number.isFinite(x)||!Number.isFinite(y)) { started=false; continue; }\n"
        "    const [px,py]=mapToCanvas(x,y,w,h,ext);\n"
        "    if(!started){ ctx.beginPath(); ctx.moveTo(px,py); started=true; }\n"
        "    else { ctx.lineTo(px,py); }\n"
        "  }\n"
        "  if(started) ctx.stroke();\n"
        "  ctx.restore();\n"
        "}\n"
        "\n"
        "function cloneArray(a){\n"
        "  const b=new Array(a.length);\n"
        "  for(let i=0;i<a.length;i++) b[i]=a[i];\n"
        "  return b;\n"
        "}\n"
        "\n"
        "function maskCurve3PrefitExtrap(samples, landTAbs, obsPts, xzWindowN, hide){\n"
        "  // 说明：curve3 的 prefit x/z 拟合是“短窗口”（默认 12 点）策略；\n"
        "  // 对很早的时间点属于外推，可能明显偏离观测并在 x-z 投影里看起来像竖线。\n"
        "  if(!hide) return samples;\n"
        "  if(landTAbs===null||landTAbs===undefined||!Number.isFinite(landTAbs)) return samples;\n"
        "  const preObs = obsPts.filter(p=> Number.isFinite(p.t_abs) && p.t_abs <= landTAbs);\n"
        "  if(preObs.length===0) return samples;\n"
        "  const n = Math.max(3, (xzWindowN|0));\n"
        "  const startIdx = Math.max(preObs.length - n, 0);\n"
        "  const tRefAbs = preObs[startIdx].t_abs;\n"
        "  if(!Number.isFinite(tRefAbs)) return samples;\n"
        "  const t = samples.t_abs;\n"
        "  const out = {t_abs: t, x: cloneArray(samples.x), y: cloneArray(samples.y), z: cloneArray(samples.z)};\n"
        "  for(let i=0;i<t.length;i++){\n"
        "    const ti=t[i];\n"
        "    if(ti===null||ti===undefined||!Number.isFinite(ti)) continue;\n"
        "    if(ti < tRefAbs && ti < landTAbs){\n"
        "      out.x[i]=null; out.y[i]=null; out.z[i]=null;\n"
        "    }\n"
        "  }\n"
        "  return out;\n"
        "}\n"
        "\n"
        "function drawLegend(ctx, items){\n"
        "  ctx.save();\n"
        "  ctx.font='12px ui-monospace,Consolas,monospace';\n"
        "  let x=46, y=54;\n"
        "  for(const it of items){\n"
        "    ctx.fillStyle=it.color;\n"
        "    ctx.fillRect(x, y-9, 12, 6);\n"
        "    ctx.fillStyle='#222';\n"
        "    ctx.fillText(it.label, x+18, y-4);\n"
        "    y += 16;\n"
        "  }\n"
        "  ctx.restore();\n"
        "}\n"
        "\n"
        "function drawVLine(ctx, w, h, ext, xVal, color, dash){\n"
        "  if(xVal===null||xVal===undefined||!Number.isFinite(xVal)) return;\n"
        "  const [px,_]=mapToCanvas(xVal, ext.ymin, w, h, ext);\n"
        "  ctx.save();\n"
        "  ctx.strokeStyle=color;\n"
        "  if(dash) ctx.setLineDash(dash);\n"
        "  ctx.beginPath();\n"
        "  ctx.moveTo(px, 10);\n"
        "  ctx.lineTo(px, h-35);\n"
        "  ctx.stroke();\n"
        "  ctx.restore();\n"
        "}\n"
        "\n"
        "function drawHLine(ctx, w, h, ext, yVal, color, dash){\n"
        "  if(yVal===null||yVal===undefined||!Number.isFinite(yVal)) return;\n"
        "  const [_,py]=mapToCanvas(ext.xmin, yVal, w, h, ext);\n"
        "  ctx.save();\n"
        "  ctx.strokeStyle=color;\n"
        "  if(dash) ctx.setLineDash(dash);\n"
        "  ctx.beginPath();\n"
        "  ctx.moveTo(40, py);\n"
        "  ctx.lineTo(w-15, py);\n"
        "  ctx.stroke();\n"
        "  ctx.restore();\n"
        "}\n"
        "\n"
        "function drawMarker(ctx, w, h, ext, xVal, yVal, color, shape){\n"
        "  if(xVal===null||yVal===null||xVal===undefined||yVal===undefined) return;\n"
        "  if(!Number.isFinite(xVal)||!Number.isFinite(yVal)) return;\n"
        "  const [px,py]=mapToCanvas(xVal,yVal,w,h,ext);\n"
        "  ctx.save();\n"
        "  ctx.fillStyle=color;\n"
        "  ctx.strokeStyle='#fff';\n"
        "  ctx.lineWidth=2;\n"
        "  const s=5.5;\n"
        "  ctx.beginPath();\n"
        "  if(shape==='diamond'){\n"
        "    ctx.moveTo(px, py-s);\n"
        "    ctx.lineTo(px+s, py);\n"
        "    ctx.lineTo(px, py+s);\n"
        "    ctx.lineTo(px-s, py);\n"
        "    ctx.closePath();\n"
        "  } else if(shape==='triangle'){\n"
        "    ctx.moveTo(px, py-s);\n"
        "    ctx.lineTo(px+s, py+s);\n"
        "    ctx.lineTo(px-s, py+s);\n"
        "    ctx.closePath();\n"
        "  } else if(shape==='x'){\n"
        "    ctx.moveTo(px-s, py-s); ctx.lineTo(px+s, py+s);\n"
        "    ctx.moveTo(px-s, py+s); ctx.lineTo(px+s, py-s);\n"
        "  } else {\n"
        "    ctx.arc(px,py,s,0,Math.PI*2);\n"
        "  }\n"
        "  if(shape==='x'){\n"
        "    ctx.strokeStyle=color;\n"
        "    ctx.lineWidth=2.5;\n"
        "    ctx.stroke();\n"
        "  } else {\n"
        "    ctx.fill();\n"
        "    ctx.stroke();\n"
        "  }\n"
        "  ctx.restore();\n"
        "}\n"
        "\n"
        "function redraw(){\n"
        "  const postN = document.getElementById('postN').value;\n"
        "  const fit = payload.fits_by_post_n[postN];\n"
        "  const pts = payload.points;\n"
        "  const t0Abs = (pts && pts.length) ? pts[0].t_abs : 0;\n"
        "  const hideExtrap = document.getElementById('hidePrefitExtrap') ? document.getElementById('hidePrefitExtrap').checked : true;\n"
        "\n"
        "  const hit2Up = (fit.curve2 && fit.curve2.hit_on_plane_up) ? fit.curve2.hit_on_plane_up : null;\n"
        "  const hit2Down = (fit.curve2 && fit.curve2.hit_on_plane_down) ? fit.curve2.hit_on_plane_down : null;\n"
        "  const land2 = (fit.curve2 && fit.curve2.predicted_land) ? fit.curve2.predicted_land : null;\n"
        "  const obsHitUp = (payload.observed && payload.observed.hit_on_plane_post_up) ? payload.observed.hit_on_plane_post_up : null;\n"
        "  const obsHitDown = (payload.observed && payload.observed.hit_on_plane_post_down) ? payload.observed.hit_on_plane_post_down : null;\n"
        "\n"
        "  // meta\n"
        "  const metaLines=[];\n"
        "  metaLines.push(`track_id=${payload.meta.track_id}  total_points=${pts.length}  filtered_start=${payload.meta.filter.start_index}`);\n"
        "  metaLines.push(`plot: curve3=${showCurve3}`);\n"
        "  metaLines.push(`bounce_t_abs_full=${fmt(payload.bounce_t_abs_full)}  target_y=${fmt(payload.target_y)}`);\n"
        "  metaLines.push(`selected postN=${postN}  subset n_total=${fit.subset.n_total} (pre=${fit.subset.n_pre}, post=${fit.subset.n_post})`);\n"
        "  const a2 = (fit.curve2 && fit.curve2.analysis) ? fit.curve2.analysis : null;\n"
        "  if(a2 && a2.residual && a2.residual.d3){\n"
        "    const ok = a2.n_pred_ok;\n"
        "    const n = a2.n_points;\n"
        "    const rmse3 = a2.residual.d3.rmse;\n"
        "    const mae3 = a2.residual.d3.mae;\n"
        "    const vld = (a2.internal && typeof a2.internal.is_curve_valid==='boolean') ? a2.internal.is_curve_valid : null;\n"
        "    metaLines.push(`curve2 fit: n=${n} ok=${ok} is_valid=${vld}  d3_rmse=${fmt(rmse3)}  d3_mae=${fmt(mae3)}`);\n"
        "  }\n"
        "  if(hit2Up && Number.isFinite(hit2Up.t_abs)) {\n"
        "    metaLines.push(`curve2 hit_pred_up: t_rel=${fmt(hit2Up.t_abs - t0Abs)}  x=${fmt(hit2Up.x)}  z=${fmt(hit2Up.z)}`);\n"
        "  }\n"
        "  if(hit2Down && Number.isFinite(hit2Down.t_abs)) {\n"
        "    metaLines.push(`curve2 hit_pred_down: t_rel=${fmt(hit2Down.t_abs - t0Abs)}  x=${fmt(hit2Down.x)}  z=${fmt(hit2Down.z)}`);\n"
        "  }\n"
        "  if(land2 && Number.isFinite(land2.t_abs)) {\n"
        "    metaLines.push(`curve2 land_pred: t_rel=${fmt(land2.t_abs - t0Abs)}  x=${fmt(land2.x)}  z=${fmt(land2.z)}`);\n"
        "  }\n"
        "  if(obsHitUp && Number.isFinite(obsHitUp.t_abs)) {\n"
        "    metaLines.push(`obs hit_est_up(post, ${obsHitUp.method}): t_rel=${fmt(obsHitUp.t_abs - t0Abs)}  x=${fmt(obsHitUp.x)}  y=${fmt(obsHitUp.y)}  z=${fmt(obsHitUp.z)}`);\n"
        "  }\n"
        "  if(obsHitDown && Number.isFinite(obsHitDown.t_abs)) {\n"
        "    metaLines.push(`obs hit_est_down(post, ${obsHitDown.method}): t_rel=${fmt(obsHitDown.t_abs - t0Abs)}  x=${fmt(obsHitDown.x)}  y=${fmt(obsHitDown.y)}  z=${fmt(obsHitDown.z)}`);\n"
        "  }\n"
        "  const obsHitPre = (payload.observed && payload.observed.hit_on_plane_pre) ? payload.observed.hit_on_plane_pre : null;\n"
        "  if(obsHitPre && Number.isFinite(obsHitPre.t_abs)) {\n"
        "    metaLines.push(`obs hit_est(pre, ${obsHitPre.method}): t_rel=${fmt(obsHitPre.t_abs - t0Abs)}  x=${fmt(obsHitPre.x)}  y=${fmt(obsHitPre.y)}  z=${fmt(obsHitPre.z)}`);\n"
        "  }\n"
        "  document.getElementById('meta').textContent = metaLines.join('\\n');\n"
        "\n"
        "  // xz\n"
        "  const xzCanvas=document.getElementById('xz');\n"
        "  const xzCtx=xzCanvas.getContext('2d');\n"
        "  xzCtx.clearRect(0,0,xzCanvas.width,xzCanvas.height);\n"
        "  const xzPts = pts.map(p=>({x:p.x,z:p.z}));\n"
        "  const land3 = (showCurve3 && fit.curve3.predicted_land) ? fit.curve3.predicted_land.t_abs : null;\n"
        "  const c3raw = showCurve3 ? fit.curve3.samples : null;\n"
        "  const c3 = showCurve3 ? maskCurve3PrefitExtrap(c3raw, land3, pts, 12, hideExtrap) : null;\n"
        "  const c2=fit.curve2.samples;\n"
        "  const xzCurvePts = [ ...c2.t_abs.map((_,i)=>({x:c2.x[i],z:c2.z[i]})) ];\n"
        "  if(showCurve3){ xzCurvePts.push(...c3.t_abs.map((_,i)=>({x:c3.x[i],z:c3.z[i]}))); }\n"
        "  const xzExt = getExtents([ ...xzPts, ...xzCurvePts ], p=>p.x, p=>p.z);\n"
        "  drawAxes(xzCtx,xzCanvas.width,xzCanvas.height,xzExt,'x','z');\n"
        "  drawScatter(xzCtx,xzCanvas.width,xzCanvas.height,xzExt,xzPts,p=>p.x,p=>p.z,'#111');\n"
        "  if(showCurve3){ drawPolyline(xzCtx,xzCanvas.width,xzCanvas.height,xzExt,c3.x,c3.z,'#d9480f'); }\n"
        "  drawPolyline(xzCtx,xzCanvas.width,xzCanvas.height,xzExt,c2.x,c2.z,'#1c7ed6');\n"
        "  if(hit2Up){ drawMarker(xzCtx,xzCanvas.width,xzCanvas.height,xzExt, hit2Up.x, hit2Up.z, '#40c057', 'triangle'); }\n"
        "  if(hit2Down){ drawMarker(xzCtx,xzCanvas.width,xzCanvas.height,xzExt, hit2Down.x, hit2Down.z, '#37b24d', 'circle'); }\n"
        "  if(land2){ drawMarker(xzCtx,xzCanvas.width,xzCanvas.height,xzExt, land2.x, land2.z, '#862e9c', 'x'); }\n"
        "  if(obsHitUp){ drawMarker(xzCtx,xzCanvas.width,xzCanvas.height,xzExt, obsHitUp.x, obsHitUp.z, '#0b7285', 'diamond'); }\n"
        "  if(obsHitDown){ drawMarker(xzCtx,xzCanvas.width,xzCanvas.height,xzExt, obsHitDown.x, obsHitDown.z, '#1098ad', 'diamond'); }\n"
        "  drawLegend(xzCtx, showCurve3 ? [{label:'obs',color:'#111'},{label:'curve3',color:'#d9480f'},{label:'curve2',color:'#1c7ed6'},{label:'hit_pred_up',color:'#40c057'},{label:'hit_pred_down',color:'#37b24d'},{label:'land_pred',color:'#862e9c'},{label:'hit_obs_est_up(post)',color:'#0b7285'},{label:'hit_obs_est_down(post)',color:'#1098ad'}] : [{label:'obs',color:'#111'},{label:'curve2',color:'#1c7ed6'},{label:'hit_pred_up',color:'#40c057'},{label:'hit_pred_down',color:'#37b24d'},{label:'land_pred',color:'#862e9c'},{label:'hit_obs_est_up(post)',color:'#0b7285'},{label:'hit_obs_est_down(post)',color:'#1098ad'}]);\n"
        "\n"
        "  // y-t\n"
        "  const ytCanvas=document.getElementById('yt');\n"
        "  const ytCtx=ytCanvas.getContext('2d');\n"
        "  ytCtx.clearRect(0,0,ytCanvas.width,ytCanvas.height);\n"
        "  const ytPts = pts.map(p=>({t:(p.t_abs - t0Abs),y:p.y}));\n"
        "  const c3t = showCurve3 ? c3.t_abs.map(t=> (t - t0Abs)) : [];\n"
        "  const c2t = c2.t_abs.map(t=> (t - t0Abs));\n"
        "  const ytCurvePts = [ ...c2t.map((t,i)=>({t:t,y:c2.y[i]})) ];\n"
        "  if(showCurve3){ ytCurvePts.push(...c3t.map((t,i)=>({t:t,y:c3.y[i]}))); }\n"
        "  const ytExt = getExtents([ ...ytPts, ...ytCurvePts ], p=>p.t, p=>p.y);\n"
        "  drawAxes(ytCtx,ytCanvas.width,ytCanvas.height,ytExt,'t_rel(s)','y');\n"
        "  drawHLine(ytCtx,ytCanvas.width,ytCanvas.height,ytExt, payload.target_y, '#74c0fc', [6,4]);\n"
        "  drawScatter(ytCtx,ytCanvas.width,ytCanvas.height,ytExt,ytPts,p=>p.t,p=>p.y,'#111');\n"
        "  if(showCurve3){ drawPolyline(ytCtx,ytCanvas.width,ytCanvas.height,ytExt,c3t,c3.y,'#d9480f'); }\n"
        "  drawPolyline(ytCtx,ytCanvas.width,ytCanvas.height,ytExt,c2t,c2.y,'#1c7ed6');\n"
        "  if(hit2Up && Number.isFinite(hit2Up.t_abs)) {\n"
        "    const th = hit2Up.t_abs - t0Abs;\n"
        "    drawVLine(ytCtx,ytCanvas.width,ytCanvas.height,ytExt, th, '#40c057', [4,4]);\n"
        "    drawMarker(ytCtx,ytCanvas.width,ytCanvas.height,ytExt, th, payload.target_y, '#40c057', 'triangle');\n"
        "  }\n"
        "  if(hit2Down && Number.isFinite(hit2Down.t_abs)) {\n"
        "    const th = hit2Down.t_abs - t0Abs;\n"
        "    drawVLine(ytCtx,ytCanvas.width,ytCanvas.height,ytExt, th, '#37b24d', [6,4]);\n"
        "    drawMarker(ytCtx,ytCanvas.width,ytCanvas.height,ytExt, th, payload.target_y, '#37b24d', 'circle');\n"
        "  }\n"
        "  if(land2 && Number.isFinite(land2.t_abs)) {\n"
        "    const tl = land2.t_abs - t0Abs;\n"
        "    drawVLine(ytCtx,ytCanvas.width,ytCanvas.height,ytExt, tl, '#862e9c', [6,4]);\n"
        "    drawMarker(ytCtx,ytCanvas.width,ytCanvas.height,ytExt, tl, 0.0, '#862e9c', 'x');\n"
        "  }\n"
        "  if(obsHitUp && Number.isFinite(obsHitUp.t_abs)) {\n"
        "    const to = obsHitUp.t_abs - t0Abs;\n"
        "    drawVLine(ytCtx,ytCanvas.width,ytCanvas.height,ytExt, to, '#0b7285', [2,6]);\n"
        "    if(Number.isFinite(obsHitUp.y)) { drawMarker(ytCtx,ytCanvas.width,ytCanvas.height,ytExt, to, obsHitUp.y, '#0b7285', 'diamond'); }\n"
        "  }\n"
        "  if(obsHitDown && Number.isFinite(obsHitDown.t_abs)) {\n"
        "    const to = obsHitDown.t_abs - t0Abs;\n"
        "    drawVLine(ytCtx,ytCanvas.width,ytCanvas.height,ytExt, to, '#1098ad', [2,6]);\n"
        "    if(Number.isFinite(obsHitDown.y)) { drawMarker(ytCtx,ytCanvas.width,ytCanvas.height,ytExt, to, obsHitDown.y, '#1098ad', 'diamond'); }\n"
        "  }\n"
        "  drawLegend(ytCtx, showCurve3 ? [{label:'obs',color:'#111'},{label:'curve3',color:'#d9480f'},{label:'curve2',color:'#1c7ed6'},{label:'target_y',color:'#74c0fc'},{label:'hit_pred_up',color:'#40c057'},{label:'hit_pred_down',color:'#37b24d'},{label:'land_pred',color:'#862e9c'},{label:'hit_obs_est_up(post)',color:'#0b7285'},{label:'hit_obs_est_down(post)',color:'#1098ad'}] : [{label:'obs',color:'#111'},{label:'curve2',color:'#1c7ed6'},{label:'target_y',color:'#74c0fc'},{label:'hit_pred_up',color:'#40c057'},{label:'hit_pred_down',color:'#37b24d'},{label:'land_pred',color:'#862e9c'},{label:'hit_obs_est_up(post)',color:'#0b7285'},{label:'hit_obs_est_down(post)',color:'#1098ad'}]);\n"
        "\n"
        "  // convergence charts（更直观）：画‘距最终值的水平误差 |Δxz|’，越低表示越收敛。\n"
        "  const steps = payload.convergence.steps;\n"
        "  const ks = steps.map((s,i)=> (s && typeof s.k==='number' && Number.isFinite(s.k)) ? s.k : i);\n"
        "\n"
        "  const hitX3 = showCurve3 ? steps.map(s=>s.curve3.hit ? s.curve3.hit.x : null) : [];\n"
        "  const hitZ3 = showCurve3 ? steps.map(s=>s.curve3.hit ? s.curve3.hit.z : null) : [];\n"
        "  const hitX2Up = steps.map(s=>s.curve2.hit_up ? s.curve2.hit_up.x : null);\n"
        "  const hitZ2Up = steps.map(s=>s.curve2.hit_up ? s.curve2.hit_up.z : null);\n"
        "  const hitX2Down = steps.map(s=>s.curve2.hit_down ? s.curve2.hit_down.x : null);\n"
        "  const hitZ2Down = steps.map(s=>s.curve2.hit_down ? s.curve2.hit_down.z : null);\n"
        "\n"
        "  const landX3 = showCurve3 ? steps.map(s=>s.curve3.land ? s.curve3.land.x : null) : [];\n"
        "  const landZ3 = showCurve3 ? steps.map(s=>s.curve3.land ? s.curve3.land.z : null) : [];\n"
        "  const landX2 = steps.map(s=>s.curve2.land ? s.curve2.land.x : null);\n"
        "  const landZ2 = steps.map(s=>s.curve2.land ? s.curve2.land.z : null);\n"
        "\n"
        "  function lastFinite(arr){\n"
        "    for(let i=arr.length-1;i>=0;i--){\n"
        "      const v=arr[i];\n"
        "      if(v===null||v===undefined) continue;\n"
        "      if(Number.isFinite(v)) return v;\n"
        "    }\n"
        "    return NaN;\n"
        "  }\n"
        "\n"
        "  function xzErrorToFinal(xs, zs){\n"
        "    const xf=lastFinite(xs);\n"
        "    const zf=lastFinite(zs);\n"
        "    const out=new Array(xs.length);\n"
        "    for(let i=0;i<xs.length;i++){\n"
        "      const x=xs[i], z=zs[i];\n"
        "      if(!Number.isFinite(x)||!Number.isFinite(z)||!Number.isFinite(xf)||!Number.isFinite(zf)){\n"
        "        out[i]=null;\n"
        "        continue;\n"
        "      }\n"
        "      const dx=x-xf, dz=z-zf;\n"
        "      out[i]=Math.sqrt(dx*dx + dz*dz);\n"
        "    }\n"
        "    return out;\n"
        "  }\n"
        "\n"
        "  function drawErrorChart(canvasId, ksArr, seriesList, yLabel){\n"
        "    const canvas=document.getElementById(canvasId);\n"
        "    const ctx=canvas.getContext('2d');\n"
        "    ctx.clearRect(0,0,canvas.width,canvas.height);\n"
        "\n"
        "    let xmin=Infinity, xmax=-Infinity, ymax=0;\n"
        "    for(const k of ksArr){\n"
        "      if(!Number.isFinite(k)) continue;\n"
        "      xmin=Math.min(xmin, k);\n"
        "      xmax=Math.max(xmax, k);\n"
        "    }\n"
        "    for(const s of seriesList){\n"
        "      for(const v of s.y){\n"
        "        if(v===null||v===undefined) continue;\n"
        "        if(!Number.isFinite(v)) continue;\n"
        "        ymax=Math.max(ymax, v);\n"
        "      }\n"
        "    }\n"
        "    if(xmin===Infinity){ xmin=0; xmax=1; }\n"
        "    if(!(ymax>0)){ ymax=1; }\n"
        "    const ext={xmin:xmin, xmax:xmax, ymin:0, ymax:ymax*1.05 + 1e-6};\n"
        "    drawAxes(ctx,canvas.width,canvas.height,ext,'k (prefix points)', yLabel);\n"
        "\n"
        "    for(const s of seriesList){\n"
        "      drawPolyline(ctx,canvas.width,canvas.height,ext,ksArr,s.y,s.color);\n"
        "    }\n"
        "    drawLegend(ctx, seriesList.map(s=>({label:s.label, color:s.color})));\n"
        "  }\n"
        "\n"
        "  const errHitDown = xzErrorToFinal(hitX2Down, hitZ2Down);\n"
        "  const errHitUp = xzErrorToFinal(hitX2Up, hitZ2Up);\n"
        "  const errLand = xzErrorToFinal(landX2, landZ2);\n"
        "\n"
        "  const hitSeries=[\n"
        "    {label:'curve2_down |Δxz|', color:'#1c7ed6', y:errHitDown},\n"
        "    {label:'curve2_up |Δxz|', color:'#37b24d', y:errHitUp},\n"
        "  ];\n"
        "  if(showCurve3){\n"
        "    const errHit3 = xzErrorToFinal(hitX3, hitZ3);\n"
        "    hitSeries.unshift({label:'curve3 |Δxz|', color:'#d9480f', y:errHit3});\n"
        "  }\n"
        "  drawErrorChart('hit', ks, hitSeries, '|Δxz| (m)');\n"
        "\n"
        "  const landSeries=[{label:'curve2 |Δxz|', color:'#1c7ed6', y:errLand}];\n"
        "  if(showCurve3){\n"
        "    const errLand3 = xzErrorToFinal(landX3, landZ3);\n"
        "    landSeries.unshift({label:'curve3 |Δxz|', color:'#d9480f', y:errLand3});\n"
        "  }\n"
        "  drawErrorChart('land', ks, landSeries, '|Δxz| (m)');\n"
        "}\n"
        "\n"
        "function init(){\n"
        "  const sel=document.getElementById('postN');\n"
        "  const keys=Object.keys(payload.fits_by_post_n);\n"
        "  for(const k of keys){\n"
        "    const opt=document.createElement('option');\n"
        "    opt.value=k; opt.textContent=k;\n"
        "    sel.appendChild(opt);\n"
        "  }\n"
        "  sel.value='all';\n"
        "  sel.addEventListener('change', redraw);\n"
        "  const axisSel=document.getElementById('axisMode');\n"
        "  if(axisSel){ axisSel.addEventListener('change', redraw); }\n"
        "  const hideSel=document.getElementById('hidePrefitExtrap');\n"
        "  if(hideSel){ hideSel.addEventListener('change', redraw); }\n"
        "  redraw();\n"
        "}\n"
        "\n"
        "init();\n"
        "</script>\n"
        "</body>\n"
        "</html>\n"
    )


def _robust_limits(vals: np.ndarray, *, mode: str) -> tuple[float, float]:
    """计算绘图用的坐标轴范围。

    Args:
        vals: 1D 数组。
        mode: "robust" 或 "full"。

    Returns:
        (vmin, vmax)
    """

    v = np.asarray(vals, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (0.0, 1.0)

    mode = str(mode).strip().lower() or "robust"
    if mode == "full":
        lo = float(np.min(v))
        hi = float(np.max(v))
        pad = (hi - lo) * 0.10 + 1e-6
        return (lo - pad, hi + pad)

    lo = float(np.percentile(v, 2.0))
    hi = float(np.percentile(v, 98.0))
    # 说明：这里的 padding 比 full 略大一些，避免分位数裁剪后边缘太“顶格”。
    pad = (hi - lo) * 0.12 + 1e-6
    return (lo - pad, hi + pad)


def _mask_curve3_prefit_extrap_python(
    *,
    samples: dict[str, Any],
    bounce_t_abs: float | None,
    land_t_abs: float | None,
    obs_points: list[ObsPoint],
    xz_window_n: int,
    hide: bool,
) -> dict[str, Any]:
    """对 curve3 的 samples 做“隐藏 prefit 外推”的掩码。

    说明：curve3 的 prefit（尤其 x/z）默认采用短窗口策略（例如 12 点），
    对更早时间属于外推，可能明显偏离观测并在 x-z 投影里看起来像“竖线”。

    这里按与 prefit.py 相同的口径取 pre 段最后 xz_window_n 个点的起始时刻 t_ref_abs，
    并把 t_abs < t_ref_abs 且 t_abs < land_t_abs 的采样点置为 None，从而让绘图断线。
    """

    if not hide:
        return samples
    if bounce_t_abs is not None and math.isfinite(float(bounce_t_abs)):
        pre_obs = [p for p in obs_points if float(p.t_abs) <= float(bounce_t_abs)]
    elif land_t_abs is not None and math.isfinite(float(land_t_abs)):
        # 兜底：没有反弹时间时退化到落地时间（效果会偏保守，可能多隐藏一些采样）。
        pre_obs = [p for p in obs_points if float(p.t_abs) <= float(land_t_abs)]
    else:
        return samples
    if not pre_obs:
        return samples

    n = max(int(xz_window_n), 3)
    start_idx = max(len(pre_obs) - n, 0)
    t_ref_abs = float(pre_obs[start_idx].t_abs)
    if not math.isfinite(t_ref_abs):
        return samples

    out = {
        "t_abs": list(samples.get("t_abs", [])),
        "x": list(samples.get("x", [])),
        "y": list(samples.get("y", [])),
        "z": list(samples.get("z", [])),
    }
    land_cap = float("inf")
    if land_t_abs is not None and math.isfinite(float(land_t_abs)):
        land_cap = float(land_t_abs)
    ts = out["t_abs"]
    for i, ti in enumerate(ts):
        t = _nan(ti)
        if not math.isfinite(t):
            continue
        if float(t) < float(t_ref_abs) and float(t) < land_cap:
            out["x"][i] = None
            out["y"][i] = None
            out["z"][i] = None
    return out


def _plot_matplotlib_png(
    *,
    title: str,
    report: dict[str, Any],
    all_points: list[ObsPoint],
    subset_points: list[ObsPoint],
    fit: dict[str, Any],
    axis_mode: str,
    show_curve3: bool,
    hide_prefit_extrap: bool,
    xz_equal_aspect: bool,
    ground_y: float,
    show_below_ground: bool,
    out_png: Path,
) -> None:
    """使用 matplotlib 输出 PNG 图。"""

    import matplotlib

    # 说明：在 CI/无 GUI 环境下强制使用 Agg，避免弹窗/后端问题。
    matplotlib.use("Agg")
    # 说明：Windows 下 DejaVu Sans 往往不含中文，设置常见中文字体回退。
    # 若系统缺失这些字体，matplotlib 会自动回退到其他字体。
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    def _plot_convergence_detailed_png(*, out_png2: Path) -> None:
        """额外输出一张“更详细、更直观、能对齐时间”的收敛图。

        画法要点：
        - 横轴使用“prefix 最后一帧的相对时间 t_last_rel(s)”，可直接对应真实时间推进。
        - 顶部再放一个辅助 x 轴，用同一组 x 坐标标注对应的 k（prefix 点数）。
        - 纵轴画误差到最终值：位置误差 |Δxz|（cm）+ 时间误差 |Δt|（ms）。
        - 额外标注 post 段开始（n_post 从 0 变为 >0 的时刻），解释“为什么 hit 早期为空”。
        """

        steps2 = report.get("convergence", {}).get("steps", [])
        if not isinstance(steps2, list) or not steps2:
            return

        # x: time axis (seconds) for each prefix
        ks2 = np.asarray([float(_nan(s.get("k"))) for s in steps2], dtype=float)
        t_last_abs2 = np.asarray([float(_nan(s.get("t_last_abs"))) for s in steps2], dtype=float)
        t_last_rel2 = t_last_abs2 - float(t0_abs)

        n_post2 = np.asarray([float(_nan(s.get("n_post"))) for s in steps2], dtype=float)
        idx_post = np.where(np.isfinite(n_post2) & (n_post2 > 0.0))[0]
        t_post_start = float(t_last_rel2[idx_post[0]]) if idx_post.size else None

        def step_series2(path: tuple[str, ...], key: str) -> np.ndarray:
            out: list[float] = []
            for s in steps2:
                cur: Any = s
                for p in path:
                    if not isinstance(cur, dict):
                        cur = None
                        break
                    cur = cur.get(p)
                if isinstance(cur, dict):
                    out.append(_nan(cur.get(key)))
                else:
                    out.append(float("nan"))
            return np.asarray(out, dtype=float)

        # curve2 predicted values per step
        hit_x2_up2 = step_series2(("curve2", "hit_up"), "x")
        hit_z2_up2 = step_series2(("curve2", "hit_up"), "z")
        hit_t2_up2 = step_series2(("curve2", "hit_up"), "t_rel")
        hit_x2_down2 = step_series2(("curve2", "hit_down"), "x")
        hit_z2_down2 = step_series2(("curve2", "hit_down"), "z")
        hit_t2_down2 = step_series2(("curve2", "hit_down"), "t_rel")

        land_x2_2 = step_series2(("curve2", "land"), "x")
        land_z2_2 = step_series2(("curve2", "land"), "z")
        land_t2_2 = step_series2(("curve2", "land"), "t_rel")

        def last_finite(a: np.ndarray) -> float:
            idx = np.where(np.isfinite(a))[0]
            return float(a[idx[-1]]) if idx.size else float("nan")

        def err_xz_cm(x: np.ndarray, z: np.ndarray) -> np.ndarray:
            xf = last_finite(x)
            zf = last_finite(z)
            e = np.sqrt((x - xf) ** 2 + (z - zf) ** 2)
            return e * 100.0

        def err_t_ms(t: np.ndarray) -> np.ndarray:
            tf = last_finite(t)
            e = np.abs(t - tf)
            return e * 1000.0

        e_hit_down_cm = err_xz_cm(hit_x2_down2, hit_z2_down2)
        e_hit_up_cm = err_xz_cm(hit_x2_up2, hit_z2_up2)
        e_land_cm = err_xz_cm(land_x2_2, land_z2_2)

        e_hit_down_ms = err_t_ms(hit_t2_down2)
        e_hit_up_ms = err_t_ms(hit_t2_up2)
        e_land_ms = err_t_ms(land_t2_2)

        # 2x2 layout: left=hit, right=land; top=pos error, bottom=time error
        fig2, axs2 = plt.subplots(2, 2, figsize=(14, 7.5), dpi=160)
        fig2.suptitle(f"{title} - 收敛（按时间对齐）")

        def add_top_k_axis(ax_):
            # 顶部辅助轴：用同一组 x 坐标展示对应 k（不可逆映射，用采样刻度）。
            ax_top = ax_.twiny()
            ax_top.set_xlim(ax_.get_xlim())
            idx = np.where(np.isfinite(t_last_rel2) & np.isfinite(ks2))[0]
            if idx.size:
                # 取最多 7 个刻度，避免拥挤。
                n_ticks = min(7, int(idx.size))
                pick = np.linspace(0, idx.size - 1, n_ticks, dtype=int)
                idx_pick = idx[pick]
                ax_top.set_xticks([float(t_last_rel2[i]) for i in idx_pick])
                ax_top.set_xticklabels([str(int(ks2[i])) for i in idx_pick])
            ax_top.set_xlabel("k (prefix points)")
            return ax_top

        def decorate(ax_):
            ax_.grid(True, alpha=0.25)
            if t_post_start is not None and math.isfinite(float(t_post_start)):
                ax_.axvline(float(t_post_start), color="#666", linestyle=":", linewidth=1.2, alpha=0.8)
                ax_.text(
                    float(t_post_start),
                    0.98,
                    "post开始",
                    transform=ax_.get_xaxis_transform(),
                    ha="left",
                    va="top",
                    fontsize=9,
                    color="#555",
                )

        # hit: pos error
        ax = axs2[0, 0]
        ax.set_title("击球点：位置误差到最终值 |Δxz| (cm)")
        ax.plot(t_last_rel2, e_hit_down_cm, color="#1c7ed6", label="curve2_down")
        ax.plot(t_last_rel2, e_hit_up_cm, color="#37b24d", label="curve2_up")
        ax.axhline(5.0, color="#999", linestyle="--", linewidth=1.0, alpha=0.5, label="5cm")
        ax.axhline(10.0, color="#aaa", linestyle="--", linewidth=1.0, alpha=0.35, label="10cm")
        ax.set_xlabel("t_last_rel (s)")
        ax.set_ylabel("|Δxz| (cm)")
        decorate(ax)
        ax.legend(loc="best", fontsize=9)
        add_top_k_axis(ax)

        # hit: time error
        ax = axs2[1, 0]
        ax.set_title("击球点：时刻误差到最终值 |Δt| (ms)")
        ax.plot(t_last_rel2, e_hit_down_ms, color="#1c7ed6", label="curve2_down")
        ax.plot(t_last_rel2, e_hit_up_ms, color="#37b24d", label="curve2_up")
        ax.axhline(10.0, color="#999", linestyle="--", linewidth=1.0, alpha=0.5, label="10ms")
        ax.axhline(20.0, color="#aaa", linestyle="--", linewidth=1.0, alpha=0.35, label="20ms")
        ax.set_xlabel("t_last_rel (s)")
        ax.set_ylabel("|Δt| (ms)")
        decorate(ax)
        ax.legend(loc="best", fontsize=9)
        add_top_k_axis(ax)

        # land: pos error
        ax = axs2[0, 1]
        ax.set_title("落地点：位置误差到最终值 |Δxz| (cm)")
        ax.plot(t_last_rel2, e_land_cm, color="#1c7ed6", label="curve2")
        ax.axhline(5.0, color="#999", linestyle="--", linewidth=1.0, alpha=0.5, label="5cm")
        ax.axhline(10.0, color="#aaa", linestyle="--", linewidth=1.0, alpha=0.35, label="10cm")
        ax.set_xlabel("t_last_rel (s)")
        ax.set_ylabel("|Δxz| (cm)")
        decorate(ax)
        ax.legend(loc="best", fontsize=9)
        add_top_k_axis(ax)

        # land: time error
        ax = axs2[1, 1]
        ax.set_title("落地点：时刻误差到最终值 |Δt| (ms)")
        ax.plot(t_last_rel2, e_land_ms, color="#1c7ed6", label="curve2")
        ax.axhline(10.0, color="#999", linestyle="--", linewidth=1.0, alpha=0.5, label="10ms")
        ax.axhline(20.0, color="#aaa", linestyle="--", linewidth=1.0, alpha=0.35, label="20ms")
        ax.set_xlabel("t_last_rel (s)")
        ax.set_ylabel("|Δt| (ms)")
        decorate(ax)
        ax.legend(loc="best", fontsize=9)
        add_top_k_axis(ax)

        fig2.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
        out_png2.parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(out_png2, bbox_inches="tight")
        plt.close(fig2)

    c3 = dict(fit["curve3"]["samples"]) if bool(show_curve3) else {"t_abs": [], "x": [], "y": [], "z": []}
    c2 = dict(fit["curve2"]["samples"])

    hit2_up = fit.get("curve2", {}).get("hit_on_plane_up")
    hit2_down = fit.get("curve2", {}).get("hit_on_plane_down")
    land2 = fit.get("curve2", {}).get("predicted_land")
    hit2_up_t_abs = _as_float(hit2_up.get("t_abs")) if isinstance(hit2_up, dict) else None
    hit2_down_t_abs = _as_float(hit2_down.get("t_abs")) if isinstance(hit2_down, dict) else None
    land2_t_abs = _as_float(land2.get("t_abs")) if isinstance(land2, dict) else None
    hit2_up_x = _as_float(hit2_up.get("x")) if isinstance(hit2_up, dict) else None
    hit2_up_z = _as_float(hit2_up.get("z")) if isinstance(hit2_up, dict) else None
    hit2_down_x = _as_float(hit2_down.get("x")) if isinstance(hit2_down, dict) else None
    hit2_down_z = _as_float(hit2_down.get("z")) if isinstance(hit2_down, dict) else None
    land2_x = _as_float(land2.get("x")) if isinstance(land2, dict) else None
    land2_z = _as_float(land2.get("z")) if isinstance(land2, dict) else None

    observed = report.get("observed") if isinstance(report.get("observed"), dict) else None
    obs_hit_up = observed.get("hit_on_plane_post_up") if isinstance(observed, dict) else None
    obs_hit_down = observed.get("hit_on_plane_post_down") if isinstance(observed, dict) else None
    obs_hit_up_t_abs = _as_float(obs_hit_up.get("t_abs")) if isinstance(obs_hit_up, dict) else None
    obs_hit_down_t_abs = _as_float(obs_hit_down.get("t_abs")) if isinstance(obs_hit_down, dict) else None
    obs_hit_up_x = _as_float(obs_hit_up.get("x")) if isinstance(obs_hit_up, dict) else None
    obs_hit_up_y = _as_float(obs_hit_up.get("y")) if isinstance(obs_hit_up, dict) else None
    obs_hit_up_z = _as_float(obs_hit_up.get("z")) if isinstance(obs_hit_up, dict) else None
    obs_hit_down_x = _as_float(obs_hit_down.get("x")) if isinstance(obs_hit_down, dict) else None
    obs_hit_down_y = _as_float(obs_hit_down.get("y")) if isinstance(obs_hit_down, dict) else None
    obs_hit_down_z = _as_float(obs_hit_down.get("z")) if isinstance(obs_hit_down, dict) else None

    if bool(show_curve3):
        land3 = fit["curve3"].get("predicted_land")
        land3_t = None if land3 is None else float(land3.get("t_abs"))
        c3 = _mask_curve3_prefit_extrap_python(
            samples=c3,
            bounce_t_abs=report.get("bounce_t_abs_full"),
            land_t_abs=land3_t,
            obs_points=subset_points,
            xz_window_n=12,
            hide=bool(hide_prefit_extrap),
        )

    t0_abs = float(all_points[0].t_abs)

    def arr(samples_: dict[str, Any], k: str) -> np.ndarray:
        return np.asarray([_nan(v) for v in samples_.get(k, [])], dtype=float)

    # obs: 区分 subset 与 excluded
    subset_ids = {id(p) for p in subset_points}
    obs_in = [p for p in all_points if id(p) in subset_ids]
    obs_out = [p for p in all_points if id(p) not in subset_ids]

    obs_in_x = np.asarray([p.x for p in obs_in], dtype=float)
    obs_in_y = np.asarray([p.y for p in obs_in], dtype=float)
    obs_in_z = np.asarray([p.z for p in obs_in], dtype=float)
    obs_in_t = np.asarray([p.t_abs - t0_abs for p in obs_in], dtype=float)

    obs_out_x = np.asarray([p.x for p in obs_out], dtype=float)
    obs_out_y = np.asarray([p.y for p in obs_out], dtype=float)
    obs_out_z = np.asarray([p.z for p in obs_out], dtype=float)
    obs_out_t = np.asarray([p.t_abs - t0_abs for p in obs_out], dtype=float)

    c3_x = arr(c3, "x")
    c3_y = arr(c3, "y")
    c3_z = arr(c3, "z")
    c3_t = arr(c3, "t_abs") - t0_abs

    c2_x = arr(c2, "x")
    c2_y = arr(c2, "y")
    c2_z = arr(c2, "z")
    c2_t = arr(c2, "t_abs") - t0_abs

    # 说明：curve3 在“第二次触地”之后通常不会再建模（只处理第一次触地/反弹），
    # 因此可能继续向下掉到 y<0，视觉上很别扭。
    # 默认把 y < ground_y 的采样点掩码掉（置为 NaN），让曲线自动断开。
    if not bool(show_below_ground):
        gy = float(ground_y)

        def mask_below_ground(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            m = np.isfinite(y) & (y < gy - 1e-6)
            if not np.any(m):
                return x, y, z
            x2 = x.copy()
            y2 = y.copy()
            z2 = z.copy()
            x2[m] = float("nan")
            y2[m] = float("nan")
            z2[m] = float("nan")
            return x2, y2, z2

        c3_x, c3_y, c3_z = mask_below_ground(c3_x, c3_y, c3_z)
        c2_x, c2_y, c2_z = mask_below_ground(c2_x, c2_y, c2_z)

    steps = report.get("convergence", {}).get("steps", [])
    ks = np.asarray([int(s.get("k", i)) for i, s in enumerate(steps)], dtype=float)

    def step_series(path: tuple[str, ...], key: str) -> np.ndarray:
        out: list[float] = []
        for s in steps:
            cur: Any = s
            for p in path:
                if not isinstance(cur, dict):
                    cur = None
                    break
                cur = cur.get(p)
            if isinstance(cur, dict):
                out.append(_nan(cur.get(key)))
            else:
                out.append(float("nan"))
        return np.asarray(out, dtype=float)

    hit_x3 = step_series(("curve3", "hit"), "x")
    hit_z3 = step_series(("curve3", "hit"), "z")
    hit_x2_up = step_series(("curve2", "hit_up"), "x")
    hit_z2_up = step_series(("curve2", "hit_up"), "z")
    hit_x2_down = step_series(("curve2", "hit_down"), "x")
    hit_z2_down = step_series(("curve2", "hit_down"), "z")

    hit_t3 = step_series(("curve3", "hit"), "t_rel")
    # 说明：时间收敛曲线这里默认只画下降段，避免上下两条时刻线叠加过于拥挤。
    hit_t2 = step_series(("curve2", "hit_down"), "t_rel")

    land_x3 = step_series(("curve3", "land"), "x")
    land_z3 = step_series(("curve3", "land"), "z")
    land_x2 = step_series(("curve2", "land"), "x")
    land_z2 = step_series(("curve2", "land"), "z")

    land_t3 = step_series(("curve3", "land"), "t_rel")
    land_t2 = step_series(("curve2", "land"), "t_rel")

    fig, axs = plt.subplots(3, 2, figsize=(14, 10), dpi=160)
    fig.suptitle(title)

    # x-z
    ax = axs[0, 0]
    ax.set_title("x-z 俯视图")
    if obs_out_x.size:
        ax.scatter(obs_out_x, obs_out_z, s=10, c="#bbbbbb", alpha=0.6, label="obs(excluded)")
    ax.scatter(obs_in_x, obs_in_z, s=14, c="black", label="obs(subset)")
    if bool(show_curve3):
        ax.plot(c3_x, c3_z, color="#d9480f", linewidth=2.0, label="curve3")
    ax.plot(c2_x, c2_z, color="#1c7ed6", linewidth=2.0, label="curve2")

    # 标注：观测击球点估计 / curve2 预测击球点（上升/下降）/ curve2 预测落地点。
    if hit2_up_x is not None and hit2_up_z is not None:
        ax.scatter(
            [float(hit2_up_x)],
            [float(hit2_up_z)],
            s=90,
            c="#40c057",
            marker="^",
            edgecolors="white",
            linewidths=1.5,
            label="hit_pred_up(curve2)",
        )
    if hit2_down_x is not None and hit2_down_z is not None:
        ax.scatter(
            [float(hit2_down_x)],
            [float(hit2_down_z)],
            s=90,
            c="#37b24d",
            marker="o",
            edgecolors="white",
            linewidths=1.5,
            label="hit_pred_down(curve2)",
        )
    if land2_x is not None and land2_z is not None:
        ax.scatter(
            [float(land2_x)],
            [float(land2_z)],
            s=100,
            c="#862e9c",
            marker="X",
            edgecolors="white",
            linewidths=1.5,
            label="land_pred(curve2)",
        )
    if obs_hit_up_x is not None and obs_hit_up_z is not None:
        ax.scatter(
            [float(obs_hit_up_x)],
            [float(obs_hit_up_z)],
            s=80,
            c="#0b7285",
            marker="D",
            edgecolors="white",
            linewidths=1.2,
            label="hit_obs_est_up(post)",
        )
    if obs_hit_down_x is not None and obs_hit_down_z is not None:
        ax.scatter(
            [float(obs_hit_down_x)],
            [float(obs_hit_down_z)],
            s=80,
            c="#1098ad",
            marker="D",
            edgecolors="white",
            linewidths=1.2,
            label="hit_obs_est_down(post)",
        )

    extra_x = np.asarray(
        [_nan(hit2_up_x), _nan(hit2_down_x), _nan(land2_x), _nan(obs_hit_up_x), _nan(obs_hit_down_x)],
        dtype=float,
    )
    extra_z = np.asarray(
        [_nan(hit2_up_z), _nan(hit2_down_z), _nan(land2_z), _nan(obs_hit_up_z), _nan(obs_hit_down_z)],
        dtype=float,
    )

    x_all = np.concatenate(
        [
            obs_in_x,
            obs_out_x,
            c2_x,
            c3_x if bool(show_curve3) else np.asarray([], dtype=float),
            extra_x,
        ]
    )
    z_all = np.concatenate(
        [
            obs_in_z,
            obs_out_z,
            c2_z,
            c3_z if bool(show_curve3) else np.asarray([], dtype=float),
            extra_z,
        ]
    )
    ax.set_xlim(_robust_limits(x_all, mode=axis_mode))
    ax.set_ylim(_robust_limits(z_all, mode=axis_mode))
    if bool(xz_equal_aspect):
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    # x-t
    ax = axs[0, 1]
    ax.set_title("x-t")
    if obs_out_t.size:
        ax.scatter(obs_out_t, obs_out_x, s=10, c="#bbbbbb", alpha=0.6, label="obs(excluded)")
    ax.scatter(obs_in_t, obs_in_x, s=14, c="black", label="obs(subset)")
    if bool(show_curve3):
        ax.plot(c3_t, c3_x, color="#d9480f", linewidth=2.0, label="curve3")
    ax.plot(c2_t, c2_x, color="#1c7ed6", linewidth=2.0, label="curve2")

    hit2_up_t = None if hit2_up_t_abs is None else float(hit2_up_t_abs - t0_abs)
    hit2_down_t = None if hit2_down_t_abs is None else float(hit2_down_t_abs - t0_abs)
    land2_t = None if land2_t_abs is None else float(land2_t_abs - t0_abs)
    obs_hit_up_t = None if obs_hit_up_t_abs is None else float(obs_hit_up_t_abs - t0_abs)
    obs_hit_down_t = None if obs_hit_down_t_abs is None else float(obs_hit_down_t_abs - t0_abs)

    # 竖线标注：预测击球点/落地点/观测击球点估计。
    if hit2_up_t is not None and math.isfinite(float(hit2_up_t)):
        ax.axvline(float(hit2_up_t), color="#40c057", linestyle="--", linewidth=1.5, alpha=0.9)
        if hit2_up_x is not None:
            ax.scatter(
                [float(hit2_up_t)],
                [float(hit2_up_x)],
                s=60,
                c="#40c057",
                marker="^",
                edgecolors="white",
                linewidths=1.2,
            )
    if hit2_down_t is not None and math.isfinite(float(hit2_down_t)):
        ax.axvline(float(hit2_down_t), color="#37b24d", linestyle="--", linewidth=1.5, alpha=0.9)
        if hit2_down_x is not None:
            ax.scatter(
                [float(hit2_down_t)],
                [float(hit2_down_x)],
                s=60,
                c="#37b24d",
                edgecolors="white",
                linewidths=1.2,
            )
    if land2_t is not None and math.isfinite(float(land2_t)):
        ax.axvline(float(land2_t), color="#862e9c", linestyle="--", linewidth=1.5, alpha=0.9)
        if land2_x is not None:
            ax.scatter(
                [float(land2_t)],
                [float(land2_x)],
                s=70,
                c="#862e9c",
                marker="X",
                edgecolors="white",
                linewidths=1.2,
            )
    if obs_hit_up_t is not None and math.isfinite(float(obs_hit_up_t)):
        ax.axvline(float(obs_hit_up_t), color="#0b7285", linestyle=":", linewidth=1.2, alpha=0.9)
    if obs_hit_down_t is not None and math.isfinite(float(obs_hit_down_t)):
        ax.axvline(float(obs_hit_down_t), color="#1098ad", linestyle=":", linewidth=1.2, alpha=0.9)

    extra_t = np.asarray(
        [_nan(hit2_up_t), _nan(hit2_down_t), _nan(land2_t), _nan(obs_hit_up_t), _nan(obs_hit_down_t)],
        dtype=float,
    )
    extra_x = np.asarray(
        [_nan(hit2_up_x), _nan(hit2_down_x), _nan(land2_x), _nan(obs_hit_up_x), _nan(obs_hit_down_x)],
        dtype=float,
    )

    t_all = np.concatenate(
        [
            obs_in_t,
            obs_out_t,
            c2_t,
            c3_t if bool(show_curve3) else np.asarray([], dtype=float),
            extra_t,
        ]
    )
    x_all = np.concatenate(
        [
            obs_in_x,
            obs_out_x,
            c2_x,
            c3_x if bool(show_curve3) else np.asarray([], dtype=float),
            extra_x,
        ]
    )
    ax.set_xlim(_robust_limits(t_all, mode=axis_mode))
    ax.set_ylim(_robust_limits(x_all, mode=axis_mode))
    ax.set_xlabel("t_rel (s)")
    ax.set_ylabel("x (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    # y-t
    ax = axs[1, 0]
    ax.set_title("y-t")
    if obs_out_t.size:
        ax.scatter(obs_out_t, obs_out_y, s=10, c="#bbbbbb", alpha=0.6, label="obs(excluded)")
    ax.scatter(obs_in_t, obs_in_y, s=14, c="black", label="obs(subset)")
    if bool(show_curve3):
        ax.plot(c3_t, c3_y, color="#d9480f", linewidth=2.0, label="curve3")
    ax.plot(c2_t, c2_y, color="#1c7ed6", linewidth=2.0, label="curve2")
    ax.axhline(float(ground_y), color="#666", linewidth=1.0, alpha=0.35)

    # 额外标注：target_y + 预测击球/落地时刻 + 观测击球点估计。
    ax.axhline(float(report.get("target_y", 0.0)), color="#74c0fc", linewidth=1.5, linestyle="--", alpha=0.85)
    if hit2_up_t is not None and math.isfinite(float(hit2_up_t)):
        ax.axvline(float(hit2_up_t), color="#40c057", linestyle="--", linewidth=1.5, alpha=0.9)
        ax.scatter(
            [float(hit2_up_t)],
            [float(report.get("target_y", 0.0))],
            s=60,
            c="#40c057",
            marker="^",
            edgecolors="white",
            linewidths=1.2,
        )
    if hit2_down_t is not None and math.isfinite(float(hit2_down_t)):
        ax.axvline(float(hit2_down_t), color="#37b24d", linestyle="--", linewidth=1.5, alpha=0.9)
        ax.scatter(
            [float(hit2_down_t)],
            [float(report.get("target_y", 0.0))],
            s=60,
            c="#37b24d",
            edgecolors="white",
            linewidths=1.2,
        )
    if land2_t is not None and math.isfinite(float(land2_t)):
        ax.axvline(float(land2_t), color="#862e9c", linestyle="--", linewidth=1.5, alpha=0.9)
        ax.scatter(
            [float(land2_t)],
            [float(ground_y)],
            s=70,
            c="#862e9c",
            marker="X",
            edgecolors="white",
            linewidths=1.2,
        )
    if obs_hit_up_t is not None and math.isfinite(float(obs_hit_up_t)):
        ax.axvline(float(obs_hit_up_t), color="#0b7285", linestyle=":", linewidth=1.2, alpha=0.9)
        if obs_hit_up_y is not None:
            ax.scatter(
                [float(obs_hit_up_t)],
                [float(obs_hit_up_y)],
                s=55,
                c="#0b7285",
                marker="D",
                edgecolors="white",
                linewidths=1.0,
            )
    if obs_hit_down_t is not None and math.isfinite(float(obs_hit_down_t)):
        ax.axvline(float(obs_hit_down_t), color="#1098ad", linestyle=":", linewidth=1.2, alpha=0.9)
        if obs_hit_down_y is not None:
            ax.scatter(
                [float(obs_hit_down_t)],
                [float(obs_hit_down_y)],
                s=55,
                c="#1098ad",
                marker="D",
                edgecolors="white",
                linewidths=1.0,
            )
    event_t = np.asarray(
        [_nan(hit2_up_t), _nan(hit2_down_t), _nan(land2_t), _nan(obs_hit_up_t), _nan(obs_hit_down_t)],
        dtype=float,
    )
    event_y = np.asarray(
        [
            _nan(float(report.get("target_y", 0.0))),
            _nan(float(ground_y)),
            _nan(obs_hit_up_y),
            _nan(obs_hit_down_y),
        ],
        dtype=float,
    )
    t_all = np.concatenate(
        [
            obs_in_t,
            obs_out_t,
            c2_t,
            c3_t if bool(show_curve3) else np.asarray([], dtype=float),
            event_t,
        ]
    )
    y_all = np.concatenate(
        [
            obs_in_y,
            obs_out_y,
            c2_y,
            c3_y if bool(show_curve3) else np.asarray([], dtype=float),
            event_y,
        ]
    )
    ax.set_xlim(_robust_limits(t_all, mode=axis_mode))
    ax.set_ylim(_robust_limits(y_all, mode=axis_mode))
    ax.set_xlabel("t_rel (s)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    # z-t
    ax = axs[1, 1]
    ax.set_title("z-t")
    if obs_out_t.size:
        ax.scatter(obs_out_t, obs_out_z, s=10, c="#bbbbbb", alpha=0.6, label="obs(excluded)")
    ax.scatter(obs_in_t, obs_in_z, s=14, c="black", label="obs(subset)")
    if bool(show_curve3):
        ax.plot(c3_t, c3_z, color="#d9480f", linewidth=2.0, label="curve3")
    ax.plot(c2_t, c2_z, color="#1c7ed6", linewidth=2.0, label="curve2")

    if hit2_up_t is not None and math.isfinite(float(hit2_up_t)):
        ax.axvline(float(hit2_up_t), color="#40c057", linestyle="--", linewidth=1.5, alpha=0.9)
        if hit2_up_z is not None:
            ax.scatter(
                [float(hit2_up_t)],
                [float(hit2_up_z)],
                s=60,
                c="#40c057",
                marker="^",
                edgecolors="white",
                linewidths=1.2,
            )
    if hit2_down_t is not None and math.isfinite(float(hit2_down_t)):
        ax.axvline(float(hit2_down_t), color="#37b24d", linestyle="--", linewidth=1.5, alpha=0.9)
        if hit2_down_z is not None:
            ax.scatter(
                [float(hit2_down_t)],
                [float(hit2_down_z)],
                s=60,
                c="#37b24d",
                edgecolors="white",
                linewidths=1.2,
            )
    if land2_t is not None and math.isfinite(float(land2_t)):
        ax.axvline(float(land2_t), color="#862e9c", linestyle="--", linewidth=1.5, alpha=0.9)
        if land2_z is not None:
            ax.scatter(
                [float(land2_t)],
                [float(land2_z)],
                s=70,
                c="#862e9c",
                marker="X",
                edgecolors="white",
                linewidths=1.2,
            )
    if obs_hit_up_t is not None and math.isfinite(float(obs_hit_up_t)):
        ax.axvline(float(obs_hit_up_t), color="#0b7285", linestyle=":", linewidth=1.2, alpha=0.9)
        if obs_hit_up_z is not None:
            ax.scatter(
                [float(obs_hit_up_t)],
                [float(obs_hit_up_z)],
                s=55,
                c="#0b7285",
                marker="D",
                edgecolors="white",
                linewidths=1.0,
            )
    if obs_hit_down_t is not None and math.isfinite(float(obs_hit_down_t)):
        ax.axvline(float(obs_hit_down_t), color="#1098ad", linestyle=":", linewidth=1.2, alpha=0.9)
        if obs_hit_down_z is not None:
            ax.scatter(
                [float(obs_hit_down_t)],
                [float(obs_hit_down_z)],
                s=55,
                c="#1098ad",
                marker="D",
                edgecolors="white",
                linewidths=1.0,
            )
    event_t = np.asarray(
        [_nan(hit2_up_t), _nan(hit2_down_t), _nan(land2_t), _nan(obs_hit_up_t), _nan(obs_hit_down_t)],
        dtype=float,
    )
    t_all = np.concatenate(
        [
            obs_in_t,
            obs_out_t,
            c2_t,
            c3_t if bool(show_curve3) else np.asarray([], dtype=float),
            event_t,
        ]
    )
    z_all = np.concatenate([obs_in_z, obs_out_z, c2_z, c3_z if bool(show_curve3) else np.asarray([], dtype=float)])
    ax.set_xlim(_robust_limits(t_all, mode=axis_mode))
    ax.set_ylim(_robust_limits(z_all, mode=axis_mode))
    ax.set_xlabel("t_rel (s)")
    ax.set_ylabel("z (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    # 收敛图（更直观）：画“到最终解的水平位置误差 |Δxz|”。
    # 说明：相比直接画 x/z 绝对值，这种画法更容易一眼看出是否“在收敛”。

    def last_finite(a: np.ndarray) -> float:
        idx = np.where(np.isfinite(a))[0]
        return float(a[idx[-1]]) if idx.size else float("nan")

    # hit convergence（curve2 up/down）
    ax = axs[2, 0]
    ax.set_title("击球点收敛（距最终值 |Δxz|）")

    hit_x2_down_f = last_finite(hit_x2_down)
    hit_z2_down_f = last_finite(hit_z2_down)
    hit_x2_up_f = last_finite(hit_x2_up)
    hit_z2_up_f = last_finite(hit_z2_up)

    err_hit_down = np.sqrt((hit_x2_down - hit_x2_down_f) ** 2 + (hit_z2_down - hit_z2_down_f) ** 2)
    err_hit_up = np.sqrt((hit_x2_up - hit_x2_up_f) ** 2 + (hit_z2_up - hit_z2_up_f) ** 2)

    ax.plot(ks, err_hit_down, color="#1c7ed6", label="curve2_down |Δxz|")
    ax.plot(ks, err_hit_up, color="#37b24d", label="curve2_up |Δxz|")

    err_all = np.concatenate([err_hit_down, err_hit_up])
    err_max = float(np.nanmax(err_all)) if np.any(np.isfinite(err_all)) else 1.0
    ax.set_xlim(_robust_limits(ks, mode="full"))
    ax.set_ylim(0.0, err_max * 1.05)
    ax.set_xlabel("k (prefix points)")
    ax.set_ylabel("|Δxz| (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    # land convergence（curve2）
    ax = axs[2, 1]
    ax.set_title("落地点收敛（距最终值 |Δxz|）")

    land_x2_f = last_finite(land_x2)
    land_z2_f = last_finite(land_z2)
    err_land = np.sqrt((land_x2 - land_x2_f) ** 2 + (land_z2 - land_z2_f) ** 2)

    ax.plot(ks, err_land, color="#1c7ed6", label="curve2 |Δxz|")

    err_max = float(np.nanmax(err_land)) if np.any(np.isfinite(err_land)) else 1.0
    ax.set_xlim(_robust_limits(ks, mode="full"))
    ax.set_ylim(0.0, err_max * 1.05)
    ax.set_xlabel("k (prefix points)")
    ax.set_ylabel("|Δxz| (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    # 额外输出：更详细、且可按时间对齐的收敛图（单独一张，不挤在主图里）。
    out_png_conv = out_png.with_name(f"{out_png.stem}.convergence_time.png")
    _plot_convergence_detailed_png(out_png2=out_png_conv)


def _subset_by_post_n(
    all_points: list[ObsPoint], *, bounce_t_abs_full: float | None, post_n: int | None
) -> tuple[list[ObsPoint], dict[str, Any]]:
    """按 post 点数 N 构造子集。

    约定：post 点定义为 t_abs > bounce_t_abs_full。

    Args:
        all_points: 全量点。
        bounce_t_abs_full: 全量拟合得到的反弹时刻（绝对时间）。
        post_n: 目标 post 点数；None 表示取全量。

    Returns:
        (subset_points, subset_meta)
    """

    if bounce_t_abs_full is None:
        # 兜底：没法稳定切分 pre/post 时，直接按“全量/不截断”。
        return (
            list(all_points),
            {
                "n_pre": len(all_points),
                "n_post": 0,
                "n_post_total": 0,
                "post_n_req": None if post_n is None else int(post_n),
                # 说明：没有 bounce 时刻无法判断 post 点是否足够，这里保守设为 True。
                "has_enough_post": True,
            },
        )

    pre = [p for p in all_points if float(p.t_abs) <= float(bounce_t_abs_full)]
    post = [p for p in all_points if float(p.t_abs) > float(bounce_t_abs_full)]

    if post_n is None:
        subset = pre + post
        has_enough_post = True
    else:
        subset = pre + post[: int(post_n)]
        has_enough_post = len(post) >= int(post_n)

    return (
        subset,
        {
            "n_pre": len(pre),
            "n_post": (len(post) if post_n is None else min(len(post), int(post_n))),
            "n_post_total": int(len(post)),
            "post_n_req": None if post_n is None else int(post_n),
            "has_enough_post": bool(has_enough_post),
        },
    )


def _build_report(
    *,
    points: list[ObsPoint],
    track_id: int | None,
    target_y: float,
    post_ns: list[int],
    plot_curve3: bool = True,
) -> dict[str, Any]:
    if not points:
        raise ValueError("no points")

    # curve3：为了与线上 legacy 输出对齐，这里把 bounce_contact_y 固定为 0。
    cfg3 = CurveV3Config(bounce_contact_y_m=0.0)

    # 说明：当用户关闭 curve3 时，我们仍然需要一个“bounce 时刻”的估计用于切分 pre/post。
    # 这里保留一次全量 curve3 拟合用来估计 bounce_t_abs_full，但后续不会再重复跑 curve3，
    # 以免收敛过程（prefix 多次拟合）导致运行时间过长。
    compute_curve3 = bool(plot_curve3)

    # 全量拟合（用于估计 bounce_t_abs_full + 收敛曲线的 post 计数）。
    pred3_full = _run_curve3(points, cfg=cfg3)
    land3_full = _curve3_predicted_land(pred3_full)
    bounce_t_abs_full = None if land3_full is None else float(land3_full["t_abs"])

    # 采样时间网格：覆盖观测区间，并向后延展一些，便于看“预测段”。
    t0_abs = float(points[0].t_abs)
    t_last_abs = float(points[-1].t_abs)
    if bounce_t_abs_full is None:
        t1_abs = t_last_abs
    else:
        t1_abs = max(t_last_abs, float(bounce_t_abs_full) + 1.2)

    t_grid_abs = _sample_abs_times(t0_abs=t0_abs, t1_abs=t1_abs, n=140)

    fits_by_post_n: dict[str, Any] = {}

    def add_fit(key: str, subset: list[ObsPoint], subset_meta: dict[str, Any]) -> None:
        if compute_curve3:
            pred3 = _run_curve3(subset, cfg=cfg3)
            land3 = _curve3_predicted_land(pred3)
            hit3 = _curve3_hit_on_plane(pred3, target_y=target_y)
            curve3_block = {
                "predicted_land": land3,
                "hit_on_plane": hit3,
                "samples": _sample_trajectory(
                    t_grid_abs=t_grid_abs,
                    point_at_abs_time=lambda t: _curve3_point_at_abs_time(pred3, t),
                ),
            }
        else:
            curve3_block = {
                "predicted_land": None,
                "hit_on_plane": None,
                "samples": {"t_abs": [], "x": [], "y": [], "z": []},
            }

        curve2 = _run_curve2(subset)
        land2 = _curve2_predicted_land(curve2)
        hit2_up = _curve2_hit_on_plane(curve2, target_y=target_y, pick="up")
        hit2_down = _curve2_hit_on_plane(curve2, target_y=target_y, pick="down")
        curve2_analysis = _curve2_fit_analysis(curve2, subset)

        fits_by_post_n[key] = {
            "subset": {
                "n_total": int(len(subset)),
                "n_pre": int(subset_meta.get("n_pre", 0)),
                "n_post": int(subset_meta.get("n_post", 0)),
            },
            "curve3": curve3_block,
            "curve2": {
                "predicted_land": land2,
                "hit_on_plane_up": hit2_up,
                "hit_on_plane_down": hit2_down,
                "analysis": curve2_analysis,
                "samples": _sample_trajectory(
                    t_grid_abs=t_grid_abs,
                    point_at_abs_time=lambda t: _curve2_point_at_abs_time(curve2, t),
                ),
            },
        }

    # 指定的 postN（若 post 点不足，则不生成该 N 的输出）。
    for n in post_ns:
        subset, meta = _subset_by_post_n(points, bounce_t_abs_full=bounce_t_abs_full, post_n=n)
        if not bool(meta.get("has_enough_post", True)):
            continue
        add_fit(str(int(n)), subset, meta)

    # 全量
    subset_all, meta_all = _subset_by_post_n(points, bounce_t_abs_full=bounce_t_abs_full, post_n=None)
    add_fit("all", subset_all, meta_all)

    # 收敛过程：对每个 prefix 重新拟合并记录 hit/land 的 xz。
    steps: list[dict[str, Any]] = []
    for k in range(5, len(points) + 1):
        prefix = points[:k]

        n_post = 0
        if bounce_t_abs_full is not None:
            n_post = sum(1 for p in prefix if float(p.t_abs) > float(bounce_t_abs_full))

        if compute_curve3:
            pred3 = _run_curve3(prefix, cfg=cfg3)
            land3 = _curve3_predicted_land(pred3)
            hit3 = _curve3_hit_on_plane(pred3, target_y=target_y)
        else:
            land3 = None
            hit3 = None

        curve2 = _run_curve2(prefix)
        land2 = _curve2_predicted_land(curve2)
        hit2_up = _curve2_hit_on_plane(curve2, target_y=target_y, pick="up")
        hit2_down = _curve2_hit_on_plane(curve2, target_y=target_y, pick="down")

        steps.append(
            {
                "k": int(k),
                "n_post": int(n_post),
                "t_last_abs": float(prefix[-1].t_abs),
                "curve3": {
                    "land": None
                    if land3 is None
                    else {
                        "x": land3["x"],
                        "z": land3["z"],
                        "t_abs": land3["t_abs"],
                        "t_rel": land3["t_rel"],
                    },
                    "hit": None
                    if hit3 is None
                    else {
                        "x": hit3["x"],
                        "z": hit3["z"],
                        "t_abs": hit3["t_abs"],
                        "t_rel": hit3["t_rel"],
                    },
                },
                "curve2": {
                    "land": None
                    if land2 is None
                    else {
                        "x": land2["x"],
                        "z": land2["z"],
                        "t_abs": land2["t_abs"],
                        "t_rel": land2["t_rel"],
                    },
                    "hit_up": None
                    if hit2_up is None
                    else {
                        "x": hit2_up["x"],
                        "z": hit2_up["z"],
                        "t_abs": hit2_up["t_abs"],
                        "t_rel": hit2_up["t_rel"],
                    },
                    "hit_down": None
                    if hit2_down is None
                    else {
                        "x": hit2_down["x"],
                        "z": hit2_down["z"],
                        "t_abs": hit2_down["t_abs"],
                        "t_rel": hit2_down["t_rel"],
                    },
                },
            }
        )

    # 说明：同一条轨迹在反弹前与反弹后都可能穿越 y=target_y。
    # 为了与 curve2 的“第二段/反弹后”口径一致，这里把 observed 的结果拆成：
    # - hit_on_plane_pre：反弹前（pre）穿越点（用于排查/对照）
    # - hit_on_plane_post_up：反弹后上升段穿越点（若存在）
    # - hit_on_plane_post_down：反弹后下降段穿越点（若存在）
    # 说明：bounce_t_abs_full 来自拟合估计，可能存在轻微偏差；这里给一个小裕量，
    # 以免把“刚反弹后”的点误判成反弹前而过滤掉。
    t_margin = 0.03

    observed_hit_pre = None
    observed_hit_post_up = None
    observed_hit_post_down = None
    if bounce_t_abs_full is not None:
        pre_pts = [p for p in points if float(p.t_abs) <= float(bounce_t_abs_full) + t_margin]
        if len(pre_pts) >= 2:
            observed_hit_pre = _estimate_observed_hit_on_plane_y(pre_pts, target_y=target_y, pick="last")
        observed_hit_post_up = _estimate_observed_hit_on_plane_y(
            points,
            target_y=target_y,
            t_min_abs=float(bounce_t_abs_full) - t_margin,
            pick="first",
        )
        observed_hit_post_down = _estimate_observed_hit_on_plane_y(
            points,
            target_y=target_y,
            t_min_abs=float(bounce_t_abs_full) - t_margin,
            pick="last",
        )
    else:
        # 没有 bounce 时间时无法稳定切分 pre/post，保守输出一个“全局”穿越点。
        observed_hit_pre = _estimate_observed_hit_on_plane_y(points, target_y=target_y, pick="first")

    # --- 误差摘要（写入 JSON，避免手算）---
    # 说明：这里的 hit_obs_est 来自离散观测点的线性插值/最近点估计，并非外部标注真值。
    # 该摘要的目的：
    # - 用“可复查的数值”支撑图上的误差/时间差解读；
    # - 便于把结果拷到表格/报告里。

    def _finite_num(v: Any) -> float | None:
        try:
            x = float(v)
        except Exception:
            return None
        if not math.isfinite(x):
            return None
        return float(x)

    def _hit_error(pred: dict[str, Any] | None, obs_est: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(pred, dict) or not isinstance(obs_est, dict):
            return None
        px = _finite_num(pred.get("x"))
        pz = _finite_num(pred.get("z"))
        pt = _finite_num(pred.get("t_abs"))
        ox = _finite_num(obs_est.get("x"))
        oz = _finite_num(obs_est.get("z"))
        ot = _finite_num(obs_est.get("t_abs"))
        if None in (px, pz, pt, ox, oz, ot):
            return None
        # 说明：上面的 None 检查对类型检查器不一定能做充分收窄，这里用断言明确约束。
        assert px is not None and pz is not None and pt is not None
        assert ox is not None and oz is not None and ot is not None
        dx = float(px - ox)
        dz = float(pz - oz)
        dxz = float(math.hypot(dx, dz))
        dt_ms = float((pt - ot) * 1000.0)
        return {
            "dx_m": dx,
            "dz_m": dz,
            "dxz_m": dxz,
            "dx_cm": dx * 100.0,
            "dz_cm": dz * 100.0,
            "dxz_cm": dxz * 100.0,
            "dt_ms": dt_ms,
            "pred_t_abs": pt,
            "obs_t_abs": ot,
            "obs_method": obs_est.get("method"),
        }

    summary: dict[str, Any] = {
        "target_y": float(target_y),
        "notes": {
            "hit_obs_est_is_not_ground_truth": True,
            "hit_obs_est_method": "linear_interp_or_closest",
        },
    }

    # 按不同 postN（以及 all）分别输出误差，便于直接对比“多喂点观测后”预测是否更稳定/更接近。
    # 说明：这里的误差仍然是 pred vs hit_obs_est（离散观测估计），不是外部标注真值。
    summary_by_post_n: dict[str, Any] = {}
    for key, fit in fits_by_post_n.items():
        if not isinstance(fit, dict):
            continue

        subset_block = fit.get("subset") if isinstance(fit.get("subset"), dict) else None
        c2 = fit.get("curve2") if isinstance(fit.get("curve2"), dict) else None
        if not isinstance(c2, dict):
            continue

        analysis = c2.get("analysis") if isinstance(c2.get("analysis"), dict) else None
        residual = None
        if isinstance(analysis, dict):
            residual = analysis.get("residual") if isinstance(analysis.get("residual"), dict) else None

        hit_pred_up = c2.get("hit_on_plane_up") if isinstance(c2.get("hit_on_plane_up"), dict) else None
        hit_pred_down = c2.get("hit_on_plane_down") if isinstance(c2.get("hit_on_plane_down"), dict) else None

        summary_by_post_n[str(key)] = {
            "subset": subset_block,
            "curve2_residual": residual,
            "hit_pred_vs_obs_est_post": {
                "up": _hit_error(hit_pred_up, observed_hit_post_up),
                "down": _hit_error(hit_pred_down, observed_hit_post_down),
            },
        }

    summary["by_post_n"] = summary_by_post_n

    # 保留顶层字段（默认取 all），方便快速查看与保持现有用法不变。
    all_block = summary_by_post_n.get("all")
    if isinstance(all_block, dict):
        summary["curve2_residual"] = all_block.get("curve2_residual")
        summary["hit_pred_vs_obs_est_post"] = all_block.get("hit_pred_vs_obs_est_post")

    return {
        "schema": "temp_curve2_curve3_compare_v2",
        "meta": {
            "track_id": None if track_id is None else int(track_id),
            "track_label": "no-track" if track_id is None else str(int(track_id)),
            "target_y": float(target_y),
            "plot": {"curve3": bool(plot_curve3), "curve2": True},
        },
        "summary": summary,
        "bounce_t_abs_full": bounce_t_abs_full,
        "target_y": float(target_y),
        "observed": {
            "hit_on_plane_pre": observed_hit_pre,
            "hit_on_plane_post_up": observed_hit_post_up,
            "hit_on_plane_post_down": observed_hit_post_down,
        },
        "points": [
            {"x": float(p.x), "y": float(p.y), "z": float(p.z), "t_abs": float(p.t_abs), "n_obs": p.n_obs}
            for p in points
        ],
        "fits_by_post_n": fits_by_post_n,
        "convergence": {"steps": steps},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Offline compare curve2 vs curve3 on existing jsonl")
    p.add_argument(
        "--jsonl",
        default=str(Path("data/tools_output/online points.jsonl")),
        help="输入 jsonl 路径",
    )
    p.add_argument("--track-id", type=int, default=None, help="目标 track_id（不指定则不按 track 过滤）")
    p.add_argument(
        "--point-source",
        choices=["auto", "track_updates", "balls"],
        default="auto",
        help="点来源：auto=优先 track_updates，其次 balls；track_updates=只用 track_updates；balls=只用 balls",
    )
    p.add_argument("--max-records", type=int, default=0, help="最多处理 N 条记录（0=不限制）")
    p.add_argument(
        "--session-gap-s",
        type=float,
        default=2.0,
        help="同一 track 在 jsonl 中相邻点时间差超过该阈值（秒）则视为新一段 session",
    )
    p.add_argument(
        "--session-pick",
        choices=["longest", "last"],
        default="longest",
        help="当同一 track_id 存在多段 session 时，选择哪一段：longest=最长段，last=最后一段",
    )
    p.add_argument("--target-y", type=float, default=0.9, help="击球高度 y（米）")
    p.add_argument(
        "--post-ns",
        default="0,3,5,7,9,11",
        help="要生成/绘制的 post 点数 N 列表（逗号分隔）；若某个 N 的 post 点不足则自动跳过",
    )
    p.add_argument(
        "--out-png",
        default="",
        help="输出 PNG 路径（可为空；若为目录或不以 .png 结尾，则按 track/postN 自动命名）",
    )
    p.add_argument(
        "--png-post-n",
        choices=["0", "3", "5", "7", "9", "11", "all"],
        default="all",
        help="PNG 输出选择哪个 postN 结果（默认 all）",
    )
    p.add_argument(
        "--png-all-post-n",
        action="store_true",
        help="若指定，则对 postNs(由 --post-ns 指定) + all 都各输出一张 PNG（post 点不足的 N 会自动跳过）",
    )
    p.add_argument(
        "--png-axis-mode",
        choices=["robust", "full"],
        default="robust",
        help="PNG 坐标轴范围：robust=分位数裁剪，full=min/max",
    )
    p.add_argument(
        "--png-show-prefit-extrap",
        action="store_true",
        default=False,
        help="PNG 显示 curve3 的 prefit 外推（默认隐藏）",
    )
    p.add_argument(
        "--png-xz-aspect",
        choices=["equal", "auto"],
        default="auto",
        help="PNG 的 x-z 图坐标比例（默认 auto，更易读；equal 会严格等比例但可能很细长）",
    )
    p.add_argument(
        "--png-ground-y",
        type=float,
        default=0.0,
        help="PNG 地面高度 y（米），用于画地面线与掩码 y<ground 的采样段（默认 0.0）",
    )
    p.add_argument(
        "--png-show-below-ground",
        action="store_true",
        default=False,
        help="PNG 显示 y<ground 的采样段（默认隐藏，用于避免曲线掉到地面以下看起来别扭）",
    )
    p.add_argument(
        "--disable-curve3",
        action="store_true",
        default=False,
        help="禁用 curve3（PNG/HTML 不画；并跳过大部分 curve3 计算，仅保留一次全量拟合用于估计 bounce 时刻）",
    )
    p.add_argument(
        "--out-json",
        default="",
        help="输出 JSON 报告路径（默认放到 temp/ 下）",
    )
    p.add_argument(
        "--out-html",
        default="",
        help="输出 HTML 可视化路径（默认放到 temp/ 下）",
    )

    args = p.parse_args(argv)

    jsonl_path = Path(str(args.jsonl)).resolve()
    if not jsonl_path.exists():
        raise SystemExit(f"jsonl not found: {jsonl_path}")

    track_id = None if args.track_id is None else int(args.track_id)
    max_records = int(args.max_records)
    session_gap_s = float(args.session_gap_s)
    session_pick = str(args.session_pick)
    target_y = float(args.target_y)

    # 解析 postNs：逗号分隔的非负整数列表，去重并保持从小到大。
    post_ns: list[int] = []
    for part in str(args.post_ns).split(","):
        s = part.strip()
        if not s:
            continue
        try:
            v = int(s)
        except Exception:
            continue
        if v < 0:
            continue
        post_ns.append(int(v))
    post_ns = sorted(set(post_ns))

    points = _iter_track_points(
        jsonl_path=jsonl_path,
        track_id=track_id,
        max_records=max_records,
        session_gap_s=session_gap_s,
        session_pick=session_pick,
        point_source=str(args.point_source),
    )
    points, filter_info = _filter_user_return_points(points)

    if len(points) < 5:
        raise SystemExit(f"not enough points after filtering: {len(points)}")

    report = _build_report(
        points=points,
        track_id=track_id,
        target_y=target_y,
        post_ns=post_ns,
        plot_curve3=not bool(args.disable_curve3),
    )
    report["meta"]["post_ns"] = [int(x) for x in post_ns]
    report["meta"]["jsonl"] = str(jsonl_path)
    report["meta"]["max_records"] = int(max_records)
    report["meta"]["session"] = {
        "gap_s": float(session_gap_s),
        "pick": str(session_pick),
    }
    report["meta"]["point_source"] = str(args.point_source)
    report["meta"]["filter"] = filter_info
    report["meta"]["disable_curve3"] = bool(args.disable_curve3)

    track_tag = "no-track" if track_id is None else f"track{int(track_id)}"
    out_json = Path(args.out_json) if str(args.out_json).strip() else Path("temp") / f"curve2_curve3_compare.{track_tag}.json"
    out_html = Path(args.out_html) if str(args.out_html).strip() else Path("temp") / f"curve2_curve3_compare.{track_tag}.html"

    out_json = out_json.resolve()
    out_html = out_html.resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    title = f"curve （{track_tag}）"
    out_html.write_text(_build_html(title=title, payload=report), encoding="utf-8")

    # PNG（可选）
    out_png_arg = str(args.out_png).strip()
    if out_png_arg:
        out_png_base = Path(out_png_arg).resolve()
        if bool(args.png_all_post_n):
            keys = [str(int(n)) for n in post_ns] + ["all"]
        else:
            keys = [str(args.png_post_n)]

        for key in keys:
            fit = report.get("fits_by_post_n", {}).get(key)
            if fit is None:
                # 说明：当 post 点不足时，对应 N 不会生成 fit，这里直接跳过不画。
                print(f"skip postN={key}: not enough post points")
                continue

            # 为 matplotlib 绘图构造“subset 点 + excluded 点”
            post_n = None if key == "all" else int(key)
            subset_points, _ = _subset_by_post_n(
                points,
                bounce_t_abs_full=report.get("bounce_t_abs_full"),
                post_n=post_n,
            )

            if out_png_base.suffix.lower() == ".png":
                if len(keys) == 1:
                    out_png = out_png_base
                else:
                    out_png = out_png_base.with_name(f"{out_png_base.stem}.post{key}.png")
            else:
                # 目录或无后缀：自动命名
                out_png = out_png_base / f"curve2_curve3_compare.{track_tag}.post{key}.png"

            _plot_matplotlib_png(
                title=f"{title}  postN={key}",
                report=report,
                all_points=points,
                subset_points=subset_points,
                fit=fit,
                axis_mode=str(args.png_axis_mode),
                show_curve3=not bool(args.disable_curve3),
                hide_prefit_extrap=not bool(args.png_show_prefit_extrap),
                xz_equal_aspect=(str(args.png_xz_aspect).lower() == "equal"),
                ground_y=float(args.png_ground_y),
                show_below_ground=bool(args.png_show_below_ground),
                out_png=out_png,
            )
            print(f"wrote: {out_png}")

    print(f"wrote: {out_json}")
    print(f"wrote: {out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
