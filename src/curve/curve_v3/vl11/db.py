"""从 v-l11.db 提取网球轨迹（abs_loc）。

数据库里有按帧记录的表 `ball_info`，其中 `abs_loc` 是一个 JSON 字符串，形如：
        [x, y, z, t]
其中 t 的单位为秒。

本模块提供：
- 将 abs_loc 解析为 `curve_v3.BallObservation`。
- 将连续点流按时间间隔切分为 N 段 shot。
- 用更稳健的 y-min 启发式（适合离线标注/抽取）把每段 shot 分为 pre/post。

设计上尽量保持轻依赖（仅使用标准库）。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..types import BallObservation
from .filtering import extract_best_inlier_run, group_quality_score
from .return_start import find_return_start_index
from .split import (
    find_bounce_index,
    split_by_gap_threshold,
    split_points_into_shots,
    split_shot_pre_post,
)
from .types import ReturnStartConfig, ShotTrajectory, TrajectoryFilterConfig


def _parse_abs_loc(abs_loc: object) -> tuple[float, float, float, float] | None:
    """解析单个 abs_loc 单元格。

    Args:
        abs_loc: sqlite 行里的值，期望是形如 "[x, y, z, t]" 的 JSON 文本。

    Returns:
        解析成功返回 (x, y, z, t)，否则返回 None。
    """

    if abs_loc is None:
        return None

    if isinstance(abs_loc, (bytes, bytearray)):
        try:
            abs_loc = abs_loc.decode("utf-8")
        except Exception:
            return None

    if not isinstance(abs_loc, str):
        abs_loc = str(abs_loc)

    s = abs_loc.strip()
    if not s:
        return None

    try:
        arr = json.loads(s)
    except Exception:
        return None

    if not isinstance(arr, list) or len(arr) != 4:
        return None

    try:
        x, y, z, t = (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))
    except Exception:
        return None

    return x, y, z, t


def load_abs_loc_points(
    db_path: str | Path,
    *,
    table: str = "ball_info",
    abs_loc_col: str = "abs_loc",
    order_by_col: str = "ts",
) -> list[BallObservation]:
    """从 sqlite DB 中加载 abs_loc 点。

    Args:
        db_path: sqlite DB 路径。
        table: 源表名。
        abs_loc_col: abs_loc JSON 所在列名。
        order_by_col: 用于排序的列（例如 ts）。

    Returns:
        按时间排序的 BallObservation 列表。

    Raises:
        FileNotFoundError: db_path 不存在。
        ValueError: 没有解析出任何有效点。
    """

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(str(db_path))

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        q = (
            f"SELECT {abs_loc_col} FROM {table} "
            f"WHERE {abs_loc_col} IS NOT NULL "
            f"ORDER BY {order_by_col}"
        )
        rows = cur.execute(q).fetchall()
    finally:
        conn.close()

    points: list[BallObservation] = []
    for (abs_loc,) in rows:
        parsed = _parse_abs_loc(abs_loc)
        if parsed is None:
            continue
        x, y, z, t = parsed
        points.append(BallObservation(x=x, y=y, z=z, t=t))

    points.sort(key=lambda p: float(p.t))
    if not points:
        raise ValueError("No valid abs_loc points parsed from DB.")

    return points


def extract_db_shots(
    db_path: str | Path,
    *,
    table: str = "ball_info",
    abs_loc_col: str = "abs_loc",
    order_by_col: str = "ts",
    expected_num_shots: int | None = 4,
    gap_s: float = 1.0,
    min_shot_points: int = 5,
    y_threshold_m: float | None = None,
    min_post_points: int = 5,
    filter_config: TrajectoryFilterConfig | None = None,
    return_start_config: ReturnStartConfig | None = None,
    only_bounce: bool = False,
) -> list[ShotTrajectory]:
    """从 **.db 中抽取 shots，并切分为 pre/post 段。"""

    points = load_abs_loc_points(
        db_path,
        table=str(table),
        abs_loc_col=str(abs_loc_col),
        order_by_col=str(order_by_col),
    )

    cfg = filter_config
    if cfg is not None and bool(cfg.enabled):
        # 先按 gap 做一个“粗分组”，并按质量打分选出 top-K 段。
        coarse_groups = split_by_gap_threshold(points, gap_s=float(gap_s))
        scored: list[tuple[list[BallObservation], float]] = []
        for g in coarse_groups:
            if len(g) < int(min_shot_points):
                continue
            score = float(group_quality_score(g, cfg))
            if score <= 0.0:
                continue
            scored.append((list(g), score))

        expected_n = None
        if expected_num_shots is not None and int(expected_num_shots) > 0:
            expected_n = int(expected_num_shots)

        if cfg.select_top_k is not None:
            k: int | None = int(cfg.select_top_k)
        else:
            k = expected_n

        if scored:
            if k is None:
                top = list(scored)
            else:
                if k <= 0:
                    k = None
                    top = list(scored)
                else:
                    top = sorted(scored, key=lambda it: float(it[1]), reverse=True)[:k]

            top.sort(key=lambda it: float(it[0][0].t) if it[0] else float("inf"))
            shots = [it[0] for it in top]
        else:
            shots = []

        # 只有在明确要求固定段数时，才做回退（避免 expected=None 时无意义回退）。
        if expected_n is not None and len(shots) < expected_n:
            shots = split_points_into_shots(
                points,
                expected_num_shots=expected_n,
                gap_s=float(gap_s),
                min_shot_points=int(min_shot_points),
            )
    else:
        shots = split_points_into_shots(
            points,
            expected_num_shots=expected_num_shots,
            gap_s=float(gap_s),
            min_shot_points=int(min_shot_points),
        )

    results: list[ShotTrajectory] = []
    for i, shot_points in enumerate(shots):
        shot_points = list(shot_points)
        bidx = find_bounce_index(shot_points, y_threshold_m=y_threshold_m)

        # 可选：用“回球开始”启发式裁剪 shot 开端。
        # 目的：把长时间的平台/误识别点段从 shot 里切掉，减少后续过滤误伤。
        if return_start_config is not None and bidx is not None and shot_points:
            start = find_return_start_index(shot_points, return_start_config)
            if start is not None and int(start) > 0 and int(start) < int(bidx):
                shot_points = shot_points[int(start) :]
                bidx = int(bidx) - int(start)

        # all_*：用于“保留整段轨迹”的输出口径。
        # 说明：
        #   - bidx_all 指向 shot_points（可能已做 return_start 裁剪）。
        #   - all_pre/all_post 按 min_post_points 判定 post 是否“存在”。
        bidx_all = bidx
        all_points = list(shot_points)
        all_pre, all_post = split_shot_pre_post(all_points, bidx_all, min_post_points=int(min_post_points))

        # 过滤阶段：先不对 post 做“最少点数”裁剪，避免过滤之前就把 post 直接清空。
        pre, post = split_shot_pre_post(shot_points, bidx, min_post_points=2)

        if cfg is not None and bool(cfg.enabled):
            pre_f = extract_best_inlier_run(pre, cfg) if pre else []
            post_f = extract_best_inlier_run(post, cfg) if post else []
            # pre_f = pre
            # post_f = post

            # 尽量保留 bounce 点（通常位于 pre 的最后一个）。
            if pre:
                bpt = pre[-1]
                if not pre_f or float(pre_f[-1].t) != float(bpt.t):
                    pre_f = list(pre_f) + [bpt]

            # 如果过滤把 post 段清空但原始 post 足够长，则回退以避免误伤。
            if post and len(post_f) < int(min_post_points) and len(post) >= int(min_post_points):
                post_f = list(post)

            # 统一按 min_post_points 判定 post 是否“存在”。
            if len(post_f) < int(min_post_points):
                post_f = []

            shot_points_f = list(pre_f) + list(post_f)
            bidx = (len(pre_f) - 1) if pre_f else None
            pre, post = list(pre_f), list(post_f)
        else:
            pre, post = split_shot_pre_post(shot_points, bidx, min_post_points=int(min_post_points))
            shot_points_f = list(shot_points)

        results.append(
            ShotTrajectory(
                shot_index=int(i),
                points=list(shot_points_f),
                bounce_index=bidx,
                pre_points=pre,
                post_points=post,
                all_points=list(all_points),
                all_bounce_index=bidx_all,
                all_pre_points=list(all_pre),
                all_post_points=list(all_post),
            )
        )

    if bool(only_bounce):
        results = [s for s in results if len(s.post_points) > 0 and s.bounce_index is not None]

    return results
