"""v-l11.db 点流的 shot 切分与 bounce 分段逻辑。

说明：
    - DB 存的是连续的 abs_loc 点流，需要先切分为多个 shot。
    - 对每个 shot，再用启发式检测 bounce 索引，并切成 pre/post。

该模块只包含纯函数，便于复用与单测。
"""

from __future__ import annotations

from typing import Sequence

from curve_v3.types import BallObservation


def split_by_gap_threshold(points: Sequence[BallObservation], *, gap_s: float) -> list[list[BallObservation]]:
    if not points:
        return []

    gap_s = float(gap_s)
    groups: list[list[BallObservation]] = [[points[0]]]
    for p in points[1:]:
        prev = groups[-1][-1]
        dt = float(p.t - prev.t)
        if dt > gap_s:
            groups.append([p])
        else:
            groups[-1].append(p)
    return groups


def _split_by_largest_gaps(
    points: Sequence[BallObservation], *, num_groups: int, min_group_points: int
) -> list[list[BallObservation]] | None:
    if num_groups <= 1:
        return [list(points)]
    if len(points) < num_groups * max(min_group_points, 1):
        return None

    dts = [float(points[i + 1].t - points[i].t) for i in range(len(points) - 1)]
    gap_indices = sorted(range(len(dts)), key=lambda i: dts[i], reverse=True)

    cut_indices: list[int] = []
    for idx in gap_indices:
        if len(cut_indices) >= num_groups - 1:
            break
        cut_indices.append(idx + 1)

    cut_indices.sort()

    groups: list[list[BallObservation]] = []
    start = 0
    for cut in cut_indices + [len(points)]:
        groups.append(list(points[start:cut]))
        start = cut

    if len(groups) != num_groups:
        return None
    if any(len(g) < min_group_points for g in groups):
        return None
    return groups


def split_points_into_shots(
    points: Sequence[BallObservation], *, expected_num_shots: int | None = 4, gap_s: float = 1.0, min_shot_points: int = 15
) -> list[list[BallObservation]]:
    """把一条长点流切分为若干段 shot。"""

    pts = list(points)
    pts.sort(key=lambda p: float(p.t))

    if expected_num_shots is None or int(expected_num_shots) <= 0:
        groups = split_by_gap_threshold(pts, gap_s=float(gap_s))
        return [g for g in groups if len(g) >= int(min_shot_points)]

    groups = _split_by_largest_gaps(pts, num_groups=int(expected_num_shots), min_group_points=int(min_shot_points))
    if groups is not None:
        return groups

    groups = split_by_gap_threshold(pts, gap_s=float(gap_s))
    return [g for g in groups if len(g) >= int(min_shot_points)]


def find_bounce_index(
    points: Sequence[BallObservation],
    *,
    y_threshold_m: float | None = None,
    slope_window: int = 2,
    min_pre_points: int = 8,
    min_post_points: int = 6,
    min_rise_m: float = 0.05,
    low_y_quantile: float = 0.10,
    max_above_low_y_m: float = 0.20,
) -> int | None:
    """用 y 的极小值 + 斜率翻转启发式检测 bounce 索引。"""

    if len(points) < 5:
        return None

    ys = [float(p.y) for p in points]

    minima: list[int] = []
    for i in range(1, len(ys) - 1):
        if ys[i] <= ys[i - 1] and ys[i] <= ys[i + 1]:
            minima.append(i)

    w = max(int(slope_window), 1)
    min_pre = max(int(min_pre_points), w)
    min_post = max(int(min_post_points), w)
    rise_req = float(min_rise_m)

    q = float(low_y_quantile)
    q = 0.10 if not (q == q) else min(max(q, 0.0), 0.5)
    ys_sorted = sorted(ys)
    q_idx = int(q * max(len(ys_sorted) - 1, 0))
    y_low = float(ys_sorted[q_idx])
    y_low_max = float(y_low + float(max_above_low_y_m))

    def has_slope_flip(i: int) -> bool:
        left = max(i - w, 0)
        right = min(i + w, len(points) - 1)
        dy_left = float(ys[i] - ys[left])
        dy_right = float(ys[right] - ys[i])
        return dy_left < -1e-6 and dy_right > 1e-6

    def has_rise(i: int) -> bool:
        j = min(i + 12, len(ys))
        return float(max(ys[i:j]) - ys[i]) >= rise_req

    candidates = [
        i
        for i in minima
        if i >= min_pre
        and (len(ys) - i - 1) >= min_post
        and float(ys[i]) <= y_low_max
        and has_slope_flip(i)
        and has_rise(i)
    ]
    if candidates:
        return int(min(candidates, key=lambda i: ys[i]))

    idx = int(min(range(len(ys)), key=lambda i: ys[i]))
    y_min = float(ys[idx])
    if y_threshold_m is not None and y_min > float(y_threshold_m):
        return None

    if idx >= min_pre:
        return idx

    return None


def split_shot_pre_post(
    points: Sequence[BallObservation], bounce_index: int | None, *, min_post_points: int = 2
) -> tuple[list[BallObservation], list[BallObservation]]:
    if bounce_index is None:
        return list(points), []

    bounce_index = int(bounce_index)
    if bounce_index < 0 or bounce_index >= len(points):
        return list(points), []

    pre = list(points[: bounce_index + 1])
    post = list(points[bounce_index + 1 :])
    if len(post) < int(min_post_points):
        post = []
    return pre, post
