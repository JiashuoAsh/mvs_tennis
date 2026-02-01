"""v-l11.db 离线抽取的噪点过滤与质量评分逻辑。

说明：
    这里的逻辑主要用于：当 DB 内存在明显非物理的异常点/乱序点时，
    尽可能从粗分组里提取出一段“更像球在飞”的连续子序列。

    该模块只包含纯函数，便于单测与复用。
"""

from __future__ import annotations

import math
from typing import Sequence

from ..types import BallObservation
from .types import TrajectoryFilterConfig


def _is_finite(x: float) -> bool:
    try:
        return bool(math.isfinite(float(x)))
    except Exception:
        return False


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median() of empty sequence")
    xs = sorted(float(v) for v in values)
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return float(xs[mid])
    return 0.5 * (float(xs[mid - 1]) + float(xs[mid]))


def _mad(values: Sequence[float], *, center: float) -> float:
    if not values:
        return 0.0
    dev = [abs(float(v) - float(center)) for v in values]
    return float(_median(dev))


def _triplet_ay(p0: BallObservation, p1: BallObservation, p2: BallObservation, cfg: TrajectoryFilterConfig) -> float | None:
    dt1 = float(p1.t - p0.t)
    dt2 = float(p2.t - p1.t)
    if not (_is_finite(dt1) and _is_finite(dt2)):
        return None
    if dt1 <= 0.0 or dt2 <= 0.0:
        return None
    if dt1 < float(cfg.min_dt_s) or dt1 > float(cfg.max_dt_s):
        return None
    if dt2 < float(cfg.min_dt_s) or dt2 > float(cfg.max_dt_s):
        return None

    v1 = float((p1.y - p0.y) / dt1)
    v2 = float((p2.y - p1.y) / dt2)
    dtm = 0.5 * (dt1 + dt2)
    if dtm <= 0.0:
        return None

    a = float((v2 - v1) / dtm)
    if not _is_finite(a):
        return None
    return float(a)


def _edge_velocity(a: BallObservation, b: BallObservation) -> tuple[float, float, float, float] | None:
    dt = float(b.t - a.t)
    if not _is_finite(dt) or dt <= 0.0:
        return None
    vx = float((b.x - a.x) / dt)
    vy = float((b.y - a.y) / dt)
    vz = float((b.z - a.z) / dt)
    if not (_is_finite(vx) and _is_finite(vy) and _is_finite(vz)):
        return None
    return dt, vx, vy, vz


def edge_is_plausible(a: BallObservation, b: BallObservation, cfg: TrajectoryFilterConfig) -> bool:
    """判断相邻两点是否满足基本的物理门禁。"""

    v = _edge_velocity(a, b)
    if v is None:
        return False
    dt, vx, vy, vz = v
    if dt < float(cfg.min_dt_s) or dt > float(cfg.max_dt_s):
        return False

    speed = math.sqrt(float(vx * vx + vy * vy + vz * vz))
    if speed > float(cfg.max_speed_m_s):
        return False

    zmin, zmax = float(cfg.z_speed_range[0]), float(cfg.z_speed_range[1])
    if abs(float(vz)) > zmax + 1e-6:
        return False
    if abs(float(vz)) < max(zmin - 1e-6, 0.0):
        # 允许极少量 vz 很小的点，但通常这类点更可能是噪点。
        return False

    return True


def extract_best_inlier_run(points: Sequence[BallObservation], cfg: TrajectoryFilterConfig) -> list[BallObservation]:
    """从一个粗分组中提取“物理一致”的最长子序列（两阶段）。

    - 第 1 阶段：仅用 dt/速度门禁得到一个粗 inlier 段。
    - 第 2 阶段：用第 1 阶段的 a_y(局部二阶差分) 中位数/MAD 做自适应门禁，
      再提取一次，以便过滤掉“速度看似合理但不满足抛体形状”的杂点。
    """

    pts = list(points)
    if not pts:
        return []
    pts.sort(key=lambda p: float(p.t))

    def extract_once(*, acc_ref: tuple[float, float] | None) -> list[BallObservation]:
        best: list[BallObservation] = []
        cur: list[BallObservation] = []
        skipped = 0

        for p in pts:
            if not cur:
                cur = [p]
                skipped = 0
                continue

            if edge_is_plausible(cur[-1], p, cfg):
                if acc_ref is not None and len(cur) >= 2:
                    a = _triplet_ay(cur[-2], cur[-1], p, cfg)
                    if a is not None:
                        a_center, a_mad = acc_ref
                        thr = float(cfg.gravity_mad_k) * max(float(a_mad), 1e-6)
                        if float(a) > -float(cfg.min_downward_acc_m_s2):
                            skipped += 1
                            if skipped > int(cfg.max_skip_points):
                                if len(cur) > len(best):
                                    best = cur
                                cur = [p]
                                skipped = 0
                            continue

                        # 只有当参考段本身体现出足够“向下加速度”时，才用 (median, MAD)
                        # 去约束 a_y 的离群程度。
                        if float(a_center) <= -float(cfg.min_downward_acc_m_s2):
                            if abs(float(a) - float(a_center)) > thr:
                                skipped += 1
                                if skipped > int(cfg.max_skip_points):
                                    if len(cur) > len(best):
                                        best = cur
                                    cur = [p]
                                    skipped = 0
                                continue

                cur.append(p)
                skipped = 0
                continue

            skipped += 1
            if skipped > int(cfg.max_skip_points):
                if len(cur) > len(best):
                    best = cur
                cur = [p]
                skipped = 0

        if len(cur) > len(best):
            best = cur

        return list(best)

    run0 = extract_once(acc_ref=None)
    acc_ref = gravity_stats(run0, cfg)
    if acc_ref is None:
        return run0

    run1 = extract_once(acc_ref=acc_ref)

    def gravity_ok(run: Sequence[BallObservation]) -> bool:
        st = gravity_stats(run, cfg)
        if st is None:
            return False
        a_med, a_mad = float(st[0]), float(st[1])
        if a_med > -float(cfg.min_downward_acc_m_s2):
            return False
        mad_ratio = float(a_mad) / max(abs(float(a_med)), 1e-6)
        return mad_ratio <= float(cfg.max_gravity_mad_ratio)

    ok0 = bool(gravity_ok(run0))
    ok1 = bool(gravity_ok(run1))
    if ok1 and not ok0:
        return run1
    if ok0 and not ok1:
        return run0

    s0 = float(run_quality_score(run0, cfg))
    s1 = float(run_quality_score(run1, cfg))
    if s1 > s0:
        return run1
    if s1 == s0 and len(run1) >= len(run0):
        return run1
    return run0


def forward_ratio(points: Sequence[BallObservation], *, expected_vz_sign: int = -1) -> float:
    """计算 z 方向的“运动一致性”比例。

    说明：
        在离线 DB 抽取里，我们需要一个很轻量的“段级一致性”指标，来判断
        一段点序列是否像“球在持续向前/向后运动”，以过滤掉杂点段。

        expected_vz_sign 的约定：
            - +1：期望 vz > 0（z 递增）
            - -1：期望 vz < 0（z 递减）
    """

    if len(points) < 2:
        return 0.0

    total = 0
    pos = 0
    neg = 0
    ok = 0

    for i in range(len(points) - 1):
        v = _edge_velocity(points[i], points[i + 1])
        if v is None:
            continue
        _, _, _, vz = v
        total += 1

        if float(vz) > 0.0:
            pos += 1
        elif float(vz) < 0.0:
            neg += 1

        if expected_vz_sign in (-1, 1):
            if float(vz) * float(expected_vz_sign) > 0.0:
                ok += 1

    if total <= 0:
        return 0.0

    if expected_vz_sign in (-1, 1):
        return float(ok) / float(total)

    return float(max(pos, neg)) / float(total)


def gravity_stats(points: Sequence[BallObservation], cfg: TrajectoryFilterConfig) -> tuple[float, float] | None:
    """估计 y 轴局部二阶差分(a_y)的 (median, MAD)。"""

    if len(points) < 5:
        return None

    acc: list[float] = []
    for i in range(1, len(points) - 1):
        a = _triplet_ay(points[i - 1], points[i], points[i + 1], cfg)
        if a is None:
            continue
        acc.append(float(a))

    if len(acc) < 5:
        return None

    a_med = float(_median(acc))
    mad = float(_mad(acc, center=a_med))
    return float(a_med), float(mad)


def run_quality_score(points: Sequence[BallObservation], cfg: TrajectoryFilterConfig) -> float:
    if len(points) < int(cfg.min_run_points):
        return -1.0

    fwd = float(forward_ratio(points))
    acc_stats = gravity_stats(points, cfg)

    forward_score = min(max(fwd, 0.0), 1.0)

    if acc_stats is None:
        gravity_score = 0.5
    else:
        a_med, a_mad = float(acc_stats[0]), float(acc_stats[1])
        if float(a_med) > -float(cfg.min_downward_acc_m_s2):
            gravity_score = 0.0
        else:
            mad_ratio = float(a_mad) / max(abs(float(a_med)), 1e-6)
            gravity_score = 1.0 if mad_ratio <= float(cfg.max_gravity_mad_ratio) else 0.0

    length_score = min(float(len(points)) / 60.0, 1.0)
    score = float(len(points))
    score *= 0.35 + 0.1 * length_score
    score *= 0.30 + 0.2 * forward_score
    score *= 0.30 + 0.7 * gravity_score
    return float(score)


def edge_plausible_ratio(points: Sequence[BallObservation], cfg: TrajectoryFilterConfig) -> float:
    """相邻边通过点级门禁的比例。"""

    if len(points) < 2:
        return 0.0
    total = 0
    ok = 0
    for i in range(len(points) - 1):
        total += 1
        if edge_is_plausible(points[i], points[i + 1], cfg):
            ok += 1
    if total <= 0:
        return 0.0
    return float(ok) / float(total)


def group_quality_score(points: Sequence[BallObservation], cfg: TrajectoryFilterConfig) -> float:
    """给一个粗分组打分，用于从很多段里挑出更像“球在飞”的那些段。"""

    base = float(run_quality_score(points, cfg))
    edge_ratio = float(edge_plausible_ratio(points, cfg))
    return base * (0.25 + 0.75 * edge_ratio)
