"""基于多视角检测框中心点进行三角化定位。

本模块的核心职责是：
- 输入：每个相机的多个候选 Detection
- 输出：0..N 个“几何一致”的 3D 球候选

多球场景下，为了抑制误检与组合爆炸，这里采用：
- 先用“两相机检测中心”生成种子候选（DLT 三角化）
- 再把 3D 点投影到其它相机，用最近邻在阈值内进行补全匹配
- 对候选做重投影误差/正深度 gating
- 对 3D 解做去重（3D-NMS）
- 最后做冲突消解，避免同一相机同一检测被多个 3D 球重复使用
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np

from tennis3d.geometry.calibration import CalibrationSet, CameraCalibration
from tennis3d.geometry.triangulation import (
    ReprojectionError,
    project_point,
    reprojection_errors,
    triangulate_dlt,
)
from tennis3d.models import Detection
from tennis3d.localization._postprocess import (
    deduplicate_by_3d_distance as _deduplicate_by_3d_distance,
    select_non_conflicting as _select_non_conflicting,
)
from tennis3d.localization._quality import compute_quality as _compute_quality


@dataclass(frozen=True)
class BallLocalization:
    """一次定位输出（单个 3D 球）。

    说明：
        - 本数据结构用于 pipeline 输出前的“中间结果”。
        - 内部使用 numpy 便于几何计算；序列化由 pipeline 统一转为纯 Python 类型。
    """

    X_w: np.ndarray
    points_uv: dict[str, tuple[float, float]]
    detections: dict[str, Detection]
    detection_indices: dict[str, int]
    reprojection_errors: list[ReprojectionError]
    X_c_by_camera: dict[str, np.ndarray]
    quality: float


@dataclass(frozen=True)
class _CameraCtx:
    name: str
    calib: CameraCalibration
    dets: list[Detection]
    centers: np.ndarray  # (N,2)

    @property
    def P(self) -> np.ndarray:
        return self.calib.P


def localize_balls(
    *,
    calib: CalibrationSet,
    detections_by_camera: Mapping[str, Sequence[Detection]],
    min_score: float = 0.25,
    require_views: int = 2,
    max_detections_per_camera: int = 10,
    max_reproj_error_px: float = 8.0,
    max_uv_match_dist_px: float = 25.0,
    merge_dist_m: float = 0.08,
) -> list[BallLocalization]:
    """从多相机检测结果中定位 0..N 个球的 3D 坐标。

    设计要点：
        - 先用两视角组合生成 3D 种子，避免全组合爆炸。
        - 对其它相机做投影补全匹配，只有跨视角几何一致才输出。
        - 通过 3D 去重与检测冲突消解，避免重复球与重复使用同一检测。

    Args:
        calib: 标定集。
        detections_by_camera: 每个相机的候选检测列表。
        min_score: 最低置信度阈值。
        require_views: 输出候选所需的最少视角数。
        max_detections_per_camera: 每个相机最多取 score topK 候选参与匹配。
        max_reproj_error_px: 重投影误差阈值（像素）；候选中任一使用视角超阈值会被过滤。
        max_uv_match_dist_px: 投影补全匹配时，投影点到 bbox center 的最大距离阈值（像素）。
        merge_dist_m: 3D 去重阈值（米）；距离小于该值认为是同一球的重复解。

    Returns:
        0..N 个 BallLocalization，按质量从高到低排序。
    """

    require_views = int(require_views)
    if require_views < 2:
        raise ValueError("require_views 必须 >= 2")

    max_detections_per_camera = int(max_detections_per_camera)
    if max_detections_per_camera <= 0:
        return []

    cams = _build_camera_contexts(
        calib=calib,
        detections_by_camera=detections_by_camera,
        min_score=float(min_score),
        max_detections_per_camera=max_detections_per_camera,
    )
    if len(cams) < 2:
        return []

    candidates: list[BallLocalization] = []
    cam_names = list(cams.keys())
    for ca, cb in combinations(cam_names, 2):
        a = cams[ca]
        b = cams[cb]
        if not a.dets or not b.dets:
            continue
        for ia in range(len(a.dets)):
            for ib in range(len(b.dets)):
                cand = _candidate_from_pair(
                    cams=cams,
                    cam_a=ca,
                    det_a_idx=ia,
                    cam_b=cb,
                    det_b_idx=ib,
                    require_views=require_views,
                    max_reproj_error_px=float(max_reproj_error_px),
                    max_uv_match_dist_px=float(max_uv_match_dist_px),
                )
                if cand is not None:
                    candidates.append(cand)

    if not candidates:
        return []

    candidates = _deduplicate_by_3d_distance(candidates, merge_dist_m=float(merge_dist_m))
    candidates = _select_non_conflicting(candidates)
    candidates.sort(key=lambda c: float(c.quality), reverse=True)
    return candidates


def _build_camera_contexts(
    *,
    calib: CalibrationSet,
    detections_by_camera: Mapping[str, Sequence[Detection]],
    min_score: float,
    max_detections_per_camera: int,
) -> dict[str, _CameraCtx]:
    out: dict[str, _CameraCtx] = {}
    for cam_name, dets in detections_by_camera.items():
        try:
            cam_calib = calib.require(str(cam_name))
        except KeyError:
            continue

        kept = [d for d in dets if float(d.score) >= float(min_score)]
        kept.sort(key=lambda d: float(d.score), reverse=True)
        kept = kept[: int(max_detections_per_camera)]
        if not kept:
            continue

        centers = np.array([[float(d.center[0]), float(d.center[1])] for d in kept], dtype=np.float64)
        out[str(cam_name)] = _CameraCtx(
            name=str(cam_name),
            calib=cam_calib,
            dets=list(kept),
            centers=centers,
        )
    return out


def _nearest_detection_index(
    *,
    ctx: _CameraCtx,
    uv: tuple[float, float],
    max_dist_px: float,
) -> int | None:
    if ctx.centers.size == 0:
        return None

    du = ctx.centers[:, 0] - float(uv[0])
    dv = ctx.centers[:, 1] - float(uv[1])
    d2 = du * du + dv * dv
    j = int(np.argmin(d2))
    if float(d2[j]) <= float(max_dist_px) * float(max_dist_px):
        return j
    return None


def _candidate_from_pair(
    *,
    cams: dict[str, _CameraCtx],
    cam_a: str,
    det_a_idx: int,
    cam_b: str,
    det_b_idx: int,
    require_views: int,
    max_reproj_error_px: float,
    max_uv_match_dist_px: float,
) -> BallLocalization | None:
    a = cams[cam_a]
    b = cams[cam_b]

    det_a = a.dets[int(det_a_idx)]
    det_b = b.dets[int(det_b_idx)]

    points_uv: dict[str, tuple[float, float]] = {
        str(cam_a): (float(det_a.center[0]), float(det_a.center[1])),
        str(cam_b): (float(det_b.center[0]), float(det_b.center[1])),
    }
    det_indices: dict[str, int] = {str(cam_a): int(det_a_idx), str(cam_b): int(det_b_idx)}
    dets_used: dict[str, Detection] = {str(cam_a): det_a, str(cam_b): det_b}

    try:
        X_w = triangulate_dlt(
            projections={str(cam_a): a.P, str(cam_b): b.P},
            points_uv=points_uv,
        )
    except Exception:
        return None

    if not _all_positive_depth(X_w=X_w, calib_by_camera={str(cam_a): a.calib, str(cam_b): b.calib}):
        return None

    # 投影补全：把三角化的 3D 点投影到其它相机，选最近且在阈值内的检测框中心。
    for cam_name, ctx in cams.items():
        if cam_name in points_uv:
            continue
        uv_hat = project_point(ctx.P, X_w)
        if not (np.isfinite(uv_hat[0]) and np.isfinite(uv_hat[1])):
            continue
        j = _nearest_detection_index(ctx=ctx, uv=uv_hat, max_dist_px=float(max_uv_match_dist_px))
        if j is None:
            continue
        det = ctx.dets[int(j)]
        points_uv[str(cam_name)] = (float(det.center[0]), float(det.center[1]))
        det_indices[str(cam_name)] = int(j)
        dets_used[str(cam_name)] = det

    if len(points_uv) < int(require_views):
        return None

    projections = {k: cams[k].P for k in points_uv.keys() if k in cams}
    calib_used = {k: cams[k].calib for k in points_uv.keys() if k in cams}
    try:
        X_w = triangulate_dlt(projections=projections, points_uv=points_uv)
    except Exception:
        return None

    errs = reprojection_errors(projections=projections, points_uv=points_uv, X_w=X_w)
    if not errs:
        return None

    max_err = float(max(e.error_px for e in errs))
    if max_err > float(max_reproj_error_px):
        return None

    if not _all_positive_depth(X_w=X_w, calib_by_camera=calib_used):
        return None

    X_c_by_camera: dict[str, np.ndarray] = {}
    for k, cam_calib in calib_used.items():
        X_c = cam_calib.R_wc @ X_w.reshape(3) + cam_calib.t_wc.reshape(3)
        X_c_by_camera[str(k)] = X_c.astype(np.float64)

    quality = _compute_quality(
        dets=dets_used,
        errs=errs,
    )

    return BallLocalization(
        X_w=X_w,
        points_uv=points_uv,
        detections=dets_used,
        detection_indices=det_indices,
        reprojection_errors=errs,
        X_c_by_camera=X_c_by_camera,
        quality=float(quality),
    )


def _all_positive_depth(*, X_w: np.ndarray, calib_by_camera: Mapping[str, CameraCalibration]) -> bool:
    """检查所有使用视角的深度为正。

    说明：
        - 标定约定为 world->camera：X_c = R_wc X_w + t_wc。
        - 深度用相机坐标系 Z 分量近似表示。
    """

    X_w = np.asarray(X_w, dtype=np.float64).reshape(3)
    for cam_calib in calib_by_camera.values():
        X_c = cam_calib.R_wc @ X_w + cam_calib.t_wc.reshape(3)
        if not np.isfinite(X_c[2]):
            return False
        if float(X_c[2]) <= 1e-6:
            return False
    return True
