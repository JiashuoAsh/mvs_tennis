"""基于多视角检测框中心点进行三角化定位。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from tennis3d.geometry.calibration import CalibrationSet, CameraCalibration
from tennis3d.geometry.triangulation import ReprojectionError, reprojection_errors, triangulate_dlt
from tennis3d.offline.models import Detection


@dataclass(frozen=True)
class BallLocalization:
    """一次定位输出（单个球）。"""

    X_w: np.ndarray
    points_uv: dict[str, tuple[float, float]]
    detections: dict[str, Detection]
    reprojection_errors: list[ReprojectionError]
    X_c_by_camera: dict[str, np.ndarray]


def localize_ball(
    *,
    calib: CalibrationSet,
    detections_by_camera: Mapping[str, Sequence[Detection]],
    min_score: float = 0.25,
    require_views: int = 2,
) -> BallLocalization | None:
    """从多相机检测结果中定位球的 3D 坐标。

    约定：
    - 每个相机可能有多个检测框，这里只取 score 最大的一个（适合单球场景）。

    Returns:
        成功则返回 BallLocalization，否则返回 None。
    """

    points_uv: dict[str, tuple[float, float]] = {}
    used: dict[str, Detection] = {}
    projections: dict[str, np.ndarray] = {}
    calib_used: dict[str, CameraCalibration] = {}

    for cam_name, dets in detections_by_camera.items():
        best = _pick_best_detection(dets)
        if best is None:
            continue
        if float(best.score) < float(min_score):
            continue

        try:
            cam_calib = calib.require(str(cam_name))
        except KeyError:
            continue

        points_uv[str(cam_name)] = (float(best.center[0]), float(best.center[1]))
        used[str(cam_name)] = best
        projections[str(cam_name)] = cam_calib.P
        calib_used[str(cam_name)] = cam_calib

    if len(points_uv) < int(require_views):
        return None

    X_w = triangulate_dlt(projections=projections, points_uv=points_uv)
    errs = reprojection_errors(projections=projections, points_uv=points_uv, X_w=X_w)

    X_c_by_camera: dict[str, np.ndarray] = {}
    for cam_name, cam_calib in calib_used.items():
        X_c = cam_calib.R_wc @ X_w.reshape(3) + cam_calib.t_wc.reshape(3)
        X_c_by_camera[str(cam_name)] = X_c.astype(np.float64)

    return BallLocalization(
        X_w=X_w,
        points_uv=points_uv,
        detections=used,
        reprojection_errors=errs,
        X_c_by_camera=X_c_by_camera,
    )


def _pick_best_detection(dets: Sequence[Detection]) -> Detection | None:
    best: Detection | None = None
    best_score = float("-inf")
    for d in dets:
        s = float(d.score)
        if s > best_score:
            best = d
            best_score = s
    return best
