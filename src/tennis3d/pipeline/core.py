"""Pipeline core: detection -> triangulation/localization.

This module is intentionally framework-free. It only needs:
- a group iterator yielding (meta, images_by_camera)
- a Detector (detect(img_bgr)->list[Detection])
- a CalibrationSet

It emits JSON-serializable dict records.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Iterator

import numpy as np

from tennis3d.detectors import Detector
from tennis3d.geometry.calibration import CalibrationSet
from tennis3d.localization.localize import localize_ball
from tennis3d.sync.aligner import PassthroughAligner, SyncAligner


def _safe_float3(x: np.ndarray) -> list[float]:
    return [float(x[0]), float(x[1]), float(x[2])]


def run_localization_pipeline(
    *,
    groups: Iterable[tuple[dict[str, Any], dict[str, np.ndarray]]],
    calib: CalibrationSet,
    detector: Detector,
    min_score: float,
    require_views: int,
    include_detection_details: bool = True,
    aligner: SyncAligner | None = None,
) -> Iterator[dict[str, Any]]:
    """Run localization pipeline over groups.

    Args:
        groups: Iterable yielding (meta, images_by_camera). meta must be JSON-serializable.
        calib: Loaded calibration set.
        detector: Detector backend.
        min_score: Minimum detection score threshold.
        require_views: Minimum number of camera views required to triangulate.
        include_detection_details: Whether to include per-camera chosen bbox/score/center.

    Yields:
        JSON-serializable records containing 3D world/camera coords, reprojection errors, etc.
    """

    if aligner is None:
        aligner = PassthroughAligner()

    for meta, images_by_camera in groups:
        aligned = aligner.align(meta or {}, images_by_camera or {})
        if aligned is None:
            continue
        meta, images_by_camera = aligned

        detections_by_camera = {}
        for serial, img in images_by_camera.items():
            if img is None:
                continue
            dets = detector.detect(img)
            if dets:
                detections_by_camera[str(serial)] = list(dets)

        loc = localize_ball(
            calib=calib,
            detections_by_camera=detections_by_camera,
            min_score=float(min_score),
            require_views=int(require_views),
        )
        if loc is None:
            continue

        out_rec: dict[str, Any] = {
            "created_at": time.time(),
            **(meta or {}),
            "used_cameras": list(loc.points_uv.keys()),
            "ball_center_uv": {k: [float(loc.points_uv[k][0]), float(loc.points_uv[k][1])] for k in loc.points_uv},
            "ball_3d_world": _safe_float3(loc.X_w),
            "ball_3d_camera": {k: _safe_float3(v) for k, v in loc.X_c_by_camera.items()},
            "reprojection_errors": [
                {
                    "camera": e.camera,
                    "uv": [float(e.uv[0]), float(e.uv[1])],
                    "uv_hat": [float(e.uv_hat[0]), float(e.uv_hat[1])],
                    "error_px": float(e.error_px),
                }
                for e in loc.reprojection_errors
            ],
        }

        if include_detection_details:
            out_rec["detections"] = {
                k: {
                    "bbox": [float(d.bbox[0]), float(d.bbox[1]), float(d.bbox[2]), float(d.bbox[3])],
                    "score": float(d.score),
                    "cls": int(d.cls),
                    "center": [float(d.center[0]), float(d.center[1])],
                }
                for k, d in loc.detections.items()
            }

        yield out_rec
