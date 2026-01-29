"""定位流水线核心：检测 -> 多视角三角化/定位。

本模块刻意保持“无框架依赖”（不依赖 FastAPI/线程池/消息队列等），只依赖三类输入：
- groups：迭代器，产出 (meta, images_by_camera)
- detector：检测器，提供 detect(img_bgr)->list[Detection]
- calib：标定集 CalibrationSet

输出为可 JSON 序列化的 dict 记录，便于落盘（jsonl）与后处理。
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Iterator

import numpy as np

from tennis3d.detectors import Detector
from tennis3d.geometry.calibration import CalibrationSet
from tennis3d.localization.localize import localize_balls
from tennis3d.sync.aligner import PassthroughAligner, SyncAligner


def _safe_float3(x: np.ndarray) -> list[float]:
    """把长度为 3 的 ndarray 转成纯 Python float 列表。

    说明：
        - JSON 序列化时，numpy 标量类型会导致不可序列化或输出不一致。
        - 这里显式转为 float，保证输出稳定。
    """

    return [float(x[0]), float(x[1]), float(x[2])]


def run_localization_pipeline(
    *,
    groups: Iterable[tuple[dict[str, Any], dict[str, np.ndarray]]],
    calib: CalibrationSet,
    detector: Detector,
    min_score: float,
    require_views: int,
    max_detections_per_camera: int,
    max_reproj_error_px: float,
    max_uv_match_dist_px: float,
    merge_dist_m: float,
    include_detection_details: bool = True,
    aligner: SyncAligner | None = None,
) -> Iterator[dict[str, Any]]:
    """对输入 groups 运行端到端定位流水线。

    流程：
        1) （可选）对齐：通过 aligner 调整 meta/images（例如按时间戳筛掉不完整组）。
        2) 检测：对每路图像调用 detector.detect 得到多个候选 Detection。
        3) 多球定位：跨视角匹配 + DLT 三角化 + 重投影误差 gating + 3D 去重 + 冲突消解。

    Args:
        groups: 迭代器，产出 (meta, images_by_camera)。meta 需可 JSON 序列化。
        calib: 已加载的标定集。
        detector: 检测器后端。
        min_score: 最低置信度阈值（低于阈值的检测会被忽略）。
        require_views: 三角化所需的最少视角数。
        max_detections_per_camera: 每个相机最多取 score topK 候选参与匹配。
        max_reproj_error_px: 重投影误差阈值（像素）。
        max_uv_match_dist_px: 投影补全匹配阈值（像素）。
        merge_dist_m: 3D 去重阈值（米）。
        include_detection_details: 是否在输出中包含每路选用的 bbox/score/center。
        aligner: 对齐器；为 None 时使用 PassthroughAligner（不做对齐）。

    Yields:
        可 JSON 序列化的记录：包含 balls 列表（0..N）。
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

        locs = localize_balls(
            calib=calib,
            detections_by_camera=detections_by_camera,
            min_score=float(min_score),
            require_views=int(require_views),
            max_detections_per_camera=int(max_detections_per_camera),
            max_reproj_error_px=float(max_reproj_error_px),
            max_uv_match_dist_px=float(max_uv_match_dist_px),
            merge_dist_m=float(merge_dist_m),
        )

        balls_out: list[dict[str, Any]] = []
        for i, loc in enumerate(locs):
            err_pxs = [float(e.error_px) for e in loc.reprojection_errors]
            med_err = float(np.median(np.asarray(err_pxs, dtype=np.float64))) if err_pxs else float("inf")
            max_err = float(max(err_pxs)) if err_pxs else float("inf")

            b: dict[str, Any] = {
                "ball_id": int(i),
                "ball_3d_world": _safe_float3(loc.X_w),
                "ball_3d_camera": {k: _safe_float3(v) for k, v in loc.X_c_by_camera.items()},
                "used_cameras": list(loc.points_uv.keys()),
                "quality": float(loc.quality),
                "num_views": int(len(loc.points_uv)),
                "median_reproj_error_px": float(med_err),
                "max_reproj_error_px": float(max_err),
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
                b["detections"] = {
                    k: {
                        "bbox": [float(d.bbox[0]), float(d.bbox[1]), float(d.bbox[2]), float(d.bbox[3])],
                        "score": float(d.score),
                        "cls": int(d.cls),
                        "center": [float(d.center[0]), float(d.center[1])],
                    }
                    for k, d in loc.detections.items()
                }

            balls_out.append(b)

        out_rec: dict[str, Any] = {
            "created_at": time.time(),
            **(meta or {}),
            "balls": balls_out,
        }

        yield out_rec
