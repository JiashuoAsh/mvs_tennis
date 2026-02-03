"""定位流水线核心：检测 -> 多视角三角化/定位。

本模块刻意保持“无框架依赖”（不依赖 FastAPI/线程池/消息队列等），只依赖三类输入：
- groups：迭代器，产出 (meta, images_by_camera)
- detector：检测器，提供 detect(img_bgr)->list[Detection]
- calib：标定集 CalibrationSet

输出为可 JSON 序列化的 dict 记录，便于落盘（jsonl）与后处理。
"""

from __future__ import annotations

import math
import time
from typing import Any, Iterable, Iterator

import numpy as np

from tennis3d.detectors import Detector
from tennis3d.geometry.calibration import CalibrationSet
from tennis3d.geometry.triangulation import estimate_triangulation_cov_world
from tennis3d.localization.localize import localize_balls
from tennis3d.pipeline.roi import RoiController
from tennis3d.preprocess import shift_detections
from tennis3d.sync.aligner import PassthroughAligner, SyncAligner


def _safe_float3(x: np.ndarray) -> list[float]:
    """把长度为 3 的 ndarray 转成纯 Python float 列表。

    说明：
        - JSON 序列化时，numpy 标量类型会导致不可序列化或输出不一致。
        - 这里显式转为 float，保证输出稳定。
    """

    return [float(x[0]), float(x[1]), float(x[2])]


def _safe_mat(x: np.ndarray) -> list[list[float]]:
    """把二维 ndarray 转成 JSON 友好的纯 Python 列表。"""

    x = np.asarray(x, dtype=np.float64)
    return [[float(v) for v in row] for row in x]


def _camera_center_world(*, calib: Any) -> np.ndarray:
    """由 world->camera 外参计算相机在 world 坐标系下的光心位置。"""

    R_wc = np.asarray(getattr(calib, "R_wc"), dtype=np.float64).reshape(3, 3)
    t_wc = np.asarray(getattr(calib, "t_wc"), dtype=np.float64).reshape(3)
    R_cw = R_wc.T
    return (-R_cw @ t_wc.reshape(3)).astype(np.float64)


def _estimate_uv_sigma_px(*, bbox: tuple[float, float, float, float] | None, score: float | None) -> float:
    """用极简启发式估计检测中心点的像素标准差（sigma）。

    说明：
        - 这里的目标是“提供可记录的协方差尺度”，便于后续离线分析与拟合加权；
          不追求严格统计最优。
        - 当上游没有 bbox/score 时，回退到常量。
        - 经验上：score 越低、不确定度越大；bbox 越小、中心点越不稳定。
    """

    base_sigma = 2.0
    if bbox is None or score is None:
        return float(base_sigma)

    x1, y1, x2, y2 = bbox
    w = max(1.0, float(x2) - float(x1))
    h = max(1.0, float(y2) - float(y1))
    size = math.sqrt(w * h)

    s = float(max(1e-3, min(1.0, float(score))))

    # size_ref 越大，表示“典型球框大小”越大，sigma 会更小。
    size_ref = 20.0
    sigma = float(base_sigma) * math.sqrt(size_ref / max(1.0, size)) / math.sqrt(s)
    # 夹紧到合理范围，避免极端值污染日志。
    return float(min(12.0, max(0.8, sigma)))


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
    roi_controller: RoiController | None = None,
) -> Iterator[dict[str, Any]]:
    """对输入 groups 运行端到端定位流水线。

    流程：
        1) (Optional) 对齐：通过 aligner 调整 meta/images（例如按时间戳筛掉不完整组）。
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

            # 可选：软件裁剪（动态 ROI）以降低 detector 输入尺寸。
            # 注意：裁剪坐标系下的 bbox 需要加回 offset 才能保持下游几何一致。
            img_for_det = img
            offset_xy = (0, 0)
            if roi_controller is not None:
                try:
                    img_for_det, offset_xy = roi_controller.preprocess_for_detection(
                        meta=meta,
                        camera=str(serial),
                        img_bgr=img,
                        calib=calib,
                    )
                except Exception:
                    img_for_det = img
                    offset_xy = (0, 0)

            dets = detector.detect(img_for_det)
            if dets and (offset_xy[0] != 0 or offset_xy[1] != 0):
                dets = shift_detections(
                    list(dets),
                    dx=int(offset_xy[0]),
                    dy=int(offset_xy[1]),
                    clip_shape=(int(img.shape[0]), int(img.shape[1])),
                )
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

            # 每相机误差字典：便于把 uv_hat / 误差合并到 obs_2d_by_camera。
            err_by_cam = {str(e.camera): e for e in (loc.reprojection_errors or [])}

            # 构造每相机像素协方差（2x2）。目前采用“对角+启发式 sigma”形式。
            cov_uv_by_camera: dict[str, np.ndarray] = {}
            obs_2d_by_camera: dict[str, Any] = {}
            for cam_name, uv in (loc.points_uv or {}).items():
                cam_name = str(cam_name)
                u, v = float(uv[0]), float(uv[1])

                det = (loc.detections or {}).get(cam_name)
                bbox = det.bbox if det is not None else None
                score = float(det.score) if det is not None else None

                det_idx = None
                if (loc.detection_indices or {}).get(cam_name) is not None:
                    det_idx = int((loc.detection_indices or {})[cam_name])

                sigma = _estimate_uv_sigma_px(bbox=bbox, score=score)
                cov = np.array([[sigma * sigma, 0.0], [0.0, sigma * sigma]], dtype=np.float64)
                cov_uv_by_camera[cam_name] = cov

                e = err_by_cam.get(cam_name)
                obs_2d_by_camera[cam_name] = {
                    "uv": [u, v],
                    "cov_uv": _safe_mat(cov),
                    "sigma_px": float(sigma),
                    "cov_source": "heuristic_bbox_score" if det is not None else "default_constant",
                    # 以下为诊断字段：存在则填充，不存在则为 None
                    "uv_hat": [float(e.uv_hat[0]), float(e.uv_hat[1])] if e is not None else None,
                    "reproj_error_px": float(e.error_px) if e is not None else None,
                    "detection_index": det_idx,
                }

            # 三角化 3D 协方差：用于后续轨迹拟合加权/可视化诊断。
            projections_used = {k: calib.require(str(k)).P for k in cov_uv_by_camera.keys() if str(k) in calib.cameras}
            cov_X = estimate_triangulation_cov_world(
                projections=projections_used,
                X_w=loc.X_w,
                cov_uv_by_camera=cov_uv_by_camera,
                min_views=2,
            )

            ball_3d_cov_world = _safe_mat(cov_X) if cov_X is not None else None
            ball_3d_std_m = None
            if cov_X is not None:
                try:
                    std = np.sqrt(np.maximum(np.diag(cov_X).astype(np.float64), 0.0))
                    ball_3d_std_m = [float(std[0]), float(std[1]), float(std[2])]
                except Exception:
                    ball_3d_std_m = None

            # 视角几何统计：最小/中位/最大 ray angle（度）。夹角越大通常三角化越稳。
            ray_angles_deg: list[float] = []
            used_cam_names = list((loc.points_uv or {}).keys())
            X_w = np.asarray(loc.X_w, dtype=np.float64).reshape(3)
            for ia in range(len(used_cam_names)):
                for ib in range(ia + 1, len(used_cam_names)):
                    ca = str(used_cam_names[ia])
                    cb = str(used_cam_names[ib])
                    if ca not in calib.cameras or cb not in calib.cameras:
                        continue
                    Cwa = _camera_center_world(calib=calib.require(ca))
                    Cwb = _camera_center_world(calib=calib.require(cb))
                    va = X_w - Cwa.reshape(3)
                    vb = X_w - Cwb.reshape(3)
                    na = float(np.linalg.norm(va))
                    nb = float(np.linalg.norm(vb))
                    if na <= 1e-9 or nb <= 1e-9:
                        continue
                    cosang = float(np.dot(va, vb) / (na * nb))
                    cosang = float(max(-1.0, min(1.0, cosang)))
                    ang = float(math.degrees(math.acos(cosang)))
                    if math.isfinite(ang):
                        ray_angles_deg.append(float(ang))

            ray_angle_min = float(min(ray_angles_deg)) if ray_angles_deg else None
            ray_angle_med = None
            ray_angle_max = float(max(ray_angles_deg)) if ray_angles_deg else None
            if ray_angles_deg:
                ray_angle_med = float(np.median(np.asarray(ray_angles_deg, dtype=np.float64)))

            b: dict[str, Any] = {
                "ball_id": int(i),
                "ball_3d_world": _safe_float3(loc.X_w),
                "ball_3d_camera": {k: _safe_float3(v) for k, v in loc.X_c_by_camera.items()},
                "used_cameras": list(loc.points_uv.keys()),
                "quality": float(loc.quality),
                "num_views": int(len(loc.points_uv)),
                "median_reproj_error_px": float(med_err),
                "max_reproj_error_px": float(max_err),
                # 新增：每相机 2D 观测（uv）与协方差（px^2），以及与重投影相关的诊断信息。
                "obs_2d_by_camera": obs_2d_by_camera,
                # 新增：3D 点协方差（世界坐标系，m^2）。若几何退化/不可逆则为 None。
                "ball_3d_cov_world": ball_3d_cov_world,
                # 新增：3D 点坐标标准差（m），用于人类可读诊断。
                "ball_3d_std_m": ball_3d_std_m,
                # 新增：三角化几何统计（用于评估视角退化）。
                "triangulation_stats": {
                    "num_pairs": int(len(ray_angles_deg)),
                    "ray_angle_deg_min": ray_angle_min,
                    "ray_angle_deg_median": ray_angle_med,
                    "ray_angle_deg_max": ray_angle_max,
                },
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

        if roi_controller is not None:
            try:
                roi_controller.update_after_output(out_rec=out_rec, calib=calib)
            except Exception:
                pass

        yield out_rec
