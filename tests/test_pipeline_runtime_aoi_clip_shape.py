from __future__ import annotations


import numpy as np

from tennis3d.geometry.calibration import CalibrationSet, CameraCalibration
from tennis3d.models import Detection
from tennis3d.pipeline.core import run_localization_pipeline


class _DummyDetector:
    def detect(self, img_bgr: np.ndarray):
        # 返回一个 bbox（内容不重要；我们只关心 shift_detections 的 clip_shape）。
        return [Detection(bbox=(10.0, 10.0, 20.0, 20.0), score=0.9, cls=0)]


class _DummyRoiController:
    def preprocess_for_detection(self, *, meta, camera: str, img_bgr: np.ndarray, calib: CalibrationSet):
        # 模拟 runtime AOI：相机输出是较小 AOI，但 offset 需要回写到满幅标定坐标系。
        return img_bgr, (200, 800)

    def update_after_output(self, *, out_rec, calib: CalibrationSet) -> None:
        return


def test_runtime_aoi_shift_clips_to_calib_image_size(monkeypatch):
    # 标定为满幅尺寸（W,H）。
    calib = CalibrationSet(
        cameras={
            "CAM0": CameraCalibration(
                name="CAM0",
                image_size=(2448, 2048),
                K=np.eye(3, dtype=np.float64),
                dist=np.zeros((0,), dtype=np.float64),
                R_wc=np.eye(3, dtype=np.float64),
                t_wc=np.zeros((3,), dtype=np.float64),
            )
        }
    )

    # AOI 图像尺寸显著小于满幅。
    img = np.zeros((1080, 1980, 3), dtype=np.uint8)

    seen = {}

    import tennis3d.pipeline.core as core

    def _spy_shift_detections(dets, *, dx: int, dy: int, clip_shape=None):
        seen["dx"] = dx
        seen["dy"] = dy
        seen["clip_shape"] = clip_shape
        return dets

    monkeypatch.setattr(core, "shift_detections", _spy_shift_detections)

    groups = [({"group_index": 0}, {"CAM0": img})]

    # require_views>=2 才合法；本测试只有一台相机，localize_balls 会直接返回空，不影响我们验证。
    out = list(
        run_localization_pipeline(
            groups=groups,
            calib=calib,
            detector=_DummyDetector(),
            min_score=0.01,
            require_views=2,
            max_detections_per_camera=10,
            max_reproj_error_px=10.0,
            max_uv_match_dist_px=10.0,
            merge_dist_m=0.08,
            roi_controller=_DummyRoiController(),
        )
    )
    assert len(out) == 1

    assert seen["dx"] == 200
    assert seen["dy"] == 800
    # clip_shape 的语义是 (H,W)
    assert seen["clip_shape"] == (2048, 2448)
