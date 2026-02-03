"""验证：软件裁剪(detector crop)会把 bbox 坐标回写到原图坐标系。

该测试的关注点不是三角化数值精度，而是坐标系一致性：
- detector 在裁剪窗口坐标系下返回 bbox
- pipeline/core 需要把 bbox 加回 (offset_x, offset_y)
- 下游 localize_balls 应看到“原图像素坐标系”的 uv
"""

from __future__ import annotations

import unittest

import numpy as np

from tennis3d.geometry.calibration import CalibrationSet, CameraCalibration
from tennis3d.geometry.triangulation import project_point
from tennis3d.models import Detection
from tennis3d.pipeline.core import run_localization_pipeline
from tennis3d.pipeline.roi import RoiController
from tennis3d.preprocess import crop_bgr


class _ConstantDetector:
    """在输入图像坐标系下返回一个固定中心的 bbox。"""

    def __init__(self, *, center_uv: tuple[float, float] = (10.0, 10.0)) -> None:
        self._cx = float(center_uv[0])
        self._cy = float(center_uv[1])

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        half = 2.0
        cx, cy = float(self._cx), float(self._cy)
        return [Detection(bbox=(cx - half, cy - half, cx + half, cy + half), score=0.9, cls=0)]


class _FixedCropController(RoiController):
    """用固定的 offset 做软件裁剪，便于验证坐标回写。"""

    def __init__(
        self,
        *,
        crop_size: int,
        origin_by_camera: dict[str, tuple[int, int]],
    ) -> None:
        self._crop_size = int(crop_size)
        self._origin_by_camera = {str(k): (int(v[0]), int(v[1])) for k, v in origin_by_camera.items()}

    def preprocess_for_detection(
        self,
        *,
        meta: dict[str, object],
        camera: str,
        img_bgr: np.ndarray,
        calib: CalibrationSet,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        ox, oy = self._origin_by_camera.get(str(camera), (0, 0))
        return crop_bgr(
            img_bgr,
            crop_width=int(self._crop_size),
            crop_height=int(self._crop_size),
            offset_x=int(ox),
            offset_y=int(oy),
        )

    def update_after_output(self, *, out_rec: dict[str, object], calib: CalibrationSet) -> None:
        return


class TestPipelineDetectorCrop(unittest.TestCase):
    def test_crop_offsets_are_added_back(self) -> None:
        # 构造两相机标定（与其他测试一致）：cam1 相对 cam0 沿 +x 平移 1m。
        fx = 800.0
        fy = 800.0
        cx = 100.0
        cy = 100.0
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        R_wc = np.eye(3, dtype=np.float64)

        cam0 = CameraCalibration(
            name="cam0",
            image_size=(200, 200),
            K=K,
            dist=np.zeros(0, dtype=np.float64),
            R_wc=R_wc,
            t_wc=np.zeros(3, dtype=np.float64),
        )
        cam1 = CameraCalibration(
            name="cam1",
            image_size=(200, 200),
            K=K,
            dist=np.zeros(0, dtype=np.float64),
            R_wc=R_wc,
            # world->camera: X_c = X_w - C，因此 C=(1,0,0) => t=-C
            t_wc=np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        )
        calib = CalibrationSet(cameras={"cam0": cam0, "cam1": cam1})

        # 选择一个前方世界点，计算其投影 uv，用于构造“期望的回写后坐标”。
        X_gt = np.array([0.2, -0.1, 4.0], dtype=np.float64)
        uv0 = project_point(cam0.P, X_gt)
        uv1 = project_point(cam1.P, X_gt)

        # detector 在裁剪窗口内固定输出中心 (10,10)，所以我们选择 origin 让回写后中心接近投影点。
        crop_size = 50
        origin0 = (int(round(float(uv0[0]) - 10.0)), int(round(float(uv0[1]) - 10.0)))
        origin1 = (int(round(float(uv1[0]) - 10.0)), int(round(float(uv1[1]) - 10.0)))

        # 注意：软件裁剪会把 origin clamp 到图像范围内。
        def _clamp_origin(ox: int, oy: int) -> tuple[int, int]:
            cw = int(min(int(crop_size), 200))
            ch = int(min(int(crop_size), 200))
            ox = int(max(0, min(int(ox), 200 - cw)))
            oy = int(max(0, min(int(oy), 200 - ch)))
            return ox, oy

        origin0_c = _clamp_origin(origin0[0], origin0[1])
        origin1_c = _clamp_origin(origin1[0], origin1[1])

        roi = _FixedCropController(crop_size=crop_size, origin_by_camera={"cam0": origin0, "cam1": origin1})
        detector = _ConstantDetector(center_uv=(10.0, 10.0))

        img0 = np.zeros((200, 200, 3), dtype=np.uint8)
        img1 = np.zeros((200, 200, 3), dtype=np.uint8)

        groups = [({"frame_id": 0}, {"cam0": img0, "cam1": img1})]

        it = run_localization_pipeline(
            groups=groups,
            calib=calib,
            detector=detector,
            min_score=0.1,
            require_views=2,
            max_detections_per_camera=5,
            max_reproj_error_px=20.0,
            max_uv_match_dist_px=50.0,
            merge_dist_m=0.01,
            include_detection_details=True,
            aligner=None,
            roi_controller=roi,
        )
        rec = next(iter(it))

        balls = rec.get("balls")
        assert isinstance(balls, list)
        self.assertEqual(len(balls), 1)

        b0 = balls[0]
        assert isinstance(b0, dict)

        obs = b0.get("obs_2d_by_camera")
        assert isinstance(obs, dict)

        uv0_out = obs["cam0"]["uv"]
        uv1_out = obs["cam1"]["uv"]

        self.assertAlmostEqual(float(uv0_out[0]), float(origin0_c[0]) + 10.0, places=6)
        self.assertAlmostEqual(float(uv0_out[1]), float(origin0_c[1]) + 10.0, places=6)
        self.assertAlmostEqual(float(uv1_out[0]), float(origin1_c[0]) + 10.0, places=6)
        self.assertAlmostEqual(float(uv1_out[1]), float(origin1_c[1]) + 10.0, places=6)

        dets = b0.get("detections")
        assert isinstance(dets, dict)
        self.assertAlmostEqual(float(dets["cam0"]["center"][0]), float(origin0_c[0]) + 10.0, places=6)
        self.assertAlmostEqual(float(dets["cam1"]["center"][0]), float(origin1_c[0]) + 10.0, places=6)


if __name__ == "__main__":
    unittest.main()
