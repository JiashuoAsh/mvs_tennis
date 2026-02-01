"""最小单测：验证定位流水线输出包含 2D 观测协方差与 3D 点协方差等增强字段。

说明：
- 该测试不追求数值最优，只验证字段存在、形状正确、数值为有限数。
- 使用自定义 detector（通过图片左上角像素标记）为不同相机返回不同的检测中心。
"""

from __future__ import annotations

import unittest

import numpy as np

from tennis3d.geometry.calibration import CalibrationSet, CameraCalibration
from tennis3d.geometry.triangulation import project_point
from tennis3d.models import Detection
from tennis3d.pipeline.core import run_localization_pipeline


class _MarkerDetector:
    """根据图像左上角像素值返回预设的 bbox。"""

    def __init__(self, uv_by_marker: dict[int, tuple[float, float]]):
        self._uv_by_marker = dict(uv_by_marker)

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        marker = int(img_bgr[0, 0, 0])
        uv = self._uv_by_marker.get(marker)
        if uv is None:
            return []
        u, v = float(uv[0]), float(uv[1])
        half = 5.0
        return [Detection(bbox=(u - half, v - half, u + half, v + half), score=0.9, cls=0)]


class TestPipelineOutputCovariance(unittest.TestCase):
    def test_pipeline_outputs_obs_and_covariances(self) -> None:
        # 构造一个简单的两相机标定：cam1 相对 cam0 沿 +x 平移 1m。
        fx = 800.0
        fy = 800.0
        cx = 50.0
        cy = 50.0
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        R_wc = np.eye(3, dtype=np.float64)

        cam0 = CameraCalibration(
            name="cam0",
            image_size=(100, 100),
            K=K,
            dist=np.zeros(0, dtype=np.float64),
            R_wc=R_wc,
            t_wc=np.zeros(3, dtype=np.float64),
        )
        cam1 = CameraCalibration(
            name="cam1",
            image_size=(100, 100),
            K=K,
            dist=np.zeros(0, dtype=np.float64),
            R_wc=R_wc,
            # world->camera: X_c = X_w - C，因此 C=(1,0,0) => t=-C
            t_wc=np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        )
        calib = CalibrationSet(cameras={"cam0": cam0, "cam1": cam1})

        # 选择一个位于前方的世界点，并为每相机生成一致的 2D 观测。
        X_gt = np.array([0.2, -0.1, 4.0], dtype=np.float64)
        uv0 = project_point(cam0.P, X_gt)
        uv1 = project_point(cam1.P, X_gt)

        detector = _MarkerDetector({0: uv0, 1: uv1})

        img0 = np.zeros((100, 100, 3), dtype=np.uint8)
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img0[0, 0, 0] = 0
        img1[0, 0, 0] = 1

        groups = [({"frame_id": 0}, {"cam0": img0, "cam1": img1})]

        it = run_localization_pipeline(
            groups=groups,
            calib=calib,
            detector=detector,
            min_score=0.1,
            require_views=2,
            max_detections_per_camera=5,
            max_reproj_error_px=5.0,
            max_uv_match_dist_px=50.0,
            merge_dist_m=0.01,
            include_detection_details=True,
            aligner=None,
        )
        rec = next(iter(it))
        self.assertIn("balls", rec)
        self.assertEqual(len(rec["balls"]), 1)

        b0 = rec["balls"][0]
        self.assertIn("obs_2d_by_camera", b0)
        self.assertIn("ball_3d_cov_world", b0)
        self.assertIn("triangulation_stats", b0)

        obs = b0["obs_2d_by_camera"]
        self.assertIn("cam0", obs)
        self.assertIn("cam1", obs)
        self.assertIn("cov_uv", obs["cam0"])
        self.assertEqual(len(obs["cam0"]["cov_uv"]), 2)
        self.assertEqual(len(obs["cam0"]["cov_uv"][0]), 2)

        cov_X = b0["ball_3d_cov_world"]
        self.assertIsNotNone(cov_X)
        self.assertEqual(len(cov_X), 3)
        self.assertEqual(len(cov_X[0]), 3)

        # 数值应为有限
        flat = [float(v) for row in cov_X for v in row]
        self.assertTrue(all(np.isfinite(flat)))


if __name__ == "__main__":
    unittest.main()
