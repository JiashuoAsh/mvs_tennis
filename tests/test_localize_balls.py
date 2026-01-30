"""单测：多球鲁棒定位（组内）。

覆盖目标：
- 同一组图像内存在多个球：应输出多个 3D（0..N）。
- 误检抑制：只有跨视角几何一致且达到 require_views 的候选才输出。
- 冲突消解：同一相机同一检测框不能被多个 3D 球重复使用。

说明：
- 这里使用合成标定与合成 3D 点，通过投影生成“理想检测框”，避免依赖模型推理。
"""

from __future__ import annotations

import unittest

import numpy as np

from tennis3d.geometry.calibration import CalibrationSet, CameraCalibration
from tennis3d.geometry.triangulation import project_point
from tennis3d.localization.localize import localize_balls
from tennis3d.models import Detection


def _make_calib_three_cams() -> CalibrationSet:
    fx = 1000.0
    fy = 1000.0
    cx = 0.0
    cy = 0.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)

    def _cam(name: str, C_w: np.ndarray) -> CameraCalibration:
        # 外参约定：X_c = R_wc X_w + t_wc，且 t_wc = -R_wc * C_w
        t = -(R @ C_w.reshape(3))
        return CameraCalibration(
            name=name,
            image_size=(1920, 1080),
            K=K,
            dist=np.zeros((0,), dtype=np.float64),
            R_wc=R,
            t_wc=t,
        )

    cams = {
        "cam0": _cam("cam0", np.array([0.0, 0.0, 0.0], dtype=np.float64)),
        "cam1": _cam("cam1", np.array([1.0, 0.0, 0.0], dtype=np.float64)),
        "cam2": _cam("cam2", np.array([0.0, 1.0, 0.0], dtype=np.float64)),
    }
    return CalibrationSet(cameras=cams)


def _det_from_uv(uv: tuple[float, float], *, score: float) -> Detection:
    u, v = float(uv[0]), float(uv[1])
    # bbox 只用于 center 计算，这里做一个小方框即可
    return Detection(bbox=(u - 2.0, v - 2.0, u + 2.0, v + 2.0), score=float(score), cls=0)


class TestLocalizeBalls(unittest.TestCase):
    def test_multi_ball_two_outputs(self) -> None:
        calib = _make_calib_three_cams()

        X1 = np.array([0.10, -0.05, 5.0], dtype=np.float64)
        X2 = np.array([-0.25, 0.10, 6.0], dtype=np.float64)

        dets_by_cam: dict[str, list[Detection]] = {}
        for cam_name, cam in calib.cameras.items():
            uv1 = project_point(cam.P, X1)
            uv2 = project_point(cam.P, X2)

            # 真球检测
            d1 = _det_from_uv(uv1, score=0.90)
            d2 = _det_from_uv(uv2, score=0.85)

            # 误检：跨视角不一致（即使分数高，require_views=3 也应被过滤）
            df = _det_from_uv((500.0, 300.0), score=0.99)

            dets_by_cam[cam_name] = [df, d1, d2]

        balls = localize_balls(
            calib=calib,
            detections_by_camera=dets_by_cam,
            min_score=0.1,
            require_views=3,
            max_detections_per_camera=5,
            max_reproj_error_px=2.0,
            max_uv_match_dist_px=5.0,
            merge_dist_m=0.05,
        )

        self.assertEqual(len(balls), 2, msg=f"expected 2 balls, got {len(balls)}")

        # 两个输出应分别接近两个 GT（顺序不固定，做最近邻匹配）
        X_out = [np.asarray(b.X_w, dtype=np.float64).reshape(3) for b in balls]
        d11 = float(np.linalg.norm(X_out[0] - X1))
        d12 = float(np.linalg.norm(X_out[0] - X2))
        d21 = float(np.linalg.norm(X_out[1] - X1))
        d22 = float(np.linalg.norm(X_out[1] - X2))
        # 要么 (0->X1 且 1->X2)，要么反过来
        ok = (d11 < 1e-3 and d22 < 1e-3) or (d12 < 1e-3 and d21 < 1e-3)
        self.assertTrue(ok, msg=f"X_out={X_out}, d={{d11:{d11}, d12:{d12}, d21:{d21}, d22:{d22}}}")

        # 冲突消解的基本属性：同一相机同一 detection 索引不应重复
        used_keys: set[tuple[str, int]] = set()
        for b in balls:
            for cam, idx in b.detection_indices.items():
                k = (str(cam), int(idx))
                self.assertNotIn(k, used_keys)
                used_keys.add(k)

    def test_false_or_incomplete_should_output_empty(self) -> None:
        calib = _make_calib_three_cams()

        X1 = np.array([0.15, 0.00, 5.5], dtype=np.float64)

        # case1: 只有单相机 -> 不能输出
        dets_by_cam_1 = {
            "cam0": [_det_from_uv(project_point(calib.cameras["cam0"].P, X1), score=0.9)],
            "cam1": [],
            "cam2": [],
        }
        balls_1 = localize_balls(
            calib=calib,
            detections_by_camera=dets_by_cam_1,
            min_score=0.1,
            require_views=2,
            max_detections_per_camera=5,
            max_reproj_error_px=2.0,
            max_uv_match_dist_px=5.0,
            merge_dist_m=0.05,
        )
        self.assertEqual(len(balls_1), 0)

        # case2: 跨视角不一致，且 require_views=3 -> 不能输出
        uv0 = project_point(calib.cameras["cam0"].P, X1)
        uv2 = project_point(calib.cameras["cam2"].P, X1)
        dets_by_cam_2 = {
            "cam0": [_det_from_uv(uv0, score=0.9)],
            # cam1 的检测刻意放很远，保证投影补全无法匹配
            "cam1": [_det_from_uv((800.0, 800.0), score=0.9)],
            "cam2": [_det_from_uv(uv2, score=0.9)],
        }
        balls_2 = localize_balls(
            calib=calib,
            detections_by_camera=dets_by_cam_2,
            min_score=0.1,
            require_views=3,
            max_detections_per_camera=5,
            max_reproj_error_px=2.0,
            max_uv_match_dist_px=5.0,
            merge_dist_m=0.05,
        )
        self.assertEqual(len(balls_2), 0)

    def test_conflict_resolution_same_detection_not_reused(self) -> None:
        calib = _make_calib_three_cams()

        # 两个球都存在，但 cam0 只有一个检测（会同时匹配两个球），冲突消解应只保留一个。
        X1 = np.array([0.10, 0.00, 5.0], dtype=np.float64)
        X2 = np.array([0.12, 0.02, 5.0], dtype=np.float64)

        uv0_1 = project_point(calib.cameras["cam0"].P, X1)
        uv0_2 = project_point(calib.cameras["cam0"].P, X2)
        # cam0 的唯一检测放在两者中间，让两个球都能在阈值内匹配到它
        uv0_mid = ((uv0_1[0] + uv0_2[0]) * 0.5, (uv0_1[1] + uv0_2[1]) * 0.5)

        dets_by_cam: dict[str, list[Detection]] = {
            "cam0": [_det_from_uv(uv0_mid, score=0.9)],
            "cam1": [
                _det_from_uv(project_point(calib.cameras["cam1"].P, X1), score=0.9),
                _det_from_uv(project_point(calib.cameras["cam1"].P, X2), score=0.9),
            ],
            "cam2": [
                _det_from_uv(project_point(calib.cameras["cam2"].P, X1), score=0.9),
                _det_from_uv(project_point(calib.cameras["cam2"].P, X2), score=0.9),
            ],
        }

        balls = localize_balls(
            calib=calib,
            detections_by_camera=dets_by_cam,
            min_score=0.1,
            require_views=3,
            max_detections_per_camera=5,
            # 放宽误差/匹配阈值，使得 cam0 的“共享检测”不会被 gating 直接过滤
            max_reproj_error_px=50.0,
            max_uv_match_dist_px=50.0,
            merge_dist_m=0.01,
        )

        self.assertEqual(len(balls), 1, msg=f"expected 1 ball after conflict resolution, got {len(balls)}")
        self.assertIn("cam0", balls[0].detection_indices)


if __name__ == "__main__":
    unittest.main()
