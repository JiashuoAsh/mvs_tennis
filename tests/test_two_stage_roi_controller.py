"""验证：两级 ROI 控制器返回的 offset 是“相机 AOI offset + 软件裁剪 offset”。

该测试聚焦坐标系一致性：
- 相机输出为 AOI 图像（其坐标原点在满幅图像的 (OffsetX,OffsetY)）
- TwoStageKinematicRoiController 在 AOI 图像内再做软件裁剪
- pipeline/core 应把 detector 输出 bbox/center 回写到满幅坐标系

注意：
- 这里不测试相机 SDK 写 Offset 的行为（需要真相机）。
- 我们仅验证：
  1) 软件裁剪中心预测时正确减去了相机 offset（在 AOI 局部坐标系裁剪）
  2) 返回给 core 的 offset 为 total_offset（满幅坐标系）
"""

from __future__ import annotations

import unittest

import numpy as np

from mvs.runtime_roi import IntNodeInfo
from tennis3d.geometry.calibration import CalibrationSet, CameraCalibration
from tennis3d.geometry.triangulation import project_point
from tennis3d.models import Detection
from tennis3d.pipeline.core import run_localization_pipeline
from tennis3d.pipeline.two_stage_roi import CameraAoiRuntimeConfig, CameraAoiState, SoftwareCropConfig, TwoStageKinematicRoiController


class _ConstantDetector:
    """在输入图像坐标系下返回一个固定中心的 bbox。"""

    def __init__(self, *, center_uv: tuple[float, float] = (10.0, 10.0)) -> None:
        self._cx = float(center_uv[0])
        self._cy = float(center_uv[1])

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        half = 2.0
        cx, cy = float(self._cx), float(self._cy)
        return [Detection(bbox=(cx - half, cy - half, cx + half, cy + half), score=0.9, cls=0)]


def _clamp_origin(*, ox: int, oy: int, crop_size: int, img_w: int, img_h: int) -> tuple[int, int]:
    cw = int(min(int(crop_size), int(img_w)))
    ch = int(min(int(crop_size), int(img_h)))
    ox = int(max(0, min(int(ox), int(img_w) - cw)))
    oy = int(max(0, min(int(oy), int(img_h) - ch)))
    return int(ox), int(oy)


class TestTwoStageRoiController(unittest.TestCase):
    def test_total_offset_includes_camera_and_crop(self) -> None:
        # 构造两相机标定：cam1 相对 cam0 沿 +x 平移 1m（与现有测试保持一致的几何关系）。
        fx = 800.0
        fy = 800.0
        cx = 100.0
        cy = 100.0
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        R_wc = np.eye(3, dtype=np.float64)

        cam0 = CameraCalibration(
            name="cam0",
            image_size=(2448, 2048),
            K=K,
            dist=np.zeros(0, dtype=np.float64),
            R_wc=R_wc,
            t_wc=np.zeros(3, dtype=np.float64),
        )
        cam1 = CameraCalibration(
            name="cam1",
            image_size=(2448, 2048),
            K=K,
            dist=np.zeros(0, dtype=np.float64),
            R_wc=R_wc,
            # world->camera: X_c = X_w - C，因此 C=(1,0,0) => t=-C
            t_wc=np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        )
        calib = CalibrationSet(cameras={"cam0": cam0, "cam1": cam1})

        # 选择一个世界点作为“上一帧 3D 观测”，让控制器能投影得到 uv。
        X_gt = np.array([0.2, -0.1, 4.0], dtype=np.float64)
        uv0 = project_point(cam0.P, X_gt)
        uv1 = project_point(cam1.P, X_gt)

        # 模拟相机侧 AOI：输出图像为 200x200，但在满幅中各自有不同 offset。
        aoi_w, aoi_h = 200, 200
        cam0_off = (80, 50)
        cam1_off = (40, 20)

        # offset 节点约束：为测试简化，inc=1 且范围足够大。
        ox_info = IntNodeInfo(cur=0, vmin=0, vmax=5000, inc=1)
        oy_info = IntNodeInfo(cur=0, vmin=0, vmax=5000, inc=1)

        aoi_state = {
            "cam0": CameraAoiState(
                aoi_width=aoi_w,
                aoi_height=aoi_h,
                offset_x=int(cam0_off[0]),
                offset_y=int(cam0_off[1]),
                offset_x_info=ox_info,
                offset_y_info=oy_info,
                initial_offset_x=int(cam0_off[0]),
                initial_offset_y=int(cam0_off[1]),
            ),
            "cam1": CameraAoiState(
                aoi_width=aoi_w,
                aoi_height=aoi_h,
                offset_x=int(cam1_off[0]),
                offset_y=int(cam1_off[1]),
                offset_x_info=ox_info,
                offset_y_info=oy_info,
                initial_offset_x=int(cam1_off[0]),
                initial_offset_y=int(cam1_off[1]),
            ),
        }

        crop_size = 50
        roi = TwoStageKinematicRoiController(
            crop_cfg=SoftwareCropConfig(
                crop_width=crop_size,
                crop_height=crop_size,
                smooth_alpha=0.0,  # 关闭平滑，便于精确断言
                max_step_px=0,
                reset_after_missed=8,
            ),
            camera_cfg=CameraAoiRuntimeConfig(enabled=False),
            aoi_state_by_camera=aoi_state,
            applier=None,
        )

        # 预热：给控制器一条 3D 输出，使其下一次 preprocess 能投影 uv。
        roi.update_after_output(out_rec={"frame_id": 0, "balls": [{"ball_3d_world": X_gt.tolist()}]}, calib=calib)

        detector = _ConstantDetector(center_uv=(10.0, 10.0))

        img0 = np.zeros((aoi_h, aoi_w, 3), dtype=np.uint8)
        img1 = np.zeros((aoi_h, aoi_w, 3), dtype=np.uint8)
        groups = [({"frame_id": 1}, {"cam0": img0, "cam1": img1})]

        it = run_localization_pipeline(
            groups=groups,
            calib=calib,
            detector=detector,
            min_score=0.1,
            require_views=2,
            max_detections_per_camera=5,
            max_reproj_error_px=50.0,
            max_uv_match_dist_px=200.0,
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

        # 期望：裁剪 origin 的计算在 AOI 局部坐标系下进行，因此需要减去相机 offset。
        # origin = round((uv_full - cam_offset) - crop/2)
        half = 0.5 * float(crop_size)

        def _expected_uv(uv_full: object, cam_off: tuple[int, int]) -> tuple[float, float]:
            # project_point 可能返回 tuple 或 ndarray；这里不强制类型，避免测试因实现细节波动而失败。
            u_local = float(uv_full[0]) - float(cam_off[0])  # type: ignore[index]
            v_local = float(uv_full[1]) - float(cam_off[1])  # type: ignore[index]
            ox = int(round(u_local - half))
            oy = int(round(v_local - half))
            ox_c, oy_c = _clamp_origin(ox=ox, oy=oy, crop_size=crop_size, img_w=aoi_w, img_h=aoi_h)
            total_ox = int(cam_off[0]) + int(ox_c)
            total_oy = int(cam_off[1]) + int(oy_c)
            # detector 在裁剪窗口坐标系输出 center=(10,10)，回写后即为 total_offset + (10,10)
            return float(total_ox) + 10.0, float(total_oy) + 10.0

        uv0_exp = _expected_uv(uv0, cam0_off)
        uv1_exp = _expected_uv(uv1, cam1_off)

        uv0_out = obs["cam0"]["uv"]
        uv1_out = obs["cam1"]["uv"]

        self.assertAlmostEqual(float(uv0_out[0]), float(uv0_exp[0]), places=6)
        self.assertAlmostEqual(float(uv0_out[1]), float(uv0_exp[1]), places=6)
        self.assertAlmostEqual(float(uv1_out[0]), float(uv1_exp[0]), places=6)
        self.assertAlmostEqual(float(uv1_out[1]), float(uv1_exp[1]), places=6)


if __name__ == "__main__":
    unittest.main()
