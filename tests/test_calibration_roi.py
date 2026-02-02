import unittest

import numpy as np

from tennis3d.geometry.calibration import (
    CalibrationSet,
    CameraCalibration,
    apply_sensor_roi_to_calibration,
)
from tennis3d.geometry.triangulation import project_point


class TestCalibrationRoi(unittest.TestCase):
    def test_apply_sensor_roi_shifts_principal_point(self) -> None:
        """ROI 裁剪应当只平移主点，不改变 fx/fy 与外参。"""

        fx = 1000.0
        fy = 1100.0
        cx = 1224.0
        cy = 1024.0
        K = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        R_wc = np.eye(3, dtype=np.float64)
        t_wc = np.zeros(3, dtype=np.float64)

        full_w, full_h = 2448, 2048
        cam = CameraCalibration(
            name="cam0",
            image_size=(full_w, full_h),
            K=K,
            dist=np.zeros((0,), dtype=np.float64),
            R_wc=R_wc,
            t_wc=t_wc,
        )
        calib = CalibrationSet(cameras={"cam0": cam})

        # 传感器 ROI：以 (ox,oy) 为左上角，输出 roi_w x roi_h 的图像。
        roi_w, roi_h = 1980, 1080
        ox, oy = 200, 400

        calib_roi = apply_sensor_roi_to_calibration(
            calib,
            image_width=roi_w,
            image_height=roi_h,
            image_offset_x=ox,
            image_offset_y=oy,
        )

        self.assertEqual(calib_roi.cameras["cam0"].image_size, (roi_w, roi_h))

        # 选一个前方点，验证：满幅投影再减 offset == ROI 标定直接投影。
        X_w = np.array([0.3, -0.2, 5.0], dtype=np.float64)
        uv_full = project_point(cam.P, X_w)
        uv_roi_expected = (uv_full[0] - float(ox), uv_full[1] - float(oy))
        uv_roi = project_point(calib_roi.cameras["cam0"].P, X_w)

        self.assertTrue(np.allclose(np.array(uv_roi), np.array(uv_roi_expected), atol=1e-9))


if __name__ == "__main__":
    unittest.main()
