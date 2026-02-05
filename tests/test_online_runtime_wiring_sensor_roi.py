import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tennis3d_online.runtime_wiring import SensorRoi, load_calibration_for_runtime


def _write_calib_json(path: Path, *, image_size: tuple[int, int], K: np.ndarray) -> None:
    """写入最小可用标定 JSON（单相机）。"""

    data = {
        "cameras": {
            "cam0": {
                "image_size": [int(image_size[0]), int(image_size[1])],
                "K": K.tolist(),
                "dist": [],
                "R_wc": np.eye(3, dtype=np.float64).tolist(),
                "t_wc": [0.0, 0.0, 0.0],
            }
        }
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


class TestOnlineRuntimeWiringSensorRoi(unittest.TestCase):
    def test_sensor_roi_override_is_used_for_calibration_shift(self) -> None:
        fx, fy, cx, cy = 1000.0, 1100.0, 1224.0, 1024.0
        K = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        with tempfile.TemporaryDirectory() as td:
            calib_path = Path(td) / "calib.json"
            _write_calib_json(calib_path, image_size=(2448, 2048), K=K)

            spec = SimpleNamespace(
                calib_path=calib_path,
                camera_aoi_runtime=False,
                image_width=1980,
                image_height=1080,
                image_offset_x=200,
                image_offset_y=400,
            )

            # 模拟“相机实际 offset 被对齐/修正”为 192。
            actual_roi = SensorRoi(width=1980, height=1080, offset_x=192, offset_y=400)
            calib = load_calibration_for_runtime(spec, sensor_roi=actual_roi)  # type: ignore[arg-type]

            cam = calib.cameras["cam0"]
            self.assertEqual(cam.image_size, (1980, 1080))
            self.assertAlmostEqual(float(cam.K[0, 2]), cx - 192.0)
            self.assertAlmostEqual(float(cam.K[1, 2]), cy - 400.0)

    def test_skip_double_shift_when_calibration_already_roi_sized(self) -> None:
        # 该用例模拟：标定文件已经是在 ROI 坐标系下完成的（image_size 已是 ROI 尺寸）。
        fx, fy, cx, cy = 900.0, 950.0, 500.0, 300.0
        K = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        with tempfile.TemporaryDirectory() as td:
            calib_path = Path(td) / "calib.json"
            _write_calib_json(calib_path, image_size=(1980, 1080), K=K)

            spec = SimpleNamespace(
                calib_path=calib_path,
                camera_aoi_runtime=False,
                image_width=1980,
                image_height=1080,
                image_offset_x=200,
                image_offset_y=400,
            )

            actual_roi = SensorRoi(width=1980, height=1080, offset_x=200, offset_y=400)
            calib = load_calibration_for_runtime(spec, sensor_roi=actual_roi)  # type: ignore[arg-type]

            cam = calib.cameras["cam0"]
            # 期望：不再重复平移主点。
            self.assertAlmostEqual(float(cam.K[0, 2]), cx)
            self.assertAlmostEqual(float(cam.K[1, 2]), cy)


if __name__ == "__main__":
    unittest.main()
