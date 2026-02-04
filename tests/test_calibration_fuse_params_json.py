"""单测：把内参目录与外参文件融合成“相机内外参融合 JSON”。

说明：
    本仓库以 data/calibration/camera_extrinsics_C_T_B.json 作为标准标定输出。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from tennis3d.geometry.calibration_fuse import (
    FuseSourceInfo,
    build_params_calib_json,
    load_extrinsics_C_T_B,
    load_intrinsics_dir,
)


class TestCalibrationFuseParamsJson(unittest.TestCase):
    def test_fuse_matches_repo_reference(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]

        intr_dir = repo_root / "data" / "calibration" / "inputs" / "2026-01-30"
        # 说明：仓库内的参考文件 camera_extrinsics_C_T_B.json 是通过 tools/generate_camera_extrinsics.py
        # 融合 intr_dir 与 data/calibration/base_to_camera_extrinsics.json 生成的。
        # 因此该测试应当使用同一份外参来源，避免“参考文件已更新但 inputs 目录下旧外参未同步”的漂移。
        extr_file = repo_root / "data" / "calibration" / "base_to_camera_extrinsics.json"
        ref_file = repo_root / "data" / "calibration" / "camera_extrinsics_C_T_B.json"

        intr = load_intrinsics_dir(intr_dir)
        extr = load_extrinsics_C_T_B(extr_file)

        # 该映射来源于仓库内的参考文件（camera_extrinsics_C_T_B.json）相机顺序与当时标定流程。
        # 若未来更换了标定数据集，应同步更新此测试。
        extr_to_cam = {
            "cam0": "DA8199303",
            "cam1": "DA8199402",
            "cam2": "DA8199243",
            "cam3": "DA8199285",
        }

        payload = build_params_calib_json(
            intrinsics_by_name=intr,
            extrinsics_by_name=extr,
            extr_to_camera_name=extr_to_cam,
            source=FuseSourceInfo(
                intrinsics_dir=str(intr_dir).replace("\\", "/"),
                extrinsics_file=str(extr_file).replace("\\", "/"),
                generated_at="2026-01-30",
            ),
            units="m",
            version=1,
            notes="",
        )

        ref = json.loads(ref_file.read_text(encoding="utf-8"))

        self.assertIn("cameras", payload)
        self.assertIn("cameras", ref)

        cams_new = payload["cameras"]
        cams_ref = ref["cameras"]
        self.assertEqual(set(cams_new.keys()), set(cams_ref.keys()))

        for cam_name in cams_ref.keys():
            a = cams_new[cam_name]
            b = cams_ref[cam_name]

            self.assertEqual(a["image_size"], b["image_size"])
            self.assertTrue(np.allclose(np.asarray(a["K"], float), np.asarray(b["K"], float)))
            self.assertTrue(np.allclose(np.asarray(a["dist"], float), np.asarray(b["dist"], float)))
            self.assertTrue(np.allclose(np.asarray(a["R_wc"], float), np.asarray(b["R_wc"], float)))
            self.assertTrue(np.allclose(np.asarray(a["t_wc"], float), np.asarray(b["t_wc"], float)))


if __name__ == "__main__":
    unittest.main()
