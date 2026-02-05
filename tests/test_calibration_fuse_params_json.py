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

        ref_file = repo_root / "data" / "calibration" / "camera_extrinsics_C_T_B.json"
        ref = json.loads(ref_file.read_text(encoding="utf-8"))

        # 说明：仓库内的参考文件 camera_extrinsics_C_T_B.json 是通过 tools/generate_camera_extrinsics.py 生成的。
        # 为了避免“参考文件已更新但测试仍固定旧 inputs 目录”的漂移，这里从参考文件的 source 字段读取输入来源。
        source = ref.get("source", {})
        if not isinstance(source, dict):
            raise AssertionError("reference file field 'source' must be an object")

        intr_dir_s = str(source.get("intrinsics_dir", "")).strip()
        extr_file_s = str(source.get("extrinsics_file", "")).strip()
        generated_at = str(source.get("generated_at", "")).strip() or ""
        if not intr_dir_s:
            raise AssertionError("reference file source.intrinsics_dir is missing")
        if not extr_file_s:
            raise AssertionError("reference file source.extrinsics_file is missing")

        intr_dir = repo_root / Path(intr_dir_s)
        extr_file = repo_root / Path(extr_file_s)

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
                generated_at=generated_at,
            ),
            units="m",
            version=1,
            notes="",
        )

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
