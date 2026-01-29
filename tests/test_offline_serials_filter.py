"""单测：离线模式支持按相机 serials 过滤。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tennis3d.config import load_offline_app_config
from tennis3d.pipeline.sources import iter_capture_image_groups


class TestIterCaptureImageGroupsSerialsFilter(unittest.TestCase):
    def test_iter_capture_image_groups_filters_by_serials(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            captures_dir = Path(td) / "captures"
            group0 = captures_dir / "group_0000000000"
            group0.mkdir(parents=True, exist_ok=True)

            img = np.zeros((8, 8, 3), dtype=np.uint8)
            img[0, 0] = (255, 0, 0)

            f0 = group0 / "cam0.bmp"
            f1 = group0 / "cam1.bmp"
            self.assertTrue(cv2.imwrite(str(f0), img))
            self.assertTrue(cv2.imwrite(str(f1), img))

            rec = {
                "group_seq": 0,
                "group_by": "frame_num",
                "trigger_index": 123,
                "frames": [
                    {"serial": "S1", "file": str(Path("group_0000000000") / "cam0.bmp")},
                    {"serial": "S2", "file": str(Path("group_0000000000") / "cam1.bmp")},
                ],
            }

            (captures_dir / "metadata.jsonl").write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")

            groups = list(iter_capture_image_groups(captures_dir=captures_dir, max_groups=0, serials=["S1"]))
            self.assertEqual(len(groups), 1)

            meta, images_by_camera = groups[0]
            self.assertEqual(meta.get("group_seq"), 0)
            self.assertEqual(meta.get("trigger_index"), 123)

            self.assertIn("S1", images_by_camera)
            self.assertNotIn("S2", images_by_camera)


class TestOfflineConfigSerials(unittest.TestCase):
    def test_load_offline_config_serials_dedup_and_strip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "cfg.json"
            p.write_text(
                json.dumps(
                    {
                        "captures_dir": "data/captures_master_slave/tennis_test",
                        "calib": "data/calibration/example_triple_camera_calib.json",
                        "serials": [" A ", "A", "B", ""],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            cfg = load_offline_app_config(p)
            self.assertEqual(cfg.serials, ["A", "B"])

    def test_load_offline_config_rejects_empty_serials_list(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "cfg.json"
            p.write_text(
                json.dumps(
                    {
                        "captures_dir": "data/captures_master_slave/tennis_test",
                        "calib": "data/calibration/example_triple_camera_calib.json",
                        "serials": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaises(RuntimeError):
                load_offline_app_config(p)


if __name__ == "__main__":
    unittest.main()
