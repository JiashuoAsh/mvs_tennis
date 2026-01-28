"""单测：YAML 标定加载。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tennis3d.geometry.calibration import load_calibration


class TestCalibrationYaml(unittest.TestCase):
    def test_load_yaml(self) -> None:
        yaml_text = """
        cameras:
          camA:
            image_size: [320, 240]
            K:
              - [300.0, 0.0, 160.0]
              - [0.0, 300.0, 120.0]
              - [0.0, 0.0, 1.0]
            dist: []
            R_wc:
              - [1.0, 0.0, 0.0]
              - [0.0, 1.0, 0.0]
              - [0.0, 0.0, 1.0]
            t_wc: [0.0, 0.0, 0.0]

          camB:
            image_size: [320, 240]
            K:
              - [300.0, 0.0, 160.0]
              - [0.0, 300.0, 120.0]
              - [0.0, 0.0, 1.0]
            dist: []
            R_wc:
              - [1.0, 0.0, 0.0]
              - [0.0, 1.0, 0.0]
              - [0.0, 0.0, 1.0]
            t_wc: [1.0, 0.0, 0.0]
        """.strip()

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "calib.yaml"
            p.write_text(yaml_text, encoding="utf-8")

            calib = load_calibration(p)
            self.assertIn("camA", calib.cameras)
            self.assertIn("camB", calib.cameras)

            camA = calib.require("camA")
            self.assertEqual(camA.image_size, (320, 240))
            self.assertTrue(np.allclose(camA.K, np.array([[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]])))
            self.assertEqual(camA.P.shape, (3, 4))


if __name__ == "__main__":
    unittest.main()
