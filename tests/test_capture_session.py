"""单测：capture session 辅助函数（不依赖真实相机）。"""

from __future__ import annotations

import unittest

from mvs import build_trigger_plan, normalize_roi


class TestCaptureSessionHelpers(unittest.TestCase):
    def test_build_trigger_plan_without_master_software(self) -> None:
        serials = ["SN0", "SN1", "SN2"]
        plan = build_trigger_plan(serials=serials, trigger_source="Software", master_serial="")
        self.assertEqual(plan.trigger_sources, ["Software", "Software", "Software"])
        self.assertEqual(plan.soft_trigger_serials, serials)
        self.assertEqual(plan.mapping_str(serials), "SN0->Software, SN1->Software, SN2->Software")

    def test_build_trigger_plan_with_master(self) -> None:
        serials = ["SN0", "SN1", "SN2"]
        plan = build_trigger_plan(serials=serials, trigger_source="Line0", master_serial="SN1")
        self.assertEqual(plan.trigger_sources, ["Line0", "Software", "Line0"])
        self.assertEqual(plan.soft_trigger_serials, ["SN1"])

    def test_build_trigger_plan_master_not_in_serials(self) -> None:
        with self.assertRaises(ValueError):
            build_trigger_plan(serials=["SN0"], trigger_source="Line0", master_serial="SN1")

    def test_normalize_roi_default_none(self) -> None:
        w, h, ox, oy = normalize_roi(image_width=0, image_height=0, image_offset_x=0, image_offset_y=0)
        self.assertIsNone(w)
        self.assertIsNone(h)
        self.assertEqual((ox, oy), (0, 0))

    def test_normalize_roi_pair_required(self) -> None:
        with self.assertRaises(ValueError):
            normalize_roi(image_width=1920, image_height=0, image_offset_x=0, image_offset_y=0)

    def test_normalize_roi_valid(self) -> None:
        w, h, ox, oy = normalize_roi(image_width=1920, image_height=1080, image_offset_x=10, image_offset_y=20)
        self.assertEqual((w, h, ox, oy), (1920, 1080, 10, 20))


if __name__ == "__main__":
    unittest.main()
