"""单测：把 captures 图片按相机重排。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mvs.session.capture_relayout import relayout_capture_by_camera


class TestCaptureRelayout(unittest.TestCase):
    def test_relayout_copy_mode_and_skip_non_group_records(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            # 关键点：目录名用 for_calib，便于覆盖真实数据里常见的 file 路径模式
            # data\...\for_calib\group_...\xxx.bmp
            captures_dir = root / "for_calib"
            captures_dir.mkdir(parents=True, exist_ok=True)

            group0 = captures_dir / "group_0000000000"
            group0.mkdir(parents=True, exist_ok=True)

            src0 = group0 / "cam0_seq000000_f1.bmp"
            src1 = group0 / "cam1_seq000000_f1.bmp"
            src0.write_bytes(b"cam0")
            src1.write_bytes(b"cam1")

            meta_path = captures_dir / "metadata.jsonl"

            event_rec = {"type": "camera_event", "serial": "DA0000000", "event_name": "ExposureStart"}
            group_rec = {
                "group_seq": 0,
                "frames": [
                    {
                        "cam_index": 0,
                        "serial": "DA8199303",
                        "frame_num": 1,
                        # 故意写成“看起来像仓库根目录相对路径”的形式，验证解析逻辑的鲁棒性。
                        "file": "data\\captures_master_slave\\for_calib\\group_0000000000\\cam0_seq000000_f1.bmp",
                    },
                    {
                        "cam_index": 1,
                        "serial": "DA8199402",
                        "frame_num": 1,
                        "file": "data\\captures_master_slave\\for_calib\\group_0000000000\\cam1_seq000000_f1.bmp",
                    },
                ],
            }

            meta_path.write_text(
                "\n".join([json.dumps(event_rec, ensure_ascii=False), json.dumps(group_rec, ensure_ascii=False)])
                + "\n",
                encoding="utf-8",
            )

            out_dir = root / "for_calib_by_camera"

            stats = relayout_capture_by_camera(
                captures_dir=captures_dir,
                output_dir=out_dir,
                mode="copy",
                overwrite=False,
                dry_run=False,
            )

            self.assertEqual(stats.groups_seen, 1)
            self.assertEqual(stats.frames_seen, 2)
            self.assertEqual(stats.files_created, 2)
            self.assertEqual(stats.files_failed, 0)
            self.assertEqual(stats.missing_source_files, 0)
            self.assertEqual(stats.link_fallback_to_copy, 0)

            dst0 = out_dir / "cam0_DA8199303" / src0.name
            dst1 = out_dir / "cam1_DA8199402" / src1.name
            self.assertTrue(dst0.exists())
            self.assertTrue(dst1.exists())
            self.assertEqual(dst0.read_bytes(), b"cam0")
            self.assertEqual(dst1.read_bytes(), b"cam1")

            # 再跑一次，验证已存在目标会被跳过。
            stats2 = relayout_capture_by_camera(
                captures_dir=captures_dir,
                output_dir=out_dir,
                mode="copy",
                overwrite=False,
                dry_run=False,
            )
            self.assertEqual(stats2.files_created, 0)
            self.assertEqual(stats2.files_skipped_existing, 2)


if __name__ == "__main__":
    unittest.main()
