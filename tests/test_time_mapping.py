"""单测：方案B（dev_timestamp -> host 时间轴）的拟合与应用。

覆盖目标：
- 能从 captures/metadata.jsonl 中按 serial 收集 (dev_timestamp, host_timestamp_ms)
- 能稳健拟合线性映射，并得到合理的残差
- 离线 source 在启用 dev_timestamp_mapping 后能产出 capture_t_abs

说明：
- 这里使用仓库自带的小型 captures 示例，避免引入额外测试数据。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mvs.time_mapping import collect_frame_pairs_from_metadata, fit_dev_to_host_ms, save_time_mappings_json
from tennis3d.pipeline import iter_capture_image_groups


class TestTimeMapping(unittest.TestCase):
    def test_fit_and_apply_dev_timestamp_mapping(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        captures_dir = repo_root / "data" / "captures_master_slave" / "tennis_offline"
        meta_path = captures_dir / "metadata.jsonl"
        self.assertTrue(meta_path.exists(), msg=f"missing test data: {meta_path}")

        pairs_by_serial = collect_frame_pairs_from_metadata(metadata_path=meta_path, max_groups=80)
        self.assertTrue(pairs_by_serial, msg="no frame pairs collected")

        mappings = {}
        for serial, pairs in pairs_by_serial.items():
            # 保护：单测不要求所有相机都满足最小点数；但样例里应有多台。
            if len(pairs) < 15:
                continue
            m = fit_dev_to_host_ms(pairs, min_points=15, hard_outlier_ms=80.0)
            mappings[str(serial)] = m

            # 基本合理性断言：斜率应为正，残差不应离谱。
            self.assertGreater(m.a, 0.0)
            self.assertLess(m.p95_ms, 80.0)

        # 样例数据应至少拟合出 2 台相机（实际通常是 4 台）。
        self.assertGreaterEqual(len(mappings), 2)

        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "time_mapping.json"
            save_time_mappings_json(out_path=out_path, mappings=mappings, metadata_path=meta_path)

            # 启用方案B，验证 capture_t_abs 可以产出且在合理 epoch 范围。
            groups = iter_capture_image_groups(
                captures_dir=captures_dir,
                max_groups=5,
                serials=None,
                time_sync_mode="dev_timestamp_mapping",
                time_mapping_path=out_path,
            )

            last_t = None
            for meta, _images in groups:
                t = meta.get("capture_t_abs")
                src = meta.get("capture_t_source")
                self.assertEqual(src, "dev_timestamp_mapping")
                if t is None:
                    self.fail("capture_t_abs is None")
                t = float(t)

                # 与样例的 created_at（约 1769759052 秒）应同量级。
                self.assertGreater(t, 1.6e9)
                self.assertLess(t, 2.2e9)

                if last_t is not None:
                    # 粗略单调性：同一段 captures 内组时间应不下降（允许少量抖动，这里只做弱约束）。
                    self.assertGreaterEqual(t + 0.050, last_t)
                last_t = t


if __name__ == "__main__":
    unittest.main()
