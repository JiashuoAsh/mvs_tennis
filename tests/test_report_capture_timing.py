"""单测：tools/report_capture_timing.py 的时序统计报告。

覆盖目标：
- 能从最小 metadata.jsonl 生成 timing_report.json 与 timing_report.md。
- send(seq) 与 group(group_seq) 能对齐时，能统计 send->arrival。
- 事件最近邻匹配能给出 dt_ticks；若提供 time-mapping 则能给出 dt_ms。
- ExposureStart/ExposureEnd 同时存在时，能统计曝光时长（ticks/ms）。

说明：
- 使用 tmp 目录构造最小 captures，避免依赖仓库内大样例数据。
- 以 importlib 动态加载工具脚本，避免把 tools 当成库强依赖。
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _load_tool_module():
    repo_root = Path(__file__).resolve().parents[1]
    tool_path = repo_root / "tools" / "report_capture_timing.py"
    if not tool_path.exists():
        raise FileNotFoundError(str(tool_path))

    spec = importlib.util.spec_from_file_location("report_capture_timing", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载工具脚本: {tool_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestReportCaptureTiming(unittest.TestCase):
    def test_basic_report_without_mapping(self) -> None:
        mod = _load_tool_module()

        with tempfile.TemporaryDirectory() as td:
            captures_dir = Path(td)
            meta_path = captures_dir / "metadata.jsonl"

            send = {
                "type": "soft_trigger_send",
                "seq": 0,
                "created_at": 1000.0,
                "host_monotonic": 10.0,
                "targets": ["S1"],
            }
            ev_start = {
                "type": "camera_event",
                "serial": "S1",
                "event_name": "ExposureStart",
                "event_timestamp": 1030,
                "created_at": 1000.1,
                "host_monotonic": 10.050,
            }
            ev_end = {
                "type": "camera_event",
                "serial": "S1",
                "event_name": "ExposureEnd",
                "event_timestamp": 2030,
                "created_at": 1000.2,
                "host_monotonic": 10.080,
            }
            group = {
                "group_seq": 0,
                "group_by": "frame_num",
                "created_at": 1001.0,
                "frames": [
                    {
                        "cam_index": 0,
                        "serial": "S1",
                        "frame_num": 1,
                        "dev_timestamp": 1000,
                        "host_timestamp": 1770211747103,
                        "arrival_monotonic": 10.100,
                        "file": None,
                    }
                ],
            }

            # 故意乱序写入，验证脚本的 pass1/pass2 不依赖记录顺序。
            meta_path.write_text(
                "\n".join(json.dumps(x, ensure_ascii=False) for x in [group, ev_start, send, ev_end]) + "\n",
                encoding="utf-8",
            )

            rc = mod.main(
                [
                    "--captures-dir",
                    str(captures_dir),
                    "--event-name",
                    "ExposureStart",
                    "--event-name",
                    "ExposureEnd",
                    "--event-match-window-ticks",
                    "100000",
                ]
            )
            self.assertEqual(rc, 0)

            out_json = captures_dir / "timing_report.json"
            out_md = captures_dir / "timing_report.md"
            self.assertTrue(out_json.exists())
            self.assertTrue(out_md.exists())

            rep = json.loads(out_json.read_text(encoding="utf-8"))

            # send -> arrival（只有一条样本，应该是 100ms）
            s2a = rep["host"]["send_to_arrival_ms"]["stats"]
            self.assertEqual(s2a["n"], 1)
            self.assertAlmostEqual(float(s2a["p50"]), 100.0, places=6)

            # ExposureStart: dt_ticks = 1030 - 1000 = 30
            s1_start = rep["events"]["by_serial_event"]["S1"]["ExposureStart"]
            self.assertEqual(s1_start["dt_ticks"]["stats"]["n"], 1)
            self.assertAlmostEqual(float(s1_start["dt_ticks"]["stats"]["p50"]), 30.0, places=6)

            # 曝光时长：2030 - 1030 = 1000 ticks
            exp = rep["exposure"]["by_serial"]["S1"]
            self.assertEqual(exp["duration_ticks"]["stats"]["n"], 1)
            self.assertAlmostEqual(float(exp["duration_ticks"]["stats"]["p50"]), 1000.0, places=6)

    def test_report_with_time_mapping_includes_ms(self) -> None:
        mod = _load_tool_module()

        with tempfile.TemporaryDirectory() as td:
            captures_dir = Path(td)
            meta_path = captures_dir / "metadata.jsonl"

            meta_path.write_text(
                "\n".join(
                    json.dumps(x, ensure_ascii=False)
                    for x in [
                        {
                            "type": "soft_trigger_send",
                            "seq": 0,
                            "created_at": 1000.0,
                            "host_monotonic": 10.0,
                            "targets": ["S1"],
                        },
                        {
                            "type": "camera_event",
                            "serial": "S1",
                            "event_name": "ExposureStart",
                            "event_timestamp": 1030,
                            "created_at": 1000.1,
                            "host_monotonic": 10.050,
                        },
                        {
                            "type": "camera_event",
                            "serial": "S1",
                            "event_name": "ExposureEnd",
                            "event_timestamp": 2030,
                            "created_at": 1000.2,
                            "host_monotonic": 10.080,
                        },
                        {
                            "group_seq": 0,
                            "group_by": "frame_num",
                            "created_at": 1001.0,
                            "frames": [
                                {
                                    "cam_index": 0,
                                    "serial": "S1",
                                    "frame_num": 1,
                                    "dev_timestamp": 1000,
                                    "host_timestamp": 1770211747103,
                                    "arrival_monotonic": 10.100,
                                    "file": None,
                                }
                            ],
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            # a=0.001 ms/tick => 30 ticks = 0.03ms, 1000 ticks = 1.0ms
            (captures_dir / "time_mapping_dev_to_host_ms.json").write_text(
                json.dumps(
                    {
                        "schema": "mvs_time_mapping_v1",
                        "created_at": 0.0,
                        "host_unit": "ms_epoch",
                        "dev_unit": "ticks",
                        "source": {"kind": "frame_dev_to_host", "metadata_path": None},
                        "cameras": {
                            "S1": {
                                "a": 0.001,
                                "b": 0.0,
                                "n_used": 1,
                                "n_total": 1,
                                "rms_ms": 0.0,
                                "p95_ms": 0.0,
                                "max_ms": 0.0,
                                "dev_min": 0,
                                "dev_max": 0,
                                "host_min_ms": 0,
                                "host_max_ms": 0,
                            }
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            rc = mod.main(
                [
                    "--captures-dir",
                    str(captures_dir),
                    "--event-name",
                    "ExposureStart",
                    "--event-name",
                    "ExposureEnd",
                    "--event-match-window-ms",
                    "10.0",
                ]
            )
            self.assertEqual(rc, 0)

            rep = json.loads((captures_dir / "timing_report.json").read_text(encoding="utf-8"))

            s1_start = rep["events"]["by_serial_event"]["S1"]["ExposureStart"]
            self.assertEqual(s1_start["dt_ms"]["stats"]["n"], 1)
            self.assertAlmostEqual(float(s1_start["dt_ms"]["stats"]["p50"]), 0.03, places=9)

            exp = rep["exposure"]["by_serial"]["S1"]
            self.assertEqual(exp["duration_ms"]["stats"]["n"], 1)
            self.assertAlmostEqual(float(exp["duration_ms"]["stats"]["p50"]), 1.0, places=9)

    def test_markdown_allows_missing_event_samples(self) -> None:
        """当某个 event_name 完全没有匹配样本时，Markdown 仍应可生成。"""

        mod = _load_tool_module()

        with tempfile.TemporaryDirectory() as td:
            captures_dir = Path(td)
            meta_path = captures_dir / "metadata.jsonl"

            # 只写 ExposureStart，不写 FrameEnd：用于触发 abs_p95=None 的路径。
            meta_path.write_text(
                "\n".join(
                    json.dumps(x, ensure_ascii=False)
                    for x in [
                        {
                            "type": "camera_event",
                            "serial": "S1",
                            "event_name": "ExposureStart",
                            "event_timestamp": 1030,
                            "created_at": 1000.1,
                            "host_monotonic": 10.050,
                        },
                        {
                            "group_seq": 0,
                            "group_by": "frame_num",
                            "created_at": 1001.0,
                            "frames": [
                                {
                                    "cam_index": 0,
                                    "serial": "S1",
                                    "frame_num": 1,
                                    "dev_timestamp": 1000,
                                    "host_timestamp": 1770211747103,
                                    "arrival_monotonic": 10.100,
                                    "file": None,
                                }
                            ],
                        },
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rc = mod.main(
                [
                    "--captures-dir",
                    str(captures_dir),
                    "--event-name",
                    "ExposureStart",
                    "--event-name",
                    "FrameEnd",
                    "--event-match-window-ticks",
                    "100000",
                ]
            )
            self.assertEqual(rc, 0)

            out_md = captures_dir / "timing_report.md"
            self.assertTrue(out_md.exists())

            # 设备域表头应输出带符号统计（避免 abs_p95 引起歧义）。
            md = out_md.read_text(encoding="utf-8")
            self.assertIn("dt_ticks_p50", md)
            self.assertIn("dt_ms_p50", md)

            rep = json.loads((captures_dir / "timing_report.json").read_text(encoding="utf-8"))
            # FrameEnd 事件没有样本，dt_ticks 统计应为空（n=0，abs_p95=None）。
            s1_end = rep["events"]["by_serial_event"]["S1"]["FrameEnd"]["dt_ticks"]["stats"]
            self.assertEqual(int(s1_end["n"]), 0)
            self.assertIsNone(s1_end["abs_p95"])
