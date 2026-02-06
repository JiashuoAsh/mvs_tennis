"""单测：mvs.analysis.capture_run 的离线分析口径（不依赖真实相机）。

说明：
- 该分析只读取 captures 目录里的 metadata.jsonl（以及可选的保存文件路径）。
- 这里构造最小 JSONL，验证核心字段能正确统计且不会因字段缺失而崩溃。
"""

from __future__ import annotations

import json
from pathlib import Path

from mvs.analysis import analyze_output_dir


def _write_jsonl(path: Path, records: list[dict]) -> None:
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_analyze_output_dir_minimal_happy_path(tmp_path: Path) -> None:
    out_dir = tmp_path / "captures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 两组、三相机，frame_num 同步递增，无丢包。
    group0 = {
        "group_seq": 0,
        "group_by": "frame_num",
        "created_at": 1_700_000_000.0,
        "frames": [
            {
                "cam_index": 0,
                "serial": "SN0",
                "frame_num": 10,
                "dev_timestamp": 1_000,
                "host_timestamp": 200,
                "width": 640,
                "height": 480,
                "pixel_type": 42,
                "lost_packet": 0,
                "arrival_monotonic": 10.0,
            },
            {
                "cam_index": 1,
                "serial": "SN1",
                "frame_num": 10,
                "dev_timestamp": 1_000,
                "host_timestamp": 201,
                "width": 640,
                "height": 480,
                "pixel_type": 42,
                "lost_packet": 0,
                "arrival_monotonic": 10.01,
            },
            {
                "cam_index": 2,
                "serial": "SN2",
                "frame_num": 10,
                "dev_timestamp": 1_000,
                "host_timestamp": 202,
                "width": 640,
                "height": 480,
                "pixel_type": 42,
                "lost_packet": 0,
                "arrival_monotonic": 10.02,
            },
        ],
    }

    group1 = {
        "group_seq": 1,
        "group_by": "frame_num",
        "created_at": 1_700_000_000.1,
        "frames": [
            {
                "cam_index": 0,
                "serial": "SN0",
                "frame_num": 11,
                "dev_timestamp": 2_000,
                "host_timestamp": 300,
                "width": 640,
                "height": 480,
                "pixel_type": 42,
                "lost_packet": 0,
                "arrival_monotonic": 10.1,
            },
            {
                "cam_index": 1,
                "serial": "SN1",
                "frame_num": 11,
                "dev_timestamp": 2_000,
                "host_timestamp": 301,
                "width": 640,
                "height": 480,
                "pixel_type": 42,
                "lost_packet": 0,
                "arrival_monotonic": 10.11,
            },
            {
                "cam_index": 2,
                "serial": "SN2",
                "frame_num": 11,
                "dev_timestamp": 2_000,
                "host_timestamp": 302,
                "width": 640,
                "height": 480,
                "pixel_type": 42,
                "lost_packet": 0,
                "arrival_monotonic": 10.12,
            },
        ],
    }

    meta_path = out_dir / "metadata.jsonl"
    _write_jsonl(meta_path, [group0, group1])

    summary, report_text, payload = analyze_output_dir(
        output_dir=out_dir,
        expected_cameras=3,
        expected_fps=None,
        fps_tolerance_ratio=0.2,
    )

    assert summary.records == 2
    assert summary.groups_complete == 2
    assert summary.groups_incomplete == 0

    assert summary.width_unique == 1
    assert summary.height_unique == 1
    assert summary.pixel_type_unique == 1

    assert summary.lost_packet_total == 0
    assert summary.groups_with_lost_packet == 0

    # frame_num 归一化 spread 应为 0。
    assert summary.groups_with_frame_num_norm_mismatch == 0

    # 报告与 payload 都应可用。
    assert "=== MVS 采集结果分析报告 ===" in report_text
    assert "summary" in payload
    assert "checks" in payload
