# -*- coding: utf-8 -*-

"""分析 mvs.apps.quad_capture 的采集结果（metadata.jsonl + 输出目录）。

本包由历史的单文件 `mvs.analysis.capture_run` 拆分而来：
- compute.py：统计计算与 payload 构建
- report.py：长文本报告渲染
- io.py：metadata.jsonl 读取
- models.py：结构化数据模型

对外稳定入口：
- `RunSummary`
- `analyze_output_dir()`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .compute import compute_run_analysis
from .models import RunSummary
from .report import render_report_text


def analyze_output_dir(
    *,
    output_dir: Path,
    expected_cameras: Optional[int],
    expected_fps: Optional[float],
    fps_tolerance_ratio: float,
) -> Tuple[RunSummary, str, Dict[str, Any]]:
    """分析采集输出目录并生成报告。

    Args:
        output_dir: mvs_quad_capture 的输出目录。
        expected_cameras: 期望相机数量；None 表示从数据自动推断。
        expected_fps: 期望 FPS；None 表示不做 FPS 合格判定。
        fps_tolerance_ratio: FPS 允许相对误差，例如 0.2 表示 ±20%。
    Returns:
        (summary, report_text, report_payload)
    """

    computed, payload = compute_run_analysis(
        output_dir=Path(output_dir),
        expected_cameras=expected_cameras,
        expected_fps=expected_fps,
        fps_tolerance_ratio=float(fps_tolerance_ratio),
    )

    report_text = render_report_text(computed)
    return computed.summary, report_text, payload


__all__ = [
    "RunSummary",
    "analyze_output_dir",
]
