"""mvs.analysis：采集数据的离线分析与诊断。

说明：
- 这里放“纯分析/统计”逻辑，尽量不依赖相机 SDK 与硬件。
- CLI 入口放在 mvs.apps（若需要 console_scripts）。
"""

from mvs.analysis.capture_run import RunSummary, analyze_output_dir

__all__ = [
    "RunSummary",
    "analyze_output_dir",
]
