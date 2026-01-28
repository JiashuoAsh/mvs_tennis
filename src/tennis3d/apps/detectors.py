"""兼容性薄封装：apps 层导出 detector 工厂。

说明：
- 实际实现放在 tennis3d.detectors，apps 仅做 CLI 组装。
"""

from tennis3d.detectors import Detector, FakeCenterDetector, create_detector

__all__ = ["Detector", "FakeCenterDetector", "create_detector"]
