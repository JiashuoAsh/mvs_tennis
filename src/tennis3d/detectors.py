"""检测器适配层。

目标：
- 给在线/离线入口提供统一的 detect(img_bgr)->list[Detection] 形式。
- Windows 无法跑 RKNN 时，可以用 fake detector 保证端到端链路可跑通。

说明：
- 这里是“业务库层”，不依赖 apps；apps 只是 CLI 外壳。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from tennis3d.offline.models import Detection


class Detector(Protocol):
    """最小检测器接口。"""

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        ...


@dataclass(frozen=True)
class FakeCenterDetector:
    """调试用假检测器：永远在图像中心给一个固定大小的 bbox。"""

    bbox_size: int = 80
    score: float = 0.99
    cls: int = 0

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        h, w = int(img_bgr.shape[0]), int(img_bgr.shape[1])
        cx, cy = 0.5 * float(w), 0.5 * float(h)
        half = 0.5 * float(self.bbox_size)
        x1 = max(0.0, cx - half)
        y1 = max(0.0, cy - half)
        x2 = min(float(w - 1), cx + half)
        y2 = min(float(h - 1), cy + half)
        return [Detection(bbox=(x1, y1, x2, y2), score=float(self.score), cls=int(self.cls))]


@dataclass(frozen=True)
class GreenBallDetector:
    """调试用检测器：用 HSV 颜色阈值找绿色球并输出 bbox。

    设计目标：
    - 不依赖深度学习/RKNN，Windows/Linux 都能跑。
    - 配合仓库内的 sample_sequence（生成的绿色圆点）可一键跑通离线 pipeline。
    """

    # HSV 范围：OpenCV H∈[0,179]
    h_min: int = 35
    h_max: int = 85
    s_min: int = 80
    v_min: int = 80
    min_area: int = 20
    score: float = 0.9
    cls: int = 0

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        if img_bgr is None or img_bgr.size == 0:
            return []

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([int(self.h_min), int(self.s_min), int(self.v_min)], dtype=np.uint8)
        upper = np.array([int(self.h_max), 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # 降噪：小图也尽量稳
        mask = cv2.medianBlur(mask, 5)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return []

        best = None
        best_area = -1.0
        for c in cnts:
            area = float(cv2.contourArea(c))
            if area > best_area:
                best_area = area
                best = c

        if best is None or best_area < float(self.min_area):
            return []

        x, y, w, h = cv2.boundingRect(best)
        x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)
        return [Detection(bbox=(x1, y1, x2, y2), score=float(self.score), cls=int(self.cls))]


def create_detector(
    *,
    name: str,
    model_path: Path | None = None,
    conf_thres: float = 0.25,
) -> Detector:
    """创建检测器。

    Args:
        name: fake/color/rknn/pt。
        model_path: 模型路径（name=rknn 或 pt 时必填）。
        conf_thres: 最低置信度阈值。

    Returns:
        Detector 实例。
    """

    name = str(name).strip().lower()
    if name == "fake":
        return FakeCenterDetector()

    if name in {"color", "green"}:
        return GreenBallDetector()

    if name == "rknn":
        if model_path is None:
            raise ValueError("detector=rknn requires --model")
        from tennis3d.offline.detector import TennisDetector

        # 重要：TennisDetector 内部会对输入做 letterbox 到 input_size。
        # pipeline/core 与 triangulation 使用的是“原图像素坐标系”，这里必须把 bbox 映射回原图。
        # 说明：不要在 TennisDetector.detect() 内做 scale back，因为 offline/pipeline.py 会单独处理。
        from tennis3d.offline.preprocess import scale_detections_back

        class _RKNNScaleBackDetector:
            def __init__(self, inner: TennisDetector):
                self._inner = inner

            def detect(self, img_bgr: np.ndarray) -> list[Detection]:
                dets_in = self._inner.detect(img_bgr)
                if not dets_in:
                    return []
                return scale_detections_back(
                    list(dets_in),
                    orig_shape=img_bgr.shape[:2],
                    input_size=int(self._inner.input_size),
                )

        inner = TennisDetector(model_path=Path(model_path), conf_thres=float(conf_thres))
        return _RKNNScaleBackDetector(inner)

    if name in {"pt", "yolo", "yolov8", "ultralytics"}:
        if model_path is None:
            raise ValueError("detector=pt requires --model")
        from tennis3d.offline.pt_detector import UltralyticsPTDetector

        return UltralyticsPTDetector(model_path=Path(model_path), conf_thres=float(conf_thres), device="cpu")

    raise ValueError(f"unknown detector: {name} (expected: fake|color|rknn|pt)")
