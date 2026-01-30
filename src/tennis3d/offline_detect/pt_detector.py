"""基于 Ultralytics YOLOv8 的 .pt 检测器（Windows/CPU 友好）。

设计目标：
- 与现有 pipeline/core 的 Detector 接口兼容：detect(img_bgr)->list[Detection]
- 只负责 .pt 模型加载与输出解析（高内聚）

注意：
- 依赖 ultralytics + torch。
- 默认强制使用 CPU（device='cpu'），避免 Windows 上 CUDA/驱动环境差异导致的问题。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tennis3d.models import Detection


@dataclass
class UltralyticsPTDetector:
    """使用 Ultralytics YOLO 加载 .pt 并输出 bbox。"""

    model_path: Path
    conf_thres: float = 0.25
    max_det: int = 20
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)
        if not self.model_path.exists():
            raise RuntimeError(f"找不到 .pt 模型文件: {self.model_path}")

        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "未安装 ultralytics，无法使用 detector=pt。\n"
                "建议：uv sync 或 uv add ultralytics\n"
                f"原始错误: {e}"
            )

        # 说明：YOLO(...) 会加载权重并构建推理图。
        self._model = YOLO(str(self.model_path))

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        if img_bgr is None or img_bgr.size == 0:
            return []

        # 说明：Ultralytics 支持直接传入 numpy.ndarray（OpenCV BGR 图像）。
        results = self._model.predict(
            source=img_bgr,
            conf=float(self.conf_thres),
            max_det=int(self.max_det),
            device=str(self.device),
            verbose=False,
            save=False,
        )

        if not results:
            return []

        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            return []

        try:
            xyxy = boxes.xyxy
            conf = boxes.conf
            cls = boxes.cls
        except Exception:
            return []

        if xyxy is None or conf is None or cls is None:
            return []

        # Boxes 里的字段通常是 torch.Tensor；统一转到 CPU numpy。
        xyxy_np = xyxy.detach().cpu().numpy()  # type: ignore[attr-defined]
        conf_np = conf.detach().cpu().numpy()  # type: ignore[attr-defined]
        cls_np = cls.detach().cpu().numpy()  # type: ignore[attr-defined]

        out: list[Detection] = []
        for i in range(int(xyxy_np.shape[0])):
            x1, y1, x2, y2 = map(float, xyxy_np[i].tolist())
            score = float(conf_np[i])
            c = int(cls_np[i])
            out.append(Detection(bbox=(x1, y1, x2, y2), score=score, cls=c))

        return out
