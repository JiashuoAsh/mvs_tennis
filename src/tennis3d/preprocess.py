"""图像预处理与可视化（高内聚：图像几何变换/坐标映射/画框）。

说明：
- 这里的函数会被 offline 推理、通用 detector 适配、以及工具脚本共同使用。
- 之所以放在顶层模块而不是 offline 子包下，是为了避免目录语义误导：
  letterbox/坐标映射属于通用视觉工具，不是离线专用。
"""

from __future__ import annotations

import cv2
import numpy as np

from tennis3d.models import Detection


def letterbox(
    img: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """YOLO 常用 letterbox。

    Returns:
        resized_img: letterbox 后图像
        scale: 缩放比例
        pad: (pad_left, pad_top) 左上角 padding（整图对称 padding 的一半）
    """

    h, w = img.shape[:2]
    new_w, new_h = new_shape

    scale = min(new_w / w, new_h / h)
    resized_w, resized_h = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_w = new_w - resized_w
    pad_h = new_h - resized_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2

    out = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_h - pad_top,
        pad_left,
        pad_w - pad_left,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )

    return out, scale, (pad_left, pad_top)


def scale_detections_back(
    dets: list[Detection],
    orig_shape: tuple[int, int],
    input_size: int,
) -> list[Detection]:
    """将 letterbox 输入坐标系下的检测框，映射回原图坐标系。"""

    h, w = orig_shape
    dummy = np.zeros((h, w, 3), dtype=np.uint8)
    _, scale, (pad_left, pad_top) = letterbox(dummy, (input_size, input_size))

    out: list[Detection] = []
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        x1 = (x1 - pad_left) / scale
        y1 = (y1 - pad_top) / scale
        x2 = (x2 - pad_left) / scale
        y2 = (y2 - pad_top) / scale

        x1 = float(np.clip(x1, 0, w - 1))
        y1 = float(np.clip(y1, 0, h - 1))
        x2 = float(np.clip(x2, 0, w - 1))
        y2 = float(np.clip(y2, 0, h - 1))

        out.append(Detection(bbox=(x1, y1, x2, y2), score=d.score, cls=d.cls))

    return out


def draw_detections(img: np.ndarray, dets: list[Detection]) -> np.ndarray:
    """在图像上画 bbox 与中心点（用于调试/可视化）。"""

    vis = img.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d.bbox)
        cx, cy = map(int, d.center)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(
            vis,
            f"{d.score:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            lineType=cv2.LINE_AA,
        )
    return vis
