"""tennis3d 公共数据模型（高内聚：只放数据结构定义）。

说明：
- 这些数据结构会被 online/offline/pipeline/tools/tests 共同使用。
- 之所以放在顶层模块而不是 offline 子包下，是为了避免“目录语义误导”：
  例如 Detection 并不属于离线专用类型。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ImageItem:
    """单张图片的元信息。"""

    camera: str
    path: Path
    ts_str: str
    ts_epoch_ms: int


@dataclass(frozen=True)
class MatchedTriple:
    """以 base 为基准，对齐的三路图片。"""

    base: ImageItem
    cam2: Optional[ImageItem]
    cam3: Optional[ImageItem]


@dataclass(frozen=True)
class Detection:
    """单个检测结果。

    bbox 坐标为原图像素坐标系下的 (x1, y1, x2, y2)。
    """

    bbox: tuple[float, float, float, float]
    score: float
    cls: int

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))
