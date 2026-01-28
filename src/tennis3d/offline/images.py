"""图片遍历与读取（高内聚：只管文件系统与 cv2 读图）。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tennis3d.offline.models import ImageItem
from tennis3d.offline.timestamps import parse_timestamp_from_name


def list_images(camera_name: str, folder: Path) -> list[ImageItem]:
    """读取目录下所有图片并按时间戳排序。"""

    items: list[ImageItem] = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".bmp", ".jpg", ".jpeg", ".png"}:
            continue

        try:
            ts_str, ts_epoch_ms = parse_timestamp_from_name(p.name)
        except ValueError:
            continue

        items.append(
            ImageItem(
                camera=camera_name,
                path=p,
                ts_str=ts_str,
                ts_epoch_ms=ts_epoch_ms,
            )
        )

    items.sort(key=lambda x: x.ts_epoch_ms)
    return items


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"读取图片失败: {path}")
    return img
