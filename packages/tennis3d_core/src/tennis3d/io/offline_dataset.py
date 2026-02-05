"""离线数据集读取。

职责：
- 读取 captures 目录下的 metadata.jsonl
- 产出 pipeline 统一格式：(meta, images_by_camera)

说明：
- 该模块只做 IO，不做同步/检测/三角化。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from tennis3d.pipeline import iter_capture_image_groups


@dataclass(frozen=True)
class OfflineDataset:
    """离线 captures 数据集。

    Args:
        captures_dir: captures 目录，必须包含 metadata.jsonl。
    """

    captures_dir: Path

    def iter_groups(self, *, max_groups: int = 0) -> Iterator[tuple[dict[str, Any], dict[str, np.ndarray]]]:
        """迭代同步组。

        Args:
            max_groups: 最多输出多少组（0 表示不限）。

        Yields:
            (meta, images_by_camera_serial)
        """

        yield from iter_capture_image_groups(captures_dir=Path(self.captures_dir), max_groups=int(max_groups))
