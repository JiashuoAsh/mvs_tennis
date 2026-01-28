"""对齐策略抽象（SyncAligner）。

说明：
- pipeline 的输入通常已经是“按组对齐”的（在线由组包器完成；离线由 metadata.jsonl 的 frames 记录完成）。
- 这里提供一个可插拔的 aligner 接口，用于未来扩展更复杂的场景：
  - 按时间戳 tolerance 做软对齐
  - 缺帧策略（丢弃/缓存/补帧）
  - 多球/多目标的关联
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class SyncAligner(Protocol):
    """同步/对齐策略接口。

    返回 None 表示丢弃该组（例如不满足对齐条件）。
    """

    def align(
        self,
        meta: dict[str, Any],
        images_by_camera: dict[str, np.ndarray],
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
        ...


@dataclass(frozen=True)
class PassthroughAligner:
    """默认对齐器：不做任何处理。"""

    def align(
        self,
        meta: dict[str, Any],
        images_by_camera: dict[str, np.ndarray],
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
        return meta, images_by_camera
