"""三路图片时间对齐（高内聚：只做对齐，不做推理/IO）。"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Optional

from tennis3d.offline.models import ImageItem, MatchedTriple


@dataclass(frozen=True)
class TimeIndex:
    items: list[ImageItem]
    ts_epoch_ms: list[int]

    @classmethod
    def from_items(cls, items: list[ImageItem]) -> "TimeIndex":
        return cls(items=items, ts_epoch_ms=[it.ts_epoch_ms for it in items])


def nearest_within(index: TimeIndex, target_ts_epoch_ms: int, tolerance_ms: int) -> Optional[ImageItem]:
    """在 index 中找与 target 最接近且差值 <= tolerance 的元素。"""

    if not index.items:
        return None

    idx = bisect.bisect_left(index.ts_epoch_ms, target_ts_epoch_ms)

    candidates: list[ImageItem] = []
    if 0 <= idx < len(index.items):
        candidates.append(index.items[idx])
    if idx - 1 >= 0:
        candidates.append(index.items[idx - 1])

    best: Optional[ImageItem] = None
    best_delta = 10**18
    for c in candidates:
        delta = abs(c.ts_epoch_ms - target_ts_epoch_ms)
        if delta < best_delta:
            best = c
            best_delta = delta

    if best is None or best_delta > tolerance_ms:
        return None
    return best


def match_triples(
    base_items: list[ImageItem],
    cam2_items: list[ImageItem],
    cam3_items: list[ImageItem],
    tolerance_ms: int,
) -> list[MatchedTriple]:
    """以 base_items 为基准，对齐另外两路。"""

    idx2 = TimeIndex.from_items(cam2_items)
    idx3 = TimeIndex.from_items(cam3_items)

    triples: list[MatchedTriple] = []
    for base in base_items:
        it2 = nearest_within(idx2, base.ts_epoch_ms, tolerance_ms)
        it3 = nearest_within(idx3, base.ts_epoch_ms, tolerance_ms)
        triples.append(MatchedTriple(base=base, cam2=it2, cam3=it3))
    return triples
