"""定位候选的后处理（内部实现）。

包含：
- 3D 去重（3D-NMS）：按质量排序后，抑制空间距离过近的重复解。
- 冲突消解：同一相机同一 detection 不能被多个 3D 球复用。

说明：
- 该模块刻意不 import `BallLocalization`，用前向引用避免循环依赖。
- 只依赖候选对象具备 `quality/X_w/detection_indices` 这几个字段即可。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    # 仅用于类型检查，避免运行时循环依赖。
    from tennis3d.localization.localize import BallLocalization


def deduplicate_by_3d_distance(
    cands: Sequence["BallLocalization"],
    *,
    merge_dist_m: float,
) -> list["BallLocalization"]:
    """对 3D 解做去重（3D-NMS）。

    Args:
        cands: 候选列表（元素需具备 X_w 与 quality 字段）。
        merge_dist_m: 合并阈值（米）。<=0 时不做去重。

    Returns:
        去重后的候选列表（保持“质量优先”的选择策略）。
    """

    merge_dist_m = float(merge_dist_m)
    if merge_dist_m <= 0:
        return list(cands)

    ordered = sorted(cands, key=lambda c: float(c.quality), reverse=True)
    kept: list["BallLocalization"] = []

    for c in ordered:
        ok = True
        for k in kept:
            d = float(np.linalg.norm(np.asarray(c.X_w).reshape(3) - np.asarray(k.X_w).reshape(3)))
            if d < merge_dist_m:
                ok = False
                break
        if ok:
            kept.append(c)

    return kept


def select_non_conflicting(cands: Sequence["BallLocalization"]) -> list["BallLocalization"]:
    """冲突消解：同一相机同一检测不能被多个 3D 球复用。"""

    ordered = sorted(cands, key=lambda c: float(c.quality), reverse=True)
    used_keys: set[tuple[str, int]] = set()
    selected: list["BallLocalization"] = []

    for c in ordered:
        keys = {(str(cam), int(idx)) for cam, idx in c.detection_indices.items()}
        if keys & used_keys:
            continue
        selected.append(c)
        used_keys |= keys

    return selected
