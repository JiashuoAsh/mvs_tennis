"""定位候选的质量评分（内部实现）。

说明：
- 该文件只负责“打分规则”，不负责候选生成/几何匹配。
- 之所以从 localize.py 拆出，是为了让 localize.py 更聚焦于算法流程编排。
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from tennis3d.geometry.triangulation import ReprojectionError
from tennis3d.models import Detection


def compute_quality(*, dets: Mapping[str, Detection], errs: Sequence[ReprojectionError]) -> float:
    """综合质量评分。

    经验规则（与旧实现保持一致）：
    - 视角数越多越好（强权重）
    - 重投影误差越小越好（负权重）
    - 检测置信度越高越好（正权重，但不应压过几何一致性）
    """

    view_cnt = float(len(dets))
    score_sum = float(sum(float(d.score) for d in dets.values()))
    err_list = [float(e.error_px) for e in errs]
    med_err = float(np.median(np.asarray(err_list, dtype=np.float64)))
    max_err = float(max(err_list))

    # 权重是经验值：优先保证“多视角+低误差”，score 只做次级排序。
    return view_cnt * 1000.0 + score_sum * 10.0 - med_err * 50.0 - max_err * 10.0
