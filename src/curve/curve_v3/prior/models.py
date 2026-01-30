"""curve_v3 的数据驱动先验（方案3）。

curve.md §6 的核心思想是：
- 保留物理结构（触地映射 + 弹道传播）。
- 数据驱动只用于增强候选 (e, kt) 的先验分布，即提供 w_m^0。

本模块定义一个小而可插拔的接口，用于输出候选权重先验。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class PriorFeatures:
    """用于预测 (e, kt) 先验的最小特征集合。

    属性:
        speed: |v-|（m/s）。
        incident_angle: 速度向量与 -y 轴之间的夹角（弧度）。
            角度越大通常意味着越“擦地”的入射。
    """

    speed: float
    incident_angle: float


def features_from_v_minus(v_minus: np.ndarray) -> PriorFeatures:
    """从 v_minus 构造 PriorFeatures。"""

    v = np.asarray(v_minus, dtype=float).reshape(3)
    speed = float(np.linalg.norm(v))

    # 地面法向为 +y；入射段通常 vy < 0。
    vt = np.array([v[0], 0.0, v[2]], dtype=float)
    vt_norm = float(np.linalg.norm(vt))
    denom = max(float(-v[1]), 1e-6)
    incident_angle = float(math.atan2(vt_norm, denom))

    return PriorFeatures(speed=speed, incident_angle=incident_angle)


class PriorModel(Protocol):
    """候选先验权重接口。"""

    def predict_candidate_weights(
        self,
        features: PriorFeatures,
        *,
        e_bins: Sequence[float],
        kt_bins: Sequence[float],
    ) -> np.ndarray:
        """返回与 (e_bins x kt_bins) 对齐的先验权重。

        期望的输出顺序：
            for e in e_bins:
                for kt in kt_bins:
                    yield weight

        Returns:
            形状为 (len(e_bins) * len(kt_bins),) 的 1D numpy 数组，且和为 1。
        """

        ...


class UniformPriorModel:
    """安全兜底：所有候选均匀分布。"""

    def predict_candidate_weights(
        self,
        features: PriorFeatures,  # noqa: ARG002
        *,
        e_bins: Sequence[float],
        kt_bins: Sequence[float],
    ) -> np.ndarray:
        m = int(len(e_bins) * len(kt_bins))
        if m <= 0:
            return np.zeros((0,), dtype=float)
        return np.full((m,), 1.0 / float(m), dtype=float)


@dataclass(frozen=True)
class RbfSamplePriorModel:
    """简单的非参数先验：用 RBF 对带标签样本做加权平均。

    这里刻意保持极简：少量样本也能工作，样本越多自然越稳。

    模型存储：
    - sample_features: shape (N,2) -> [speed, incident_angle]
    - sample_weights: shape (N,M) -> 候选网格上的 soft label
    """

    sample_features: np.ndarray
    sample_weights: np.ndarray
    rbf_sigma: float = 1.0

    def __post_init__(self) -> None:
        sf = np.asarray(self.sample_features, dtype=float)
        sw = np.asarray(self.sample_weights, dtype=float)
        if sf.ndim != 2 or sf.shape[1] != 2:
            raise ValueError("sample_features must have shape (N,2)")
        if sw.ndim != 2 or sw.shape[0] != sf.shape[0]:
            raise ValueError("sample_weights must have shape (N,M) matching features")

    def predict_candidate_weights(
        self,
        features: PriorFeatures,
        *,
        e_bins: Sequence[float],  # noqa: ARG002
        kt_bins: Sequence[float],  # noqa: ARG002
    ) -> np.ndarray:
        m = int(self.sample_weights.shape[1])
        if m <= 0:
            return np.zeros((0,), dtype=float)

        q = np.array([float(features.speed), float(features.incident_angle)], dtype=float)
        sigma2 = max(float(self.rbf_sigma) ** 2, 1e-9)

        d = self.sample_features - q[None, :]
        d2 = np.sum(d * d, axis=1)
        k = np.exp(-0.5 * d2 / sigma2)

        w = k[:, None] * self.sample_weights
        out = np.sum(w, axis=0)
        s = float(np.sum(out))
        if s <= 0:
            return np.full((m,), 1.0 / float(m), dtype=float)
        return out / s

    @staticmethod
    def from_json(path: str | Path) -> tuple[Sequence[float], Sequence[float], "RbfSamplePriorModel"]:
        """从 JSON 加载先验模型。

        Returns:
            (e_bins, kt_bins, model)
        """

        path = Path(path)
        obj = json.loads(path.read_text(encoding="utf-8"))

        e_bins = obj.get("e_bins")
        kt_bins = obj.get("kt_bins")
        samples = obj.get("samples")
        rbf_sigma = float(obj.get("rbf_sigma", 1.0))

        if not isinstance(e_bins, list) or not isinstance(kt_bins, list) or not isinstance(samples, list):
            raise ValueError("Invalid prior JSON: expected keys e_bins, kt_bins, samples")

        feats: list[list[float]] = []
        weights: list[list[float]] = []
        for s in samples:
            if not isinstance(s, dict):
                continue
            f = s.get("feature")
            w = s.get("weights")
            if not (isinstance(f, list) and len(f) == 2 and isinstance(w, list)):
                continue
            feats.append([float(f[0]), float(f[1])])
            weights.append([float(x) for x in w])

        sf = np.asarray(feats, dtype=float)
        sw = np.asarray(weights, dtype=float)
        model = RbfSamplePriorModel(sample_features=sf, sample_weights=sw, rbf_sigma=rbf_sigma)
        return [float(x) for x in e_bins], [float(x) for x in kt_bins], model

    def to_json(
        self,
        *,
        path: str | Path,
        e_bins: Sequence[float],
        kt_bins: Sequence[float],
    ) -> None:
        """把模型保存为 JSON。"""

        path = Path(path)
        obj = {
            "e_bins": [float(x) for x in e_bins],
            "kt_bins": [float(x) for x in kt_bins],
            "rbf_sigma": float(self.rbf_sigma),
            "samples": [
                {
                    "feature": [float(self.sample_features[i, 0]), float(self.sample_features[i, 1])],
                    "weights": [float(x) for x in self.sample_weights[i, :].tolist()],
                }
                for i in range(int(self.sample_features.shape[0]))
            ],
        }
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
