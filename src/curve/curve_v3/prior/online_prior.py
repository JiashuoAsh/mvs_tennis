"""在线参数沉淀：把 posterior 信息回灌到 prior。

`docs/curve.md` 的 v1.1 方案强调两点：
1) 第一阶段 prior 用少量离散候选覆盖不可辨识因素（避免灾难性偏差）。
2) 第二阶段 posterior 用很少点数快速校正，并将后验结果回灌，让同场地/球况下
    prior 逐步收敛（走廊变窄，名义误差变小）。

本模块实现一个“最小可用”的在线权重池：
- 只沉淀到离散候选权重（而不是学习连续参数或端到端曲线）。
- 默认用 EMA（指数滑动平均）更新：w <- (1-α)w + α w_post。
- 可选持久化为 JSON，便于跨进程/跨运行复用。

设计取舍：
- 不引入外部依赖（仅 NumPy + 标准库）。
- 若配置/候选网格发生变化（bins 不一致），会拒绝加载旧权重，避免静默错配。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from curve_v3.types import Candidate


def _as_float_list(xs: Sequence[float]) -> list[float]:
    """把可迭代对象转成 Python float 列表。"""

    return [float(x) for x in xs]


def _float_tuple_key(e: float, kt: float, ang: float, ndigits: int = 12) -> tuple[float, float, float]:
    """构造稳定的浮点三元组 key。

    说明：用 round 做一个稳定 key，避免浮点 JSON 读回后 1e-16 级别的差异导致 key 不一致。
    """

    return (
        float(round(float(e), ndigits)),
        float(round(float(kt), ndigits)),
        float(round(float(ang), ndigits)),
    )


@dataclass
class OnlinePriorWeights:
    """离散候选权重的在线沉淀池（全维：e * kt * angle）。

    Args:
        e_bins: e 候选分档。
        kt_bins: kt 候选分档。
        angle_bins_rad: 切向偏转角分档（弧度）。
        weights: 全网格权重，shape=(M,)，且和为 1。
        ema_alpha: EMA 更新系数 α，范围 (0, 1]。
        eps: 避免 0 权重导致“永远学不回来”的下限。
        num_updates: 已更新次数（用于调试/诊断）。
    """

    e_bins: tuple[float, ...]
    kt_bins: tuple[float, ...]
    angle_bins_rad: tuple[float, ...]
    weights: np.ndarray
    ema_alpha: float = 0.05
    eps: float = 1e-8
    num_updates: int = 0

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=float).reshape(-1)
        m = int(len(self.e_bins) * len(self.kt_bins) * len(self.angle_bins_rad))
        if w.size != m:
            raise ValueError(f"weights size mismatch: expected {m}, got {w.size}")

        w = np.maximum(w, float(self.eps))
        s = float(np.sum(w))
        if s <= 0.0:
            w = np.full((m,), 1.0 / float(m), dtype=float)
        else:
            w = w / s

        object.__setattr__(self, "weights", w)

    @property
    def m(self) -> int:
        """全网格候选数量 M。"""

        return int(len(self.e_bins) * len(self.kt_bins) * len(self.angle_bins_rad))

    def _grid_index(self, *, e: float, kt: float, ang: float) -> int | None:
        key = _float_tuple_key(e, kt, ang)

        # 构造一张小 mapping，避免每次 update 都 O(M) 查找。
        # 由于 bins 很小（通常 3*3*(1~3)），用 dict 很轻量。
        if not hasattr(self, "_idx_map"):
            idx_map: dict[tuple[float, float, float], int] = {}
            idx = 0
            for ee in self.e_bins:
                for kk in self.kt_bins:
                    for aa in self.angle_bins_rad:
                        idx_map[_float_tuple_key(ee, kk, aa)] = idx
                        idx += 1
            object.__setattr__(self, "_idx_map", idx_map)

        idx_map2 = getattr(self, "_idx_map")
        return idx_map2.get(key)

    def apply_to_base_weights(self, *, base_weights_ekt: np.ndarray) -> np.ndarray:
        """把在线沉淀权重乘到 (e*kt) 的 base prior 上，并展开到全维。

        注意：这里默认 angle 的 base prior 为均匀分配；在线沉淀再对 angle 做偏置。

        Args:
            base_weights_ekt: shape=(len(e_bins)*len(kt_bins),)，且应为非负。

        Returns:
            shape=(M,) 的全维候选权重（已归一化）。
        """

        base = np.asarray(base_weights_ekt, dtype=float).reshape(-1)
        m0 = int(len(self.e_bins) * len(self.kt_bins))
        if base.size != m0:
            raise ValueError(f"base_weights_ekt size mismatch: expected {m0}, got {base.size}")

        base = np.maximum(base, 0.0)
        s0 = float(np.sum(base))
        if s0 <= 0.0:
            base = np.full((m0,), 1.0 / float(m0), dtype=float)
        else:
            base = base / s0

        # 展开到 e*kt*angle，并与在线权重相乘再归一化。
        k = int(len(self.angle_bins_rad))
        base_full = np.repeat(base / float(k), repeats=k)

        w = np.maximum(np.asarray(self.weights, dtype=float).reshape(-1), float(self.eps))
        combined = base_full * w
        s = float(np.sum(combined))
        if s <= 0.0:
            return np.full((self.m,), 1.0 / float(self.m), dtype=float)
        return combined / s

    def update_from_candidates(self, *, candidates: Sequence[Candidate]) -> None:
        """用当前候选（通常已融合 posterior 的权重）更新在线权重。

        Args:
            candidates: 候选列表，要求字段 (e, kt, kt_angle_rad, weight) 可用。
        """

        if not candidates:
            return

        w_post = np.zeros((self.m,), dtype=float)
        for c in candidates:
            idx = self._grid_index(
                e=float(c.e),
                kt=float(c.kt),
                ang=float(getattr(c, "kt_angle_rad", 0.0)),
            )
            if idx is None:
                continue
            w_post[idx] += float(c.weight)

        s = float(np.sum(w_post))
        if s <= 0.0:
            return
        w_post = w_post / s

        alpha = float(self.ema_alpha)
        alpha = min(max(alpha, 1e-6), 1.0)

        w_old = np.asarray(self.weights, dtype=float).reshape(-1)
        w_new = (1.0 - alpha) * w_old + alpha * w_post
        w_new = np.maximum(w_new, float(self.eps))
        w_new = w_new / max(float(np.sum(w_new)), 1e-12)

        object.__setattr__(self, "weights", w_new)
        object.__setattr__(self, "num_updates", int(self.num_updates) + 1)

    def to_json_obj(self) -> dict:
        """序列化为 JSON 兼容的 dict。"""

        return {
            "e_bins": _as_float_list(self.e_bins),
            "kt_bins": _as_float_list(self.kt_bins),
            "angle_bins_rad": _as_float_list(self.angle_bins_rad),
            "weights": [float(x) for x in np.asarray(self.weights, dtype=float).reshape(-1).tolist()],
            "ema_alpha": float(self.ema_alpha),
            "eps": float(self.eps),
            "num_updates": int(self.num_updates),
        }

    def save_json(self, path: str | Path) -> None:
        """保存为 JSON 文件。"""

        path = Path(path)
        path.write_text(json.dumps(self.to_json_obj(), ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load_json(path: str | Path) -> "OnlinePriorWeights":
        """从 JSON 文件加载。"""

        path = Path(path)
        obj = json.loads(path.read_text(encoding="utf-8"))

        e_bins = tuple(float(x) for x in obj.get("e_bins", []))
        kt_bins = tuple(float(x) for x in obj.get("kt_bins", []))
        ang_bins = tuple(float(x) for x in obj.get("angle_bins_rad", []))
        weights = np.asarray(obj.get("weights", []), dtype=float).reshape(-1)

        ema_alpha = float(obj.get("ema_alpha", 0.05))
        eps = float(obj.get("eps", 1e-8))
        num_updates = int(obj.get("num_updates", 0))

        return OnlinePriorWeights(
            e_bins=e_bins,
            kt_bins=kt_bins,
            angle_bins_rad=ang_bins,
            weights=weights,
            ema_alpha=ema_alpha,
            eps=eps,
            num_updates=num_updates,
        )

    @staticmethod
    def create_uniform(
        *,
        e_bins: Sequence[float],
        kt_bins: Sequence[float],
        angle_bins_rad: Sequence[float],
        ema_alpha: float = 0.05,
        eps: float = 1e-8,
    ) -> "OnlinePriorWeights":
        """创建均匀权重池。"""

        e_bins_t = tuple(float(x) for x in e_bins)
        kt_bins_t = tuple(float(x) for x in kt_bins)
        ang_bins_t = tuple(float(x) for x in angle_bins_rad)
        m = int(len(e_bins_t) * len(kt_bins_t) * len(ang_bins_t))
        if m <= 0:
            raise ValueError("Empty candidate grid")
        w = np.full((m,), 1.0 / float(m), dtype=float)
        return OnlinePriorWeights(
            e_bins=e_bins_t,
            kt_bins=kt_bins_t,
            angle_bins_rad=ang_bins_t,
            weights=w,
            ema_alpha=float(ema_alpha),
            eps=float(eps),
            num_updates=0,
        )


def load_or_create_online_prior(
    *,
    path: str | Path | None,
    e_bins: Sequence[float],
    kt_bins: Sequence[float],
    angle_bins_rad: Sequence[float],
    ema_alpha: float,
    eps: float,
) -> OnlinePriorWeights:
    """从路径加载在线权重池；失败时创建均匀分布。

    注意：若 bins 不一致则拒绝加载，避免错配。
    """

    if path is None:
        return OnlinePriorWeights.create_uniform(
            e_bins=e_bins,
            kt_bins=kt_bins,
            angle_bins_rad=angle_bins_rad,
            ema_alpha=ema_alpha,
            eps=eps,
        )

    p = Path(path)
    if not p.exists():
        return OnlinePriorWeights.create_uniform(
            e_bins=e_bins,
            kt_bins=kt_bins,
            angle_bins_rad=angle_bins_rad,
            ema_alpha=ema_alpha,
            eps=eps,
        )

    try:
        inst = OnlinePriorWeights.load_json(p)
    except Exception:
        return OnlinePriorWeights.create_uniform(
            e_bins=e_bins,
            kt_bins=kt_bins,
            angle_bins_rad=angle_bins_rad,
            ema_alpha=ema_alpha,
            eps=eps,
        )

    want_e = tuple(float(x) for x in e_bins)
    want_k = tuple(float(x) for x in kt_bins)
    want_a = tuple(float(x) for x in angle_bins_rad)
    if inst.e_bins != want_e or inst.kt_bins != want_k or inst.angle_bins_rad != want_a:
        # bins 不一致，直接丢弃旧文件，避免静默错配。
        return OnlinePriorWeights.create_uniform(
            e_bins=e_bins,
            kt_bins=kt_bins,
            angle_bins_rad=angle_bins_rad,
            ema_alpha=ema_alpha,
            eps=eps,
        )

    # 用最新配置覆盖 EMA 参数（便于热更新）。
    object.__setattr__(inst, "ema_alpha", float(ema_alpha))
    object.__setattr__(inst, "eps", float(eps))
    return inst
