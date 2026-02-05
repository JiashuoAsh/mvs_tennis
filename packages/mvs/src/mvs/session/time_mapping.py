# -*- coding: utf-8 -*-

"""时间戳对齐/映射（方案B）。

本模块的目标：在没有 PTP 的情况下，把相机侧时间戳（dev/event）映射到主机时间轴。

在本仓库的 captures/metadata.jsonl 中：
- frames[*].dev_timestamp：来自 SDK 帧信息的设备时间戳（相机侧计数）。
- frames[*].host_timestamp：来自 SDK 帧信息的主机时间戳（本项目实测多为 epoch 毫秒）。

方案B的落地做法：
- 对每台相机单独拟合线性映射：host_ms ≈ a * dev_ts + b
- 使用稳健拟合（简单离群剔除）获得可用的 (a,b)

注意：
- 每次“新会话”（相机重启/断开重连/程序重启）通常都需要重新拟合一次，
  因为 dev_timestamp 的起点/漂移与链路延迟都会变化。
- 本模块不引入第三方依赖（仅用标准库）。
"""

from __future__ import annotations

from collections import deque
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .metadata_io import iter_metadata_records


@dataclass(frozen=True, slots=True)
class LinearTimeMapping:
    """线性时间映射：host_ms = a * dev_ts + b。

    约定：
        - host_ms 的单位为“epoch 毫秒”（与 metadata.jsonl 中 frames[*].host_timestamp 一致）。
        - dev_ts 为相机侧计数（整数 tick），单位未知，但在同一台相机/同一会话内单调递增。
    """

    a: float
    b: float

    host_unit: str = "ms_epoch"
    dev_unit: str = "ticks"

    n_used: int = 0
    n_total: int = 0

    rms_ms: float = 0.0
    p95_ms: float = 0.0
    max_ms: float = 0.0

    dev_min: int = 0
    dev_max: int = 0
    host_min_ms: int = 0
    host_max_ms: int = 0

    def map_dev_to_host_ms(self, dev_ts: int) -> float:
        return self.a * float(int(dev_ts)) + self.b

    def map_dev_to_host_seconds(self, dev_ts: int) -> float:
        return self.map_dev_to_host_ms(dev_ts) / 1000.0


def _median_sorted(xs: list[float]) -> float:
    n = len(xs)
    if n <= 0:
        raise ValueError("median of empty list")
    return float(xs[n // 2])


def _percentile_abs_sorted(abs_sorted: list[float], q: float) -> float:
    """已排序（升序）的 abs 残差列表上取分位数。

    说明：
        这里采用简单的 nearest-rank（不插值），保证实现简单且可复现。
    """

    if not abs_sorted:
        return 0.0
    if q <= 0:
        return float(abs_sorted[0])
    if q >= 1:
        return float(abs_sorted[-1])

    # nearest-rank: ceil(q*n) - 1
    n = len(abs_sorted)
    k = int(math.ceil(float(q) * n) - 1)
    if k < 0:
        k = 0
    if k >= n:
        k = n - 1
    return float(abs_sorted[k])


def _fit_centered_ols(pairs: list[tuple[int, int]]) -> tuple[float, float]:
    """中心化最小二乘拟合：y = a*x + b。

    pairs:
        (x=dev_ts, y=host_ms)

    返回：
        (a, b)

    说明：
        - x/y 的绝对量级可能很大（host_ms ~ 1e12），直接用未中心化求和容易数值不稳定。
        - 这里先减去均值再计算协方差/方差，降低灾难性消减。
    """

    if len(pairs) < 2:
        raise ValueError("need >=2 pairs to fit")

    xs = [float(int(x)) for x, _ in pairs]
    ys = [float(int(y)) for _, y in pairs]

    x0 = math.fsum(xs) / float(len(xs))
    y0 = math.fsum(ys) / float(len(ys))

    num = 0.0
    den = 0.0
    for x, y in zip(xs, ys, strict=True):
        dx = x - x0
        dy = y - y0
        num += dx * dy
        den += dx * dx

    if den <= 0:
        raise ValueError("degenerate x variance (dev_ts not changing)")

    a = num / den
    b = y0 - a * x0
    return float(a), float(b)


def fit_dev_to_host_ms(
    pairs: list[tuple[int, int]],
    *,
    min_points: int = 30,
    max_rounds: int = 3,
    min_keep_ratio: float = 0.6,
    hard_outlier_ms: float = 50.0,
) -> LinearTimeMapping:
    """从 (dev_ts, host_ms) 配对点拟合线性映射。

    策略：
        1) OLS 拟合 (a,b)
        2) 计算残差 r = y - (a*x + b)
        3) 用 MAD 估计噪声尺度并剔除离群点，重复若干轮

    Args:
        pairs: (dev_timestamp, host_timestamp_ms)
        min_points: 最少样本点（不足则抛错）
        max_rounds: 离群剔除迭代轮数
        min_keep_ratio: 每轮至少保留的比例，避免过度剔除
        hard_outlier_ms: 硬阈值（毫秒）；剔除阈值至少不小于它

    Returns:
        LinearTimeMapping
    """

    if len(pairs) < int(min_points):
        raise ValueError(f"not enough pairs: n={len(pairs)} < min_points={int(min_points)}")

    cur = list(pairs)
    n_total = len(cur)

    for _ in range(int(max_rounds)):
        a, b = _fit_centered_ols(cur)
        residuals = [float(int(y)) - (a * float(int(x)) + b) for x, y in cur]

        abs_res = sorted(abs(r) for r in residuals)
        med_abs = _median_sorted(abs_res)

        # MAD 近似标准差：sigma ~= 1.4826 * MAD
        # 这里用 abs 残差的中位数当做 MAD（对称噪声下近似成立），足够工程上做一次离群剔除。
        sigma = 1.4826 * float(med_abs)
        cutoff = max(float(hard_outlier_ms), 6.0 * float(sigma))

        kept: list[tuple[int, int]] = []
        for (x, y), r in zip(cur, residuals, strict=True):
            if abs(float(r)) <= float(cutoff):
                kept.append((int(x), int(y)))

        # 保护：避免“剔除过猛”导致映射漂。
        if len(kept) < max(int(min_points), int(math.ceil(float(min_keep_ratio) * len(cur)))):
            break

        if len(kept) == len(cur):
            break

        cur = kept

    a, b = _fit_centered_ols(cur)
    residuals = [float(int(y)) - (a * float(int(x)) + b) for x, y in cur]
    abs_res_sorted = sorted(abs(r) for r in residuals)

    rms = math.sqrt(math.fsum(r * r for r in residuals) / float(len(residuals))) if residuals else 0.0
    p95 = _percentile_abs_sorted(abs_res_sorted, 0.95)
    mx = float(abs_res_sorted[-1]) if abs_res_sorted else 0.0

    xs_i = [int(x) for x, _ in cur]
    ys_i = [int(y) for _, y in cur]

    return LinearTimeMapping(
        a=float(a),
        b=float(b),
        n_used=int(len(cur)),
        n_total=int(n_total),
        rms_ms=float(rms),
        p95_ms=float(p95),
        max_ms=float(mx),
        dev_min=int(min(xs_i)) if xs_i else 0,
        dev_max=int(max(xs_i)) if xs_i else 0,
        host_min_ms=int(min(ys_i)) if ys_i else 0,
        host_max_ms=int(max(ys_i)) if ys_i else 0,
    )


def collect_frame_pairs_from_metadata(
    *,
    metadata_path: Path,
    max_groups: int = 0,
    serials: list[str] | None = None,
) -> dict[str, list[tuple[int, int]]]:
    """从 metadata.jsonl 收集每台相机的 (dev_ts, host_ms) 配对点。"""

    meta_path = Path(metadata_path).resolve()
    if not meta_path.exists():
        raise FileNotFoundError(str(meta_path))

    serials_set: set[str] | None = None
    if serials is not None:
        serials_norm = [str(s).strip() for s in (serials or []) if str(s).strip()]
        serials_set = set(serials_norm) if serials_norm else set()

    out: dict[str, list[tuple[int, int]]] = {}
    groups_seen = 0

    for rec in iter_metadata_records(meta_path):
        if "frames" not in rec:
            continue

        frames = rec.get("frames")
        if not isinstance(frames, list) or not frames:
            continue

        groups_seen += 1
        if int(max_groups) > 0 and groups_seen > int(max_groups):
            break

        for fr in frames:
            if not isinstance(fr, dict):
                continue

            serial = str(fr.get("serial", "")).strip()
            if not serial:
                continue
            if serials_set is not None and serial not in serials_set:
                continue

            dev_ts = fr.get("dev_timestamp")
            host_ts = fr.get("host_timestamp")
            if dev_ts is None or host_ts is None:
                continue

            try:
                x = int(dev_ts)
                y = int(host_ts)
            except Exception:
                continue

            if y <= 0:
                continue

            out.setdefault(serial, []).append((x, y))

    return out


def save_time_mappings_json(
    *,
    out_path: Path,
    mappings: dict[str, LinearTimeMapping],
    metadata_path: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """把映射参数写入 JSON，供离线/在线复用。"""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "schema": "mvs_time_mapping_v1",
        "created_at": float(time.time()),
        "host_unit": "ms_epoch",
        "dev_unit": "ticks",
        "source": {
            "kind": "frame_dev_to_host",
            "metadata_path": str(Path(metadata_path).resolve()) if metadata_path is not None else None,
        },
        "cameras": {
            serial: {
                "a": float(m.a),
                "b": float(m.b),
                "n_used": int(m.n_used),
                "n_total": int(m.n_total),
                "rms_ms": float(m.rms_ms),
                "p95_ms": float(m.p95_ms),
                "max_ms": float(m.max_ms),
                "dev_min": int(m.dev_min),
                "dev_max": int(m.dev_max),
                "host_min_ms": int(m.host_min_ms),
                "host_max_ms": int(m.host_max_ms),
            }
            for serial, m in sorted(mappings.items())
        },
    }

    if extra:
        payload["extra"] = dict(extra)

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_time_mappings_json(path: Path) -> dict[str, LinearTimeMapping]:
    """从 JSON 读取映射参数。"""

    p = Path(path).resolve()
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("time mapping json must be an object")

    schema = str(data.get("schema", ""))
    if schema != "mvs_time_mapping_v1":
        raise ValueError(f"unsupported schema: {schema}")

    host_unit = str(data.get("host_unit", ""))
    if host_unit != "ms_epoch":
        raise ValueError(f"unsupported host_unit: {host_unit} (expected ms_epoch)")

    cams = data.get("cameras")
    if not isinstance(cams, dict):
        raise ValueError("time mapping json missing 'cameras' dict")

    out: dict[str, LinearTimeMapping] = {}
    for serial, v in cams.items():
        if not isinstance(v, dict):
            continue
        a_raw = v.get("a")
        b_raw = v.get("b")
        if a_raw is None or b_raw is None:
            continue
        try:
            out[str(serial)] = LinearTimeMapping(
                a=float(a_raw),
                b=float(b_raw),
                n_used=int(v.get("n_used", 0)),
                n_total=int(v.get("n_total", 0)),
                rms_ms=float(v.get("rms_ms", 0.0)),
                p95_ms=float(v.get("p95_ms", 0.0)),
                max_ms=float(v.get("max_ms", 0.0)),
                dev_min=int(v.get("dev_min", 0)),
                dev_max=int(v.get("dev_max", 0)),
                host_min_ms=int(v.get("host_min_ms", 0)),
                host_max_ms=int(v.get("host_max_ms", 0)),
            )
        except Exception:
            continue

    return out


class OnlineDevToHostMapper:
    """在线：dev_timestamp -> host_ms 的滑窗映射器（方案B）。

    设计目标：
    - 启动期快速“可用”：收集少量组后立刻拟合一次
    - 采集中持续更新：用滑动窗口定期重拟合，吸收温漂/漂移

    说明：
    - 该映射器只依赖 frames 的 (dev_timestamp, host_timestamp_ms)，不依赖事件回调。
    - host_timestamp 在本仓库实测为 epoch 毫秒。
    - 为了避免过度计算，映射不会每组都拟合；可以通过 update_every_groups 控制频率。
    """

    def __init__(
        self,
        *,
        warmup_groups: int = 20,
        window_groups: int = 200,
        update_every_groups: int = 5,
        min_points: int = 20,
        hard_outlier_ms: float = 50.0,
    ) -> None:
        self._warmup_groups = max(0, int(warmup_groups))
        self._window_groups = max(2, int(window_groups))
        self._update_every_groups = max(1, int(update_every_groups))
        self._min_points = max(2, int(min_points))
        self._hard_outlier_ms = float(hard_outlier_ms)

        # 每台相机：最近 window_groups 个配对点（每个 group 贡献 1 个点）。
        self._pairs: dict[str, "deque[tuple[int, int]]"] = {}

        # 每台相机：最新拟合结果。
        self._mappings: dict[str, LinearTimeMapping] = {}

        self.groups_seen: int = 0
        self.last_fit_groups_seen: int = 0

    def observe_pair(self, *, serial: str, dev_ts: int, host_ms: int) -> None:
        """观察一个配对点（来自某组中的某台相机的一帧）。"""

        s = str(serial).strip()
        if not s:
            return
        try:
            x = int(dev_ts)
            y = int(host_ms)
        except Exception:
            return
        if y <= 0:
            return

        dq = self._pairs.get(s)
        if dq is None:
            dq = deque(maxlen=int(self._window_groups))
            self._pairs[s] = dq
        dq.append((x, y))

    def on_group_end(self) -> None:
        """每处理完一个 group 调用一次，用于触发拟合更新。"""

        self.groups_seen += 1
        if self.groups_seen < int(self._warmup_groups):
            return

        if (self.groups_seen - int(self.last_fit_groups_seen)) < int(self._update_every_groups):
            return

        self._refit_all()

    def _refit_all(self) -> None:
        for serial, dq in list(self._pairs.items()):
            pairs = list(dq)
            if len(pairs) < int(self._min_points):
                continue
            try:
                m = fit_dev_to_host_ms(
                    pairs,
                    min_points=int(self._min_points),
                    hard_outlier_ms=float(self._hard_outlier_ms),
                )
            except Exception:
                continue
            self._mappings[str(serial)] = m

        self.last_fit_groups_seen = int(self.groups_seen)

    def get_mapping(self, serial: str) -> LinearTimeMapping | None:
        return self._mappings.get(str(serial).strip())

    def ready_count(self, serials: Iterable[str]) -> int:
        n = 0
        for s in serials:
            m = self.get_mapping(str(s))
            if m is None:
                continue
            if int(m.n_used) >= int(self._min_points):
                n += 1
        return int(n)

    def worst_p95_ms(self, serials: Iterable[str]) -> float | None:
        vals: list[float] = []
        for s in serials:
            m = self.get_mapping(str(s))
            if m is None:
                continue
            vals.append(float(m.p95_ms))
        return max(vals) if vals else None

    def worst_rms_ms(self, serials: Iterable[str]) -> float | None:
        vals: list[float] = []
        for s in serials:
            m = self.get_mapping(str(s))
            if m is None:
                continue
            vals.append(float(m.rms_ms))
        return max(vals) if vals else None
