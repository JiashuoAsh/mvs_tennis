# -*- coding: utf-8 -*-

"""时间映射（dev_timestamp_mapping）组内对齐质量报告（纯逻辑）。

背景：
- online/offline 输出 JSONL 会在每条记录 meta 中携带若干时间映射诊断字段。
- `tools/time_mapping_report.py` 需要对这些字段做统计汇总。

本模块把“统计计算”下沉到 src，避免核心逻辑散落在 tools 里。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from tennis3d.pipeline.time_utils import median_float


@dataclass(frozen=True, slots=True)
class SpreadSummary:
    """一组标量的分位数摘要（nearest-rank，不插值）。"""

    p50: float | None
    p95: float | None
    max: float | None


@dataclass(frozen=True, slots=True)
class CameraDeltaSummary:
    """每相机相对组内中位数偏差的摘要。"""

    median: float | None
    abs_p95: float | None


@dataclass(frozen=True, slots=True)
class TimeMappingReport:
    groups_used: int
    raw_host_spread_ms: SpreadSummary | None
    mapped_spread_ms: SpreadSummary | None
    mapped_delta_by_camera: dict[str, CameraDeltaSummary]


def _percentile_nearest_rank(sorted_values: list[float], q: float) -> float | None:
    """nearest-rank 分位数（不插值），q in [0,1]。"""

    if not sorted_values:
        return None
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])

    n = len(sorted_values)
    k = int(math.ceil(float(q) * float(n)) - 1)
    k = max(0, min(n - 1, k))
    return float(sorted_values[k])


def iter_jsonl_dicts(path: Path) -> Iterator[dict[str, Any]]:
    """稳健读取 JSONL（跳过空行/坏行/非 dict 顶层）。"""

    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _as_float_map(x: Any) -> dict[str, float]:
    if not isinstance(x, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in x.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            continue
    return out


def build_time_mapping_report(*, records: Iterable[dict[str, Any]], max_groups: int = 0) -> TimeMappingReport:
    """从记录流构建时间映射报告。"""

    raw_spread_ms: list[float] = []
    mapped_spread_ms: list[float] = []
    mapped_deltas_by_camera: dict[str, list[float]] = {}

    seen = 0
    for rec in records:
        seen += 1
        if int(max_groups) > 0 and seen > int(max_groups):
            break

        raw_spread = rec.get("time_mapping_host_ms_spread_ms")
        if raw_spread is not None:
            try:
                raw_spread_ms.append(float(raw_spread))
            except Exception:
                pass

        mapped_spread = rec.get("time_mapping_mapped_host_ms_spread_ms")
        if mapped_spread is not None:
            try:
                mapped_spread_ms.append(float(mapped_spread))
            except Exception:
                pass

        delta_by_cam = _as_float_map(rec.get("time_mapping_mapped_host_ms_delta_to_median_by_camera"))

        # 兼容：如果没有 delta 字段，则用 mapped_host_ms_by_camera 现算一次。
        if len(delta_by_cam) < 2:
            mapped = _as_float_map(rec.get("time_mapping_mapped_host_ms_by_camera"))
            if len(mapped) >= 2:
                med = median_float(list(mapped.values()))
                if med is not None:
                    delta_by_cam = {cam: float(t) - float(med) for cam, t in mapped.items()}

        for cam, d in delta_by_cam.items():
            mapped_deltas_by_camera.setdefault(cam, []).append(float(d))

    groups_used = max(len(mapped_spread_ms), max((len(v) for v in mapped_deltas_by_camera.values()), default=0))

    raw_summary = None
    if raw_spread_ms:
        xs = sorted(raw_spread_ms)
        raw_summary = SpreadSummary(
            p50=_percentile_nearest_rank(xs, 0.50),
            p95=_percentile_nearest_rank(xs, 0.95),
            max=float(xs[-1]),
        )

    mapped_summary = None
    if mapped_spread_ms:
        xs = sorted(mapped_spread_ms)
        mapped_summary = SpreadSummary(
            p50=_percentile_nearest_rank(xs, 0.50),
            p95=_percentile_nearest_rank(xs, 0.95),
            max=float(xs[-1]),
        )

    per_cam: dict[str, CameraDeltaSummary] = {}
    for cam in sorted(mapped_deltas_by_camera.keys()):
        ds = sorted(mapped_deltas_by_camera[cam])
        if not ds:
            continue
        abs_ds = sorted(abs(x) for x in ds)
        per_cam[str(cam)] = CameraDeltaSummary(
            median=_percentile_nearest_rank(ds, 0.50),
            abs_p95=_percentile_nearest_rank(abs_ds, 0.95),
        )

    return TimeMappingReport(
        groups_used=int(groups_used),
        raw_host_spread_ms=raw_summary,
        mapped_spread_ms=mapped_summary,
        mapped_delta_by_camera=per_cam,
    )


def build_time_mapping_report_from_jsonl(*, jsonl_path: Path, max_groups: int = 0) -> TimeMappingReport:
    """从 JSONL 文件构建报告。"""

    return build_time_mapping_report(records=iter_jsonl_dicts(Path(jsonl_path)), max_groups=int(max_groups))
