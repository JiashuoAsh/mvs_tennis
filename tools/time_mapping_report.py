"""基于 tennis3d online 输出 JSONL 的时间映射诊断报告。

用途：
- 你在 online 模式启用 time_sync_mode=dev_timestamp_mapping 后，
  `iter_mvs_image_groups` 会在每条记录 meta 中写入：
  - time_mapping_mapped_host_ms_by_camera
  - time_mapping_mapped_host_ms_spread_ms
  - time_mapping_mapped_host_ms_delta_to_median_by_camera

该脚本读取 jsonl 并输出：
- 组内 spread（max-min）的 p50/p95/max（毫秒）
- 每台相机相对组内中位数的偏差（delta）的 p50/p95（毫秒）

说明：
- 不依赖第三方库，仅用标准库。
- 该报告反映的是“同一 group 内各相机时间戳对齐程度”，
  与 `time_mapping_worst_rms_ms/p95_ms`（拟合残差）是不同指标。

运行示例：
    python tools/time_mapping_report.py --jsonl data/tools_output/online.jsonl

"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _percentile_nearest_rank(sorted_values: list[float], q: float) -> float | None:
    """nearest-rank 分位数（不插值），q in [0,1]。"""

    if not sorted_values:
        return None
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])

    n = len(sorted_values)
    # nearest-rank: ceil(q*n) - 1
    k = int(math.ceil(float(q) * float(n)) - 1)
    k = max(0, min(n - 1, k))
    return float(sorted_values[k])


def _median_no_interp(values: list[float]) -> float | None:
    """中位数（不插值）。

    说明：与 `tennis3d.pipeline.sources._median_float` 的行为一致：
    - 偶数个元素时取“上中位数”（sorted[n//2]）。
    """

    if not values:
        return None
    xs = sorted(float(v) for v in values)
    return float(xs[len(xs) // 2])


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Report intragroup time deltas after dev->host mapping")
    p.add_argument("--jsonl", required=True, help="Path to online output jsonl")
    p.add_argument("--max-groups", type=int, default=0, help="Limit number of groups (0=no limit)")
    args = p.parse_args(argv)

    jsonl_path = Path(str(args.jsonl)).resolve()
    if not jsonl_path.exists():
        raise SystemExit(f"jsonl not found: {jsonl_path}")

    raw_spread_ms: list[float] = []
    mapped_spread_ms: list[float] = []
    mapped_deltas_by_camera: dict[str, list[float]] = {}

    seen = 0
    for rec in _iter_jsonl(jsonl_path):
        seen += 1
        if int(args.max_groups) > 0 and seen > int(args.max_groups):
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

        delta_map = rec.get("time_mapping_mapped_host_ms_delta_to_median_by_camera")
        delta_by_cam = _as_float_map(delta_map)

        # 兼容：如果没有 delta 字段，则用 mapped_host_ms_by_camera 现算一次。
        if len(delta_by_cam) < 2:
            mapped = _as_float_map(rec.get("time_mapping_mapped_host_ms_by_camera"))
            if len(mapped) >= 2:
                med = _median_no_interp(list(mapped.values()))
                if med is not None:
                    delta_by_cam = {cam: float(t) - float(med) for cam, t in mapped.items()}

        for cam, d in delta_by_cam.items():
            mapped_deltas_by_camera.setdefault(cam, []).append(float(d))

    if not mapped_spread_ms and not mapped_deltas_by_camera:
        print("未找到可用的 group（需要每组至少 2 台相机具备映射后的时间戳字段）。")
        return 0

    groups_used = max(len(mapped_spread_ms), max((len(v) for v in mapped_deltas_by_camera.values()), default=0))
    print(f"可用组数：{groups_used}")

    if raw_spread_ms:
        xs = sorted(raw_spread_ms)
        p50 = _percentile_nearest_rank(xs, 0.50)
        p95 = _percentile_nearest_rank(xs, 0.95)
        mx = xs[-1]
        print(f"原始 host_timestamp 组内跨度（ms）：p50={p50:.3f} p95={p95:.3f} max={mx:.3f}")

    if mapped_spread_ms:
        xs = sorted(mapped_spread_ms)
        p50 = _percentile_nearest_rank(xs, 0.50)
        p95 = _percentile_nearest_rank(xs, 0.95)
        mx = xs[-1]
        print(f"映射后（dev_timestamp_mapping）组内跨度（ms）：p50={p50:.3f} p95={p95:.3f} max={mx:.3f}")

    for cam in sorted(mapped_deltas_by_camera.keys()):
        ds = sorted(mapped_deltas_by_camera[cam])
        if not ds:
            continue
        abs_ds = sorted(abs(x) for x in ds)
        d_med = _percentile_nearest_rank(ds, 0.50)
        d_abs_p95 = _percentile_nearest_rank(abs_ds, 0.95)
        print(f"相机 {cam} 相对组内中位数偏差（ms）：median={d_med:.3f} abs_p95={d_abs_p95:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
