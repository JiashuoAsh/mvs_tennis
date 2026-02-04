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
- 统计计算逻辑位于 `src/tennis3d/sync/time_mapping_report.py`；本脚本仅提供 CLI。
- 该报告反映的是“同一 group 内各相机时间戳对齐程度”，
    与 `time_mapping_worst_rms_ms/p95_ms`（拟合残差）是不同指标。

运行示例：
    python tools/time_mapping_report.py --jsonl data/tools_output/online.jsonl

"""

from __future__ import annotations

import argparse
from pathlib import Path

from tennis3d.sync.time_mapping_report import build_time_mapping_report_from_jsonl


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Report intragroup time deltas after dev->host mapping")
    p.add_argument("--jsonl", required=True, help="Path to online output jsonl")
    p.add_argument("--max-groups", type=int, default=0, help="Limit number of groups (0=no limit)")
    args = p.parse_args(argv)

    jsonl_path = Path(str(args.jsonl)).resolve()
    if not jsonl_path.exists():
        raise SystemExit(f"jsonl not found: {jsonl_path}")

    report = build_time_mapping_report_from_jsonl(jsonl_path=jsonl_path, max_groups=int(args.max_groups))
    if report.mapped_spread_ms is None and not report.mapped_delta_by_camera:
        print("未找到可用的 group（需要每组至少 2 台相机具备映射后的时间戳字段）。")
        return 0

    print(f"可用组数：{report.groups_used}")

    if report.raw_host_spread_ms is not None:
        s = report.raw_host_spread_ms
        print(f"原始 host_timestamp 组内跨度（ms）：p50={s.p50:.3f} p95={s.p95:.3f} max={s.max:.3f}")

    if report.mapped_spread_ms is not None:
        s = report.mapped_spread_ms
        print(f"映射后（dev_timestamp_mapping）组内跨度（ms）：p50={s.p50:.3f} p95={s.p95:.3f} max={s.max:.3f}")

    for cam, s in report.mapped_delta_by_camera.items():
        print(f"相机 {cam} 相对组内中位数偏差（ms）：median={s.median:.3f} abs_p95={s.abs_p95:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
