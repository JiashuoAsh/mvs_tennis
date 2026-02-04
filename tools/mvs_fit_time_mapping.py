# -*- coding: utf-8 -*-

"""离线拟合时间映射（方案B）。

用法（示例）：
- 从某个 captures 目录拟合并写出映射文件：
  uv run python tools/mvs_fit_time_mapping.py --captures-dir data/captures_master_slave/tennis_offline

输出：
- 默认写入：<captures-dir>/time_mapping_dev_to_host_ms.json

说明：
- 本工具只使用 frames 的 (dev_timestamp, host_timestamp) 进行拟合。
- host_timestamp 在本仓库实测为 epoch 毫秒；如果你的数据不是该单位，请先确认并调整。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mvs import collect_frame_pairs_from_metadata, fit_dev_to_host_ms, save_time_mappings_json


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fit per-camera dev_timestamp -> host_timestamp(ms) time mapping")
    p.add_argument(
        "--captures-dir",
        required=True,
        help="captures directory (contains metadata.jsonl)",
    )
    p.add_argument(
        "--out",
        default="",
        help="output mapping json path (default: <captures-dir>/time_mapping_dev_to_host_ms.json)",
    )
    p.add_argument(
        "--max-groups",
        type=int,
        default=0,
        help="use at most N groups from metadata.jsonl (0 = all)",
    )
    p.add_argument(
        "--serial",
        nargs="*",
        default=None,
        help="optional camera serial whitelist (subset)",
    )
    # 说明：样例数据可能只有几十组；默认 20 既能跑通，又能避免样本太少时斜率不稳定。
    p.add_argument("--min-points", type=int, default=20, help="min pairs per camera")
    p.add_argument("--hard-outlier-ms", type=float, default=50.0, help="hard outlier threshold in ms")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    captures_dir = Path(str(args.captures_dir)).resolve()
    meta_path = captures_dir / "metadata.jsonl"
    if not meta_path.exists():
        raise SystemExit(f"metadata.jsonl not found: {meta_path}")

    out_path = Path(str(args.out)).expanduser().resolve() if str(args.out).strip() else (captures_dir / "time_mapping_dev_to_host_ms.json")

    serials = [str(s).strip() for s in (args.serial or []) if str(s).strip()] or None

    pairs_by_serial = collect_frame_pairs_from_metadata(
        metadata_path=meta_path,
        max_groups=int(args.max_groups),
        serials=serials,
    )

    mappings = {}
    failed = {}
    for serial, pairs in sorted(pairs_by_serial.items()):
        try:
            m = fit_dev_to_host_ms(
                pairs,
                min_points=int(args.min_points),
                hard_outlier_ms=float(args.hard_outlier_ms),
            )
            mappings[str(serial)] = m
        except Exception as exc:
            failed[str(serial)] = str(exc)

    if not mappings:
        raise SystemExit(f"no mappings fitted. failed={failed or '-'}")

    save_time_mappings_json(
        out_path=out_path,
        mappings=mappings,
        metadata_path=meta_path,
        extra={
            "max_groups": int(args.max_groups),
            "min_points": int(args.min_points),
            "hard_outlier_ms": float(args.hard_outlier_ms),
            "failed": failed,
        },
    )

    print(f"Wrote: {out_path}")
    for serial, m in sorted(mappings.items()):
        print(
            f"- {serial}: n_used={m.n_used}/{m.n_total} a={m.a:.6e} b={m.b:.3f} "
            f"rms_ms={m.rms_ms:.3f} p95_ms={m.p95_ms:.3f} max_ms={m.max_ms:.3f}"
        )

    if failed:
        print("Some cameras failed:")
        for serial, err in sorted(failed.items()):
            print(f"- {serial}: {err}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
