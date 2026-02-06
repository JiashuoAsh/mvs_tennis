# -*- coding: utf-8 -*-

"""命令行入口：分析一次采集输出目录（metadata.jsonl）。

说明：
- 这是 entry layer：负责参数解析与打印/落盘。
- 核心分析逻辑在 `mvs.analysis.analyze_output_dir`。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from mvs.analysis import analyze_output_dir


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="分析 mvs 采集输出目录（metadata.jsonl）")
    p.add_argument(
        "--output-dir",
        required=False,
        default="data/captures",
        help="采集输出目录（包含 metadata.jsonl）",
    )
    p.add_argument(
        "--expected-cameras",
        type=int,
        default=3,
        help="期望相机数量（不填则自动从数据推断）",
    )
    p.add_argument(
        "--expected-fps",
        type=float,
        default=15,
        help="期望 FPS（用于合格判定，不填则不判定 FPS）",
    )
    p.add_argument(
        "--fps-tolerance",
        type=float,
        default=0.2,
        # argparse 内部会对 help 文本做 % 格式化；要显示字面量 % 必须写成 %%。
        help="FPS 允许相对误差（默认 0.2=±20%%）",
    )
    p.add_argument(
        "--write-json",
        default="analysis_summary.json",
        help="将汇总写入 JSON 文件（例如 analysis_summary.json；为空则不写）",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    # 尽量固定 UTF-8 输出，避免重定向到文件时出现乱码。
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    args = _parse_args(argv)

    out_dir = Path(args.output_dir)
    if not out_dir.exists() or not out_dir.is_dir():
        print(f"output_dir not found: {out_dir}")
        return 2

    try:
        _, report_text, payload = analyze_output_dir(
            output_dir=out_dir,
            expected_cameras=args.expected_cameras,
            expected_fps=args.expected_fps,
            fps_tolerance_ratio=float(args.fps_tolerance),
        )
    except Exception as exc:
        print(f"analyze failed: {exc}")
        return 2

    print(report_text)

    write_json_path = str(args.write_json or "").strip()
    if write_json_path:
        out_path = Path(write_json_path)
        try:
            # 允许用户把输出写到尚不存在的目录（例如 tools_output/tmp.json）。
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n[已写出] {out_path}")
        except Exception as exc:
            print(f"write json failed: {exc}")
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
