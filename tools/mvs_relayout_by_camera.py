# -*- coding: utf-8 -*-

"""把 captures 输出从“按 group”重排为“按相机”。

示例：
- 输入：data/captures_master_slave/for_calib（包含 metadata.jsonl + group_*/cam*.bmp）
- 输出：data/captures_master_slave/for_calib_by_camera/
    cam0_DA8199303/
    cam1_DA8199402/
    ...

默认优先使用 hardlink（省空间）；若 hardlink/symlink 因跨盘或权限失败，会自动回退为 copy。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mvs.session.capture_relayout import relayout_capture_by_camera


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="按相机重排 captures 图片（基于 metadata.jsonl）。")
    p.add_argument(
        "--captures-dir",
        type=str,
        required=True,
        help="输入 captures 目录（必须包含 metadata.jsonl）。",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="输出目录（默认：<captures_dir>_by_camera）。",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="hardlink",
        choices=["hardlink", "copy", "symlink"],
        help="落盘方式：hardlink/copy/symlink。默认 hardlink。",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="若目标文件已存在则覆盖。",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="仅演练，不实际写文件。",
    )
    p.add_argument(
        "--max-groups",
        type=int,
        default=0,
        help="仅处理前 N 个 group（0=不限制）。",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    # Windows 下默认控制台编码可能不是 UTF-8；当把输出重定向到文件时，
    # 这里显式切到 UTF-8，便于后续工具/脚本读取。
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = _build_parser().parse_args(argv)

    captures_dir = Path(args.captures_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path(f"{captures_dir}_by_camera").resolve()

    stats = relayout_capture_by_camera(
        captures_dir=captures_dir,
        output_dir=output_dir,
        mode=str(args.mode),
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        max_groups=int(args.max_groups),
    )

    print("重排完成：")
    print(f"  captures_dir          : {captures_dir}")
    print(f"  output_dir            : {output_dir}")
    print(f"  groups_seen           : {stats.groups_seen}")
    print(f"  frames_seen           : {stats.frames_seen}")
    print(f"  files_created         : {stats.files_created}")
    print(f"  files_skipped_existing: {stats.files_skipped_existing}")
    print(f"  missing_source_files  : {stats.missing_source_files}")
    print(f"  files_failed          : {stats.files_failed}")
    print(f"  link_fallback_to_copy : {stats.link_fallback_to_copy}")

    if stats.files_failed > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
