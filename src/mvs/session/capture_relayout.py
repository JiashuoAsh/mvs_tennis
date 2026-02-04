# -*- coding: utf-8 -*-

"""把一次 captures 目录的图片从“按组”重排为“按相机”。

背景：
- `python -m mvs.apps.quad_capture` 这类采集通常把输出组织为 group_* 目录；每个 group 里包含多台相机同一次触发的帧。
- 在某些标定/质检流程里，更希望把同一台相机的所有帧放在一起（便于逐相机挑帧/筛选/可视化）。

本模块读取 `metadata.jsonl` 中的 group 记录（包含 frames 列表），并把每个 frame 对应的文件
放到输出目录下的“相机子目录”里。

注意：
- `metadata.jsonl` 可能包含非 group 的事件记录（例如 camera_event/soft_trigger_send），会被自动跳过。
- frame 的 `file` 字段在不同采集脚本/版本里可能是：绝对路径、相对 captures_dir、或相对仓库根目录。
  这里实现了较稳妥的路径解析策略，尽量在不依赖额外配置的情况下找到源文件。
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from .metadata_io import iter_metadata_records


@dataclass(frozen=True, slots=True)
class RelayoutStats:
    """重排过程的统计信息（用于打印/测试/诊断）。"""

    groups_seen: int
    frames_seen: int
    files_created: int
    files_skipped_existing: int
    missing_source_files: int
    files_failed: int
    link_fallback_to_copy: int


def _iter_group_records(meta_path: Path) -> Iterator[Dict[str, Any]]:
    """从 metadata.jsonl 迭代出包含 frames 的 group 记录。"""

    for rec in iter_metadata_records(meta_path):
        frames = rec.get("frames")
        if not isinstance(frames, list) or not frames:
            continue
        yield rec


def _resolve_frame_file(*, captures_dir: Path, file_value: Any) -> Optional[Path]:
    """把 metadata.jsonl 里 frame 的 file 字段解析成可用的本地路径。

    解析策略（按优先级尝试）：
    1) 若是绝对路径：直接使用。
    2) 若相对路径能在 captures_dir 下命中：使用 captures_dir/file。
    3) 若 file 路径中包含 captures_dir 的目录名（例如 .../for_calib/...）：
       丢弃前缀，仅保留从该目录名之后的尾部路径，拼回 captures_dir。
    4) 若能在当前工作目录下命中：使用 cwd/file。

    Returns:
        解析成功且文件存在时返回 Path，否则返回 None。
    """

    if not isinstance(file_value, str) or not file_value.strip():
        return None

    p = Path(file_value)

    # 1) 绝对路径
    if p.is_absolute() and p.exists():
        return p

    # 2) 相对 captures_dir
    cand = (captures_dir / p)
    if cand.exists():
        return cand.resolve()

    # 3) 如果包含 captures_dir.name，则从该段开始截断。
    try:
        parts = list(p.parts)
        if captures_dir.name in parts:
            idx = max(i for i, part in enumerate(parts) if part == captures_dir.name)
            tail = Path(*parts[idx + 1 :])
            cand2 = (captures_dir / tail)
            if cand2.exists():
                return cand2.resolve()
    except Exception:
        # 这里不希望因为奇怪路径导致整个流程中断。
        pass

    # 4) 相对当前工作目录（通常是仓库根目录）
    cand3 = (Path.cwd() / p)
    if cand3.exists():
        return cand3.resolve()

    return None


def _safe_unlink(path: Path) -> None:
    """删除已有目标文件（只处理常见情况）。"""

    try:
        path.unlink()
    except FileNotFoundError:
        return


def _materialize_one(
    *,
    src: Path,
    dst: Path,
    mode: str,
    overwrite: bool,
    dry_run: bool,
) -> str:
    """把 src 以指定方式落到 dst。

    Returns:
        - "created": 成功创建（link/copy/symlink 任一）。
        - "skipped": 目标已存在且不覆盖。
        - "fallback_copy": link/symlink 失败后回退 copy。

    Raises:
        OSError: 在 copy 模式或回退 copy 也失败时抛出。
        ValueError: mode 非法。
    """

    mode = str(mode).strip().lower()
    if mode not in {"hardlink", "copy", "symlink"}:
        raise ValueError(f"Unsupported mode: {mode}")

    if dst.exists():
        if not overwrite:
            return "skipped"
        _safe_unlink(dst)

    if dry_run:
        return "created"

    dst.parent.mkdir(parents=True, exist_ok=True)

    if mode == "copy":
        shutil.copy2(src, dst)
        return "created"

    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "created"
        except OSError:
            # 常见原因：跨盘、权限、目标已存在（已处理）等。
            shutil.copy2(src, dst)
            return "fallback_copy"

    # mode == "symlink"
    try:
        os.symlink(src, dst, target_is_directory=False)
        return "created"
    except OSError:
        # Windows 上 symlink 可能需要管理员/开发者模式。
        shutil.copy2(src, dst)
        return "fallback_copy"


def relayout_capture_by_camera(
    *,
    captures_dir: Path,
    output_dir: Path,
    mode: str = "hardlink",
    overwrite: bool = False,
    dry_run: bool = False,
    max_groups: int = 0,
) -> RelayoutStats:
    """把 captures_dir 的图片按相机重排到 output_dir。

    Args:
        captures_dir: 输入目录，必须包含 metadata.jsonl 和 group_* 图片。
        output_dir: 输出目录，会创建相机子目录。
        mode: "hardlink" | "copy" | "symlink"。默认 hardlink（省空间）。
        overwrite: 若目标文件已存在，是否覆盖。
        dry_run: 仅演练，不实际写文件（但仍返回统计）。
        max_groups: 仅处理前 N 个 group（0=不限制）。

    Returns:
        RelayoutStats: 统计信息。
    """

    captures_dir = Path(captures_dir).resolve()
    output_dir = Path(output_dir).resolve()

    meta_path = captures_dir / "metadata.jsonl"
    if not meta_path.exists():
        raise RuntimeError(f"metadata.jsonl not found: {meta_path}")

    groups_seen = 0
    frames_seen = 0
    files_created = 0
    files_skipped_existing = 0
    missing_source_files = 0
    files_failed = 0
    link_fallback_to_copy = 0

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for rec in _iter_group_records(meta_path):
        groups_seen += 1
        if int(max_groups) > 0 and groups_seen > int(max_groups):
            break

        frames = rec.get("frames") or []
        for fr in frames:
            if not isinstance(fr, dict):
                continue

            frames_seen += 1

            cam_index_raw = fr.get("cam_index")
            if cam_index_raw is None:
                continue
            try:
                cam_index = int(cam_index_raw)
            except Exception:
                # cam_index 缺失时无法分目录，直接跳过。
                continue

            serial = str(fr.get("serial", "")).strip()
            if not serial:
                # serial 缺失时仍可按 cam_index 分组，但目录名会不稳定；这里选择跳过。
                continue

            src = _resolve_frame_file(captures_dir=captures_dir, file_value=fr.get("file"))
            if src is None or not src.exists():
                missing_source_files += 1
                continue

            cam_dir = output_dir / f"cam{cam_index}_{serial}"
            dst = cam_dir / src.name

            try:
                outcome = _materialize_one(
                    src=src,
                    dst=dst,
                    mode=mode,
                    overwrite=overwrite,
                    dry_run=dry_run,
                )
            except Exception:
                files_failed += 1
                continue

            if outcome == "skipped":
                files_skipped_existing += 1
            else:
                files_created += 1
                if outcome == "fallback_copy":
                    link_fallback_to_copy += 1

    return RelayoutStats(
        groups_seen=int(groups_seen),
        frames_seen=int(frames_seen),
        files_created=int(files_created),
        files_skipped_existing=int(files_skipped_existing),
        missing_source_files=int(missing_source_files),
        files_failed=int(files_failed),
        link_fallback_to_copy=int(link_fallback_to_copy),
    )
