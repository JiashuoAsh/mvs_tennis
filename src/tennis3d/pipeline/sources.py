"""流水线输入源：产出“每组图像”。

本模块提供两类 source：
- `iter_capture_image_groups`：离线，从 captures/metadata.jsonl 读取分组并加载对应图像。
- `iter_mvs_image_groups`：在线，从 MVS `QuadCapture` 读取分组并转换为 BGR 图像。

二者都产出 (meta, images_by_camera)：
- meta：可 JSON 序列化的 dict（用于携带 group_index/trigger_index 等信息）。
- images_by_camera：key 为相机序列号（serial），value 为 OpenCV BGR 的 numpy.ndarray。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from mvs.binding import MvsBinding
from mvs.pipeline import QuadCapture
from mvs.image import frame_to_bgr


def _find_repo_root(start: Path) -> Path:
    """从给定路径向上查找仓库根目录。

    约定：仓库根目录包含 pyproject.toml。
    """

    start = Path(start).resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return start


def iter_capture_image_groups(
    *,
    captures_dir: Path,
    max_groups: int = 0,
    serials: list[str] | None = None,
) -> Iterator[tuple[dict[str, Any], dict[str, np.ndarray]]]:
    """从 captures/metadata.jsonl 迭代读取每组图像。

    Notes:
        - metadata.jsonl 中可能混有非 group 记录（例如事件/日志），这里会跳过。
        - frames[*].file 可能是绝对路径，也可能是相对路径：
          1) 相对 captures_dir；
          2) 相对仓库根目录（例如以 data/ 开头）。
          本函数会做一次稳健解析，尽量找到真实存在的文件。

    Args:
        captures_dir: captures 目录（包含 metadata.jsonl 与各帧图像文件）。
        max_groups: 最多处理的组数（0 表示不限）。
        serials: 可选的相机序列号白名单；为 None 时不过滤。

    Yields:
        (meta, images_by_camera_serial)
    """

    captures_dir = Path(captures_dir).resolve()
    meta_path = captures_dir / "metadata.jsonl"
    if not meta_path.exists():
        raise RuntimeError(f"metadata.jsonl not found: {meta_path}")

    # 说明：captures 的 frames.file 可能是两种相对路径：
    # 1) 相对 captures_dir 的路径：group_xxx/cam0_xxx.bmp
    # 2) 相对仓库根目录的路径：data/captures_xxx/.../cam0_xxx.bmp
    # 这里做一次稳健解析，避免路径被重复拼接。
    repo_root = _find_repo_root(captures_dir)
    groups_done = 0

    serials_set: set[str] | None = None
    if serials is not None:
        serials_norm = [str(s).strip() for s in (serials or []) if str(s).strip()]
        serials_set = set(serials_norm) if serials_norm else set()

    with meta_path.open("r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            if not isinstance(rec, dict) or "frames" not in rec:
                continue

            frames = rec.get("frames")
            if not isinstance(frames, list) or not frames:
                continue

            images_by_camera: dict[str, np.ndarray] = {}
            for fr in frames:
                if not isinstance(fr, dict):
                    continue

                serial = str(fr.get("serial", "")).strip()
                if serials_set is not None and serial not in serials_set:
                    continue
                file = fr.get("file")
                if not serial or not isinstance(file, str) or not file:
                    continue

                file_path = Path(file)
                if not file_path.is_absolute():
                    # 优先按 captures_dir 解析；若不存在，再按仓库根目录解析。
                    candidate = (captures_dir / file_path).resolve()
                    if candidate.exists():
                        file_path = candidate
                    else:
                        candidate2 = (repo_root / file_path).resolve()
                        file_path = candidate2 if candidate2.exists() else candidate

                if not file_path.exists():
                    continue

                img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
                if img is None:
                    continue

                images_by_camera[serial] = img

            meta = {
                "group_seq": rec.get("group_seq"),
                "group_by": rec.get("group_by"),
                "trigger_index": rec.get("trigger_index"),
            }

            yield meta, images_by_camera

            groups_done += 1
            if int(max_groups) > 0 and groups_done >= int(max_groups):
                break


def iter_mvs_image_groups(
    *,
    cap: QuadCapture,
    binding: MvsBinding,
    max_groups: int = 0,
    timeout_s: float = 0.5,
) -> Iterator[tuple[dict[str, Any], dict[str, np.ndarray]]]:
    """从在线 MVS `QuadCapture` 迭代读取每组图像。

    Args:
        cap: 已打开的 QuadCapture。
        binding: 已加载的 MVS binding（用于像素格式解码）。
        max_groups: 最多处理的组数（0 表示不限）。
        timeout_s: 等待 cap.get_next_group 的超时时间。

    Yields:
        (meta, images_by_camera_serial)
    """

    group_index = 0

    while True:
        if int(max_groups) > 0 and group_index >= int(max_groups):
            break

        group = cap.get_next_group(timeout_s=float(timeout_s))
        if group is None:
            continue

        images_by_camera: dict[str, np.ndarray] = {}
        for fr in group:
            bgr = frame_to_bgr(binding=binding, cam=cap.cameras[fr.cam_index].cam, frame=fr)
            images_by_camera[str(fr.serial)] = bgr

        meta: dict[str, Any] = {
            "group_index": int(group_index),
        }

        yield meta, images_by_camera
        group_index += 1
