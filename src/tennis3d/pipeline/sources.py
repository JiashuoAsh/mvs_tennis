"""Pipeline sources: produce per-group images.

Two sources are provided:
- iter_capture_image_groups: offline from captures/metadata.jsonl
- iter_mvs_image_groups: online from QuadCapture groups

Both yield (meta, images_by_camera) where images are OpenCV BGR numpy arrays.
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


def iter_capture_image_groups(
    *,
    captures_dir: Path,
    max_groups: int = 0,
) -> Iterator[tuple[dict[str, Any], dict[str, np.ndarray]]]:
    """Iterate image groups from captures/metadata.jsonl.

    Notes:
        - The metadata.jsonl may contain non-group event records; they are skipped.
        - If a frame file path is relative, it is resolved against captures_dir.

    Args:
        captures_dir: Directory containing metadata.jsonl and frames.
        max_groups: Stop after N groups (0 = no limit).

    Yields:
        (meta, images_by_camera_serial)
    """

    captures_dir = Path(captures_dir).resolve()
    meta_path = captures_dir / "metadata.jsonl"
    if not meta_path.exists():
        raise RuntimeError(f"metadata.jsonl not found: {meta_path}")

    groups_done = 0

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
                file = fr.get("file")
                if not serial or not isinstance(file, str) or not file:
                    continue

                file_path = Path(file)
                if not file_path.is_absolute():
                    file_path = (captures_dir / file_path).resolve()

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
    """Iterate image groups from an online MVS QuadCapture.

    Args:
        cap: Opened QuadCapture.
        binding: Loaded MVS binding.
        max_groups: Stop after N groups (0 = no limit).
        timeout_s: Wait timeout for cap.get_next_group.

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
