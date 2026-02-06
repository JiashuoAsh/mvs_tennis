# -*- coding: utf-8 -*-

"""生成离线可跑通的 sample_sequence（图片 + metadata.jsonl）。

动机：
- 仓库不直接提交二进制 BMP（体积/差异不友好），但又希望新用户能在无相机环境下
  直接跑通 `tennis3d.apps.offline_localize_from_captures` 的默认路径。
- 因此把“生成样例数据”的核心逻辑放在库侧（src/），`tools/` 与 app 入口只做薄壳调用。

约定：
- 输出目录结构与 `mvs.apps.quad_capture` 类似：
  <captures_dir>/metadata.jsonl
  <captures_dir>/group_0000000000/cam0.bmp ...
- 相机名使用 cam0/cam1/cam2，与 `data/calibration/sample_cams.yaml` 对齐。

注意：
- 该模块依赖 opencv-python 与 numpy（仓库运行依赖已包含）。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class SampleCam:
    """样例相机（用于生成图片与 metadata）。"""

    name: str
    t_wc: tuple[float, float, float]


def _project_uv(*, K: np.ndarray, t_wc: np.ndarray, X_w: np.ndarray) -> tuple[float, float]:
    """在 R=I 的假设下，把世界点投影到像素坐标（与 sample_cams.yaml 对齐）。"""

    X_c = X_w.reshape(3) + t_wc.reshape(3)
    x, y, z = float(X_c[0]), float(X_c[1]), float(X_c[2])
    if abs(z) < 1e-9:
        return (float("nan"), float("nan"))

    u = float(K[0, 0] * (x / z) + K[0, 2])
    v = float(K[1, 1] * (y / z) + K[1, 2])
    return (u, v)


def ensure_sample_sequence(
    *,
    captures_dir: Path,
    groups: int = 2,
    overwrite: bool = False,
) -> Path:
    """确保 sample_sequence 数据存在；若缺失则生成。

    Args:
        captures_dir: 输出 captures 目录。
        groups: 生成多少组（每组包含每路相机各 1 张图片）。
        overwrite: 是否覆盖已存在的数据。

    Returns:
        生成/已存在的 captures_dir（绝对路径）。
    """

    captures_dir = Path(captures_dir).resolve()
    meta_path = captures_dir / "metadata.jsonl"

    if meta_path.exists() and not overwrite:
        return captures_dir

    if captures_dir.exists() and overwrite:
        # 说明：不做递归删除，避免误删用户目录；只覆盖 metadata.jsonl 并补齐缺失图片。
        pass

    captures_dir.mkdir(parents=True, exist_ok=True)

    w, h = 320, 240
    # 与 data/calibration/sample_cams.yaml 对齐
    K = np.array([[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    cams = [
        SampleCam("cam0", (1.0, 0.0, 0.0)),
        SampleCam("cam1", (0.0, 0.0, 0.0)),
        SampleCam("cam2", (-1.0, 0.0, 0.0)),
    ]

    # 两帧世界坐标点（单位随便，只要几何一致即可）
    world_points = [
        np.array([0.0, 0.0, 4.0], dtype=np.float64),
        np.array([0.25, -0.15, 3.5], dtype=np.float64),
    ]

    g = max(1, int(groups))
    if len(world_points) < g:
        # 说明：如果用户想生成更多组，就循环使用点位即可。
        world_points = (world_points * ((g + len(world_points) - 1) // len(world_points)))[:g]

    base_host_ms = int(time.time() * 1000)
    base_dev = 1_000_000

    records: list[dict[str, object]] = []
    green = (0, 255, 0)  # BGR

    for gi in range(g):
        group_dir = captures_dir / f"group_{gi:010d}"
        group_dir.mkdir(parents=True, exist_ok=True)

        frames: list[dict[str, object]] = []
        X_w = world_points[gi]

        for cam_index, cam in enumerate(cams):
            img = np.zeros((h, w, 3), dtype=np.uint8)
            u, v = _project_uv(K=K, t_wc=np.array(cam.t_wc, dtype=np.float64), X_w=X_w)
            if np.isfinite(u) and np.isfinite(v):
                cx = int(np.clip(round(u), 0, w - 1))
                cy = int(np.clip(round(v), 0, h - 1))
                cv2.circle(img, (cx, cy), 10, green, thickness=-1)

            rel_file = Path(f"group_{gi:010d}") / f"{cam.name}.bmp"
            out_path = captures_dir / rel_file
            ok = cv2.imwrite(str(out_path), img)
            if not ok:
                raise RuntimeError(f"写入失败: {out_path}")

            frames.append(
                {
                    "cam_index": int(cam_index),
                    "serial": str(cam.name),
                    "frame_num": int(gi + 1),
                    # 说明：dev_timestamp/host_timestamp 只是为了让离线 pipeline 能计算 capture_t_abs。
                    "dev_timestamp": int(base_dev + gi * 10_000 + cam_index * 100),
                    "host_timestamp": int(base_host_ms + gi * 50 + cam_index * 2),
                    "width": int(w),
                    "height": int(h),
                    "lost_packet": 0,
                    "file": str(rel_file.as_posix()),
                }
            )

        records.append(
            {
                "group_seq": int(gi),
                "group_by": "sequence",
                "created_at": float(time.time()),
                "frames": frames,
            }
        )

    # 覆盖写出：保持生成结果确定。
    with meta_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return captures_dir
