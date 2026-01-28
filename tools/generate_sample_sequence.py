# -*- coding: utf-8 -*-

"""生成离线可跑通的 sample_sequence（图片 + metadata.jsonl）。

说明：
- 该脚本用于把二进制图片（BMP）写入仓库，因为编辑器工具无法直接创建二进制文件。
- 输出目录：data/captures/sample_sequence/
- 配合：data/calibration/sample_cams.yaml + detector=color

运行后可用：
- python -m tennis3d.apps.offline_localize_from_captures
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class Cam:
    name: str
    t_wc: tuple[float, float, float]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _project_uv(*, K: np.ndarray, t_wc: np.ndarray, X_w: np.ndarray) -> tuple[float, float]:
    """在 R=I 的假设下，把世界点投影到像素坐标（与 sample_cams.yaml 对齐）。"""

    X_c = X_w.reshape(3) + t_wc.reshape(3)
    x, y, z = float(X_c[0]), float(X_c[1]), float(X_c[2])
    if abs(z) < 1e-9:
        return (float("nan"), float("nan"))

    u = float(K[0, 0] * (x / z) + K[0, 2])
    v = float(K[1, 1] * (y / z) + K[1, 2])
    return (u, v)


def main() -> int:
    root = _repo_root()
    seq_dir = root / "data" / "captures" / "sample_sequence"
    (seq_dir / "group_0000000000").mkdir(parents=True, exist_ok=True)
    (seq_dir / "group_0000000001").mkdir(parents=True, exist_ok=True)

    w, h = 320, 240
    K = np.array([[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    cams = [
        Cam("cam0", (1.0, 0.0, 0.0)),
        Cam("cam1", (0.0, 0.0, 0.0)),
        Cam("cam2", (-1.0, 0.0, 0.0)),
    ]

    # 两帧世界坐标点（单位随便，只要几何一致即可）
    frames = [
        np.array([0.0, 0.0, 4.0], dtype=np.float64),
        np.array([0.25, -0.15, 3.5], dtype=np.float64),
    ]

    green = (0, 255, 0)  # BGR

    written = 0
    for gi, X_w in enumerate(frames):
        group_dir = seq_dir / f"group_{gi:010d}"
        for cam in cams:
            img = np.zeros((h, w, 3), dtype=np.uint8)

            u, v = _project_uv(K=K, t_wc=np.array(cam.t_wc, dtype=np.float64), X_w=X_w)
            if np.isfinite(u) and np.isfinite(v):
                cx = int(np.clip(round(u), 0, w - 1))
                cy = int(np.clip(round(v), 0, h - 1))
                cv2.circle(img, (cx, cy), 10, green, thickness=-1)

            out_path = group_dir / f"{cam.name}.bmp"
            ok = cv2.imwrite(str(out_path), img)
            if not ok:
                raise RuntimeError(f"写入失败: {out_path}")
            written += 1

    # Use ASCII to avoid Windows console encoding issues.
    print(f"Generated sample_sequence images: {written} -> {seq_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
