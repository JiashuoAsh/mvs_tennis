"""生成“可跑通链路”的假标定数据。

用途：
- 你还没有真实内参/外参时，先用这个生成一份 JSON，跑通：采集 -> 检测 -> 三角化 -> 3D 输出。

约定：
- 外参使用 world->camera：$X_c = R_{wc} X_w + t_{wc}$。
- 这里默认所有相机 R_wc=I，只在 world 的 x 轴上拉开 baseline。
  这是“能跑通”的最小假设，不代表真实安装。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class FakeCalibrationConfig:
    """生成假标定的参数。"""

    image_width: int = 2448
    image_height: int = 2048
    fx: float = 2000.0
    fy: float = 2000.0
    baseline_m: float = 0.30


def build_fake_calibration_json(
    *,
    camera_names: Iterable[str],
    config: FakeCalibrationConfig = FakeCalibrationConfig(),
) -> dict:
    """生成标定 JSON dict（可直接 json.dump）。

    Args:
        camera_names: 相机名称/序列号列表（建议用 mvs 里传入的 serial 字符串）。
        config: 假标定参数。

    Returns:
        符合 tennis3d.geometry.calibration.load_calibration 的 JSON dict。
    """

    names = [str(x).strip() for x in camera_names if str(x).strip()]
    if len(names) < 2:
        raise ValueError("camera_names must contain at least 2 cameras")

    w = int(config.image_width)
    h = int(config.image_height)
    if w <= 0 or h <= 0:
        raise ValueError("invalid image size")

    fx = float(config.fx)
    fy = float(config.fy)
    if fx <= 0 or fy <= 0:
        raise ValueError("fx/fy must be positive")

    cx = 0.5 * float(w)
    cy = 0.5 * float(h)

    K = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    R_wc = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    dist = [0.0, 0.0, 0.0, 0.0, 0.0]

    # 让相机在 x 方向对称分布：例如 N=3 => [-1,0,1] * baseline
    n = len(names)
    mid = 0.5 * float(n - 1)

    cameras = {}
    for i, name in enumerate(names):
        x = (float(i) - mid) * float(config.baseline_m)
        cameras[name] = {
            "image_size": [w, h],
            "K": K,
            "dist": dist,
            "R_wc": R_wc,
            "t_wc": [x, 0.0, 0.0],
        }

    return {
        "version": 1,
        "units": "m",
        "world_frame": {"x": "right", "y": "down", "z": "forward"},
        "notes": "fake calibration generated for pipeline smoke test; replace with real calibration for production",
        "cameras": cameras,
    }
