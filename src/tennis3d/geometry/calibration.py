"""标定数据读取与投影矩阵构建（高内聚：只做几何与 IO）。

说明：
- 标定支持 JSON 与 YAML（.json/.yaml/.yml）。
- 外参采用 world->camera：$X_c = R_{wc} X_w + t_{wc}$。
- 投影矩阵采用像素坐标形式：$P = K [R_{wc} | t_{wc}]$。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    """单个相机的内外参。"""

    name: str
    image_size: tuple[int, int]  # (width, height)
    K: np.ndarray  # (3, 3)
    dist: np.ndarray  # (N,) Optional，当前不参与计算
    R_wc: np.ndarray  # (3, 3)
    t_wc: np.ndarray  # (3,)

    @property
    def P(self) -> np.ndarray:
        """像素坐标投影矩阵：$P = K [R|t]$。"""

        Rt = np.concatenate([self.R_wc, self.t_wc.reshape(3, 1)], axis=1)
        return self.K @ Rt


@dataclass(frozen=True)
class CalibrationSet:
    """整套标定（多相机）。"""

    cameras: dict[str, CameraCalibration]

    def require(self, camera_name: str) -> CameraCalibration:
        if camera_name not in self.cameras:
            raise KeyError(f"标定文件中找不到相机: {camera_name}")
        return self.cameras[camera_name]


def _as_np_matrix(x: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"{name} 形状应为 {shape}，实际为 {arr.shape}")
    return arr


def _as_np_vector(x: Any, length: int, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.shape[0] != length:
        raise ValueError(f"{name} 长度应为 {length}，实际为 {arr.shape[0]}")
    return arr


def _load_mapping(path: Path) -> dict[str, Any]:
    """读取 JSON/YAML 并返回顶层 dict。

    Args:
        path: 文件路径。

    Returns:
        顶层对象（必须是 dict）。
    """

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "需要 PyYAML 才能读取 .yaml/.yml 标定文件，请先安装依赖：pip install pyyaml"
            ) from exc

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        raise RuntimeError(f"不支持的标定文件类型: {path}（仅支持 .json/.yaml/.yml）")

    if not isinstance(data, dict):
        raise RuntimeError("标定文件顶层必须是对象（dict）")
    return data


def load_calibration(path: Path) -> CalibrationSet:
    """从 JSON/YAML 加载标定。"""

    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"找不到标定文件: {path}")

    data = _load_mapping(path)

    cams = data.get("cameras")
    if not isinstance(cams, dict) or not cams:
        raise RuntimeError("标定 JSON 缺少 cameras 字段或为空")

    out: dict[str, CameraCalibration] = {}

    for cam_name, c in cams.items():
        if not isinstance(c, dict):
            raise RuntimeError(f"相机 {cam_name} 标定项不是对象")

        image_size = c.get("image_size")
        if (
            not isinstance(image_size, list)
            or len(image_size) != 2
            or not all(isinstance(v, (int, float)) for v in image_size)
        ):
            raise RuntimeError(f"相机 {cam_name} 缺少 image_size 或格式错误")

        w, h = int(image_size[0]), int(image_size[1])

        K = _as_np_matrix(c.get("K"), (3, 3), f"{cam_name}.K")
        R_wc = _as_np_matrix(c.get("R_wc"), (3, 3), f"{cam_name}.R_wc")
        t_wc = _as_np_vector(c.get("t_wc"), 3, f"{cam_name}.t_wc")

        dist_raw = c.get("dist", [])
        dist = np.asarray(dist_raw, dtype=np.float64).reshape(-1)

        out[cam_name] = CameraCalibration(
            name=str(cam_name),
            image_size=(w, h),
            K=K,
            dist=dist,
            R_wc=R_wc,
            t_wc=t_wc,
        )

    return CalibrationSet(cameras=out)


def apply_sensor_roi_to_calibration(
    calib: CalibrationSet,
    *,
    image_width: int,
    image_height: int,
    image_offset_x: int = 0,
    image_offset_y: int = 0,
) -> CalibrationSet:
    """根据传感器 ROI 裁剪，把“满幅标定”转换为 ROI 标定。

    背景：
        海康 MVS 的 Width/Height + OffsetX/OffsetY 本质是“裁剪 ROI”，不是缩放。
        因此：
        - 焦距 fx/fy（以像素为单位）不变
        - 主点 cx/cy 需要按 ROI 左上角偏移平移

    等价关系：
        若满幅像素坐标为 (u_full, v_full)，ROI 图像坐标为 (u_roi, v_roi)，则
        $u_{roi} = u_{full} - offset_x$，$v_{roi} = v_{full} - offset_y$。

        也等价于把内参主点改成：
        $c'_x = c_x - offset_x$，$c'_y = c_y - offset_y$。

    注意：
        - 本函数仅处理“裁剪”，不处理后续可能存在的图像缩放/letterbox。
          （本仓库的 detector 适配已对 letterbox 做了 scale_back；此处不重复。）
        - 如果你的相机在硬件侧做了 binning/decimation（像素尺寸变化），则 fx/fy 也需要缩放；
          该情形不在本函数覆盖范围内。
    """

    w = int(image_width)
    h = int(image_height)
    ox = int(image_offset_x)
    oy = int(image_offset_y)

    if w <= 0 or h <= 0:
        raise ValueError("image_width/image_height 必须为正（ROI 必须是裁剪窗口）")

    out: dict[str, CameraCalibration] = {}
    for cam_name, cam in calib.cameras.items():
        K = np.array(cam.K, dtype=np.float64, copy=True)
        # 关键点：ROI 裁剪相当于把像素坐标原点平移到 ROI 左上角。
        K[0, 2] = float(K[0, 2]) - float(ox)
        K[1, 2] = float(K[1, 2]) - float(oy)

        out[str(cam_name)] = CameraCalibration(
            name=str(cam.name),
            image_size=(w, h),
            K=K,
            dist=cam.dist,
            R_wc=cam.R_wc,
            t_wc=cam.t_wc,
        )

    return CalibrationSet(cameras=out)
