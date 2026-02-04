"""把“内参目录 + 外参文件”融合成可被 load_calibration() 直接读取的标定 JSON。

目标：
- 输入：
  - 内参：形如 <SERIAL>_intrinsics.json（camera_matrix/dist_coeffs/image_size）
  - 外参：base_to_camera_extrinsics.json（C_T_B: Base->Cam 的 4x4 齐次矩阵）
- 输出：
    - 与 data/calibration/camera_extrinsics_C_T_B.json 相同 schema 的 JSON（顶层 cameras，且每个相机包含 image_size/K/dist/R_wc/t_wc）

约定：
- 外参方向为 world->camera：X_c = R_wc X_w + t_wc。
- 若把 world 定义为 base，则外参文件中的 C_T_B（Base->Cam）可直接取：
  R_wc = C_T_B[:3,:3]
  t_wc = C_T_B[:3, 3]

注意：
- 本模块不做“相机命名规则”的猜测。
  若外参文件 key（例如 cam0/cam1）与内参文件 key（例如相机序列号）不同，必须显式提供映射。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FuseSourceInfo:
    """输出 JSON 的 source 字段信息。"""

    intrinsics_dir: str
    extrinsics_file: str
    generated_at: str


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError(f"JSON 顶层必须是对象: {path}")
    return obj


def load_intrinsics_dir(intrinsics_dir: Path) -> dict[str, dict[str, Any]]:
    """读取 *_intrinsics.json，返回 {camera_name: {image_size,K,dist}}。

    当前仓库内参文件格式约定（见 data/calibration/inputs）：
    - camera_matrix: 3x3
    - dist_coeffs: list[float]
    - image_size: [w,h]

    Returns:
        dict：key 为相机名（通常是 serial），value 为 load_calibration 可直接使用的字段。
    """

    intrinsics_dir = Path(intrinsics_dir)
    if not intrinsics_dir.exists():
        raise RuntimeError(f"找不到内参目录: {intrinsics_dir}")

    out: dict[str, dict[str, Any]] = {}
    for p in sorted(intrinsics_dir.glob("*_intrinsics.json")):
        name = p.name[: -len("_intrinsics.json")]
        data = _read_json(p)

        K = data.get("camera_matrix")
        dist = data.get("dist_coeffs")
        image_size = data.get("image_size")

        if K is None or dist is None or image_size is None:
            raise RuntimeError(f"内参文件缺少字段 camera_matrix/dist_coeffs/image_size: {p}")

        out[name] = {
            "image_size": image_size,
            "K": K,
            "dist": dist,
        }

    if not out:
        raise RuntimeError(f"内参目录中未找到 *_intrinsics.json: {intrinsics_dir}")

    return out


def load_extrinsics_C_T_B(extrinsics_file: Path) -> dict[str, np.ndarray]:
    """读取 base_to_camera_extrinsics.json，返回 {extr_name: 4x4 齐次矩阵}。"""

    extrinsics_file = Path(extrinsics_file)
    data = _read_json(extrinsics_file)

    C_T_B = data.get("C_T_B")
    if not isinstance(C_T_B, dict) or not C_T_B:
        raise RuntimeError(f"外参文件缺少 C_T_B 或为空: {extrinsics_file}")

    out: dict[str, np.ndarray] = {}
    for k, v in C_T_B.items():
        A = np.asarray(v, dtype=np.float64)
        if A.shape != (4, 4):
            raise RuntimeError(f"外参矩阵 {k} 形状应为 (4,4)，实际为 {A.shape}: {extrinsics_file}")
        out[str(k)] = A

    return out


def _build_camera_entry(*, intr: dict[str, Any], C_T_B: np.ndarray) -> dict[str, Any]:
    """把单相机内参与 4x4 外参融合成 load_calibration 所需字段。"""

    R_wc = C_T_B[:3, :3]
    t_wc = C_T_B[:3, 3]

    return {
        "image_size": intr["image_size"],
        "K": intr["K"],
        "dist": intr.get("dist", []),
        "R_wc": R_wc.tolist(),
        "t_wc": t_wc.tolist(),
    }


def build_params_calib_json(
    *,
    intrinsics_by_name: dict[str, dict[str, Any]],
    extrinsics_by_name: dict[str, np.ndarray],
    extr_to_camera_name: dict[str, str] | None = None,
    source: FuseSourceInfo | None = None,
    units: str = "m",
    version: int = 1,
    notes: str | None = None,
) -> dict[str, Any]:
    """生成融合标定 JSON（返回 dict，不负责写文件）。

    当前仓库约定的标准落盘文件名为 data/calibration/camera_extrinsics_C_T_B.json。

    Args:
        intrinsics_by_name: key 为相机名（通常是 serial）。
        extrinsics_by_name: key 为外参中的相机名（例如 cam0/cam1）。
        extr_to_camera_name: 映射：{外参名 -> 输出相机名(也用于内参查找)}。
            - 若为 None，则要求外参名与内参名完全一致。
        source: 可选的 source 字段信息。

    Returns:
        dict：可直接 json.dump 写出。
    """

    if extr_to_camera_name is None:
        # 不做猜测：要求两者命名一致。
        extr_to_camera_name = {k: k for k in extrinsics_by_name.keys()}

    cameras: dict[str, Any] = {}

    for extr_name, cam_name in extr_to_camera_name.items():
        if extr_name not in extrinsics_by_name:
            raise RuntimeError(f"外参中找不到相机: {extr_name}")
        if cam_name not in intrinsics_by_name:
            raise RuntimeError(
                f"内参目录中找不到相机: {cam_name}（由映射 {extr_name} -> {cam_name} 得到）"
            )

        cameras[cam_name] = _build_camera_entry(
            intr=intrinsics_by_name[cam_name],
            C_T_B=extrinsics_by_name[extr_name],
        )

    out: dict[str, Any] = {
        "version": int(version),
        "units": str(units),
        "notes": notes or "",
        "source": (
            {
                "intrinsics_dir": source.intrinsics_dir,
                "extrinsics_file": source.extrinsics_file,
                "generated_at": source.generated_at,
            }
            if source is not None
            else {}
        ),
        "cameras": cameras,
    }

    return out


def write_params_calib_json(*, out_path: Path, payload: dict[str, Any]) -> None:
    """把 payload 写成 UTF-8 JSON，保持人类可读。"""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    out_path.write_text(text + "\n", encoding="utf-8")
