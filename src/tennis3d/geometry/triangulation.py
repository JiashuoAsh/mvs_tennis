"""三角化与重投影误差计算（高内聚：只做几何）。

核心目标：
- 输入：多相机投影矩阵 $P_i$ 与像素坐标点 $(u, v)$
- 输出：世界坐标系下 3D 点 $X_w$ 以及各相机重投影误差

实现说明：
- 使用 DLT（Direct Linear Transform）最小二乘求解，支持 2~N 视角。
- 不依赖 OpenCV，避免环境差异带来的问题。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReprojectionError:
    """单相机的重投影误差。"""

    camera: str
    uv: tuple[float, float]
    uv_hat: tuple[float, float]
    error_px: float


def triangulate_dlt(
    *,
    projections: dict[str, np.ndarray],
    points_uv: dict[str, tuple[float, float]],
) -> np.ndarray:
    """使用 DLT 三角化得到世界坐标系下 3D 点。"""

    keys = [k for k in points_uv.keys() if k in projections]
    if len(keys) < 2:
        raise ValueError("三角化至少需要两个相机的观测点")

    A_rows: list[np.ndarray] = []
    for k in keys:
        P = np.asarray(projections[k], dtype=np.float64)
        if P.shape != (3, 4):
            raise ValueError(f"P 形状应为 (3,4)，相机 {k} 实际为 {P.shape}")

        u, v = points_uv[k]
        u = float(u)
        v = float(v)

        A_rows.append(u * P[2, :] - P[0, :])
        A_rows.append(v * P[2, :] - P[1, :])

    A = np.stack(A_rows, axis=0)

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1, :]
    if abs(float(X[3])) < 1e-12:
        raise ValueError("三角化失败：齐次坐标 w 过小")

    X = X / X[3]
    return X[:3].astype(np.float64)


def project_point(P: np.ndarray, X_w: np.ndarray) -> tuple[float, float]:
    """将世界点投影到像素坐标。"""

    P = np.asarray(P, dtype=np.float64)
    X_w = np.asarray(X_w, dtype=np.float64).reshape(3)

    Xh = np.array([X_w[0], X_w[1], X_w[2], 1.0], dtype=np.float64)
    x = P @ Xh
    if abs(float(x[2])) < 1e-12:
        return (float("nan"), float("nan"))

    return (float(x[0] / x[2]), float(x[1] / x[2]))


def reprojection_errors(
    *,
    projections: dict[str, np.ndarray],
    points_uv: dict[str, tuple[float, float]],
    X_w: np.ndarray,
) -> list[ReprojectionError]:
    """计算各相机的重投影误差（像素）。"""

    errs: list[ReprojectionError] = []
    for cam, (u, v) in points_uv.items():
        if cam not in projections:
            continue
        uv_hat = project_point(projections[cam], X_w)
        du = float(u) - float(uv_hat[0])
        dv = float(v) - float(uv_hat[1])
        errs.append(
            ReprojectionError(
                camera=str(cam),
                uv=(float(u), float(v)),
                uv_hat=uv_hat,
                error_px=float(np.hypot(du, dv)),
            )
        )
    return errs
