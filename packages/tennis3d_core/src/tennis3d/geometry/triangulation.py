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


def projection_jacobian(P: np.ndarray, X_w: np.ndarray) -> np.ndarray:
    """计算像素投影 (u,v) 对世界坐标 (x,y,z) 的雅可比 J，形状为 (2,3)。

    说明：
        - 采用解析形式，避免数值差分的稳定性与性能问题。
        - 该雅可比用于把像素域观测噪声协方差传播到 3D 点协方差：
          Σ_X ≈ (J^T W J)^{-1}。
        - 这里假设外参/内参已被包含在 P 中，因此 J 的单位为 px/m。
    """

    P = np.asarray(P, dtype=np.float64)
    if P.shape != (3, 4):
        raise ValueError(f"P 形状应为 (3,4)，实际为 {P.shape}")

    X_w = np.asarray(X_w, dtype=np.float64).reshape(3)
    Xh = np.array([X_w[0], X_w[1], X_w[2], 1.0], dtype=np.float64)

    p0 = P[0, :]
    p1 = P[1, :]
    p2 = P[2, :]

    a = float(p0 @ Xh)
    b = float(p1 @ Xh)
    c = float(p2 @ Xh)
    if abs(c) < 1e-12:
        return np.full((2, 3), np.nan, dtype=np.float64)

    # 对 x,y,z 的偏导数；齐次分量 w=1 的导数为 0。
    da = p0[:3]
    db = p1[:3]
    dc = p2[:3]

    c2 = float(c * c)
    # u=a/c, v=b/c
    du = (da * c - a * dc) / c2
    dv = (db * c - b * dc) / c2
    J = np.stack([du, dv], axis=0).astype(np.float64)
    return J


def estimate_triangulation_cov_world(
    *,
    projections: dict[str, np.ndarray],
    X_w: np.ndarray,
    cov_uv_by_camera: dict[str, np.ndarray],
    min_views: int = 2,
) -> np.ndarray | None:
    """估计三角化 3D 点的协方差（世界坐标系，单位 m^2）。

    设计目标：
        - 输出一个“工程可用”的不确定度尺度，用于后续拟合加权/诊断。
        - 不追求严格的最优估计（未显式进行重投影误差最小化迭代），而是用线性化近似。

    近似模型：
        把每个相机的像素观测视为：u = π(P, X) + noise，noise ~ N(0, Σ_uv)。
        在线性化点 X_w 处，堆叠各相机的雅可比 J_i，得到：

        Σ_X ≈ (Σ_i J_i^T Σ_uv_i^{-1} J_i)^{-1}

    Args:
        projections: 相机投影矩阵 P（3x4）。
        X_w: 世界坐标 3D 点（3,）。
        cov_uv_by_camera: 每相机像素观测协方差（2x2，单位 px^2）。
        min_views: 最少视角数；低于该值返回 None。

    Returns:
        3x3 协方差矩阵（单位 m^2）；若不可逆/数值异常则返回 None。
    """

    keys = [k for k in cov_uv_by_camera.keys() if k in projections]
    if len(keys) < int(min_views):
        return None

    X_w = np.asarray(X_w, dtype=np.float64).reshape(3)

    JT_W_J = np.zeros((3, 3), dtype=np.float64)
    for k in keys:
        P = np.asarray(projections[k], dtype=np.float64)
        J = projection_jacobian(P, X_w)
        if not np.all(np.isfinite(J)):
            continue

        cov_uv = np.asarray(cov_uv_by_camera[k], dtype=np.float64)
        if cov_uv.shape != (2, 2):
            continue

        # 协方差必须正定/半正定。这里仅做最基本的数值健壮性保护。
        try:
            W = np.linalg.inv(cov_uv)
        except Exception:
            continue
        if not np.all(np.isfinite(W)):
            continue

        JT_W_J += J.T @ W @ J

    if not np.all(np.isfinite(JT_W_J)):
        return None

    # 若矩阵不可逆（几何退化/视角不足/协方差异常），返回 None。
    try:
        cov_X = np.linalg.inv(JT_W_J)
    except Exception:
        return None
    if not np.all(np.isfinite(cov_X)):
        return None

    # 数值误差可能导致非对称，强制对称化。
    cov_X = 0.5 * (cov_X + cov_X.T)
    return cov_X.astype(np.float64)


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
