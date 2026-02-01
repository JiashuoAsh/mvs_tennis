"""最小单测：验证三角化与重投影误差的基本正确性。"""

from __future__ import annotations

import unittest

import numpy as np

from tennis3d.geometry.triangulation import (
    estimate_triangulation_cov_world,
    projection_jacobian,
    project_point,
    reprojection_errors,
    triangulate_dlt,
)


class TestTriangulation(unittest.TestCase):
    def test_triangulate_two_views(self) -> None:
        fx = 1000.0
        fy = 1000.0
        cx = 0.0
        cy = 0.0
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

        R = np.eye(3, dtype=np.float64)

        # cam0：原点
        t0 = np.zeros(3, dtype=np.float64)
        P0 = K @ np.concatenate([R, t0.reshape(3, 1)], axis=1)

        # cam1：相机中心在 (1,0,0) -> world->camera: X_c = X_w - C => t = -C
        t1 = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        P1 = K @ np.concatenate([R, t1.reshape(3, 1)], axis=1)

        X_gt = np.array([0.2, -0.1, 5.0], dtype=np.float64)
        uv0 = project_point(P0, X_gt)
        uv1 = project_point(P1, X_gt)

        X = triangulate_dlt(projections={"cam0": P0, "cam1": P1}, points_uv={"cam0": uv0, "cam1": uv1})
        self.assertTrue(np.allclose(X, X_gt, atol=1e-6), msg=f"X={X}, gt={X_gt}")

        errs = reprojection_errors(projections={"cam0": P0, "cam1": P1}, points_uv={"cam0": uv0, "cam1": uv1}, X_w=X)
        self.assertEqual(len(errs), 2)
        self.assertTrue(all(e.error_px < 1e-5 for e in errs))

    def test_projection_jacobian_matches_finite_difference(self) -> None:
        fx = 800.0
        fy = 900.0
        cx = 10.0
        cy = 20.0
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        t = np.zeros(3, dtype=np.float64)
        P = K @ np.concatenate([R, t.reshape(3, 1)], axis=1)

        X = np.array([0.3, -0.2, 4.5], dtype=np.float64)
        J = projection_jacobian(P, X)
        self.assertEqual(J.shape, (2, 3))
        self.assertTrue(np.all(np.isfinite(J)))

        # 数值差分验证（仅 sanity check，容忍小误差）。
        eps = 1e-4
        uv0 = np.asarray(project_point(P, X), dtype=np.float64)
        J_num = np.zeros((2, 3), dtype=np.float64)
        for k in range(3):
            d = np.zeros(3, dtype=np.float64)
            d[k] = eps
            uv1 = np.asarray(project_point(P, X + d), dtype=np.float64)
            J_num[:, k] = (uv1 - uv0) / eps

        self.assertTrue(np.allclose(J, J_num, atol=1e-2), msg=f"J={J}, J_num={J_num}")

    def test_estimate_triangulation_cov_scales_with_pixel_noise(self) -> None:
        fx = 1000.0
        fy = 1000.0
        cx = 0.0
        cy = 0.0
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        R = np.eye(3, dtype=np.float64)

        t0 = np.zeros(3, dtype=np.float64)
        P0 = K @ np.concatenate([R, t0.reshape(3, 1)], axis=1)

        t1 = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        P1 = K @ np.concatenate([R, t1.reshape(3, 1)], axis=1)

        X = np.array([0.2, -0.1, 5.0], dtype=np.float64)

        cov1 = estimate_triangulation_cov_world(
            projections={"cam0": P0, "cam1": P1},
            X_w=X,
            cov_uv_by_camera={
                "cam0": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
                "cam1": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            },
        )
        self.assertIsNotNone(cov1)
        assert cov1 is not None
        self.assertEqual(cov1.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(cov1)))

        cov2 = estimate_triangulation_cov_world(
            projections={"cam0": P0, "cam1": P1},
            X_w=X,
            cov_uv_by_camera={
                "cam0": np.array([[4.0, 0.0], [0.0, 4.0]], dtype=np.float64),
                "cam1": np.array([[4.0, 0.0], [0.0, 4.0]], dtype=np.float64),
            },
        )
        self.assertIsNotNone(cov2)
        assert cov2 is not None

        # 像素噪声方差放大 4 倍，3D 协方差应近似放大 4 倍（线性化模型下）。
        r = float(np.trace(cov2) / np.trace(cov1))
        self.assertTrue(3.0 < r < 5.0, msg=f"trace ratio={r}")


if __name__ == "__main__":
    unittest.main()
