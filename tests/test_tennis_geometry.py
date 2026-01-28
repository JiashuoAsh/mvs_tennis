"""最小单测：验证三角化与重投影误差的基本正确性。"""

from __future__ import annotations

import unittest

import numpy as np

from tennis3d.geometry.triangulation import project_point, reprojection_errors, triangulate_dlt


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


if __name__ == "__main__":
    unittest.main()
