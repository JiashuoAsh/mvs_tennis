"""离线模块内部若需要三角化，可直接复用 geometry.triangulation。"""

from tennis3d.geometry.triangulation import ReprojectionError, project_point, reprojection_errors, triangulate_dlt

__all__ = ["ReprojectionError", "project_point", "triangulate_dlt", "reprojection_errors"]
