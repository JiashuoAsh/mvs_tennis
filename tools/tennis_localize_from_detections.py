"""从离线检测结果计算三相机网球3D位置。

使用场景：
- 你已经有三相机图片的“时间对齐结果 + 网球检测框”（例如 tennis_detections.json）。
- 你也有（或先用示例）相机内外参。
- 目标是输出每组对齐帧对应的 3D 网球位置（世界坐标系）。

说明：
- 本脚本不负责跑 .rknn 检测（Windows 上通常跑不了），只做几何。
 - 输入 JSON 结构默认兼容 tennis3d.offline_detect.pipeline 的输出。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from tennis3d.geometry.calibration import load_calibration
from tennis3d.geometry.triangulation import reprojection_errors, triangulate_dlt


def _pick_best_detection(det_list: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    best = None
    best_score = float("-inf")
    for d in det_list:
        s = float(d.get("score", -1.0))
        if s > best_score:
            best = d
            best_score = s
    return best


def _extract_detections_for_camera(value: Any) -> list[dict[str, Any]]:
    # 支持两种：
    # 1) 直接是 list[det]
    # 2) dump-raw-outputs 时是 {"detections": [...], "raw_outputs": [...]}
    if value is None:
        return []
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    if isinstance(value, dict) and isinstance(value.get("detections"), list):
        return [x for x in value["detections"] if isinstance(x, dict)]
    return []


def build_arg_parser() -> argparse.ArgumentParser:
    # 说明：Windows 终端编码差异较大，这里尽量使用 ASCII，避免 --help 乱码。
    p = argparse.ArgumentParser(description="Triangulate tennis ball 3D position from bboxes + calibration")
    p.add_argument(
        "--detections-json",
        default=str(Path(__file__).resolve().parents[1] / "data" / "tools_output" / "tennis_detections.json"),
        help="Detections JSON (output of tennis3d.offline_detect.pipeline)",
    )
    p.add_argument(
        "--calib",
        default=str(
            Path(__file__).resolve().parents[1] / "data" / "calibration" / "example_triple_camera_calib.json"
        ),
        help="Calibration JSON path",
    )
    p.add_argument(
        "--out-json",
        default=str(Path(__file__).resolve().parents[1] / "data" / "tools_output" / "tennis_positions_3d.json"),
        help="Output JSON for 3D results",
    )
    p.add_argument(
        "--min-score",
        type=float,
        default=0.25,
        help="Ignore detections below this confidence",
    )
    p.add_argument(
        "--require-views",
        type=int,
        default=2,
        help="Minimum number of camera views required (default: 2)",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process at most N records (for quick check)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    detections_path = Path(args.detections_json).resolve()
    calib_path = Path(args.calib).resolve()
    out_path = Path(args.out_json).resolve()

    if not detections_path.exists():
        raise RuntimeError(f"找不到 detections-json: {detections_path}")

    calib = load_calibration(calib_path)

    with detections_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise RuntimeError("detections-json 顶层必须是 list")

    if args.max_frames is not None:
        records = records[: max(0, int(args.max_frames))]

    out_records: list[dict[str, Any]] = []

    for rec in records:
        dets_by_cam = rec.get("detections") if isinstance(rec, dict) else None
        if not isinstance(dets_by_cam, dict):
            continue

        points_uv: dict[str, tuple[float, float]] = {}
        used_det: dict[str, Any] = {}
        projections: dict[str, np.ndarray] = {}
        calib_by_cam: dict[str, Any] = {}

        for cam_name, raw in dets_by_cam.items():
            det_list = _extract_detections_for_camera(raw)
            best = _pick_best_detection(det_list)
            if best is None:
                continue
            if float(best.get("score", 0.0)) < float(args.min_score):
                continue

            center = best.get("center")
            if not isinstance(center, list) or len(center) != 2:
                # 也允许 bbox 推中心
                bbox = best.get("bbox")
                if isinstance(bbox, list) and len(bbox) == 4:
                    x1, y1, x2, y2 = map(float, bbox)
                    center = [0.5 * (x1 + x2), 0.5 * (y1 + y2)]
                else:
                    continue

            u, v = float(center[0]), float(center[1])

            try:
                cam_calib = calib.require(cam_name)
            except KeyError:
                # 该相机没有标定，跳过
                continue

            points_uv[cam_name] = (u, v)
            used_det[cam_name] = best
            projections[cam_name] = cam_calib.P
            calib_by_cam[cam_name] = cam_calib

        if len(points_uv) < int(args.require_views):
            continue

        X_w = triangulate_dlt(projections=projections, points_uv=points_uv)
        errs = reprojection_errors(
            projections=projections,
            points_uv=points_uv,
            X_w=X_w,
        )

        # 同时给出各相机坐标系下的 3D 点：X_c = R_wc * X_w + t_wc
        ball_3d_by_camera: dict[str, list[float]] = {}
        for cam_name, cam_calib in calib_by_cam.items():
            X_c = cam_calib.R_wc @ X_w.reshape(3) + cam_calib.t_wc.reshape(3)
            ball_3d_by_camera[cam_name] = [
                float(X_c[0]),
                float(X_c[1]),
                float(X_c[2]),
            ]

        out_records.append(
            {
                "base_ts_str": rec.get("base_ts_str"),
                "base_ts_epoch_ms": rec.get("base_ts_epoch_ms"),
                "used_cameras": list(points_uv.keys()),
                "ball_center_uv": {k: [points_uv[k][0], points_uv[k][1]] for k in points_uv},
                "ball_3d_world": [float(X_w[0]), float(X_w[1]), float(X_w[2])],
                "ball_3d_camera": ball_3d_by_camera,
                "reprojection_errors": [
                    {
                        "camera": e.camera,
                        "uv": [e.uv[0], e.uv[1]],
                        "uv_hat": [e.uv_hat[0], e.uv_hat[1]],
                        "error_px": e.error_px,
                    }
                    for e in errs
                ],
                "detections": used_det,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_records, f, ensure_ascii=False, indent=2)

    print(f"Done. 3D records: {len(out_records)}")
    print(f"Done. OUT: {out_path}")


if __name__ == "__main__":
    main()

