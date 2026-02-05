"""离线检测流水线（高内聚：对齐 + 推理 + 输出组装）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from tennis3d.models import Detection, ImageItem
from tennis3d.offline_detect.alignment import match_triples
from tennis3d.offline_detect.detector import TennisDetector
from tennis3d.offline_detect.images import list_images, read_image
from tennis3d.offline_detect.outputs import maybe_write_csv, write_json
from tennis3d.preprocess import draw_detections, scale_detections_back


@dataclass(frozen=True)
class CameraDirs:
    cam1_name: str
    cam1_dir: Path
    cam2_name: str
    cam2_dir: Path
    cam3_name: str
    cam3_dir: Path


def default_camera_dirs(data_root: Path) -> CameraDirs:
    return CameraDirs(
        cam1_name="MV-CS050-02(DA8199285)",
        cam1_dir=data_root / "MV-CS050-02(DA8199285)",
        cam2_name="MV-CS050-03(DA8199303)",
        cam2_dir=data_root / "MV-CS050-03(DA8199303)",
        cam3_name="MV-CS050-CTRL(DA8199402)",
        cam3_dir=data_root / "MV-CS050-CTRL(DA8199402)",
    )


def _as_json_detection(d: Detection) -> dict[str, Any]:
    return {
        "bbox": list(d.bbox),
        "score": d.score,
        "cls": d.cls,
        "center": list(d.center),
    }


def _as_matched_json(it: ImageItem, base: ImageItem) -> dict[str, Any]:
    return {
        "path": str(it.path),
        "ts_str": it.ts_str,
        "ts_epoch_ms": it.ts_epoch_ms,
        "delta_ms": int(it.ts_epoch_ms - base.ts_epoch_ms),
    }


def run_pipeline(
    *,
    data_root: Path,
    model_path: Path,
    tolerance_ms: int,
    out_json: Path,
    out_csv: Optional[Path],
    save_vis_dir: Optional[Path],
    dump_raw_outputs: bool,
    align_only: bool,
    max_frames: Optional[int],
    require_all: bool,
    rgb: bool,
) -> None:
    cam = default_camera_dirs(data_root)

    for d in [cam.cam1_dir, cam.cam2_dir, cam.cam3_dir]:
        if not d.exists():
            raise RuntimeError(f"找不到相机目录: {d}")

    cam1 = list_images(cam.cam1_name, cam.cam1_dir)
    cam2 = list_images(cam.cam2_name, cam.cam2_dir)
    cam3 = list_images(cam.cam3_name, cam.cam3_dir)

    triples = match_triples(cam1, cam2, cam3, tolerance_ms=tolerance_ms)
    if require_all:
        triples = [t for t in triples if t.cam2 is not None and t.cam3 is not None]
    if max_frames is not None:
        triples = triples[: max(0, int(max_frames))]

    detector = None if align_only else TennisDetector(model_path=model_path, rgb=rgb)

    if save_vis_dir is not None:
        save_vis_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for t in triples:
        base = t.base
        matched: dict[str, Any] = {
            base.camera: _as_matched_json(base, base),
        }
        if t.cam2 is not None:
            matched[t.cam2.camera] = _as_matched_json(t.cam2, base)
        if t.cam3 is not None:
            matched[t.cam3.camera] = _as_matched_json(t.cam3, base)

        images: list[ImageItem] = [base] + [x for x in [t.cam2, t.cam3] if x is not None]

        det_by_cam: dict[str, Any] = {}

        for it in images:
            if align_only:
                det_by_cam[it.camera] = []
                continue

            assert detector is not None

            img = read_image(it.path)
            dets_in = detector.detect(img)
            dets = scale_detections_back(
                dets_in,
                orig_shape=img.shape[:2],
                input_size=detector.input_size,
            )

            if dump_raw_outputs:
                raw = detector.infer_raw(img)
                det_by_cam[it.camera] = {
                    "detections": [_as_json_detection(d) for d in dets],
                    "raw_outputs": [
                        {
                            "shape": list(o.shape),
                            "dtype": str(o.dtype),
                            "min": float(np.min(o)) if o.size else None,
                            "max": float(np.max(o)) if o.size else None,
                        }
                        for o in raw
                    ],
                }
            else:
                det_by_cam[it.camera] = [_as_json_detection(d) for d in dets]

            # CSV：每张图只取最高分一个球
            best = max(dets, key=lambda x: x.score, default=None)
            csv_rows.append(
                {
                    "base_ts_str": base.ts_str,
                    "base_ts_epoch_ms": base.ts_epoch_ms,
                    "camera": it.camera,
                    "image": it.path.name,
                    "image_path": str(it.path),
                    "matched_ts_str": it.ts_str,
                    "matched_ts_epoch_ms": it.ts_epoch_ms,
                    "delta_ms": int(it.ts_epoch_ms - base.ts_epoch_ms),
                    "has_ball": 1 if best is not None else 0,
                    "cx": best.center[0] if best is not None else None,
                    "cy": best.center[1] if best is not None else None,
                    "x1": best.bbox[0] if best is not None else None,
                    "y1": best.bbox[1] if best is not None else None,
                    "x2": best.bbox[2] if best is not None else None,
                    "y2": best.bbox[3] if best is not None else None,
                    "score": best.score if best is not None else None,
                    "cls": best.cls if best is not None else None,
                }
            )

            if save_vis_dir is not None:
                vis = draw_detections(img, dets)
                out_name = f"{it.camera}_{it.path.stem}.jpg"
                cv2.imwrite(str(save_vis_dir / out_name), vis)

        records.append(
            {
                "base_ts_str": base.ts_str,
                "base_ts_epoch_ms": base.ts_epoch_ms,
                "matched": matched,
                "detections": det_by_cam,
            }
        )

    write_json(out_json, records)
    maybe_write_csv(out_csv, csv_rows)
