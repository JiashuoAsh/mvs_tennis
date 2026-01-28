"""离线：从 MVS captures/metadata.jsonl 读取同步组，检测并三角化输出 3D。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from tennis3d.detectors import create_detector
from tennis3d.config import load_offline_app_config
from tennis3d.geometry.calibration import load_calibration
from tennis3d.pipeline import iter_capture_image_groups, run_localization_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    # 说明：Windows 终端编码差异较大，这里尽量使用 ASCII，避免 --help 乱码。
    p = argparse.ArgumentParser(description="Offline: localize tennis ball 3D from captures/metadata.jsonl")
    p.add_argument(
        "--config",
        default="",
        help="Optional offline config file (.json/.yaml/.yml). If set, other CLI args are ignored.",
    )
    p.add_argument(
        "--captures-dir",
        default=str(Path(__file__).resolve().parents[3] / "data" / "captures" / "sample_sequence"),
        help="captures directory (contains metadata.jsonl and image files)",
    )
    p.add_argument(
        "--calib",
        default=str(Path(__file__).resolve().parents[3] / "data" / "calibration" / "sample_cams.yaml"),
        help="Calibration path (.json/.yaml/.yml)",
    )
    p.add_argument(
        "--detector",
        choices=["fake", "color", "rknn"],
        default="color",
        help="Detector backend (color works on Windows without RKNN runtime)",
    )
    p.add_argument(
        "--model",
        default="",
        help="RKNN model path (required when --detector rknn)",
    )
    p.add_argument("--min-score", type=float, default=0.25, help="Ignore detections below this confidence")
    p.add_argument("--require-views", type=int, default=2, help="Minimum camera views required")
    p.add_argument("--max-groups", type=int, default=0, help="Process at most N groups (0 = no limit)")
    p.add_argument(
        "--out-jsonl",
        default=str(Path(__file__).resolve().parents[3] / "data" / "tools_output" / "offline_positions_3d.jsonl"),
        help="Output JSONL path",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if str(getattr(args, "config", "") or "").strip():
        cfg = load_offline_app_config(Path(str(args.config)).resolve())
        captures_dir = Path(cfg.captures_dir).resolve()
        calib_path = Path(cfg.calib).resolve()
        detector_name = str(cfg.detector)
        model_path = Path(cfg.model).resolve() if cfg.model is not None else None
        min_score = float(cfg.min_score)
        require_views = int(cfg.require_views)
        max_groups = int(cfg.max_groups)
        out_path = Path(cfg.out_jsonl).resolve()
    else:
        captures_dir = Path(args.captures_dir).resolve()
        calib_path = Path(args.calib).resolve()
        detector_name = str(args.detector)
        model_path = (Path(args.model).resolve() if str(args.model).strip() else None)
        min_score = float(args.min_score)
        require_views = int(args.require_views)
        max_groups = int(args.max_groups)
        out_path = Path(args.out_jsonl).resolve()

    calib = load_calibration(calib_path)
    detector = create_detector(
        name=detector_name,
        model_path=model_path,
        conf_thres=float(min_score),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    groups_done = 0
    results_done = 0

    base_groups_iter = iter_capture_image_groups(captures_dir=captures_dir, max_groups=max_groups)

    def _counting_groups():
        nonlocal groups_done
        for meta, images in base_groups_iter:
            groups_done += 1
            yield meta, images

    with out_path.open("w", encoding="utf-8") as f_out:
        for out_rec in run_localization_pipeline(
            groups=_counting_groups(),
            calib=calib,
            detector=detector,
            min_score=float(min_score),
            require_views=int(require_views),
            include_detection_details=True,
        ):
            f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            results_done += 1

    print(f"Done. groups={groups_done} results={results_done} out={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
