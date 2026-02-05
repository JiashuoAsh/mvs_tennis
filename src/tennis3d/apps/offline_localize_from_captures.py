"""离线：从 MVS captures/metadata.jsonl 读取同步组，检测并三角化输出 3D。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from tennis3d.detectors import create_detector
from tennis3d.config import load_offline_app_config
from tennis3d.geometry.calibration import load_calibration
from tennis3d.io.sample_sequence import ensure_sample_sequence
from tennis3d.pipeline import iter_capture_image_groups, run_localization_pipeline
from tennis3d.trajectory import CurveStageConfig, apply_curve_stage


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
        "--serial",
        nargs="*",
        default=None,
        help="Optional camera serials to use (subset). Example: --serial DA8199303 DA8199402",
    )
    p.add_argument(
        "--detector",
        choices=["fake", "color", "rknn", "pt"],
        default="color",
        help=(
            "Detector backend (pt uses Ultralytics YOLOv8 .pt; default device=cpu; "
            "color works on Windows without RKNN runtime)"
        ),
    )
    p.add_argument(
        "--model",
        default="",
        help="Model path (required when --detector rknn or pt)",
    )
    p.add_argument(
        "--pt-device",
        default="cpu",
        help=(
            "Ultralytics device for --detector pt (default=cpu). "
            "CUDA examples: cuda:0 / 0 / cuda"
        ),
    )
    p.add_argument("--min-score", type=float, default=0.25, help="Ignore detections below this confidence")
    p.add_argument("--require-views", type=int, default=2, help="Minimum camera views required")
    p.add_argument(
        "--max-detections-per-camera",
        type=int,
        default=10,
        help="TopK detections kept per camera (to limit combinations)",
    )
    p.add_argument(
        "--max-reproj-error-px",
        type=float,
        default=8.0,
        help="Max reprojection error in pixels for a ball candidate",
    )
    p.add_argument(
        "--max-uv-match-dist-px",
        type=float,
        default=25.0,
        help="Max pixel distance when matching projected 3D point to a detection center",
    )
    p.add_argument(
        "--merge-dist-m",
        type=float,
        default=0.08,
        help="3D merge distance in meters for deduplicating ball candidates",
    )
    p.add_argument("--max-groups", type=int, default=0, help="Process at most N groups (0 = no limit)")
    p.add_argument(
        "--out-jsonl",
        default=str(Path(__file__).resolve().parents[3] / "data" / "tools_output" / "offline_positions_3d.jsonl"),
        help="Output JSONL path",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    default_sample_dir = (Path(__file__).resolve().parents[3] / "data" / "captures" / "sample_sequence").resolve()

    if str(getattr(args, "config", "") or "").strip():
        cfg = load_offline_app_config(Path(str(args.config)).resolve())
        captures_dir = Path(cfg.captures_dir).resolve()
        calib_path = Path(cfg.calib).resolve()
        serials = list(cfg.serials) if cfg.serials is not None else None
        detector_name = str(cfg.detector)
        model_path = Path(cfg.model).resolve() if cfg.model is not None else None
        pt_device = str(getattr(cfg, "pt_device", "cpu") or "cpu").strip() or "cpu"
        min_score = float(cfg.min_score)
        require_views = int(cfg.require_views)
        max_detections_per_camera = int(cfg.max_detections_per_camera)
        max_reproj_error_px = float(cfg.max_reproj_error_px)
        max_uv_match_dist_px = float(cfg.max_uv_match_dist_px)
        merge_dist_m = float(cfg.merge_dist_m)
        max_groups = int(cfg.max_groups)
        out_path = Path(cfg.out_jsonl).resolve()
        curve_cfg = cfg.curve
        time_sync_mode = str(cfg.time_sync_mode)
        time_mapping_path = Path(cfg.time_mapping_path).resolve() if cfg.time_mapping_path is not None else None
    else:
        captures_dir = Path(args.captures_dir).resolve()
        calib_path = Path(args.calib).resolve()
        serials = [s.strip() for s in (args.serial or []) if str(s).strip()] or None
        detector_name = str(args.detector)
        model_path = (Path(args.model).resolve() if str(args.model).strip() else None)
        pt_device = str(getattr(args, "pt_device", "cpu") or "cpu").strip() or "cpu"
        min_score = float(args.min_score)
        require_views = int(args.require_views)
        max_detections_per_camera = int(args.max_detections_per_camera)
        max_reproj_error_px = float(args.max_reproj_error_px)
        max_uv_match_dist_px = float(args.max_uv_match_dist_px)
        merge_dist_m = float(args.merge_dist_m)
        max_groups = int(args.max_groups)
        out_path = Path(args.out_jsonl).resolve()
        curve_cfg = CurveStageConfig()
        time_sync_mode = "frame_host_timestamp"
        time_mapping_path = None

    # 说明：当用户直接运行离线入口且未提供 --captures-dir 时，我们希望“一键跑通”。
    # sample_sequence 不随仓库提交二进制图片，因此当默认目录缺失 metadata.jsonl 时，
    # 这里会自动生成一份最小可运行数据集。
    if captures_dir == default_sample_dir and not (captures_dir / "metadata.jsonl").exists():
        ensure_sample_sequence(captures_dir=captures_dir)

    calib = load_calibration(calib_path)

    # 说明：离线 3D 定位至少需要 2 路视角；如果用户显式指定 serials，则在此处做一次前置校验。
    if serials is not None:
        if len(serials) < max(2, int(require_views)):
            raise RuntimeError(
                f"serials 数量不足：serials={len(serials)} require_views={int(require_views)}（至少需要 2 且 >= require_views）"
            )

        calib_serials = set(calib.cameras.keys())
        serials_in_calib = [s for s in serials if s in calib_serials]
        if len(serials_in_calib) < int(require_views):
            avail = ",".join(sorted(calib_serials))
            got = ",".join(serials)
            raise RuntimeError(
                "指定的 serials 与标定不匹配或数量不足："
                f"serials=[{got}] require_views={int(require_views)} calib_cameras=[{avail}]"
            )

        serials = serials_in_calib
    detector = create_detector(
        name=detector_name,
        model_path=model_path,
        conf_thres=float(min_score),
        pt_device=pt_device,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    groups_done = 0
    records_done = 0
    balls_done = 0

    base_groups_iter = iter_capture_image_groups(
        captures_dir=captures_dir,
        max_groups=max_groups,
        serials=serials,
        time_sync_mode=str(time_sync_mode),
        time_mapping_path=time_mapping_path,
    )

    def _counting_groups():
        nonlocal groups_done
        for meta, images in base_groups_iter:
            groups_done += 1
            yield meta, images

    with out_path.open("w", encoding="utf-8") as f_out:
        records = run_localization_pipeline(
            groups=_counting_groups(),
            calib=calib,
            detector=detector,
            min_score=float(min_score),
            require_views=int(require_views),
            max_detections_per_camera=int(max_detections_per_camera),
            max_reproj_error_px=float(max_reproj_error_px),
            max_uv_match_dist_px=float(max_uv_match_dist_px),
            merge_dist_m=float(merge_dist_m),
            include_detection_details=True,
        )

        # Optional：对 3D 输出做轨迹拟合增强（落点/落地时间/走廊）。
        records = apply_curve_stage(records, curve_cfg)

        for out_rec in records:
            f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            records_done += 1
            balls = out_rec.get("balls") or []
            if isinstance(balls, list):
                balls_done += int(len(balls))

    print(f"Done. groups={groups_done} records={records_done} balls={balls_done} out={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
