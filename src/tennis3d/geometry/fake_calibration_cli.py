"""命令行：生成假标定 JSON 文件。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from tennis3d.geometry.fake_calibration import FakeCalibrationConfig, build_fake_calibration_json


def build_arg_parser() -> argparse.ArgumentParser:
    # 说明：Windows 终端编码差异较大，这里尽量使用 ASCII，避免 --help 乱码。
    p = argparse.ArgumentParser(description="Generate fake multi-camera calibration JSON")
    p.add_argument(
        "--camera",
        action="extend",
        nargs="+",
        default=[],
        help="camera names/serials (2+). Example: --camera A B C",
    )
    p.add_argument("--image-width", type=int, default=2448, help="image width")
    p.add_argument("--image-height", type=int, default=2048, help="image height")
    p.add_argument("--fx", type=float, default=2000.0, help="focal length fx")
    p.add_argument("--fy", type=float, default=2000.0, help="focal length fy")
    p.add_argument("--baseline-m", type=float, default=0.30, help="baseline in meters")
    p.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[3] / "data" / "calibration" / "fake_multi_camera_calib.json"),
        help="output calibration JSON path",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    cameras = [str(x).strip() for x in (args.camera or []) if str(x).strip()]

    cfg = FakeCalibrationConfig(
        image_width=int(args.image_width),
        image_height=int(args.image_height),
        fx=float(args.fx),
        fy=float(args.fy),
        baseline_m=float(args.baseline_m),
    )

    data = build_fake_calibration_json(camera_names=cameras, config=cfg)
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Done. OUT: {out_path}")
    print(f"Done. cameras: {len(cameras)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
