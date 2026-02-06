"""在线模式 CLI 参数解析。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal


_TERMINAL_PRINT_MODE = Literal["best", "all", "none"]


def build_arg_parser() -> argparse.ArgumentParser:
    # 说明：Windows 终端编码差异较大，这里尽量使用 ASCII，避免 --help 乱码。
    p = argparse.ArgumentParser(description="Online: localize tennis ball 3D from MVS stream")
    p.add_argument(
        "--config",
        default="",
        help="Optional online config file (.json/.yaml/.yml). If set, other CLI args are ignored.",
    )
    p.add_argument(
        "--mvimport-dir",
        default=None,
        help=(
            "MVS official Python sample bindings directory (MvImport). "
            "Optional; or set env MVS_MVIMPORT_DIR"
        ),
    )
    p.add_argument(
        "--dll-dir",
        default=None,
        help="directory containing MvCameraControl.dll (optional); or set env MVS_DLL_DIR",
    )
    p.add_argument(
        "--serial",
        action="extend",
        nargs="+",
        default=[],
        help="camera serials (ordered) e.g. --serial A B C",
    )
    p.add_argument(
        "--trigger-source",
        default="Software",
        help=(
            "trigger source. If --master-serial is set, this is applied to slaves only "
            "(e.g. Line0/Line1/...). Master always uses Software."
        ),
    )
    p.add_argument(
        "--master-serial",
        default="",
        help=(
            "optional master camera serial. When set: master uses Software trigger; "
            "slaves use --trigger-source (typically Line0)."
        ),
    )
    p.add_argument(
        "--master-line-out",
        default="Line1",
        help=(
            "master output line selector (e.g. Line1). Empty means do not set via script."
        ),
    )
    p.add_argument(
        "--master-line-source",
        default="",
        help=(
            "master output line source (e.g. ExposureStartActive). Empty means do not set via script."
        ),
    )
    p.add_argument(
        "--master-line-mode",
        default="Output",
        help="master output line mode (usually Output)",
    )
    p.add_argument(
        "--soft-trigger-fps",
        type=float,
        default=5.0,
        help=(
            "soft trigger fps. In pure Software mode, sends to all cameras; "
            "in master/slave mode, sends to master only."
        ),
    )
    p.add_argument(
        "--trigger-activation",
        default="FallingEdge",
        help="trigger activation (RisingEdge/FallingEdge)",
    )
    p.add_argument(
        "--trigger-cache-enable",
        action="store_true",
        help="try to enable TriggerCacheEnable (some models may not support)",
    )
    p.add_argument(
        "--group-by",
        choices=["frame_num", "sequence"],
        default="frame_num",
        help="grouping key",
    )
    p.add_argument("--timeout-ms", type=int, default=1000, help="single frame grab timeout")
    p.add_argument("--group-timeout-ms", type=int, default=1000, help="wait time for assembling a full group")
    p.add_argument("--max-pending-groups", type=int, default=256, help="max pending groups")
    p.add_argument("--max-groups", type=int, default=0, help="stop after N groups (0 = no limit)")
    p.add_argument(
        "--max-wait-seconds",
        type=float,
        default=0.0,
        help=(
            "exit if no full group is received for this many seconds (0 = no limit). "
            "Useful for debugging hardware trigger wiring."
        ),
    )

    # 相机图像参数（Optional）。0/空字符串表示不设置，沿用相机当前配置。
    p.add_argument("--pixel-format", default="", help="PixelFormat (e.g. BayerRG8). Empty means do not set.")
    p.add_argument("--image-width", type=int, default=0, help="ROI width (0 = do not set)")
    p.add_argument("--image-height", type=int, default=0, help="ROI height (0 = do not set)")
    p.add_argument("--image-offset-x", type=int, default=0, help="ROI offset X")
    p.add_argument("--image-offset-y", type=int, default=0, help="ROI offset Y")

    # 说明：该模块位于 src/tennis3d_online/cli.py，所以回到仓库根目录是 parents[2]。
    repo_root = Path(__file__).resolve().parents[2]
    p.add_argument(
        "--calib",
        default=str(repo_root / "data" / "calibration" / "example_triple_camera_calib.json"),
        help="Calibration path (.json/.yaml/.yml)",
    )
    p.add_argument(
        "--detector",
        choices=["fake", "color", "rknn", "pt"],
        default="fake",
        help="Detector backend",
    )
    p.add_argument("--model", default="", help="Model path (required when --detector rknn or pt)")
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

    # 软件裁剪（动态 ROI）：在 detector 前裁剪小窗口，提高吞吐/精度；bbox 会自动回写到原图坐标系。
    p.add_argument(
        "--detector-crop-size",
        type=int,
        default=0,
        help=(
            "optional software crop size (square). 0 disables. "
            "Typical: 640 (run detector on 640x640 window)"
        ),
    )
    p.add_argument(
        "--detector-crop-smooth-alpha",
        type=float,
        default=0.2,
        help="crop smoothing alpha in [0,1]. Larger = more stable",
    )
    p.add_argument(
        "--detector-crop-max-step-px",
        type=int,
        default=120,
        help="max crop movement per group in pixels (0 = no limit)",
    )
    p.add_argument(
        "--detector-crop-reset-after-missed",
        type=int,
        default=8,
        help="reset crop prediction after N consecutive groups with no balls",
    )

    # 相机侧 AOI（运行中动态平移 OffsetX/OffsetY）。
    p.add_argument(
        "--camera-aoi-runtime",
        action="store_true",
        help="enable runtime camera AOI panning (OffsetX/OffsetY)",
    )
    p.add_argument(
        "--camera-aoi-update-every-groups",
        type=int,
        default=2,
        help="update camera offsets every N groups (>=1)",
    )
    p.add_argument(
        "--camera-aoi-min-move-px",
        type=int,
        default=8,
        help="skip AOI update if movement < this pixels",
    )
    p.add_argument(
        "--camera-aoi-smooth-alpha",
        type=float,
        default=0.3,
        help="camera AOI smoothing alpha in [0,1]",
    )
    p.add_argument(
        "--camera-aoi-max-step-px",
        type=int,
        default=160,
        help="max AOI movement per update in pixels (0 = no limit)",
    )
    p.add_argument(
        "--camera-aoi-recenter-after-missed",
        type=int,
        default=30,
        help="after N missed groups, slowly recenter AOI to initial offset (0 disables)",
    )
    p.add_argument(
        "--out-jsonl",
        default="",
        help="Optional output JSONL path (if empty, print only)",
    )
    p.add_argument(
        "--out-jsonl-only-when-balls",
        action="store_true",
        help="if set: write JSONL only when balls is non-empty (reduces disk IO)",
    )
    p.add_argument(
        "--out-jsonl-flush-every-records",
        type=int,
        default=1,
        help=(
            "flush JSONL file every N records (1 = flush every record; 0 = disable count-based flush)"
        ),
    )
    p.add_argument(
        "--out-jsonl-flush-interval-s",
        type=float,
        default=0.0,
        help=(
            "flush JSONL file if time since last flush exceeds this seconds (0 = disable time-based flush)"
        ),
    )
    p.add_argument(
        "--terminal-print-mode",
        choices=["best", "all", "none"],
        default="best",
        help=(
            "terminal output mode: best prints only the top-1 ball per group; "
            "all prints all balls in the group; none prints nothing"
        ),
    )
    p.add_argument(
        "--terminal-status-interval-s",
        type=float,
        default=0.0,
        help=(
            "print a periodic status line every N seconds (0 = disable). Useful when no balls are detected."
        ),
    )

    # 在线时间轴（方案B）：实时滑窗映射 dev_timestamp -> host 时间轴。
    p.add_argument(
        "--time-sync-mode",
        choices=["frame_host_timestamp", "dev_timestamp_mapping"],
        default="frame_host_timestamp",
        help="time axis mode for capture_t_abs",
    )
    p.add_argument("--time-mapping-warmup-groups", type=int, default=20, help="warmup groups before first fit")
    p.add_argument("--time-mapping-window-groups", type=int, default=200, help="sliding window size in groups")
    p.add_argument(
        "--time-mapping-update-every-groups",
        type=int,
        default=5,
        help="refit mapping every N groups after warmup",
    )
    p.add_argument("--time-mapping-min-points", type=int, default=20, help="min pairs per camera to fit")
    p.add_argument("--time-mapping-hard-outlier-ms", type=float, default=50.0, help="hard outlier cutoff in ms")
    return p
