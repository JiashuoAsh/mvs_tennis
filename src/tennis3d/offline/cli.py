"""命令行入口（高内聚：只管参数解析与调用 pipeline）。"""

from __future__ import annotations

import argparse
from pathlib import Path

from tennis3d.offline.pipeline import run_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="三相机离线网球检测（按时间戳对齐）")
    p.add_argument(
        "--data-root",
        default=str(Path(__file__).resolve().parents[3] / "data"),
        help="数据根目录（默认取仓库根目录）",
    )
    p.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parents[3] / "data" / "models" / "tennis_20241227.rknn"),
        help=".rknn 模型路径（默认 process/tennis_20241227.rknn）",
    )
    p.add_argument("--tolerance-ms", type=int, default=60, help="时间对齐容差（毫秒）")
    p.add_argument(
        "--out-json",
        default=str(Path(__file__).resolve().parents[3] / "data" / "tools_output" / "tennis_detections.json"),
        help="输出 JSON 路径",
    )
    p.add_argument(
        "--out-csv",
        default=str(Path(__file__).resolve().parents[3] / "data" / "tools_output" / "tennis_detections.csv"),
        help="输出 CSV 路径（每张图只写最高分单球）",
    )
    p.add_argument(
        "--no-csv",
        action="store_true",
        help="不输出 CSV",
    )
    p.add_argument(
        "--save-vis",
        default=None,
        help="保存可视化结果目录（不填则不保存）",
    )
    p.add_argument(
        "--dump-raw-outputs",
        action="store_true",
        help="在 JSON 中附带原始输出 shape/min/max（用于调试输出格式）",
    )
    p.add_argument(
        "--bgr",
        action="store_true",
        help="模型输入使用 BGR（默认会先转为 RGB）",
    )
    p.add_argument(
        "--align-only",
        action="store_true",
        help="只做三路时间对齐，不做模型推理（用于 Windows 先验证对齐结果）",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="最多处理多少组对齐帧（用于快速试跑）",
    )
    p.add_argument(
        "--require-all",
        action="store_true",
        help="只输出三路都对齐成功的帧组（有缺帧则跳过）",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    data_root = Path(args.data_root).resolve()
    model_path = Path(args.model).resolve()
    out_json = Path(args.out_json).resolve()
    out_csv = None if args.no_csv else Path(args.out_csv).resolve()
    save_vis_dir = Path(args.save_vis).resolve() if args.save_vis else None

    # align-only 模式下不做推理，因此不要求模型存在。
    if not bool(args.align_only) and not model_path.exists():
        raise RuntimeError(f"找不到模型文件: {model_path}")

    run_pipeline(
        data_root=data_root,
        model_path=model_path,
        tolerance_ms=int(args.tolerance_ms),
        out_json=out_json,
        out_csv=out_csv,
        save_vis_dir=save_vis_dir,
        dump_raw_outputs=bool(args.dump_raw_outputs),
        align_only=bool(args.align_only),
        max_frames=args.max_frames,
        require_all=bool(args.require_all),
        rgb=not bool(args.bgr),
    )

    # 尽量使用 ASCII 输出，避免不同终端编码导致的乱码
    print(f"Done. JSON: {out_json}")
    if out_csv is not None:
        print(f"Done. CSV:  {out_csv}")
    if save_vis_dir is not None:
        print(f"Done. VIS:  {save_vis_dir}")
