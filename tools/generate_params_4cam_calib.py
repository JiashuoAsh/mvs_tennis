"""生成符合 tennis3d.geometry.calibration.load_calibration() 的标定 JSON。

输出格式参考：data/calibration/params_4cam_calib.json

典型用法（示例）：
- 已有内参目录（*_intrinsics.json）
- 已有外参文件（base_to_camera_extrinsics.json，包含 C_T_B: Base->Cam）
- 外参相机名是 cam0/cam1/...，内参相机名是 serial，需要显式映射：
  --map cam0=DA8199303 --map cam1=DA8199402 ...

注意：
- 本脚本不对映射做猜测；映射缺失时会失败并给出提示。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# 说明：tools/ 下脚本常在“未 pip install -e .”的环境直接从仓库根目录运行。
# 为了稳定导入 src-layout 包（tennis3d/ 等），这里显式把 ./src 加入 sys.path。
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
_src_str = str(_SRC_DIR)
if _src_str not in sys.path:
    sys.path.insert(0, _src_str)

from tennis3d.geometry.calibration_fuse import (
    FuseSourceInfo,
    build_params_calib_json,
    load_extrinsics_C_T_B,
    load_intrinsics_dir,
    write_params_calib_json,
)


def _parse_map(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--map 参数格式应为 <extr>=<camera>，实际为: {it}")
        k, v = it.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k or not v:
            raise SystemExit(f"--map 参数非法（键或值为空）: {it}")
        if k in out and out[k] != v:
            raise SystemExit(f"--map 重复键且值不一致: {k} -> {out[k]} vs {v}")
        out[k] = v
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="融合内参与外参，生成 load_calibration 可读取的标定 JSON（params_4cam_calib 风格）"
    )

    parser.add_argument(
        "--intrinsics-dir",
        type=str,
        required=True,
        help="内参目录，包含 *_intrinsics.json（如 data/calibration/inputs/2026-01-30）",
    )
    parser.add_argument(
        "--extrinsics-file",
        type=str,
        required=True,
        help="外参文件，包含 C_T_B（如 data/calibration/base_to_camera_extrinsics.json）",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="输出 JSON 路径（如 data/calibration/camera_extrinsics_C_T_B.json）",
    )
    parser.add_argument(
        "--units",
        type=str,
        default="m",
        help="长度单位（默认 m）",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="输出 JSON 的 version 字段（默认 1）",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="写入 notes（可留空）",
    )
    parser.add_argument(
        "--generated-at",
        type=str,
        default="",
        help="写入 source.generated_at（默认使用当前时间 YYYY-MM-DD）",
    )
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        help="外参名到相机名(内参名)的映射：cam0=DA8199303。可重复提供多次。",
    )

    args = parser.parse_args(argv)

    intr_dir = Path(args.intrinsics_dir)
    extr_file = Path(args.extrinsics_file)
    out_path = Path(args.out)

    intr = load_intrinsics_dir(intr_dir)
    extr = load_extrinsics_C_T_B(extr_file)

    mapping = _parse_map(list(args.map)) if args.map else None

    if mapping is None:
        # 如果命名一致，可以不传映射；否则直接报错并提示。
        if set(extr.keys()) != set(intr.keys()):
            raise SystemExit(
                "外参相机名与内参相机名不一致，且未提供 --map。\n"
                f"外参相机名: {sorted(extr.keys())}\n"
                f"内参相机名: {sorted(intr.keys())}\n"
                "请通过 --map 指定映射，例如：--map cam0=DA8199303"
            )

    generated_at = args.generated_at.strip() or datetime.now().strftime("%Y-%m-%d")

    payload = build_params_calib_json(
        intrinsics_by_name=intr,
        extrinsics_by_name=extr,
        extr_to_camera_name=mapping,
        source=FuseSourceInfo(
            intrinsics_dir=str(intr_dir).replace("\\", "/"),
            extrinsics_file=str(extr_file).replace("\\", "/"),
            generated_at=generated_at,
        ),
        units=args.units,
        version=args.version,
        notes=args.notes,
    )

    write_params_calib_json(out_path=out_path, payload=payload)


if __name__ == "__main__":
    main()
