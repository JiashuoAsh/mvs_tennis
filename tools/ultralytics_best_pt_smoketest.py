"""用 Ultralytics YOLOv8（CPU）验证 data/models/best.pt 的最小可用性。

设计目标：
- 只做一件事：加载 best.pt，并对一张图片做一次推理。
- 尽量复用仓库内已有 captures 数据，避免你再手动找图片。
- 输出两份结果：
  1) 控制台摘要（ASCII，避免 Windows 终端编码问题）
  2) JSON（bbox/score/cls/name），用于后续三角化/排障

注意：
- 该脚本依赖 ultralytics + torch（CPU）。建议通过 uv 安装 dev 依赖。

说明：
- 本文件名刻意不以 test_ 开头，避免被 pytest 误收集为单元测试。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator

import cv2

if TYPE_CHECKING:  # pragma: no cover
    from tennis3d.models import Detection


@dataclass(frozen=True)
class YoloBox:
    """单个检测框的序列化结构。"""

    xyxy: list[float]
    conf: float
    cls: int
    name: str


def _find_repo_root(start: Path) -> Path:
    """向上查找包含 pyproject.toml 的目录，作为仓库根目录。"""

    start = Path(start).resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return start


# 说明：tools/*.py 通常直接用 `python tools/xxx.py` 运行。
# 这时 Python 不会自动把 src/ 加入 sys.path，导致 `import tennis3d` 失败。
# 这里显式把 src/ 加到 sys.path，保证工具脚本可直接运行。
REPO_ROOT = _find_repo_root(Path(__file__).resolve())
_SRC_DIR = REPO_ROOT / "src"
if _SRC_DIR.exists():
    sys.path.insert(0, str(_SRC_DIR))


def _iter_capture_groups(metadata_path: Path) -> Iterator[dict[str, Any]]:
    """迭代 metadata.jsonl 中的 group 记录（含 frames 字段）。"""

    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            if not isinstance(rec, dict):
                continue
            if "frames" not in rec:
                continue
            frames = rec.get("frames")
            if not isinstance(frames, list) or not frames:
                continue
            yield rec


def _resolve_image_path(repo_root: Path, captures_dir: Path, file_field: str) -> Path:
    """把 metadata.jsonl 里的 frames.file 转成真实存在的文件路径。"""

    raw = str(file_field).strip()
    if not raw:
        raise RuntimeError("frames.file is empty")

    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p

    # 说明：tennis_test 数据里常见的是 "data\\captures_master_slave\\..."，这是相对 repo_root 的。
    cand1 = (repo_root / p).resolve()
    if cand1.exists():
        return cand1

    cand2 = (captures_dir / p).resolve()
    if cand2.exists():
        return cand2

    # 最后兜底：如果路径里已经包含 group_xxx/cam?.bmp 的相对部分
    parts = list(p.parts)
    for i, part in enumerate(parts):
        if part.startswith("group_"):
            cand3 = (captures_dir / Path(*parts[i:])).resolve()
            if cand3.exists():
                return cand3

    raise RuntimeError(f"image not found for file field: {raw}")


def _safe_relpath(path: Path, base: Path) -> Path:
    """尽量计算相对路径；失败时退化为文件名。"""

    try:
        return path.resolve().relative_to(base.resolve())
    except Exception:
        return Path(path.name)


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _load_yolo_model(model_path: Path):
    """加载 Ultralytics YOLO 模型（CPU）。"""

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "ultralytics is not installed (or you are using the wrong Python).\n"
            "Hint: run with your venv python, e.g. .venv/Scripts/python.exe\n"
            f"Original error: {e}"
        )

    # 说明：在 Windows/CPU 环境下，device='cpu' 可强制不走 CUDA。
    return YOLO(str(model_path))


def _predict_boxes(
    *,
    model,
    img_bgr,
    conf: float,
    max_det: int,
    device: str = "cpu",
) -> tuple[list[YoloBox], list["Detection"]]:
    """执行一次推理，并同时返回可序列化的 boxes 与用于画框的 Detection。"""

    # 说明：Ultralytics 支持直接传入 numpy.ndarray（OpenCV BGR 图像）。
    results = model.predict(
        source=img_bgr,
        conf=float(conf),
        max_det=int(max_det),
        device=str(device),
        verbose=False,
        save=False,
    )

    if not results:
        return [], []

    r0 = results[0]
    boxes = getattr(r0, "boxes", None)
    if boxes is None or getattr(boxes, "xyxy", None) is None:
        return [], []

    xyxy = boxes.xyxy
    conf_t = boxes.conf
    cls_t = boxes.cls

    xyxy_list = xyxy.detach().cpu().numpy().tolist() if hasattr(xyxy, "detach") else xyxy.tolist()
    conf_list = conf_t.detach().cpu().numpy().tolist() if hasattr(conf_t, "detach") else conf_t.tolist()
    cls_list = cls_t.detach().cpu().numpy().tolist() if hasattr(cls_t, "detach") else cls_t.tolist()

    names: dict[int, str] = {}
    try:
        names = dict(getattr(model, "names", {}) or {})
    except Exception:
        names = {}

    # 延迟 import：避免在 ultralytics 未安装时就先因为 tennis3d import 失败。
    from tennis3d.models import Detection

    boxes_out: list[YoloBox] = []
    dets_out: list[Detection] = []
    for i in range(len(xyxy_list)):
        c = int(cls_list[i])
        xyxy_i = [float(x) for x in xyxy_list[i]]
        score_i = float(conf_list[i])
        boxes_out.append(
            YoloBox(
                xyxy=xyxy_i,
                conf=score_i,
                cls=c,
                name=str(names.get(c, str(c))),
            )
        )
        dets_out.append(Detection(bbox=(xyxy_i[0], xyxy_i[1], xyxy_i[2], xyxy_i[3]), score=score_i, cls=c))

    return boxes_out, dets_out


def _pick_first_image_from_captures(captures_dir: Path) -> Path:
    """从 captures/metadata.jsonl 中挑一张实际存在的图片路径。"""

    captures_dir = Path(captures_dir).resolve()
    meta_path = captures_dir / "metadata.jsonl"
    if not meta_path.exists():
        raise RuntimeError(f"metadata.jsonl not found: {meta_path}")

    repo_root = REPO_ROOT

    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            # 只关心 group 记录
            if not isinstance(rec, dict) or "frames" not in rec:
                continue

            frames = rec.get("frames")
            if not isinstance(frames, list) or not frames:
                continue

            for fr in frames:
                if not isinstance(fr, dict):
                    continue
                file = fr.get("file")
                if not isinstance(file, str) or not file:
                    continue

                p = Path(file)
                if p.is_absolute():
                    if p.exists():
                        return p
                    continue

                # 兼容两种相对路径：
                # 1) 相对 captures_dir：group_xxx/cam0_xxx.bmp
                # 2) 相对 repo_root：data/captures_xxx/.../cam0_xxx.bmp
                candidate = (captures_dir / p).resolve()
                if candidate.exists():
                    return candidate

                candidate2 = (repo_root / p).resolve()
                if candidate2.exists():
                    return candidate2

    raise RuntimeError(f"no readable image found in captures: {captures_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ultralytics YOLOv8 .pt tennis detection utils (CPU)")

    repo_root = REPO_ROOT

    p.add_argument(
        "--model",
        default=str(repo_root / "data" / "models" / "best.pt"),
        help="Path to best.pt",
    )
    p.add_argument(
        "--captures-dir",
        default=str(repo_root / "data" / "captures_master_slave" / "tennis_offline"),
        help="captures directory containing metadata.jsonl",
    )
    p.add_argument(
        "--image",
        default="",
        help="Optional explicit image path (overrides --captures-dir auto pick)",
    )
    p.add_argument("--conf", type=float, default=0.005, help="confidence threshold")
    p.add_argument("--max-det", type=int, default=20, help="max detections")

    p.add_argument(
        "--all",
        action="store_true",
        help="process ALL images in metadata.jsonl and write JSONL + visualizations",
    )
    p.add_argument("--max-groups", type=int, default=0, help="process at most N groups when --all (0=all)")
    p.add_argument("--max-images", type=int, default=0, help="process at most N images when --all (0=all)")
    p.add_argument(
        "--out-jsonl",
        default=str(repo_root / "data" / "tools_output" / "tennis_test_ultralytics_detections.jsonl"),
        help="output JSONL path when --all",
    )
    p.add_argument(
        "--out-vis-dir",
        default=str(repo_root / "data" / "tools_output" / "tennis_test_ultralytics_vis"),
        help="output directory for annotated images (when --all or --save-vis)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing visualization images",
    )
    p.add_argument(
        "--out-json",
        default=str(repo_root / "data" / "tools_output" / "best_pt_detections_ultralytics.json"),
        help="output json path",
    )
    p.add_argument(
        "--save-vis",
        action="store_true",
        help="save annotated visualization image(s)",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise RuntimeError(f"model not found: {model_path}")

    model = _load_yolo_model(model_path)

    # 延迟 import：只在真正需要画框时才加载。
    from tennis3d.preprocess import draw_detections

    out_vis_dir = Path(args.out_vis_dir).resolve()

    if bool(args.all):
        captures_dir = Path(args.captures_dir).resolve()
        meta_path = captures_dir / "metadata.jsonl"
        if not meta_path.exists():
            raise RuntimeError(f"metadata.jsonl not found: {meta_path}")

        out_jsonl = Path(args.out_jsonl).resolve()

        max_groups = int(args.max_groups)
        max_images = int(args.max_images)

        groups_done = 0
        images_done = 0
        images_with_ball = 0
        out_records: list[dict[str, Any]] = []

        for group in _iter_capture_groups(meta_path):
            frames = group.get("frames")
            if not isinstance(frames, list):
                continue

            if max_groups > 0 and groups_done >= max_groups:
                break
            groups_done += 1

            group_seq = int(group.get("group_seq", -1))

            for fr in frames:
                if not isinstance(fr, dict):
                    continue

                file_field = fr.get("file")
                if not isinstance(file_field, str):
                    continue

                img_path = _resolve_image_path(REPO_ROOT, captures_dir, file_field)
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f"failed to read image: {img_path}")

                boxes_out, dets_for_vis = _predict_boxes(
                    model=model,
                    img_bgr=img,
                    conf=float(args.conf),
                    max_det=int(args.max_det),
                    device="cpu",
                )

                # 说明：全量模式默认保存可视化（符合你“要可视化检测框”的需求）。
                vis = draw_detections(img, dets_for_vis)
                rel = _safe_relpath(img_path, captures_dir)
                out_path = (out_vis_dir / rel).with_suffix(".jpg")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if out_path.exists() and not bool(args.overwrite):
                    pass
                else:
                    ok = cv2.imwrite(str(out_path), vis)
                    if not ok:
                        raise RuntimeError(f"failed to write visualization: {out_path}")
                vis_path = str(out_path)

                out_records.append(
                    {
                        "group_seq": group_seq,
                        "cam_index": int(fr.get("cam_index", -1)),
                        "frame_num": int(fr.get("frame_num", -1)),
                        "serial": str(fr.get("serial", "")),
                        "trigger_index": int(fr.get("trigger_index", -1)),
                        "host_timestamp": int(fr.get("host_timestamp", -1)),
                        "image": str(img_path),
                        "vis": vis_path,
                        "num_boxes": len(boxes_out),
                        "boxes": [asdict(b) for b in boxes_out],
                    }
                )

                images_done += 1
                if boxes_out:
                    images_with_ball += 1

                if max_images > 0 and images_done >= max_images:
                    break

            if max_images > 0 and images_done >= max_images:
                break

        _write_jsonl(out_jsonl, out_records)

        print(f"Done. groups={groups_done} images={images_done} images_with_ball={images_with_ball}")
        print(f"Done. model={model_path}")
        print(f"Done. vis_dir={out_vis_dir}")
        print(f"Done. out_jsonl={out_jsonl}")
        return 0

    # 单图模式：输出一个 JSON，可选保存可视化。
    if str(args.image).strip():
        image_path = Path(str(args.image)).resolve()
    else:
        image_path = _pick_first_image_from_captures(Path(args.captures_dir))

    if not image_path.exists():
        raise RuntimeError(f"image not found: {image_path}")

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"failed to read image: {image_path}")

    boxes_out, dets_for_vis = _predict_boxes(
        model=model,
        img_bgr=img,
        conf=float(args.conf),
        max_det=int(args.max_det),
        device="cpu",
    )

    out_json_path = Path(args.out_json).resolve()
    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    out_obj: dict[str, Any] = {
        "model": str(model_path),
        "image": str(image_path),
        "conf": float(args.conf),
        "max_det": int(args.max_det),
        "num_boxes": len(boxes_out),
        "boxes": [asdict(b) for b in boxes_out],
    }

    if bool(args.save_vis):
        out_vis_dir.mkdir(parents=True, exist_ok=True)
        vis = draw_detections(img, dets_for_vis)
        out_path = (out_vis_dir / f"{Path(image_path).stem}.jpg").resolve()
        if (not out_path.exists()) or bool(args.overwrite):
            ok = cv2.imwrite(str(out_path), vis)
            if not ok:
                raise RuntimeError(f"failed to write visualization: {out_path}")
        out_obj["vis"] = str(out_path)

    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    # ASCII 输出，避免 Windows 终端乱码
    print(f"OK. model={model_path}")
    print(f"OK. image={image_path}")
    print(f"OK. boxes={len(boxes_out)} out_json={out_json_path}")
    if bool(args.save_vis):
        print(f"OK. vis_dir={out_vis_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
