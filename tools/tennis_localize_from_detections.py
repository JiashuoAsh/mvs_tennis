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
import datetime as _dt
import json
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np

from tennis3d.geometry.calibration import load_calibration
from tennis3d.localization import localize_balls
from tennis3d.models import Detection


class _UltralyticsGroupAgg(TypedDict):
    """ultralytics --all JSONL 的按组聚合中间结构。"""

    group_seq: int
    host_timestamps: list[int]
    detections: dict[str, list[dict[str, Any]]]
    images: dict[str, str]


def _as_detection(obj: Any) -> Detection | None:
    """把 JSON 字典转换为 Detection。

    兼容字段：
        - bbox: [x1,y1,x2,y2]
        - 或 x1/y1/x2/y2

    Returns:
        Detection | None：解析失败则返回 None。
    """

    if not isinstance(obj, dict):
        return None

    def _to_float(v: Any) -> float | None:
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

    bbox = obj.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        try:
            x1, y1, x2, y2 = map(float, bbox)
        except Exception:
            return None
    else:
        x1v = _to_float(obj.get("x1"))
        y1v = _to_float(obj.get("y1"))
        x2v = _to_float(obj.get("x2"))
        y2v = _to_float(obj.get("y2"))
        if x1v is None or y1v is None or x2v is None or y2v is None:
            return None
        x1, y1, x2, y2 = x1v, y1v, x2v, y2v

    score_v = _to_float(obj.get("score"))
    if score_v is None:
        # ultralytics 原始输出字段名是 conf；聚合后会变成 score，这里做个兜底
        score_v = _to_float(obj.get("conf"))
    if score_v is None:
        return None
    score = score_v

    cls_raw = obj.get("cls")
    try:
        cls_i = int(cls_raw) if cls_raw is not None else -1
    except Exception:
        cls_i = -1

    return Detection(bbox=(float(x1), float(y1), float(x2), float(y2)), score=float(score), cls=int(cls_i))


def _safe_float3(x: Any) -> list[float] | None:
    try:
        a = np.asarray(x, dtype=np.float64).reshape(3)
        return [float(a[0]), float(a[1]), float(a[2])]
    except Exception:
        return None


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


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL（每行一个 JSON 对象）。

    说明：
        - ultralytics_best_pt_smoketest.py 的 --all 输出是严格 JSONL。
        - 该文件与 captures/metadata.jsonl 不同，不需要多行对象解析。
    """

    out: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _bbox_center(bbox_xyxy: Any) -> tuple[float, float] | None:
    if not isinstance(bbox_xyxy, list) or len(bbox_xyxy) != 4:
        return None
    try:
        x1, y1, x2, y2 = map(float, bbox_xyxy)
    except Exception:
        return None
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _group_ultralytics_all_jsonl(
    records: list[dict[str, Any]],
    *,
    min_score: float,
) -> list[dict[str, Any]]:
    """把 ultralytics --all 的逐图 JSONL 聚合成按 group 的 detections 记录。

    输入（每行）关键字段：
        - group_seq: 组序号
        - serial: 相机序列号（应与标定文件的相机 key 一致）
        - host_timestamp: epoch ms（用于构造 base_ts）
        - image: 图片路径（用于回溯定位）
        - boxes: [{xyxy, conf, cls, name}, ...]

    输出格式与本脚本旧逻辑兼容：
        - 顶层 list
        - 每条包含 detections: {camera_key: [det, ...]}
        - det 至少包含 bbox/score/cls/center
    """

    groups: dict[int, _UltralyticsGroupAgg] = {}
    for r in records:
        if not isinstance(r, dict):
            continue

        group_seq_raw = r.get("group_seq")
        if group_seq_raw is None:
            continue
        try:
            group_seq = int(group_seq_raw)
        except Exception:
            continue

        serial = str(r.get("serial", "") or "").strip()
        if not serial:
            continue

        host_ts_raw = r.get("host_timestamp")
        host_ts: int | None = None
        if host_ts_raw is not None:
            try:
                host_ts = int(host_ts_raw)
            except Exception:
                host_ts = None

        g = groups.get(group_seq)
        if g is None:
            g = cast(
                _UltralyticsGroupAgg,
                {
                    "group_seq": int(group_seq),
                    "host_timestamps": [],
                    "detections": {},
                    "images": {},
                },
            )
            groups[group_seq] = g

        if host_ts is not None:
            g["host_timestamps"].append(int(host_ts))

        image_raw = r.get("image")
        if isinstance(image_raw, str) and str(image_raw).strip():
            g["images"][serial] = str(image_raw)

        boxes = r.get("boxes")
        if not isinstance(boxes, list) or not boxes:
            continue

        det_list: list[dict[str, Any]] = []
        for b in boxes:
            if not isinstance(b, dict):
                continue
            conf_raw = b.get("conf")
            if conf_raw is None:
                continue
            try:
                score = float(conf_raw)
            except Exception:
                continue
            if score < float(min_score):
                continue
            center = _bbox_center(b.get("xyxy"))
            if center is None:
                continue

            bbox = b.get("xyxy")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue

            cls_raw = b.get("cls")
            try:
                cls_i = int(cls_raw) if cls_raw is not None else -1
            except Exception:
                cls_i = -1

            det_list.append(
                {
                    "bbox": [float(x) for x in bbox],
                    "score": float(score),
                    "cls": int(cls_i),
                    "center": [float(center[0]), float(center[1])],
                }
            )

        if not det_list:
            continue

        dets_by_cam = g["detections"].get(serial)
        if dets_by_cam is None:
            g["detections"][serial] = det_list
        else:
            # 理论上同一相机同一组只会出现一次；若重复出现则合并。
            dets_by_cam.extend(det_list)

    out: list[dict[str, Any]] = []
    for group_seq in sorted(groups.keys()):
        g = groups[group_seq]
        ts_list = g.get("host_timestamps") if isinstance(g.get("host_timestamps"), list) else []

        base_ts_epoch_ms = None
        if ts_list:
            try:
                ts_sorted = sorted(int(x) for x in ts_list)
                base_ts_epoch_ms = int(ts_sorted[len(ts_sorted) // 2])
            except Exception:
                base_ts_epoch_ms = None

        base_ts_str = None
        if base_ts_epoch_ms is not None:
            try:
                base_ts_str = _dt.datetime.fromtimestamp(base_ts_epoch_ms / 1000.0).isoformat(
                    timespec="milliseconds"
                )
            except Exception:
                base_ts_str = None

        out.append(
            {
                "group_seq": int(group_seq),
                "base_ts_str": base_ts_str,
                "base_ts_epoch_ms": base_ts_epoch_ms,
                "detections": g.get("detections"),
                "images": g.get("images"),
            }
        )

    return out


def build_arg_parser() -> argparse.ArgumentParser:
    # 说明：Windows 终端编码差异较大，这里尽量使用 ASCII，避免 --help 乱码。
    p = argparse.ArgumentParser(description="Triangulate tennis ball 3D position from bboxes + calibration")
    p.add_argument(
        "--detections-json",
        default=str(Path(__file__).resolve().parents[1] / "data" / "tools_output" / "tennis_ultralytics_detections.jsonl"),
        help=(
            "Detections input path (.json or .jsonl). "
            "Default: data/tools_output/tennis_ultralytics_detections.jsonl (from tools/ultralytics_best_pt_smoketest.py --all)."
        ),
    )
    p.add_argument(
        "--calib",
        default=str(
            Path(__file__).resolve().parents[1] / "data" / "calibration" / "camera_extrinsics_C_T_B.json"
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
        default=0.005,
        help="Ignore detections below this confidence",
    )
    p.add_argument(
        "--require-views",
        type=int,
        default=2,
        help="Minimum number of camera views required (default: 2)",
    )
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
    p.add_argument(
        "--max-balls-per-group",
        type=int,
        default=0,
        help="Keep at most N balls per group (0=all)",
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
        raise RuntimeError(
            "找不到 detections 输入文件："
            f"{detections_path}\n"
            "提示：你可以先运行 ultralytics 离线检测生成它，例如：\n"
            "  uv run python tools/ultralytics_best_pt_smoketest.py --all --captures-dir data/captures_master_slave/tennis_offline --model data/models/best.pt\n"
        )

    calib = load_calibration(calib_path)

    # 说明：
    # - .json  ：tennis3d.offline_detect.pipeline 输出（顶层 list）
    # - .jsonl ：ultralytics_best_pt_smoketest.py --all 输出（逐图记录，需要聚合成按 group 的记录）
    if detections_path.suffix.lower() == ".jsonl":
        raw = _iter_jsonl(detections_path)
        records = _group_ultralytics_all_jsonl(raw, min_score=float(args.min_score))
    else:
        with detections_path.open("r", encoding="utf-8") as f:
            records = json.load(f)

        if not isinstance(records, list):
            raise RuntimeError("detections-json 顶层必须是 list")

    if args.max_frames is not None:
        records = records[: max(0, int(args.max_frames))]

    out_records: list[dict[str, Any]] = []
    balls_total = 0

    for rec in records:
        dets_by_cam = rec.get("detections") if isinstance(rec, dict) else None
        # 说明：保持输出“每个 group 一条记录”，但每条记录里 balls 是 0..N。
        # 这比“每相机只取一个最高分框”更稳健：可以处理多球/误检/跨视角不一致。
        balls_out: list[dict[str, Any]] = []

        if isinstance(dets_by_cam, dict):
            dets_by_camera: dict[str, list[Detection]] = {}
            det_count_by_camera: dict[str, int] = {}
            for cam_name, raw in dets_by_cam.items():
                cam_name = str(cam_name)
                det_list_raw = _extract_detections_for_camera(raw)
                dets: list[Detection] = []
                for d in det_list_raw:
                    det = _as_detection(d)
                    if det is None:
                        continue
                    dets.append(det)
                det_count_by_camera[cam_name] = int(len(dets))
                if dets:
                    dets_by_camera[cam_name] = dets

            locs = localize_balls(
                calib=calib,
                detections_by_camera=dets_by_camera,
                min_score=float(args.min_score),
                require_views=int(args.require_views),
                max_detections_per_camera=int(args.max_detections_per_camera),
                max_reproj_error_px=float(args.max_reproj_error_px),
                max_uv_match_dist_px=float(args.max_uv_match_dist_px),
                merge_dist_m=float(args.merge_dist_m),
            )

            if int(args.max_balls_per_group) > 0:
                locs = locs[: int(args.max_balls_per_group)]

            for i, loc in enumerate(locs):
                err_pxs = [float(e.error_px) for e in (loc.reprojection_errors or [])]
                med_err = float(np.median(np.asarray(err_pxs, dtype=np.float64))) if err_pxs else None
                max_err = float(max(err_pxs)) if err_pxs else None

                b: dict[str, Any] = {
                    "ball_id": int(i),
                    "ball_3d_world": _safe_float3(loc.X_w),
                    "ball_3d_camera": {k: _safe_float3(v) for k, v in (loc.X_c_by_camera or {}).items()},
                    "used_cameras": list((loc.points_uv or {}).keys()),
                    "num_views": int(len(loc.points_uv or {})),
                    "quality": float(loc.quality),
                    "median_reproj_error_px": float(med_err) if med_err is not None else None,
                    "max_reproj_error_px": float(max_err) if max_err is not None else None,
                    "ball_center_uv": {
                        k: [float(uv[0]), float(uv[1])] for k, uv in (loc.points_uv or {}).items()
                    },
                    "reprojection_errors": [
                        {
                            "camera": e.camera,
                            "uv": [float(e.uv[0]), float(e.uv[1])],
                            "uv_hat": [float(e.uv_hat[0]), float(e.uv_hat[1])],
                            "error_px": float(e.error_px),
                        }
                        for e in (loc.reprojection_errors or [])
                    ],
                    "detections": {
                        k: {
                            "bbox": [float(d.bbox[0]), float(d.bbox[1]), float(d.bbox[2]), float(d.bbox[3])],
                            "score": float(d.score),
                            "cls": int(d.cls),
                            "center": [float(d.center[0]), float(d.center[1])],
                        }
                        for k, d in (loc.detections or {}).items()
                    },
                    "detection_indices": {k: int(v) for k, v in (loc.detection_indices or {}).items()},
                }
                balls_out.append(b)

            balls_total += int(len(balls_out))
        else:
            det_count_by_camera = {}

        out_records.append(
            {
                "group_seq": rec.get("group_seq") if isinstance(rec, dict) else None,
                "base_ts_str": rec.get("base_ts_str") if isinstance(rec, dict) else None,
                "base_ts_epoch_ms": rec.get("base_ts_epoch_ms") if isinstance(rec, dict) else None,
                "images": rec.get("images") if isinstance(rec, dict) else None,
                "num_detections_by_camera": det_count_by_camera,
                "balls": balls_out,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_records, f, ensure_ascii=False, indent=2)

    print(f"Done. groups: {len(out_records)}")
    print(f"Done. balls: {int(balls_total)}")
    print(f"Done. OUT: {out_path}")


if __name__ == "__main__":
    main()

