"""tennis3d 对外稳定调用入口（public API）。

设计目标：
- 让其他程序以稳定的方式 `import tennis3d` / `from tennis3d.api import ...` 调用核心能力。
- 保持 apps/CLI 只是“参数解析 + 调用”，核心逻辑复用 `tennis3d.pipeline`。

注意：
- 这里不做过度封装；优先提供少量、清晰、可组合的函数。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator

from tennis3d.models import Detector
from tennis3d_detectors import create_detector
from tennis3d.geometry.calibration import CalibrationSet, load_calibration
from tennis3d.pipeline import iter_capture_image_groups, run_localization_pipeline
from tennis3d.sync.aligner import SyncAligner


def build_calibration(calib_path: Path) -> CalibrationSet:
    """加载标定文件。

    Args:
        calib_path: 标定文件路径（.json/.yaml/.yml）。

    Returns:
        CalibrationSet。
    """

    return load_calibration(Path(calib_path).resolve())


def build_detector(
    *,
    name: str,
    model_path: Path | None = None,
    conf_thres: float = 0.25,
    pt_device: str = "cpu",
) -> Detector:
    """创建检测器实例。

    Args:
        name: fake/color/rknn/pt。
        model_path: 模型路径（rknn/pt 时必需）。
        conf_thres: 最低置信度阈值。
        pt_device: detector=pt 时的推理设备（例如 cpu/cuda:0/0）。

    Returns:
        Detector。
    """

    return create_detector(
        name=str(name),
        model_path=(Path(model_path) if model_path is not None else None),
        conf_thres=float(conf_thres),
        pt_device=str(pt_device or "cpu").strip() or "cpu",
    )


def iter_localization_from_captures(
    *,
    captures_dir: Path,
    calib: CalibrationSet,
    detector: Detector,
    min_score: float = 0.25,
    require_views: int = 2,
    max_detections_per_camera: int = 10,
    max_reproj_error_px: float = 10.0,
    max_uv_match_dist_px: float = 25.0,
    merge_dist_m: float = 0.08,
    max_groups: int = 0,
    serials: list[str] | None = None,
    include_detection_details: bool = True,
    aligner: SyncAligner | None = None,
) -> Iterator[dict[str, Any]]:
    """从离线 captures/metadata.jsonl 读取图像组并输出 3D 定位结果记录。

    这是离线应用 `tennis3d.apps.offline_localize_from_captures` 的“可复用库接口”。

    Args:
        captures_dir: captures 目录（包含 metadata.jsonl 与图像文件）。
        calib: 已加载的标定集。
        detector: 检测器实例。
        min_score: 最低置信度阈值。
        require_views: 三角化所需最少视角数（>=2）。
        max_detections_per_camera: 每个相机保留的 topK 候选数。
        max_reproj_error_px: 重投影误差阈值（像素）。
        max_uv_match_dist_px: 投影补全匹配阈值（像素）。
        merge_dist_m: 3D 去重阈值（米）。
        max_groups: 最多处理多少组（0 表示不限）。
        serials: 可选相机序列号白名单（子集）。
        include_detection_details: 是否在输出中包含每路选用的 bbox/score/center。
        aligner: 可选对齐器（默认不对齐）。

    Yields:
        每组一个可 JSON 序列化的 dict 记录。
    """

    groups: Iterable[tuple[dict[str, Any], Any]] = iter_capture_image_groups(
        captures_dir=Path(captures_dir).resolve(),
        max_groups=int(max_groups),
        serials=serials,
    )

    return run_localization_pipeline(
        groups=groups,
        calib=calib,
        detector=detector,
        min_score=float(min_score),
        require_views=int(require_views),
        max_detections_per_camera=int(max_detections_per_camera),
        max_reproj_error_px=float(max_reproj_error_px),
        max_uv_match_dist_px=float(max_uv_match_dist_px),
        merge_dist_m=float(merge_dist_m),
        include_detection_details=bool(include_detection_details),
        aligner=aligner,
    )
