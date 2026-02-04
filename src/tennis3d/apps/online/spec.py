"""在线模式运行规格（从 CLI/配置文件统一抽象）。

该模块的目标是把“参数来源”（CLI vs 配置文件）与“运行逻辑”（打开相机、跑 pipeline）解耦。

说明：
- 该模块不触达硬件与 IO（不打开相机、不写文件），只做参数整理与校验。
- 运行逻辑在 `tennis3d.apps.online.runtime`。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence, cast

from tennis3d.config import OnlineAppConfig
from tennis3d.trajectory import CurveStageConfig

from mvs.roi import normalize_roi

from .cli import _TERMINAL_PRINT_MODE


_GROUP_BY = Literal["trigger_index", "frame_num", "sequence"]


@dataclass(frozen=True, slots=True)
class OnlineRunSpec:
    """在线模式运行规格。

    约定：
    - 该结构体是 entry 层与 runtime 层之间的“稳定边界”。
    - 字段尽量保持语义化与单位明确（ms/s、px、m）。

    Attributes:
        serials: 相机序列号列表（有序）。
        mvimport_dir: MVS 官方示例绑定目录（MvImport），None 表示依赖环境变量/自动探测。
        dll_dir: 包含 MvCameraControl.dll 的目录，None 表示依赖环境变量/自动探测。
        calib_path: 标定文件路径。
        detector_name: 检测器后端名（fake/color/rknn/pt）。
        model_path: 模型路径（某些 detector 需要）。
        pt_device: detector=pt 时的推理设备（cpu/cuda:0/...）。
        min_score: 置信度阈值（低于则忽略）。
        require_views: 需要的最少视角数。
        max_detections_per_camera: 每相机最多保留的候选数（TopK）。
        max_reproj_error_px: 最大重投影误差（像素）。
        max_uv_match_dist_px: 投影点与检测点匹配最大距离（像素）。
        merge_dist_m: 3D 候选去重/合并距离（米）。
        group_by: 组包键。
        timeout_ms: 单帧取流超时。
        group_timeout_ms: 组包等待超时。
        max_pending_groups: 最大 pending 组数。
        max_groups: 最大组数（0 表示无限）。
        max_wait_seconds: 若连续这么久拿不到完整组，则退出（0 表示无限）。
        out_path: JSONL 输出路径（None 表示不写盘）。
        out_jsonl_only_when_balls: 仅当 balls 非空时才写盘。
        out_jsonl_flush_every_records: 每 N 条 flush 一次（0 表示禁用按条数 flush）。
        out_jsonl_flush_interval_s: 超过该秒数则 flush（0 表示禁用按时间 flush）。
        terminal_print_mode: 终端逐组输出策略（best/all/none）。
        terminal_status_interval_s: 周期性状态输出间隔（秒，0 关闭）。
        trigger_source: 触发源（在主从模式下作用于从机）。
        master_serial: 主机序列号（空表示非主从）。
        master_line_out: 主机输出线选择器。
        master_line_source: 主机输出线源。
        master_line_mode: 主机输出线模式。
        soft_trigger_fps: 软件触发帧率。
        trigger_activation: 触发沿。
        trigger_cache_enable: 尝试启用 TriggerCacheEnable。
        curve_cfg: 轨迹拟合 stage 配置。
        pixel_format: PixelFormat（空表示不设置）。
        image_width: 相机 ROI 宽（None 表示不设置）。
        image_height: 相机 ROI 高（None 表示不设置）。
        image_offset_x: 相机 ROI 偏移 X。
        image_offset_y: 相机 ROI 偏移 Y。
        time_sync_mode: 时间轴模式。
        time_mapping_warmup_groups: 在线映射 warmup 组数。
        time_mapping_window_groups: 在线映射滑窗组数。
        time_mapping_update_every_groups: 在线映射更新频率。
        time_mapping_min_points: 拟合最小点数。
        time_mapping_hard_outlier_ms: 硬离群阈值（毫秒）。
        detector_crop_size: 软件裁剪窗口大小（0 关闭）。
        detector_crop_smooth_alpha: 软件裁剪平滑系数。
        detector_crop_max_step_px: 软件裁剪最大移动步长（像素）。
        detector_crop_reset_after_missed: 连续 missed 后重置预测。
        camera_aoi_runtime: 是否启用相机 AOI 运行中平移。
        camera_aoi_update_every_groups: 相机 AOI 更新频率。
        camera_aoi_min_move_px: 相机 AOI 最小移动阈值。
        camera_aoi_smooth_alpha: 相机 AOI 平滑系数。
        camera_aoi_max_step_px: 相机 AOI 最大步长。
        camera_aoi_recenter_after_missed: 连续 missed 后逐步回中。
    """

    serials: list[str]
    mvimport_dir: str | None
    dll_dir: str | None

    calib_path: Path

    detector_name: str
    model_path: Path | None
    pt_device: str

    min_score: float
    require_views: int
    max_detections_per_camera: int
    max_reproj_error_px: float
    max_uv_match_dist_px: float
    merge_dist_m: float

    group_by: _GROUP_BY
    timeout_ms: int
    group_timeout_ms: int
    max_pending_groups: int
    max_groups: int
    max_wait_seconds: float

    out_path: Path | None
    out_jsonl_only_when_balls: bool
    out_jsonl_flush_every_records: int
    out_jsonl_flush_interval_s: float

    terminal_print_mode: _TERMINAL_PRINT_MODE
    terminal_status_interval_s: float

    trigger_source: str
    master_serial: str
    master_line_out: str
    master_line_source: str
    master_line_mode: str
    soft_trigger_fps: float
    trigger_activation: str
    trigger_cache_enable: bool

    curve_cfg: CurveStageConfig

    pixel_format: str
    image_width: int | None
    image_height: int | None
    image_offset_x: int
    image_offset_y: int

    time_sync_mode: str
    time_mapping_warmup_groups: int
    time_mapping_window_groups: int
    time_mapping_update_every_groups: int
    time_mapping_min_points: int
    time_mapping_hard_outlier_ms: float

    detector_crop_size: int
    detector_crop_smooth_alpha: float
    detector_crop_max_step_px: int
    detector_crop_reset_after_missed: int

    camera_aoi_runtime: bool
    camera_aoi_update_every_groups: int
    camera_aoi_min_move_px: int
    camera_aoi_smooth_alpha: float
    camera_aoi_max_step_px: int
    camera_aoi_recenter_after_missed: int

    def validate(self) -> None:
        """校验运行规格。

        Raises:
            ValueError: 当参数不满足约束时。
        """

        if not self.serials:
            raise ValueError("Please provide --serial (one or more).")

        if self.detector_crop_size < 0:
            raise ValueError("--detector-crop-size must be >= 0")
        if self.detector_crop_max_step_px < 0:
            raise ValueError("--detector-crop-max-step-px must be >= 0")
        if self.detector_crop_reset_after_missed < 0:
            raise ValueError("--detector-crop-reset-after-missed must be >= 0")
        if not (0.0 <= float(self.detector_crop_smooth_alpha) <= 1.0):
            raise ValueError("--detector-crop-smooth-alpha must be in [0,1]")

        if self.camera_aoi_update_every_groups < 0:
            raise ValueError("--camera-aoi-update-every-groups must be >= 0")
        if bool(self.camera_aoi_runtime) and self.camera_aoi_update_every_groups < 1:
            raise ValueError(
                "--camera-aoi-update-every-groups must be >= 1 when --camera-aoi-runtime is set"
            )
        if self.camera_aoi_min_move_px < 0:
            raise ValueError("--camera-aoi-min-move-px must be >= 0")
        if self.camera_aoi_max_step_px < 0:
            raise ValueError("--camera-aoi-max-step-px must be >= 0")
        if self.camera_aoi_recenter_after_missed < 0:
            raise ValueError("--camera-aoi-recenter-after-missed must be >= 0")
        if not (0.0 <= float(self.camera_aoi_smooth_alpha) <= 1.0):
            raise ValueError("--camera-aoi-smooth-alpha must be in [0,1]")

        if self.out_jsonl_flush_every_records < 0:
            raise ValueError("--out-jsonl-flush-every-records must be >= 0")
        if self.out_jsonl_flush_interval_s < 0:
            raise ValueError("--out-jsonl-flush-interval-s must be >= 0")
        if self.terminal_status_interval_s < 0:
            raise ValueError("--terminal-status-interval-s must be >= 0")

        if self.master_serial and self.master_serial not in self.serials:
            raise ValueError("--master-serial must be one of the provided --serial values.")


def build_spec_from_config(cfg: OnlineAppConfig) -> OnlineRunSpec:
    """从在线配置文件构建运行规格。"""

    out_path = Path(cfg.out_jsonl).resolve() if cfg.out_jsonl is not None else None

    spec = OnlineRunSpec(
        serials=list(cfg.serials),
        mvimport_dir=str(cfg.mvimport_dir) if cfg.mvimport_dir is not None else None,
        dll_dir=str(cfg.dll_dir) if cfg.dll_dir is not None else None,
        calib_path=Path(cfg.calib).resolve(),
        detector_name=str(cfg.detector),
        model_path=Path(cfg.model).resolve() if cfg.model is not None else None,
        pt_device=str(getattr(cfg, "pt_device", "cpu") or "cpu").strip() or "cpu",
        min_score=float(cfg.min_score),
        require_views=int(cfg.require_views),
        max_detections_per_camera=int(cfg.max_detections_per_camera),
        max_reproj_error_px=float(cfg.max_reproj_error_px),
        max_uv_match_dist_px=float(cfg.max_uv_match_dist_px),
        merge_dist_m=float(cfg.merge_dist_m),
        group_by=cast(_GROUP_BY, str(cfg.group_by)),
        timeout_ms=int(cfg.timeout_ms),
        group_timeout_ms=int(cfg.group_timeout_ms),
        max_pending_groups=int(cfg.max_pending_groups),
        max_groups=int(cfg.max_groups),
        max_wait_seconds=float(getattr(cfg, "max_wait_seconds", 0.0)),
        out_path=out_path,
        out_jsonl_only_when_balls=bool(getattr(cfg, "out_jsonl_only_when_balls", False)),
        out_jsonl_flush_every_records=int(getattr(cfg, "out_jsonl_flush_every_records", 1)),
        out_jsonl_flush_interval_s=float(getattr(cfg, "out_jsonl_flush_interval_s", 0.0)),
        terminal_print_mode=cast(
            _TERMINAL_PRINT_MODE, str(getattr(cfg, "terminal_print_mode", "best"))
        ),
        terminal_status_interval_s=float(getattr(cfg, "terminal_status_interval_s", 0.0)),
        trigger_source=str(cfg.trigger.trigger_source),
        master_serial=str(cfg.trigger.master_serial),
        master_line_out=str(cfg.trigger.master_line_out),
        master_line_source=str(cfg.trigger.master_line_source),
        master_line_mode=str(cfg.trigger.master_line_mode),
        soft_trigger_fps=float(cfg.trigger.soft_trigger_fps),
        trigger_activation=str(cfg.trigger.trigger_activation),
        trigger_cache_enable=bool(cfg.trigger.trigger_cache_enable),
        curve_cfg=cfg.curve,
        pixel_format=str(getattr(cfg, "pixel_format", "") or "").strip(),
        image_width=getattr(cfg, "image_width", None),
        image_height=getattr(cfg, "image_height", None),
        image_offset_x=int(getattr(cfg, "image_offset_x", 0)),
        image_offset_y=int(getattr(cfg, "image_offset_y", 0)),
        time_sync_mode=str(cfg.time_sync_mode),
        time_mapping_warmup_groups=int(cfg.time_mapping_warmup_groups),
        time_mapping_window_groups=int(cfg.time_mapping_window_groups),
        time_mapping_update_every_groups=int(cfg.time_mapping_update_every_groups),
        time_mapping_min_points=int(cfg.time_mapping_min_points),
        time_mapping_hard_outlier_ms=float(cfg.time_mapping_hard_outlier_ms),
        detector_crop_size=int(getattr(cfg, "detector_crop_size", 0)),
        detector_crop_smooth_alpha=float(getattr(cfg, "detector_crop_smooth_alpha", 0.2)),
        detector_crop_max_step_px=int(getattr(cfg, "detector_crop_max_step_px", 120)),
        detector_crop_reset_after_missed=int(getattr(cfg, "detector_crop_reset_after_missed", 8)),
        camera_aoi_runtime=bool(getattr(cfg, "camera_aoi_runtime", False)),
        camera_aoi_update_every_groups=int(getattr(cfg, "camera_aoi_update_every_groups", 2)),
        camera_aoi_min_move_px=int(getattr(cfg, "camera_aoi_min_move_px", 8)),
        camera_aoi_smooth_alpha=float(getattr(cfg, "camera_aoi_smooth_alpha", 0.3)),
        camera_aoi_max_step_px=int(getattr(cfg, "camera_aoi_max_step_px", 160)),
        camera_aoi_recenter_after_missed=int(getattr(cfg, "camera_aoi_recenter_after_missed", 30)),
    )

    # 说明：配置文件通常已经在 loader 里校验过，但这里仍做一次统一校验，保证 CLI/配置一致。
    spec.validate()
    return spec


def build_spec_from_args(args: Any) -> OnlineRunSpec:
    """从 argparse Namespace 构建运行规格。"""

    serials = [s.strip() for s in (getattr(args, "serial", None) or []) if str(s).strip()]

    mvimport_dir = str(getattr(args, "mvimport_dir", None) or "").strip() or None
    dll_dir = getattr(args, "dll_dir", None)

    calib_path = Path(getattr(args, "calib")).resolve()
    detector_name = str(getattr(args, "detector"))
    model_raw = str(getattr(args, "model", "") or "").strip()
    model_path = Path(model_raw).resolve() if model_raw else None
    pt_device = str(getattr(args, "pt_device", "cpu") or "cpu").strip() or "cpu"

    out_jsonl = str(getattr(args, "out_jsonl", "") or "").strip()
    out_path = Path(out_jsonl).resolve() if out_jsonl else None

    pixel_format = str(getattr(args, "pixel_format", "") or "").strip()
    try:
        image_width, image_height, image_offset_x, image_offset_y = normalize_roi(
            image_width=int(getattr(args, "image_width", 0)),
            image_height=int(getattr(args, "image_height", 0)),
            image_offset_x=int(getattr(args, "image_offset_x", 0)),
            image_offset_y=int(getattr(args, "image_offset_y", 0)),
        )
    except ValueError as exc:
        # 说明：保持旧行为：直接复用原错误文本。
        raise ValueError(str(exc)) from exc

    spec = OnlineRunSpec(
        serials=serials,
        mvimport_dir=mvimport_dir,
        dll_dir=dll_dir,
        calib_path=calib_path,
        detector_name=detector_name,
        model_path=model_path,
        pt_device=pt_device,
        min_score=float(getattr(args, "min_score")),
        require_views=int(getattr(args, "require_views")),
        max_detections_per_camera=int(getattr(args, "max_detections_per_camera")),
        max_reproj_error_px=float(getattr(args, "max_reproj_error_px")),
        max_uv_match_dist_px=float(getattr(args, "max_uv_match_dist_px")),
        merge_dist_m=float(getattr(args, "merge_dist_m")),
        group_by=cast(_GROUP_BY, str(getattr(args, "group_by"))),
        timeout_ms=int(getattr(args, "timeout_ms")),
        group_timeout_ms=int(getattr(args, "group_timeout_ms")),
        max_pending_groups=int(getattr(args, "max_pending_groups")),
        max_groups=int(getattr(args, "max_groups")),
        max_wait_seconds=float(getattr(args, "max_wait_seconds", 0.0)),
        out_path=out_path,
        out_jsonl_only_when_balls=bool(getattr(args, "out_jsonl_only_when_balls", False)),
        out_jsonl_flush_every_records=int(getattr(args, "out_jsonl_flush_every_records", 1)),
        out_jsonl_flush_interval_s=float(getattr(args, "out_jsonl_flush_interval_s", 0.0)),
        terminal_print_mode=cast(
            _TERMINAL_PRINT_MODE, str(getattr(args, "terminal_print_mode", "best"))
        ),
        terminal_status_interval_s=float(getattr(args, "terminal_status_interval_s", 0.0)),
        trigger_source=str(getattr(args, "trigger_source")),
        master_serial=str(getattr(args, "master_serial", "") or "").strip(),
        master_line_out=str(getattr(args, "master_line_out")),
        master_line_source=str(getattr(args, "master_line_source")),
        master_line_mode=str(getattr(args, "master_line_mode")),
        soft_trigger_fps=float(getattr(args, "soft_trigger_fps")),
        trigger_activation=str(getattr(args, "trigger_activation")),
        trigger_cache_enable=bool(getattr(args, "trigger_cache_enable")),
        curve_cfg=CurveStageConfig(),
        pixel_format=pixel_format,
        image_width=image_width,
        image_height=image_height,
        image_offset_x=int(image_offset_x),
        image_offset_y=int(image_offset_y),
        time_sync_mode=str(getattr(args, "time_sync_mode")),
        time_mapping_warmup_groups=int(getattr(args, "time_mapping_warmup_groups")),
        time_mapping_window_groups=int(getattr(args, "time_mapping_window_groups")),
        time_mapping_update_every_groups=int(getattr(args, "time_mapping_update_every_groups")),
        time_mapping_min_points=int(getattr(args, "time_mapping_min_points")),
        time_mapping_hard_outlier_ms=float(getattr(args, "time_mapping_hard_outlier_ms")),
        detector_crop_size=int(getattr(args, "detector_crop_size", 0)),
        detector_crop_smooth_alpha=float(getattr(args, "detector_crop_smooth_alpha", 0.2)),
        detector_crop_max_step_px=int(getattr(args, "detector_crop_max_step_px", 120)),
        detector_crop_reset_after_missed=int(getattr(args, "detector_crop_reset_after_missed", 8)),
        camera_aoi_runtime=bool(getattr(args, "camera_aoi_runtime", False)),
        camera_aoi_update_every_groups=int(getattr(args, "camera_aoi_update_every_groups", 2)),
        camera_aoi_min_move_px=int(getattr(args, "camera_aoi_min_move_px", 8)),
        camera_aoi_smooth_alpha=float(getattr(args, "camera_aoi_smooth_alpha", 0.3)),
        camera_aoi_max_step_px=int(getattr(args, "camera_aoi_max_step_px", 160)),
        camera_aoi_recenter_after_missed=int(getattr(args, "camera_aoi_recenter_after_missed", 30)),
    )

    spec.validate()
    return spec
