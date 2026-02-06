"""在线模式运行规格（从 CLI/配置文件统一抽象）。

说明：
- 该模块不触达硬件与 IO（不打开相机、不写文件），只做参数整理与校验。
- 运行逻辑在 `tennis3d_online.runtime`。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from mvs import normalize_roi

from tennis3d.config import OnlineAppConfig
from tennis3d.trajectory import CurveStageConfig

from .cli import _TERMINAL_PRINT_MODE


_GROUP_BY = Literal["frame_num", "sequence"]


@dataclass(frozen=True, slots=True)
class OnlineRunSpec:
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
    terminal_timing: bool

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

    exposure_auto: str
    exposure_time_us: float | None
    gain_auto: str
    gain: float | None

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

        if self.exposure_time_us is not None and float(self.exposure_time_us) <= 0:
            raise ValueError("exposure_time_us must be > 0 (or None to disable)")
        if self.gain is not None and float(self.gain) < 0:
            raise ValueError("gain must be >= 0 (or None to disable)")


def build_spec_from_config(cfg: OnlineAppConfig) -> OnlineRunSpec:
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
        terminal_timing=bool(getattr(cfg, "terminal_timing", False)),
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
        exposure_auto=str(getattr(cfg, "exposure_auto", "Off") if cfg is not None else "Off"),
        exposure_time_us=(
            float(getattr(cfg, "exposure_time_us", 10000.0))
            if getattr(cfg, "exposure_time_us", 10000.0) is not None
            else None
        ),
        gain_auto=str(getattr(cfg, "gain_auto", "Off") if cfg is not None else "Off"),
        gain=(
            float(getattr(cfg, "gain", 12.0))
            if getattr(cfg, "gain", 12.0) is not None
            else None
        ),
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

    spec.validate()
    return spec


def build_spec_from_args(args: Any) -> OnlineRunSpec:
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
        terminal_timing=bool(getattr(args, "terminal_timing", False)),
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
        exposure_auto=str(getattr(args, "exposure_auto", "Off") if args is not None else "Off"),
        exposure_time_us=(
            float(getattr(args, "exposure_time_us", 10000.0))
            if getattr(args, "exposure_time_us", 10000.0) is not None
            else None
        ),
        gain_auto=str(getattr(args, "gain_auto", "Off") if args is not None else "Off"),
        gain=(
            float(getattr(args, "gain", 12.0))
            if getattr(args, "gain", 12.0) is not None
            else None
        ),
        time_sync_mode=str(getattr(args, "time_sync_mode")),
        time_mapping_warmup_groups=int(getattr(args, "time_mapping_warmup_groups")),
        time_mapping_window_groups=int(getattr(args, "time_mapping_window_groups")),
        time_mapping_update_every_groups=int(getattr(args, "time_mapping_update_every_groups")),
        time_mapping_min_points=int(getattr(args, "time_mapping_min_points")),
        time_mapping_hard_outlier_ms=float(getattr(args, "time_mapping_hard_outlier_ms")),
        detector_crop_size=int(getattr(args, "detector_crop_size", 0)),
        detector_crop_smooth_alpha=float(getattr(args, "detector_crop_smooth_alpha", 0.2)),
        detector_crop_max_step_px=int(getattr(args, "detector_crop_max_step_px", 120)),
        detector_crop_reset_after_missed=int(
            getattr(args, "detector_crop_reset_after_missed", 8)
        ),
        camera_aoi_runtime=bool(getattr(args, "camera_aoi_runtime", False)),
        camera_aoi_update_every_groups=int(getattr(args, "camera_aoi_update_every_groups", 2)),
        camera_aoi_min_move_px=int(getattr(args, "camera_aoi_min_move_px", 8)),
        camera_aoi_smooth_alpha=float(getattr(args, "camera_aoi_smooth_alpha", 0.3)),
        camera_aoi_max_step_px=int(getattr(args, "camera_aoi_max_step_px", 160)),
        camera_aoi_recenter_after_missed=int(getattr(args, "camera_aoi_recenter_after_missed", 30)),
    )

    spec.validate()
    return spec
