"""配置模型（dataclass）与 YAML/JSON 加载。

目标：
- 用 dataclass 表达在线/离线入口所需的关键配置
- 支持从 `.yaml/.yml/.json` 加载

说明：
- 当前 CLI 仍是主入口；配置文件用于“可复用的一组参数”。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from tennis3d.trajectory.curve_stage import CurveStageConfig


def _load_mapping(path: Path) -> dict[str, Any]:
    path = Path(path)
    suf = path.suffix.lower()

    if suf == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    elif suf in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("PyYAML 未安装，无法读取 YAML 配置") from exc

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        raise RuntimeError(f"不支持的配置文件类型: {path}（仅支持 .json/.yaml/.yml）")

    if not isinstance(data, dict):
        raise RuntimeError("配置文件顶层必须是对象（dict）")

    return data


def _as_path(x: Any) -> Path:
    return Path(str(x)).expanduser()


def _as_optional_path(x: Any) -> Path | None:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    return Path(s).expanduser()


def _as_optional_positive_int(x: Any) -> int | None:
    """把可能为 None/0/空的值解析成可选正整数。

    约定：
        - None/""/0 -> None（表示“不设置”）
        - >0 -> int
    """

    if x is None:
        return None
    try:
        v = int(x)
    except Exception:
        s = str(x).strip()
        if not s:
            return None
        v = int(s)

    if v <= 0:
        return None
    return int(v)


def _as_int(x: Any, default: int = 0) -> int:
    if x is None:
        return int(default)
    try:
        return int(x)
    except Exception:
        s = str(x).strip()
        return int(s) if s else int(default)


_DETECTOR = Literal["fake", "color", "rknn", "pt"]
_GROUP_BY = Literal["trigger_index", "frame_num", "sequence"]
_TIME_SYNC_MODE = Literal["frame_host_timestamp", "dev_timestamp_mapping"]
_TERMINAL_PRINT_MODE = Literal["best", "all", "none"]


def _as_detector(x: Any, default: str) -> _DETECTOR:
    s = str(x if x is not None else default).strip().lower()
    if s not in {"fake", "color", "rknn", "pt"}:
        raise RuntimeError(f"unknown detector: {s} (expected: fake|color|rknn|pt)")
    return cast(_DETECTOR, s)


def _as_group_by(x: Any, default: str) -> _GROUP_BY:
    s = str(x if x is not None else default).strip()
    if s not in {"trigger_index", "frame_num", "sequence"}:
        raise RuntimeError(f"unknown group_by: {s} (expected: trigger_index|frame_num|sequence)")
    return cast(_GROUP_BY, s)


def _as_time_sync_mode(x: Any, default: str) -> _TIME_SYNC_MODE:
    s = str(x if x is not None else default).strip()
    if s not in {"frame_host_timestamp", "dev_timestamp_mapping"}:
        raise RuntimeError(
            f"unknown time_sync_mode: {s} (expected: frame_host_timestamp|dev_timestamp_mapping)"
        )
    return cast(_TIME_SYNC_MODE, s)


def _as_terminal_print_mode(x: Any, default: str) -> _TERMINAL_PRINT_MODE:
    s = str(x if x is not None else default).strip().lower()
    if s not in {"best", "all", "none"}:
        raise RuntimeError(f"unknown terminal_print_mode: {s} (expected: best|all|none)")
    return cast(_TERMINAL_PRINT_MODE, s)


def _as_curve_stage_config(x: Any) -> CurveStageConfig:
    """解析可选的 curve stage 配置段。

    说明：
        - 该段用于把 3D 定位输出进一步做轨迹拟合（落点/落地时间/走廊）。
        - 默认 disabled，不影响既有行为。
    """

    if x is None:
        return CurveStageConfig()
    if not isinstance(x, dict):
        raise RuntimeError("config field 'curve' must be an object")

    primary = str(x.get("primary", "v3")).strip().lower()
    if primary not in {"v3", "v2", "v3_legacy"}:
        raise RuntimeError("curve.primary must be one of: v3 | v2 | v3_legacy")

    conf_from = str(x.get("conf_from", "quality")).strip().lower()
    if conf_from not in {"quality", "constant"}:
        raise RuntimeError("curve.conf_from must be 'quality' or 'constant'")

    # 可选浮点字段：先取 raw，再转换，避免 None/Unknown 触发类型告警。
    obs_max_median_reproj_error_px = x.get("obs_max_median_reproj_error_px")
    obs_max_median_reproj_error_px_f = (
        float(obs_max_median_reproj_error_px) if obs_max_median_reproj_error_px is not None else None
    )
    obs_max_ball_3d_std_m = x.get("obs_max_ball_3d_std_m")
    obs_max_ball_3d_std_m_f = float(obs_max_ball_3d_std_m) if obs_max_ball_3d_std_m is not None else None

    cfg = CurveStageConfig(
        enabled=bool(x.get("enabled", False)),
        primary=primary,
        compare_v2=bool(x.get("compare_v2", False)),
        compare_v3_legacy=bool(x.get("compare_v3_legacy", False)),
        max_tracks=int(x.get("max_tracks", 4)),
        association_dist_m=float(x.get("association_dist_m", 0.6)),
        max_missed_s=float(x.get("max_missed_s", 0.6)),
        min_dt_s=float(x.get("min_dt_s", 1e-6)),
        corridor_y_min=float(x.get("corridor_y_min", 0.6)),
        corridor_y_max=float(x.get("corridor_y_max", 1.6)),
        corridor_y_step=float(x.get("corridor_y_step", 0.1)),
        conf_from=conf_from,
        constant_conf=float(x.get("constant_conf", 1.0)),
        y_offset_m=float(x.get("y_offset_m", 0.0)),
        y_negate=bool(x.get("y_negate", False)),
        is_bot_fire=int(x.get("is_bot_fire", -1)),

        # 观测过滤（默认关闭）
        obs_min_views=int(x.get("obs_min_views", 0)),
        obs_min_quality=float(x.get("obs_min_quality", 0.0)),
        obs_max_median_reproj_error_px=obs_max_median_reproj_error_px_f,
        obs_max_ball_3d_std_m=obs_max_ball_3d_std_m_f,

        # episode（默认关闭）
        episode_enabled=bool(x.get("episode_enabled", False)),
        episode_buffer_s=float(x.get("episode_buffer_s", 0.6)),
        # 说明：episode_start 判定固定为“z 方向门控 + y 轴重力一致性”，
        # 因此这里的最小点数要求至少 3（建议更高）。
        episode_min_obs=int(x.get("episode_min_obs", 5)),
        episode_z_dir=int(x.get("episode_z_dir", 0)),
        episode_min_abs_dz_m=float(x.get("episode_min_abs_dz_m", 0.25)),
        episode_min_abs_vz_mps=float(x.get("episode_min_abs_vz_mps", 1.0)),
        episode_gravity_mps2=float(x.get("episode_gravity_mps2", 9.8)),
        episode_gravity_tol_mps2=float(x.get("episode_gravity_tol_mps2", 3.0)),
        episode_stationary_speed_mps=float(x.get("episode_stationary_speed_mps", 0.25)),
        episode_end_if_stationary_s=float(x.get("episode_end_if_stationary_s", 0.35)),
        episode_end_after_predicted_land_s=float(x.get("episode_end_after_predicted_land_s", 0.2)),
        reset_predictor_on_episode_start=bool(x.get("reset_predictor_on_episode_start", True)),
        reset_predictor_on_episode_end=bool(x.get("reset_predictor_on_episode_end", True)),
        feed_curve_only_when_episode_active=bool(x.get("feed_curve_only_when_episode_active", False)),

        # episode 多 track 处置策略（默认关闭）
        episode_lock_single_track=bool(x.get("episode_lock_single_track", False)),
    )

    if cfg.max_tracks <= 0:
        raise RuntimeError("curve.max_tracks must be > 0")
    if cfg.association_dist_m <= 0:
        raise RuntimeError("curve.association_dist_m must be > 0")
    if cfg.max_missed_s < 0:
        raise RuntimeError("curve.max_missed_s must be >= 0")
    if cfg.min_dt_s <= 0:
        raise RuntimeError("curve.min_dt_s must be > 0")
    if cfg.corridor_y_step <= 0:
        raise RuntimeError("curve.corridor_y_step must be > 0")
    if cfg.corridor_y_max <= cfg.corridor_y_min:
        raise RuntimeError("curve.corridor_y_max must be > curve.corridor_y_min")

    if cfg.obs_min_views < 0:
        raise RuntimeError("curve.obs_min_views must be >= 0")
    if cfg.obs_min_quality < 0:
        raise RuntimeError("curve.obs_min_quality must be >= 0")
    if cfg.obs_max_median_reproj_error_px is not None and cfg.obs_max_median_reproj_error_px <= 0:
        raise RuntimeError("curve.obs_max_median_reproj_error_px must be > 0")
    if cfg.obs_max_ball_3d_std_m is not None and cfg.obs_max_ball_3d_std_m <= 0:
        raise RuntimeError("curve.obs_max_ball_3d_std_m must be > 0")

    # episode 相关约束：即使未启用也做基本健壮性校验，避免写错配置后静默异常。
    if cfg.episode_buffer_s < 0:
        raise RuntimeError("curve.episode_buffer_s must be >= 0")
    if cfg.episode_min_obs < 3:
        raise RuntimeError("curve.episode_min_obs must be >= 3")
    if cfg.episode_z_dir not in {-1, 0, 1}:
        raise RuntimeError("curve.episode_z_dir must be -1, 0 or 1")
    if cfg.episode_min_abs_dz_m < 0:
        raise RuntimeError("curve.episode_min_abs_dz_m must be >= 0")
    if cfg.episode_min_abs_vz_mps < 0:
        raise RuntimeError("curve.episode_min_abs_vz_mps must be >= 0")
    if cfg.episode_gravity_mps2 <= 0:
        raise RuntimeError("curve.episode_gravity_mps2 must be > 0")
    if cfg.episode_gravity_tol_mps2 < 0:
        raise RuntimeError("curve.episode_gravity_tol_mps2 must be >= 0")
    if cfg.episode_stationary_speed_mps < 0:
        raise RuntimeError("curve.episode_stationary_speed_mps must be >= 0")
    if cfg.episode_end_if_stationary_s < 0:
        raise RuntimeError("curve.episode_end_if_stationary_s must be >= 0")
    if cfg.episode_end_after_predicted_land_s < 0:
        raise RuntimeError("curve.episode_end_after_predicted_land_s must be >= 0")

    return cfg


@dataclass(frozen=True)
class OfflineAppConfig:
    captures_dir: Path
    calib: Path
    # 可选：仅使用这些相机序列号（serial）参与检测/定位；None 表示使用 captures 中出现的全部相机。
    serials: list[str] | None = None
    detector: _DETECTOR = "color"
    model: Path | None = None
    # detector=pt 时可选：Ultralytics 推理设备。
    # 说明：
    # - 默认 cpu（保持旧行为）。
    # - CUDA 环境可写 cuda:0 / 0 / cuda。
    # - 该字段仅影响 detector=pt，其它 detector 会忽略。
    pt_device: str = "cpu"
    min_score: float = 0.25
    require_views: int = 2
    max_detections_per_camera: int = 10
    max_reproj_error_px: float = 8.0
    max_uv_match_dist_px: float = 25.0
    merge_dist_m: float = 0.08
    max_groups: int = 0
    out_jsonl: Path = Path("data/tools_output/offline_positions_3d.jsonl")

    # 可选：方案B（对齐映射）
    # - frame_host_timestamp：沿用原逻辑，用组内 frames[*].host_timestamp 中位数作为 capture_t_abs
    # - dev_timestamp_mapping：使用预先拟合的 dev_timestamp -> host_ms 映射，把组时间贴近曝光时刻
    time_sync_mode: _TIME_SYNC_MODE = "frame_host_timestamp"
    time_mapping_path: Path | None = None

    # 可选：轨迹拟合后处理（落点/落地时间/走廊）。默认 disabled。
    curve: CurveStageConfig = CurveStageConfig()


@dataclass(frozen=True)
class OnlineTriggerConfig:
    trigger_source: str = "Software"
    master_serial: str = ""
    master_line_out: str = "Line1"
    master_line_source: str = ""
    master_line_mode: str = "Output"
    soft_trigger_fps: float = 5.0
    trigger_activation: str = "RisingEdge"
    trigger_cache_enable: bool = False


@dataclass(frozen=True)
class OnlineAppConfig:
    # MVS 官方 Python 示例绑定目录（MvImport）。可选；不填则依赖环境变量/自动探测。
    mvimport_dir: Path | None
    dll_dir: Path | None
    serials: list[str]

    group_by: _GROUP_BY = "frame_num"
    timeout_ms: int = 1000
    group_timeout_ms: int = 1000
    max_pending_groups: int = 256
    max_groups: int = 0

    # 采集等待策略：若 >0，则在连续这么久拿不到“完整组包”时退出。
    # 说明：该参数主要用于排障（例如硬触发线路未接好时避免无限等待）。
    max_wait_seconds: float = 0.0

    # 可选：相机图像参数（ROI/像素格式）。
    # 约定：
    # - pixel_format 为空表示不设置（沿用相机当前配置）。
    # - image_width/image_height 同时设置才生效；否则将报错。
    pixel_format: str = ""
    image_width: int | None = None
    image_height: int | None = None
    image_offset_x: int = 0
    image_offset_y: int = 0

    calib: Path = Path("data/calibration/example_triple_camera_calib.json")
    detector: _DETECTOR = "fake"
    model: Path | None = None
    # detector=pt 时可选：Ultralytics 推理设备（默认 cpu）。
    # 说明：CUDA 环境可写 cuda:0 / 0 / cuda。
    pt_device: str = "cpu"
    min_score: float = 0.25
    require_views: int = 2
    max_detections_per_camera: int = 10
    max_reproj_error_px: float = 8.0
    max_uv_match_dist_px: float = 25.0
    merge_dist_m: float = 0.08
    out_jsonl: Path | None = None

    # JSONL 写盘策略（性能相关）：
    # - out_jsonl_only_when_balls=true：仅当 balls 非空时才写盘，避免无球阶段刷文件。
    # - out_jsonl_flush_every_records：每写入 N 条记录就 flush 一次（1 表示每条都 flush，最安全但最慢）。
    # - out_jsonl_flush_interval_s：距离上次 flush 超过该秒数则 flush（0 表示禁用基于时间的 flush）。
    # 说明：默认保持旧行为（每条记录 flush），避免改变既有语义。
    out_jsonl_only_when_balls: bool = False
    out_jsonl_flush_every_records: int = 1
    out_jsonl_flush_interval_s: float = 0.0

    # 终端输出策略：
    # - best：只打印每个 group 的最佳球（更安静，默认）
    # - all：打印该 group 的所有球（便于调参/排障）
    # - none：完全静默（不打印逐组球信息；适合追求吞吐或重定向到文件时）
    terminal_print_mode: _TERMINAL_PRINT_MODE = "best"

    # 可选：周期性输出“状态心跳”，用于无球阶段确认程序仍在跑，以及观察吞吐。
    # - 0 表示关闭（默认）。
    # - >0 表示每隔这么多秒打印一行统计。
    terminal_status_interval_s: float = 0.0

    # 在线时间轴：默认仍用 frames[*].host_timestamp 中位数。
    # 若需要更贴近曝光时刻，可启用方案B在线滑窗映射（dev_timestamp -> host_ms）。
    time_sync_mode: _TIME_SYNC_MODE = "frame_host_timestamp"
    time_mapping_warmup_groups: int = 20
    time_mapping_window_groups: int = 200
    time_mapping_update_every_groups: int = 5
    time_mapping_min_points: int = 20
    time_mapping_hard_outlier_ms: float = 50.0

    # 可选：轨迹拟合后处理（落点/落地时间/走廊）。默认 disabled。
    curve: CurveStageConfig = CurveStageConfig()

    # 可选：软件裁剪（动态 ROI）。
    # 说明：
    # - 该裁剪发生在 detector 前，不需要逐帧修改相机 ROI 或标定。
    # - detector 输出 bbox 会自动加回裁剪 offset，保证下游仍是“原图像素坐标系”。
    # - detector_crop_size=0 表示关闭（默认）。
    detector_crop_size: int = 0
    detector_crop_smooth_alpha: float = 0.2
    detector_crop_max_step_px: int = 120
    detector_crop_reset_after_missed: int = 8

    # 可选：相机侧 AOI（OffsetX/OffsetY）运行中平移。
    # 说明：
    # - 该能力依赖具体机型/固件：有些机型 StartGrabbing 后会锁定 OffsetX/OffsetY。
    # - 一旦启用，**不要**对 calib 做 apply_sensor_roi_to_calibration 的一次性主点平移；
    #   而是让 RoiController 返回每相机的 total_offset，把 bbox/uv 回写到满幅坐标系。
    camera_aoi_runtime: bool = False
    camera_aoi_update_every_groups: int = 2
    camera_aoi_min_move_px: int = 8
    camera_aoi_smooth_alpha: float = 0.3
    camera_aoi_max_step_px: int = 160
    camera_aoi_recenter_after_missed: int = 30

    trigger: OnlineTriggerConfig = OnlineTriggerConfig()


def load_offline_app_config(path: Path) -> OfflineAppConfig:
    """加载离线入口配置。"""

    data = _load_mapping(Path(path))

    serials_raw = data.get("serials")
    serials: list[str] | None = None
    if serials_raw is not None:
        if not isinstance(serials_raw, list) or not serials_raw:
            raise RuntimeError("offline config field 'serials' must be a non-empty list")
        # 去重但保持顺序，避免用户写重复项导致误判数量。
        seen: set[str] = set()
        serials = []
        for x in serials_raw:
            s = str(x).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            serials.append(s)
        if not serials:
            raise RuntimeError("offline config field 'serials' is empty after stripping")

    curve = _as_curve_stage_config(data.get("curve"))

    captures_dir = _as_path(data.get("captures_dir"))
    time_sync_mode = _as_time_sync_mode(data.get("time_sync_mode"), "frame_host_timestamp")
    time_mapping_path = _as_optional_path(data.get("time_mapping_path"))
    if time_sync_mode == "dev_timestamp_mapping" and time_mapping_path is None:
        # 约定：若启用方案B但未显式指定映射文件，则默认在 captures_dir 下寻找。
        time_mapping_path = captures_dir / "time_mapping_dev_to_host_ms.json"

    return OfflineAppConfig(
        captures_dir=captures_dir,
        calib=_as_path(data.get("calib")),
        serials=serials,
        detector=_as_detector(data.get("detector"), "color"),
        model=_as_optional_path(data.get("model")),
        pt_device=str(data.get("pt_device", "cpu") or "cpu").strip() or "cpu",
        min_score=float(data.get("min_score", 0.25)),
        require_views=int(data.get("require_views", 2)),
        max_detections_per_camera=int(data.get("max_detections_per_camera", 10)),
        max_reproj_error_px=float(data.get("max_reproj_error_px", 8.0)),
        max_uv_match_dist_px=float(data.get("max_uv_match_dist_px", 25.0)),
        merge_dist_m=float(data.get("merge_dist_m", 0.08)),
        max_groups=int(data.get("max_groups", 0)),
        out_jsonl=_as_path(data.get("out_jsonl", "data/tools_output/offline_positions_3d.jsonl")),
        time_sync_mode=time_sync_mode,
        time_mapping_path=time_mapping_path,
        curve=curve,
    )


def load_online_app_config(path: Path) -> OnlineAppConfig:
    """加载在线入口配置。"""

    data = _load_mapping(Path(path))

    serials_raw = data.get("serials")
    if not isinstance(serials_raw, list) or not serials_raw:
        raise RuntimeError("online config requires non-empty 'serials' list")
    serials = [str(x).strip() for x in serials_raw if str(x).strip()]

    trig = data.get("trigger", {})
    if trig is None:
        trig = {}
    if not isinstance(trig, dict):
        raise RuntimeError("online config field 'trigger' must be an object")

    trigger = OnlineTriggerConfig(
        trigger_source=str(trig.get("trigger_source", data.get("trigger_source", "Software"))).strip(),
        master_serial=str(trig.get("master_serial", data.get("master_serial", ""))).strip(),
        master_line_out=str(trig.get("master_line_out", data.get("master_line_out", "Line1"))).strip(),
        master_line_source=str(trig.get("master_line_source", data.get("master_line_source", ""))).strip(),
        master_line_mode=str(trig.get("master_line_mode", data.get("master_line_mode", "Output"))).strip(),
        soft_trigger_fps=float(trig.get("soft_trigger_fps", data.get("soft_trigger_fps", 5.0))),
        trigger_activation=str(trig.get("trigger_activation", data.get("trigger_activation", "RisingEdge"))).strip(),
        trigger_cache_enable=bool(trig.get("trigger_cache_enable", data.get("trigger_cache_enable", False))),
    )

    curve = _as_curve_stage_config(data.get("curve"))

    time_sync_mode = _as_time_sync_mode(data.get("time_sync_mode"), "frame_host_timestamp")
    time_mapping_warmup_groups = int(data.get("time_mapping_warmup_groups", 20))
    time_mapping_window_groups = int(data.get("time_mapping_window_groups", 200))
    time_mapping_update_every_groups = int(data.get("time_mapping_update_every_groups", 5))
    time_mapping_min_points = int(data.get("time_mapping_min_points", 20))
    time_mapping_hard_outlier_ms = float(data.get("time_mapping_hard_outlier_ms", 50.0))

    terminal_print_mode = _as_terminal_print_mode(data.get("terminal_print_mode"), "best")
    terminal_status_interval_s = float(data.get("terminal_status_interval_s", 0.0))

    out_jsonl_only_when_balls = bool(data.get("out_jsonl_only_when_balls", False))
    out_jsonl_flush_every_records = int(data.get("out_jsonl_flush_every_records", 1))
    out_jsonl_flush_interval_s = float(data.get("out_jsonl_flush_interval_s", 0.0))

    # 约束：flush_every_records 必须为非负整数。
    # - 0 表示禁用“按条数 flush”，仅依赖 flush_interval_s 或最终 close。
    # - 1 表示每条都 flush（旧行为）。
    if out_jsonl_flush_every_records < 0:
        raise RuntimeError("online config out_jsonl_flush_every_records must be >= 0")
    if out_jsonl_flush_interval_s < 0:
        raise RuntimeError("online config out_jsonl_flush_interval_s must be >= 0")
    if terminal_status_interval_s < 0:
        raise RuntimeError("online config terminal_status_interval_s must be >= 0")

    pixel_format = str(data.get("pixel_format", "") or "").strip()
    image_width = _as_optional_positive_int(data.get("image_width"))
    image_height = _as_optional_positive_int(data.get("image_height"))
    image_offset_x = _as_int(data.get("image_offset_x"), 0)
    image_offset_y = _as_int(data.get("image_offset_y"), 0)

    detector_crop_size = _as_int(data.get("detector_crop_size"), 0)
    detector_crop_smooth_alpha = float(data.get("detector_crop_smooth_alpha", 0.2))
    detector_crop_max_step_px = _as_int(data.get("detector_crop_max_step_px"), 120)
    detector_crop_reset_after_missed = _as_int(data.get("detector_crop_reset_after_missed"), 8)

    camera_aoi_runtime = bool(data.get("camera_aoi_runtime", False))
    camera_aoi_update_every_groups = _as_int(data.get("camera_aoi_update_every_groups"), 2)
    camera_aoi_min_move_px = _as_int(data.get("camera_aoi_min_move_px"), 8)
    camera_aoi_smooth_alpha = float(data.get("camera_aoi_smooth_alpha", 0.3))
    camera_aoi_max_step_px = _as_int(data.get("camera_aoi_max_step_px"), 160)
    camera_aoi_recenter_after_missed = _as_int(data.get("camera_aoi_recenter_after_missed"), 30)

    if detector_crop_size < 0:
        raise RuntimeError("online config detector_crop_size must be >= 0")
    if detector_crop_max_step_px < 0:
        raise RuntimeError("online config detector_crop_max_step_px must be >= 0")
    if detector_crop_reset_after_missed < 0:
        raise RuntimeError("online config detector_crop_reset_after_missed must be >= 0")
    if not (0.0 <= float(detector_crop_smooth_alpha) <= 1.0):
        raise RuntimeError("online config detector_crop_smooth_alpha must be in [0,1]")

    if camera_aoi_update_every_groups < 0:
        raise RuntimeError("online config camera_aoi_update_every_groups must be >= 0")
    if bool(camera_aoi_runtime) and camera_aoi_update_every_groups < 1:
        raise RuntimeError("online config camera_aoi_update_every_groups must be >= 1 when camera_aoi_runtime=true")
    if camera_aoi_min_move_px < 0:
        raise RuntimeError("online config camera_aoi_min_move_px must be >= 0")
    if camera_aoi_max_step_px < 0:
        raise RuntimeError("online config camera_aoi_max_step_px must be >= 0")
    if camera_aoi_recenter_after_missed < 0:
        raise RuntimeError("online config camera_aoi_recenter_after_missed must be >= 0")
    if not (0.0 <= float(camera_aoi_smooth_alpha) <= 1.0):
        raise RuntimeError("online config camera_aoi_smooth_alpha must be in [0,1]")

    # 约束：宽高必须同时设置，避免出现“只裁宽不裁高”的歧义。
    if (image_width is None) ^ (image_height is None):
        raise RuntimeError(
            "online config ROI 参数错误：image_width 与 image_height 必须同时设置，或同时不设置。"
        )

    return OnlineAppConfig(
        mvimport_dir=_as_optional_path(data.get("mvimport_dir")),
        dll_dir=_as_optional_path(data.get("dll_dir")),
        serials=serials,
        group_by=_as_group_by(data.get("group_by"), "frame_num"),
        timeout_ms=int(data.get("timeout_ms", 1000)),
        group_timeout_ms=int(data.get("group_timeout_ms", 1000)),
        max_pending_groups=int(data.get("max_pending_groups", 256)),
        max_groups=int(data.get("max_groups", 0)),
        max_wait_seconds=float(data.get("max_wait_seconds", 0.0)),
        pixel_format=pixel_format,
        image_width=image_width,
        image_height=image_height,
        image_offset_x=int(image_offset_x),
        image_offset_y=int(image_offset_y),
        calib=_as_path(data.get("calib", "data/calibration/example_triple_camera_calib.json")),
        detector=_as_detector(data.get("detector"), "fake"),
        model=_as_optional_path(data.get("model")),
        pt_device=str(data.get("pt_device", "cpu") or "cpu").strip() or "cpu",
        min_score=float(data.get("min_score", 0.25)),
        require_views=int(data.get("require_views", 2)),
        max_detections_per_camera=int(data.get("max_detections_per_camera", 10)),
        max_reproj_error_px=float(data.get("max_reproj_error_px", 8.0)),
        max_uv_match_dist_px=float(data.get("max_uv_match_dist_px", 25.0)),
        merge_dist_m=float(data.get("merge_dist_m", 0.08)),
        out_jsonl=_as_optional_path(data.get("out_jsonl")),
        out_jsonl_only_when_balls=out_jsonl_only_when_balls,
        out_jsonl_flush_every_records=out_jsonl_flush_every_records,
        out_jsonl_flush_interval_s=out_jsonl_flush_interval_s,
        terminal_print_mode=terminal_print_mode,
        terminal_status_interval_s=terminal_status_interval_s,
        time_sync_mode=time_sync_mode,
        time_mapping_warmup_groups=time_mapping_warmup_groups,
        time_mapping_window_groups=time_mapping_window_groups,
        time_mapping_update_every_groups=time_mapping_update_every_groups,
        time_mapping_min_points=time_mapping_min_points,
        time_mapping_hard_outlier_ms=time_mapping_hard_outlier_ms,
        trigger=trigger,
        curve=curve,

        detector_crop_size=int(detector_crop_size),
        detector_crop_smooth_alpha=float(detector_crop_smooth_alpha),
        detector_crop_max_step_px=int(detector_crop_max_step_px),
        detector_crop_reset_after_missed=int(detector_crop_reset_after_missed),

        camera_aoi_runtime=bool(camera_aoi_runtime),
        camera_aoi_update_every_groups=int(camera_aoi_update_every_groups),
        camera_aoi_min_move_px=int(camera_aoi_min_move_px),
        camera_aoi_smooth_alpha=float(camera_aoi_smooth_alpha),
        camera_aoi_max_step_px=int(camera_aoi_max_step_px),
        camera_aoi_recenter_after_missed=int(camera_aoi_recenter_after_missed),
    )
