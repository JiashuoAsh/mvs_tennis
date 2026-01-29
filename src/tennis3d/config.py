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


_DETECTOR = Literal["fake", "color", "rknn", "pt"]
_GROUP_BY = Literal["trigger_index", "frame_num", "sequence"]


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


@dataclass(frozen=True)
class OfflineAppConfig:
    captures_dir: Path
    calib: Path
    # 可选：仅使用这些相机序列号（serial）参与检测/定位；None 表示使用 captures 中出现的全部相机。
    serials: list[str] | None = None
    detector: _DETECTOR = "color"
    model: Path | None = None
    min_score: float = 0.25
    require_views: int = 2
    max_detections_per_camera: int = 10
    max_reproj_error_px: float = 8.0
    max_uv_match_dist_px: float = 25.0
    merge_dist_m: float = 0.08
    max_groups: int = 0
    out_jsonl: Path = Path("data/tools_output/offline_positions_3d.jsonl")


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
    dll_dir: Path | None
    serials: list[str]

    group_by: _GROUP_BY = "frame_num"
    timeout_ms: int = 1000
    group_timeout_ms: int = 1000
    max_pending_groups: int = 256
    max_groups: int = 0

    calib: Path = Path("data/calibration/example_triple_camera_calib.json")
    detector: _DETECTOR = "fake"
    model: Path | None = None
    min_score: float = 0.25
    require_views: int = 2
    max_detections_per_camera: int = 10
    max_reproj_error_px: float = 8.0
    max_uv_match_dist_px: float = 25.0
    merge_dist_m: float = 0.08
    out_jsonl: Path | None = None

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

    return OfflineAppConfig(
        captures_dir=_as_path(data.get("captures_dir")),
        calib=_as_path(data.get("calib")),
        serials=serials,
        detector=_as_detector(data.get("detector"), "color"),
        model=_as_optional_path(data.get("model")),
        min_score=float(data.get("min_score", 0.25)),
        require_views=int(data.get("require_views", 2)),
        max_detections_per_camera=int(data.get("max_detections_per_camera", 10)),
        max_reproj_error_px=float(data.get("max_reproj_error_px", 8.0)),
        max_uv_match_dist_px=float(data.get("max_uv_match_dist_px", 25.0)),
        merge_dist_m=float(data.get("merge_dist_m", 0.08)),
        max_groups=int(data.get("max_groups", 0)),
        out_jsonl=_as_path(data.get("out_jsonl", "data/tools_output/offline_positions_3d.jsonl")),
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

    return OnlineAppConfig(
        dll_dir=_as_optional_path(data.get("dll_dir")),
        serials=serials,
        group_by=_as_group_by(data.get("group_by"), "frame_num"),
        timeout_ms=int(data.get("timeout_ms", 1000)),
        group_timeout_ms=int(data.get("group_timeout_ms", 1000)),
        max_pending_groups=int(data.get("max_pending_groups", 256)),
        max_groups=int(data.get("max_groups", 0)),
        calib=_as_path(data.get("calib", "data/calibration/example_triple_camera_calib.json")),
        detector=_as_detector(data.get("detector"), "fake"),
        model=_as_optional_path(data.get("model")),
        min_score=float(data.get("min_score", 0.25)),
        require_views=int(data.get("require_views", 2)),
        max_detections_per_camera=int(data.get("max_detections_per_camera", 10)),
        max_reproj_error_px=float(data.get("max_reproj_error_px", 8.0)),
        max_uv_match_dist_px=float(data.get("max_uv_match_dist_px", 25.0)),
        merge_dist_m=float(data.get("merge_dist_m", 0.08)),
        out_jsonl=_as_optional_path(data.get("out_jsonl")),
        trigger=trigger,
    )
