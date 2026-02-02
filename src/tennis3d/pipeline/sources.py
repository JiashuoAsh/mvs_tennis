"""流水线输入源：产出“每组图像”。

本模块提供两类 source：
- `iter_capture_image_groups`：离线，从 captures/metadata.jsonl 读取分组并加载对应图像。
- `iter_mvs_image_groups`：在线，从 MVS `QuadCapture` 读取分组并转换为 BGR 图像。

二者都产出 (meta, images_by_camera)：
- meta：可 JSON 序列化的 dict（用于携带 group_index/trigger_index 等信息）。
- images_by_camera：key 为相机序列号（serial），value 为 OpenCV BGR 的 numpy.ndarray。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from mvs.binding import MvsBinding
from mvs.metadata_io import iter_metadata_records
from mvs.pipeline import QuadCapture
from mvs.image import frame_to_bgr
from mvs.time_mapping import LinearTimeMapping, OnlineDevToHostMapper, load_time_mappings_json


class OnlineGroupWaitTimeout(RuntimeError):
    """在线采集在限定时间内未收到任何完整组包。"""


def _median_int(values: list[int]) -> int | None:
    """返回整型列表的中位数（不做插值）。

    说明：
        这里用于把同一组内多相机的 host_timestamp 做一个稳健聚合。
        - 选择中位数能减小单路线程抖动/偶发延迟的影响。
        - 不做插值：保持输出仍是“真实出现过的 timestamp”。
    """

    if not values:
        return None
    xs = sorted(int(v) for v in values)
    return int(xs[len(xs) // 2])


def _median_float(values: list[float]) -> float | None:
    """返回浮点列表的中位数（不做插值）。"""

    if not values:
        return None
    xs = sorted(float(v) for v in values)
    return float(xs[len(xs) // 2])


def _host_timestamp_to_seconds(ts: Any) -> float | None:
    """把 host_timestamp（可能是 ms/ns/s）转换成 epoch 秒。

    说明：
        MVS 抓取侧记录的 host_timestamp 在不同设备/配置下可能有不同单位。
        本项目目前主要遇到：
        - 毫秒 epoch（典型值约 1.7e12）
        - 秒 epoch（典型值约 1.7e9）
        - 纳秒 epoch（典型值约 1.7e18，较少见）

        这里用数量级启发式做一次归一化，供后续曲线拟合的 t_abs 使用。
    """

    try:
        v = int(ts)
    except Exception:
        return None

    if v <= 0:
        return None

    # ns epoch
    if v >= 10**16:
        return float(v) / 1e9
    # ms epoch
    if v >= 10**11:
        return float(v) / 1e3
    # s epoch
    if v >= 10**8:
        return float(v)

    return None


def _host_timestamp_to_ms_epoch(ts: Any) -> float | None:
    """把 host_timestamp（可能是 ms/ns/s）转换成 epoch 毫秒。

    说明：
        - 在线时间映射（dev_timestamp_mapping）拟合出来的结果单位约定为 epoch 毫秒。
        - 为了评估“组内各相机时间差”，这里把每台相机的 host_timestamp 统一归一化到 ms。
        - 该转换使用与 `_host_timestamp_to_seconds` 相同的数量级启发式。
    """

    try:
        v = int(ts)
    except Exception:
        return None

    if v <= 0:
        return None

    # ns epoch
    if v >= 10**16:
        return float(v) / 1e6
    # ms epoch
    if v >= 10**11:
        return float(v)
    # s epoch
    if v >= 10**8:
        return float(v) * 1000.0

    return None


def _spread_ms(values_ms: list[float]) -> float | None:
    """返回一组时间戳（毫秒）的跨度：max - min。"""

    if len(values_ms) < 2:
        return None
    lo = min(float(x) for x in values_ms)
    hi = max(float(x) for x in values_ms)
    return float(hi - lo)


def _delta_to_median_by_camera(values_by_camera: dict[str, float]) -> dict[str, float] | None:
    """把每相机时间戳转成“相对组内中位数”的偏差（毫秒）。"""

    if len(values_by_camera) < 2:
        return None

    med = _median_float(list(values_by_camera.values()))
    if med is None:
        return None

    return {str(k): float(v) - float(med) for k, v in values_by_camera.items()}


def _find_repo_root(start: Path) -> Path:
    """从给定路径向上查找仓库根目录。

    约定：仓库根目录包含 pyproject.toml。
    """

    start = Path(start).resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return start


def iter_capture_image_groups(
    *,
    captures_dir: Path,
    max_groups: int = 0,
    serials: list[str] | None = None,
    time_sync_mode: str = "frame_host_timestamp",
    time_mapping_path: Path | None = None,
) -> Iterator[tuple[dict[str, Any], dict[str, np.ndarray]]]:
    """从 captures/metadata.jsonl 迭代读取每组图像。

    Notes:
        - metadata.jsonl 中可能混有非 group 记录（例如事件/日志），这里会跳过。
        - frames[*].file 可能是绝对路径，也可能是相对路径：
          1) 相对 captures_dir；
          2) 相对仓库根目录（例如以 data/ 开头）。
          本函数会做一次稳健解析，尽量找到真实存在的文件。

    Args:
        captures_dir: captures 目录（包含 metadata.jsonl 与各帧图像文件）。
        max_groups: 最多处理的组数（0 表示不限）。
        serials: 可选的相机序列号白名单；为 None 时不过滤。

    Yields:
        (meta, images_by_camera_serial)
    """

    captures_dir = Path(captures_dir).resolve()
    meta_path = captures_dir / "metadata.jsonl"
    if not meta_path.exists():
        raise RuntimeError(f"metadata.jsonl not found: {meta_path}")

    # 说明：captures 的 frames.file 可能是两种相对路径：
    # 1) 相对 captures_dir 的路径：group_xxx/cam0_xxx.bmp
    # 2) 相对仓库根目录的路径：data/captures_xxx/.../cam0_xxx.bmp
    # 这里做一次稳健解析，避免路径被重复拼接。
    repo_root = _find_repo_root(captures_dir)
    groups_done = 0

    # 可选：方案B（dev_timestamp -> host_ms 的线性映射）。
    mappings: dict[str, LinearTimeMapping] | None = None
    if str(time_sync_mode).strip() == "dev_timestamp_mapping":
        if time_mapping_path is None:
            raise RuntimeError("time_sync_mode=dev_timestamp_mapping 需要提供 time_mapping_path")
        mappings = load_time_mappings_json(Path(time_mapping_path).resolve())

    serials_set: set[str] | None = None
    if serials is not None:
        serials_norm = [str(s).strip() for s in (serials or []) if str(s).strip()]
        serials_set = set(serials_norm) if serials_norm else set()

    for rec in iter_metadata_records(meta_path):
        if "frames" not in rec:
            continue

        frames = rec.get("frames")
        if not isinstance(frames, list) or not frames:
            continue

        # 用每帧的 host_timestamp 估计这一组的绝对时间（默认行为）。
        # 说明：host_timestamp 为主机侧观测，存在抖动，但通常在 1ms 量级内可接受。
        host_ts_list: list[int] = []
        for fr in frames:
            if not isinstance(fr, dict):
                continue
            ts_raw = fr.get("host_timestamp")
            if ts_raw is None:
                continue
            try:
                host_ts_list.append(int(ts_raw))
            except Exception:
                pass
        host_ts_med = _median_int(host_ts_list)

        capture_t_abs: float | None = _host_timestamp_to_seconds(host_ts_med) if host_ts_med is not None else None
        capture_t_source: str | None = "frame_host_timestamp" if capture_t_abs is not None else None

        # 可选：方案B——用映射后的时间替代默认 host_timestamp 聚合。
        if mappings is not None:
            mapped_ms_list: list[float] = []
            for fr in frames:
                if not isinstance(fr, dict):
                    continue
                serial = str(fr.get("serial", "")).strip()
                dev_ts = fr.get("dev_timestamp")
                if not serial or dev_ts is None:
                    continue
                m = mappings.get(serial)
                if m is None:
                    continue
                try:
                    mapped_ms_list.append(float(m.map_dev_to_host_ms(int(dev_ts))))
                except Exception:
                    continue

            mapped_ms_med = _median_float(mapped_ms_list)
            if mapped_ms_med is not None:
                capture_t_abs = float(mapped_ms_med) / 1000.0
                capture_t_source = "dev_timestamp_mapping"

        images_by_camera: dict[str, np.ndarray] = {}
        for fr in frames:
            if not isinstance(fr, dict):
                continue

            serial = str(fr.get("serial", "")).strip()
            if serials_set is not None and serial not in serials_set:
                continue
            file = fr.get("file")
            if not serial or not isinstance(file, str) or not file:
                continue

            file_path = Path(file)
            if not file_path.is_absolute():
                # 优先按 captures_dir 解析；若不存在，再按仓库根目录解析。
                candidate = (captures_dir / file_path).resolve()
                if candidate.exists():
                    file_path = candidate
                else:
                    candidate2 = (repo_root / file_path).resolve()
                    file_path = candidate2 if candidate2.exists() else candidate

            if not file_path.exists():
                continue

            img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
            if img is None:
                continue

            images_by_camera[serial] = img

        meta = {
            "group_seq": rec.get("group_seq"),
            "group_by": rec.get("group_by"),
            "trigger_index": rec.get("trigger_index"),
            # capture_t_abs：用于下游把 3D 点序列映射到绝对时间轴（curve 需要 t_abs）。
            # 若缺失则由下游回退到 created_at（处理时间）。
            "capture_t_abs": float(capture_t_abs) if capture_t_abs is not None else None,
            "capture_t_source": str(capture_t_source) if capture_t_source is not None else None,
            "capture_host_timestamp": int(host_ts_med) if host_ts_med is not None else None,
        }

        yield meta, images_by_camera

        groups_done += 1
        if int(max_groups) > 0 and groups_done >= int(max_groups):
            break


def iter_mvs_image_groups(
    *,
    cap: QuadCapture,
    binding: MvsBinding,
    max_groups: int = 0,
    timeout_s: float = 0.5,
    max_wait_seconds: float = 0.0,
    time_sync_mode: str = "frame_host_timestamp",
    time_mapping_warmup_groups: int = 20,
    time_mapping_window_groups: int = 200,
    time_mapping_update_every_groups: int = 5,
    time_mapping_min_points: int = 20,
    time_mapping_hard_outlier_ms: float = 50.0,
) -> Iterator[tuple[dict[str, Any], dict[str, np.ndarray]]]:
    """从在线 MVS `QuadCapture` 迭代读取每组图像。

    Args:
        cap: 已打开的 QuadCapture。
        binding: 已加载的 MVS binding（用于像素格式解码）。
        max_groups: 最多处理的组数（0 表示不限）。
        timeout_s: 等待 cap.get_next_group 的超时时间。

    Yields:
        (meta, images_by_camera_serial)
    """

    group_index = 0
    last_progress = time.monotonic()

    # 在线方案B：用滑窗持续拟合 dev_timestamp -> host_ms。
    mapper: OnlineDevToHostMapper | None = None
    if str(time_sync_mode).strip() == "dev_timestamp_mapping":
        mapper = OnlineDevToHostMapper(
            warmup_groups=int(time_mapping_warmup_groups),
            window_groups=int(time_mapping_window_groups),
            update_every_groups=int(time_mapping_update_every_groups),
            min_points=int(time_mapping_min_points),
            hard_outlier_ms=float(time_mapping_hard_outlier_ms),
        )

    while True:
        if int(max_groups) > 0 and group_index >= int(max_groups):
            break

        group = cap.get_next_group(timeout_s=float(timeout_s))
        if group is None:
            max_wait = float(max_wait_seconds)
            if max_wait > 0 and (time.monotonic() - last_progress) > max_wait:
                raise OnlineGroupWaitTimeout(
                    f"在线采集等待超时：超过 {max_wait:.3f}s 未收到任何完整组包。"
                )
            continue

        last_progress = time.monotonic()

        images_by_camera: dict[str, np.ndarray] = {}
        host_ts_list: list[int] = []
        host_ms_by_camera: dict[str, float] = {}
        serials_in_group: list[str] = []
        for fr in group:
            bgr = frame_to_bgr(binding=binding, cam=cap.cameras[fr.cam_index].cam, frame=fr)
            serial = str(fr.serial)
            images_by_camera[serial] = bgr
            serials_in_group.append(serial)
            try:
                host_ts_list.append(int(fr.host_timestamp))
            except Exception:
                pass

            # 说明：用于评估组内各相机“到达/时间戳”的离散程度。
            # 这里把 host_timestamp 统一归一化到 ms_epoch。
            hm = _host_timestamp_to_ms_epoch(getattr(fr, "host_timestamp", None))
            if hm is not None:
                host_ms_by_camera[serial] = float(hm)

            if mapper is not None:
                # 关键点：每组每相机提供一个 (dev,host) 配对点。
                mapper.observe_pair(serial=serial, dev_ts=int(fr.dev_timestamp), host_ms=int(fr.host_timestamp))

        host_ts_med = _median_int(host_ts_list)
        capture_t_abs = _host_timestamp_to_seconds(host_ts_med) if host_ts_med is not None else None
        capture_t_source: str | None = "frame_host_timestamp" if capture_t_abs is not None else None

        # 说明：仅在 time_sync_mode=dev_timestamp_mapping 时会填充。
        mapped_ms_by_camera: dict[str, float] = {}

        # 让映射器在“每个 group 结束”这个时刻决定是否重拟合一次。
        if mapper is not None:
            mapper.on_group_end()

            mapped_ms_list: list[float] = []
            for fr in group:
                m = mapper.get_mapping(str(fr.serial))
                if m is None:
                    continue
                try:
                    ms = float(m.map_dev_to_host_ms(int(fr.dev_timestamp)))
                except Exception:
                    continue
                mapped_ms_list.append(ms)
                mapped_ms_by_camera[str(fr.serial)] = float(ms)
            mapped_ms_med = _median_float(mapped_ms_list)
            if mapped_ms_med is not None:
                capture_t_abs = float(mapped_ms_med) / 1000.0
                capture_t_source = "dev_timestamp_mapping"

        meta: dict[str, Any] = {
            "group_index": int(group_index),
            "capture_t_abs": float(capture_t_abs) if capture_t_abs is not None else None,
            "capture_t_source": str(capture_t_source) if capture_t_source is not None else None,
            "capture_host_timestamp": int(host_ts_med) if host_ts_med is not None else None,
            "time_sync_mode": str(time_sync_mode).strip() or None,
        }

        # 说明：组内时间差评估（不依赖球是否被检测到）。
        # - host_ms_*：基于原始 host_timestamp（归一化到 ms_epoch）的组内离散程度。
        # - mapped_host_ms_*：基于时间映射后的 host_ms（ms_epoch）的组内离散程度。
        if host_ms_by_camera:
            meta["time_mapping_host_ms_by_camera"] = dict(host_ms_by_camera)
            meta["time_mapping_host_ms_spread_ms"] = _spread_ms(list(host_ms_by_camera.values()))
            meta["time_mapping_host_ms_delta_to_median_by_camera"] = _delta_to_median_by_camera(host_ms_by_camera)

        if mapper is not None:
            meta["time_mapping_groups_seen"] = int(mapper.groups_seen)
            meta["time_mapping_ready_count"] = int(mapper.ready_count(serials_in_group))
            worst_p95 = mapper.worst_p95_ms(serials_in_group)
            worst_rms = mapper.worst_rms_ms(serials_in_group)
            meta["time_mapping_worst_p95_ms"] = float(worst_p95) if worst_p95 is not None else None
            meta["time_mapping_worst_rms_ms"] = float(worst_rms) if worst_rms is not None else None

            # 注意：映射未就绪时，mapped_ms_by_camera 可能缺项。
            if mapped_ms_by_camera:
                meta["time_mapping_mapped_host_ms_by_camera"] = dict(mapped_ms_by_camera)
                meta["time_mapping_mapped_host_ms_spread_ms"] = _spread_ms(list(mapped_ms_by_camera.values()))
                meta["time_mapping_mapped_host_ms_delta_to_median_by_camera"] = _delta_to_median_by_camera(
                    mapped_ms_by_camera
                )

        yield meta, images_by_camera
        group_index += 1
