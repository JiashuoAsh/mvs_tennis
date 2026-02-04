"""在线输入源：从 MVS `QuadCapture` 产出每组图像。

职责：
- 从 `cap.get_next_group()` 持续读取同步组包。
- 将每帧转换为 OpenCV BGR 图像。
- 计算组级别的时间轴信息（capture_t_abs / capture_host_timestamp）。
- 可选：在线滑窗拟合 dev_timestamp -> host_ms（time_sync_mode=dev_timestamp_mapping）。

依赖方向：
- 本模块属于 pipeline 层，可依赖 `mvs.*` 的采集与时间映射实现，但不应依赖 apps/CLI 入口层。
"""

from __future__ import annotations

import time
from typing import Any, Iterator

import numpy as np

from mvs import MvsBinding, OnlineDevToHostMapper, QuadCapture
from mvs.capture.image import frame_to_bgr

from .time_utils import (
    delta_to_median_by_camera,
    host_timestamp_to_ms_epoch,
    host_timestamp_to_seconds,
    median_float,
    median_int,
    spread_ms,
)


class OnlineGroupWaitTimeout(RuntimeError):
    """在线采集在限定时间内未收到任何完整组包。"""


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
        max_wait_seconds: 超过该时长仍无任何组包则抛出 OnlineGroupWaitTimeout（0 表示不限）。
        time_sync_mode: 时间轴策略（frame_host_timestamp / dev_timestamp_mapping）。

    Yields:
        (meta, images_by_camera_serial)
    """

    group_index = 0
    last_progress = time.monotonic()

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
                raise OnlineGroupWaitTimeout(f"在线采集等待超时：超过 {max_wait:.3f}s 未收到任何完整组包。")
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

            hm = host_timestamp_to_ms_epoch(getattr(fr, "host_timestamp", None))
            if hm is not None:
                host_ms_by_camera[serial] = float(hm)

            if mapper is not None:
                # 关键点：每组每相机提供一个 (dev,host) 配对点。
                mapper.observe_pair(serial=serial, dev_ts=int(fr.dev_timestamp), host_ms=int(fr.host_timestamp))

        host_ts_med = median_int(host_ts_list)
        capture_t_abs = host_timestamp_to_seconds(host_ts_med) if host_ts_med is not None else None
        capture_t_source: str | None = "frame_host_timestamp" if capture_t_abs is not None else None

        mapped_ms_by_camera: dict[str, float] = {}

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

            mapped_ms_med = median_float(mapped_ms_list)
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

        if host_ms_by_camera:
            meta["time_mapping_host_ms_by_camera"] = dict(host_ms_by_camera)
            meta["time_mapping_host_ms_spread_ms"] = spread_ms(list(host_ms_by_camera.values()))
            meta["time_mapping_host_ms_delta_to_median_by_camera"] = delta_to_median_by_camera(host_ms_by_camera)

        if mapper is not None:
            meta["time_mapping_groups_seen"] = int(mapper.groups_seen)
            meta["time_mapping_ready_count"] = int(mapper.ready_count(serials_in_group))
            worst_p95 = mapper.worst_p95_ms(serials_in_group)
            worst_rms = mapper.worst_rms_ms(serials_in_group)
            meta["time_mapping_worst_p95_ms"] = float(worst_p95) if worst_p95 is not None else None
            meta["time_mapping_worst_rms_ms"] = float(worst_rms) if worst_rms is not None else None

            if mapped_ms_by_camera:
                meta["time_mapping_mapped_host_ms_by_camera"] = dict(mapped_ms_by_camera)
                meta["time_mapping_mapped_host_ms_spread_ms"] = spread_ms(list(mapped_ms_by_camera.values()))
                meta["time_mapping_mapped_host_ms_delta_to_median_by_camera"] = delta_to_median_by_camera(
                    mapped_ms_by_camera
                )

        yield meta, images_by_camera
        group_index += 1
