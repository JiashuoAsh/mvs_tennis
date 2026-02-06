# -*- coding: utf-8 -*-

"""按指定分组键把多相机帧汇聚成一组。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from .grab import FramePacket


@dataclass
class TriggerGroupAssembler:
    """按指定键汇聚 N 台相机帧。

    目标是把“同一次触发”的多相机帧凑成一组，交给上层做保存/推理。

    Attributes:
        num_cameras: 参与分组的相机数量。
        group_timeout_s: 从首次见到某个分组键起，等待凑齐一组的超时时间。
        max_pending_groups: 最多允许同时缓存多少个未凑齐的分组（防止内存无限增长）。
        dropped_groups: 因超时或缓存上限被丢弃的分组计数（用于诊断）。
    """

    num_cameras: int
    group_timeout_s: float = 0.2
    max_pending_groups: int = 256
    group_by: Literal["frame_num", "sequence"] = "frame_num"

    def __post_init__(self) -> None:
        self._pending: Dict[int, Dict[int, FramePacket]] = {}
        self._first_seen: Dict[int, float] = {}
        self._base_frame_num_by_cam: Dict[int, int] = {}
        self._seq_by_cam: Dict[int, int] = {}
        self.dropped_groups = 0
        self.frames_seen_total = 0
        self.frames_seen_by_cam: Dict[int, int] = {}

    @property
    def pending_groups(self) -> int:
        return int(len(self._pending))

    def pending_oldest_age_s(self) -> float:
        if not self._first_seen:
            return 0.0
        now = time.monotonic()
        oldest = min(self._first_seen.values())
        return float(max(0.0, now - float(oldest)))

    def _group_key(self, pkt: FramePacket) -> int:
        """计算分组键，并在需要时做跨相机序列归一化。

                说明：
                - 不同相机的 frame_num 可能不是从同一个起始值开始；但只要各相机都在同一触发节奏下工作，
                    frame_num 的变化通常同速。
                - 这里用“首次看到的 frame_num”作为基准，把后续 frame_num 转成从 0 开始的序列。
        """

        if self.group_by == "frame_num":
            base = self._base_frame_num_by_cam.get(pkt.cam_index)
            if base is None:
                self._base_frame_num_by_cam[pkt.cam_index] = int(pkt.frame_num)
                base = int(pkt.frame_num)

            # frame_num 同样可按 32-bit wrap 语义处理。
            return (int(pkt.frame_num) - int(base)) & 0xFFFFFFFF

        if self.group_by == "sequence":
            # 以“每台相机进入分组器的帧顺序”作为分组键。
            # 适用于：所有相机由同一触发脉冲驱动且基本不丢帧的场景。
            seq = int(self._seq_by_cam.get(pkt.cam_index, 0))
            self._seq_by_cam[pkt.cam_index] = seq + 1
            return seq

        raise ValueError(f"Unsupported group_by: {self.group_by}")

    def add(self, pkt: FramePacket) -> Optional[List[FramePacket]]:
        """喂入一帧，尝试返回完整分组。

        Args:
            pkt: 任意相机的一帧。

        Returns:
            当某个分组键的分组已凑齐 `num_cameras` 帧时，按 cam_index 顺序返回列表；
            否则返回 None。
        """

        now = time.monotonic()

        # 仅用于诊断：统计每台相机收到的帧数。
        self.frames_seen_total += 1
        self.frames_seen_by_cam[pkt.cam_index] = int(self.frames_seen_by_cam.get(pkt.cam_index, 0)) + 1

        group_key = self._group_key(pkt)

        group = self._pending.get(group_key)
        if group is None:
            group = {}
            self._pending[group_key] = group
            self._first_seen[group_key] = now

        group[pkt.cam_index] = pkt

        if len(group) == self.num_cameras:
            frames = [group[i] for i in range(self.num_cameras)]
            del self._pending[group_key]
            self._first_seen.pop(group_key, None)
            return frames

        self._prune(now)
        return None

    def _prune(self, now: float) -> None:
        """清理超时分组与过量缓存。

        Notes:
            - 超时和超量都会计入 `dropped_groups`。
            - 清理策略优先丢弃最早开始等待的分组（oldest-first）。
        """

        expired = [k for k, t0 in self._first_seen.items() if (now - t0) > self.group_timeout_s]
        for k in expired:
            self._pending.pop(k, None)
            self._first_seen.pop(k, None)
            self.dropped_groups += 1

        if len(self._pending) <= self.max_pending_groups:
            return

        oldest = sorted(self._first_seen.items(), key=lambda kv: kv[1])
        for k, _ in oldest[: max(0, len(self._pending) - self.max_pending_groups)]:
            self._pending.pop(k, None)
            self._first_seen.pop(k, None)
            self.dropped_groups += 1
