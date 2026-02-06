# -*- coding: utf-8 -*-

"""主从触发/时间戳偏移诊断脚本。

这个脚本用于回答一个非常具体的问题：
- 你有 1 台 master（软件触发），它通过硬件线触发另外 3 台 slave。
- 你做了 dev_timestamp -> host_ms 的时间映射后，发现 3 台 slave 的时间几乎一致，
  但 master 总是与它们相差约 10ms，且非常规律。

本脚本会从 `captures/metadata.jsonl` 里提取三类信息并做对照：
1) 每组 frames[*].host_timestamp（主机侧时间戳，可能是 ms/ns/s epoch，脚本会做归一化）
2) 每帧 frames[*].dev_timestamp（设备侧 tick）
3) 相机事件 type=camera_event（默认关注 ExposureStart），并尝试把事件与帧进行“同相机同组”匹配

如果提供 time mapping（`time_mapping_dev_to_host_ms.json`），脚本还会计算：
- 映射后的每相机 host_ms（ms_epoch）
- 组内各相机相对中位数的偏差分布（median / abs_p95）

注意：
- 这个脚本是“诊断工具”，不改任何采集逻辑。
- 它的输出可以帮助你区分：
  - 字段缺失/单位识别错误；还是
  - 触发链路本身就存在固定延迟（例如 StrobeDelay / 脉冲源选择导致 slave 曝光开始更晚）。

用法示例：
    uv run python tools/debug_master_slave_timing.py --captures-dir data/captures_master_slave/tennis_offline \
      --master DA8199303

如果你已经拟合了时间映射（推荐），再加上：
    uv run python tools/debug_master_slave_timing.py --captures-dir data/captures_master_slave/tennis_offline \
      --master DA8199303 \
    --time-mapping data/captures_master_slave/tennis_offline/time_mapping_dev_to_host_ms.json

"""

from __future__ import annotations

import argparse
import bisect
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from mvs import LinearTimeMapping, load_time_mappings_json
from mvs.session.metadata_io import iter_metadata_records
from tennis3d.pipeline.time_utils import host_timestamp_to_ms_epoch, median_float


@dataclass(frozen=True, slots=True)
class _FrameRec:
    """从 group 记录里抽取出的帧关键信息。"""

    serial: str
    cam_index: int
    frame_num: int
    dev_ts: int
    host_ts_raw: int | None
    host_ms_epoch: float | None


@dataclass(frozen=True, slots=True)
class _EventRec:
    """从 camera_event 记录里抽取出的关键信息。"""

    serial: str
    event_name: str
    event_ts: int
    created_at: float | None
    host_monotonic: float | None


def _percentile_nearest_rank(sorted_values: list[float], q: float) -> float | None:
    """nearest-rank 分位数（不插值），q in [0,1]。"""

    if not sorted_values:
        return None
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])

    n = len(sorted_values)
    k = int(math.ceil(float(q) * float(n)) - 1)
    k = max(0, min(n - 1, k))
    return float(sorted_values[k])


def _iter_groups_and_events(
    *,
    metadata_path: Path,
    event_name: str,
    max_groups: int,
) -> tuple[list[dict[str, Any]], list[_EventRec]]:
    """从 metadata.jsonl 里分别提取 group 记录与指定事件。"""

    groups: list[dict[str, Any]] = []
    events: list[_EventRec] = []

    groups_seen = 0
    for rec in iter_metadata_records(Path(metadata_path).resolve()):
        if not isinstance(rec, dict):
            continue

        # 事件记录：type=camera_event
        if str(rec.get("type", "")) == "camera_event":
            if str(rec.get("event_name", "")) != str(event_name):
                continue
            serial = str(rec.get("serial", "")).strip()
            if not serial:
                continue
            ts = rec.get("event_timestamp")
            if ts is None:
                continue
            try:
                ev_ts = int(ts)
            except Exception:
                continue

            created_at = rec.get("created_at")
            host_mono = rec.get("host_monotonic")
            events.append(
                _EventRec(
                    serial=serial,
                    event_name=str(event_name),
                    event_ts=ev_ts,
                    created_at=float(created_at) if created_at is not None else None,
                    host_monotonic=float(host_mono) if host_mono is not None else None,
                )
            )
            continue

        # 组记录：含 frames
        frames = rec.get("frames")
        if not isinstance(frames, list) or not frames:
            continue

        groups.append(rec)
        groups_seen += 1
        if int(max_groups) > 0 and groups_seen >= int(max_groups):
            break

    # 事件按 (serial, event_ts) 排序，便于后续二分匹配
    events.sort(key=lambda x: (x.serial, x.event_ts))
    return groups, events


def _extract_frames_from_group(group: dict[str, Any]) -> list[_FrameRec]:
    out: list[_FrameRec] = []
    frames = group.get("frames")
    if not isinstance(frames, list):
        return out

    for fr in frames:
        if not isinstance(fr, dict):
            continue

        serial = str(fr.get("serial", "")).strip()
        if not serial:
            continue

        dev_ts_raw = fr.get("dev_timestamp")
        if dev_ts_raw is None:
            continue

        try:
            dev_ts = int(dev_ts_raw)
        except Exception:
            continue

        host_ts_raw = fr.get("host_timestamp")
        host_ts_int: int | None
        if host_ts_raw is None:
            host_ts_int = None
        else:
            try:
                host_ts_int = int(host_ts_raw)
            except Exception:
                host_ts_int = None

        host_ms_epoch = host_timestamp_to_ms_epoch(host_ts_int) if host_ts_int is not None else None

        out.append(
            _FrameRec(
                serial=serial,
                cam_index=int(fr.get("cam_index", -1) or -1),
                frame_num=int(fr.get("frame_num", -1) or -1),
                dev_ts=dev_ts,
                host_ts_raw=host_ts_int,
                host_ms_epoch=host_ms_epoch,
            )
        )

    return out


def _events_by_serial(events: Iterable[_EventRec]) -> dict[str, list[_EventRec]]:
    out: dict[str, list[_EventRec]] = {}
    for ev in events:
        out.setdefault(ev.serial, []).append(ev)
    # 已在上游全局排序过，但这里再保证一下每个 serial 内按 event_ts 排
    for s in list(out.keys()):
        out[s].sort(key=lambda x: x.event_ts)
    return out


def _find_nearest_event_for_frame(
    *,
    evs_sorted: list[_EventRec],
    dev_ts: int,
) -> _EventRec | None:
    """在同一相机的事件列表中，为给定 dev_ts 找最近的事件（按 event_ts）。"""

    if not evs_sorted:
        return None

    keys = [e.event_ts for e in evs_sorted]
    i = bisect.bisect_left(keys, int(dev_ts))

    best: _EventRec | None = None
    best_abs = None
    for j in (i - 1, i, i + 1):
        if j < 0 or j >= len(evs_sorted):
            continue
        cand = evs_sorted[j]
        d = abs(int(cand.event_ts) - int(dev_ts))
        if best is None or best_abs is None or d < best_abs:
            best = cand
            best_abs = d

    return best


def _ticks_per_ms(mapping: LinearTimeMapping | None) -> float | None:
    if mapping is None:
        return None
    # host_ms = a * ticks + b  => ticks_per_ms = 1/a
    if mapping.a <= 0:
        return None
    return 1.0 / float(mapping.a)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Debug master/slave timing offsets from captures metadata")
    p.add_argument("--captures-dir", required=True, help="captures directory containing metadata.jsonl")
    p.add_argument("--time-mapping", default="", help="time_mapping_dev_to_host_ms.json path (optional)")
    p.add_argument("--master", default="", help="master camera serial (optional, for nicer reporting)")
    p.add_argument("--event-name", default="ExposureStart", help="camera_event name to analyze")
    p.add_argument("--max-groups", type=int, default=0, help="limit number of groups (0=no limit)")
    p.add_argument(
        "--event-match-window-ms",
        type=float,
        default=5.0,
        help="max allowed |event_ts - frame_dev_ts| in milliseconds (requires mapping; otherwise ignored)",
    )
    p.add_argument("--show-first", type=int, default=3, help="print detailed info for first N groups")
    args = p.parse_args(argv)

    captures_dir = Path(str(args.captures_dir)).resolve()
    meta_path = captures_dir / "metadata.jsonl"
    if not meta_path.exists():
        raise SystemExit(f"metadata.jsonl not found: {meta_path}")

    mapping_path = Path(str(args.time_mapping)).resolve() if str(args.time_mapping).strip() else None
    mappings: dict[str, LinearTimeMapping] | None = None
    if mapping_path is not None:
        if not mapping_path.exists():
            raise SystemExit(f"time mapping json not found: {mapping_path}")
        mappings = load_time_mappings_json(mapping_path)

    groups, events = _iter_groups_and_events(
        metadata_path=meta_path,
        event_name=str(args.event_name),
        max_groups=int(args.max_groups),
    )

    if not groups:
        print("未找到任何 group 记录（metadata.jsonl 中缺少 frames 记录）。")
        return 0

    ev_by_serial = _events_by_serial(events)

    # 统计：每相机在 host/ms/mapped_ms 下相对组内中位数的偏差
    delta_host_ms_by_serial: dict[str, list[float]] = {}
    delta_mapped_ms_by_serial: dict[str, list[float]] = {}

    # 统计：ExposureStart 事件与帧 dev_timestamp 的匹配误差（tick / ms）
    event_frame_dt_ticks_by_serial: dict[str, list[int]] = {}
    event_frame_dt_ms_by_serial: dict[str, list[float]] = {}

    master = str(args.master).strip()

    def _add_delta(dst: dict[str, list[float]], serial: str, v: float) -> None:
        dst.setdefault(str(serial), []).append(float(v))

    for gi, g in enumerate(groups):
        frames = _extract_frames_from_group(g)
        if len(frames) < 2:
            continue

        # --- host_timestamp（归一化为 ms_epoch）---
        host_ms = {fr.serial: fr.host_ms_epoch for fr in frames if fr.host_ms_epoch is not None}
        if len(host_ms) >= 2:
            med = median_float(list(host_ms.values()))
            if med is not None:
                for s, v in host_ms.items():
                    _add_delta(delta_host_ms_by_serial, s, float(v) - float(med))

        # --- dev_timestamp 映射后的 host_ms（ms_epoch）---
        mapped_ms: dict[str, float] = {}
        if mappings is not None:
            for fr in frames:
                m = mappings.get(fr.serial)
                if m is None:
                    continue
                mapped_ms[fr.serial] = float(m.map_dev_to_host_ms(fr.dev_ts))

            if len(mapped_ms) >= 2:
                med = median_float(list(mapped_ms.values()))
                if med is not None:
                    for s, v in mapped_ms.items():
                        _add_delta(delta_mapped_ms_by_serial, s, float(v) - float(med))

        # --- 事件匹配：同 serial 下用 event_timestamp 找最接近 frame.dev_timestamp 的事件 ---
        # 说明：只有当 event_timestamp 与 dev_timestamp 使用同一 tick 时，这个匹配才会非常“紧”。
        for fr in frames:
            evs = ev_by_serial.get(fr.serial, [])
            ev = _find_nearest_event_for_frame(evs_sorted=evs, dev_ts=fr.dev_ts)
            if ev is None:
                continue

            dt_ticks = int(ev.event_ts) - int(fr.dev_ts)

            # 如果提供了 mapping，则可以把 tick 差换算到 ms。
            dt_ms: float | None = None
            if mappings is not None:
                m = mappings.get(fr.serial)
                tpm = _ticks_per_ms(m)
                if tpm is not None and tpm > 0:
                    dt_ms = float(dt_ticks) / float(tpm)

                    # 使用 ms 窗口过滤掉错误匹配（例如拿到了上一帧/下一帧的事件）。
                    if abs(float(dt_ms)) > float(args.event_match_window_ms):
                        continue

            event_frame_dt_ticks_by_serial.setdefault(fr.serial, []).append(int(dt_ticks))
            if dt_ms is not None:
                event_frame_dt_ms_by_serial.setdefault(fr.serial, []).append(float(dt_ms))

        # --- 详细打印（前 N 组）---
        if gi < int(args.show_first):
            print(f"\n[组 {gi}] group_by={g.get('group_by')} group_seq={g.get('group_seq')}")
            if host_ms:
                lo = min(host_ms.values())
                hi = max(host_ms.values())
                print(f"- host_ms_epoch spread={hi - lo:.3f}ms host_ms_by_cam={host_ms}")
            else:
                print("- host_ms_epoch: 不可用（host_timestamp 缺失或单位无法识别）")

            if mappings is not None:
                if mapped_ms:
                    lo = min(mapped_ms.values())
                    hi = max(mapped_ms.values())
                    print(f"- mapped_host_ms spread={hi - lo:.3f}ms mapped_ms_by_cam={mapped_ms}")
                else:
                    print("- mapped_host_ms: 不可用（time_mapping 缺项或该组缺 dev_timestamp）")

            # master 视角：看它相对其他相机的偏移（用映射后的时间更贴近曝光时刻）
            if master and mappings is not None and master in mapped_ms:
                others = [v for s, v in mapped_ms.items() if s != master]
                if others:
                    others_med = median_float(others)
                    if others_med is not None:
                        print(f"- master({master}) vs others_median: dt={mapped_ms[master] - others_med:.3f}ms")

    # --- 汇总输出 ---
    serials = sorted({fr["serial"] for g in groups for fr in (g.get("frames") or []) if isinstance(fr, dict) and fr.get("serial")})
    print("\n==================== 汇总 ====================")
    print(f"groups_total={len(groups)} serials={serials}")

    def _print_delta_stats(title: str, deltas: dict[str, list[float]]) -> None:
        if not deltas:
            print(f"\n{title}: 无可用数据")
            return

        print(f"\n{title}（相对组内中位数，单位 ms）：")
        for s in sorted(deltas.keys()):
            xs = sorted(float(v) for v in deltas[s])
            if not xs:
                continue
            med = _percentile_nearest_rank(xs, 0.50)
            abs_p95 = _percentile_nearest_rank(sorted(abs(v) for v in xs), 0.95)
            print(f"- {s}: median={med:.3f} abs_p95={abs_p95:.3f} n={len(xs)}")

    _print_delta_stats("原始 host_timestamp 归一化后的偏移", delta_host_ms_by_serial)

    if mappings is None:
        print("\n映射后偏移：未提供 --time-mapping，跳过")
    else:
        _print_delta_stats("dev_timestamp_mapping 映射后的偏移", delta_mapped_ms_by_serial)

    # 事件匹配质量：如果 dev_timestamp 与 ExposureStart event_timestamp 同源，dt 通常应非常小。
    if event_frame_dt_ticks_by_serial:
        print("\nExposureStart 事件与帧 dev_timestamp 的差（event_ts - dev_ts）：")
        for s in sorted(event_frame_dt_ticks_by_serial.keys()):
            xs_ticks = sorted(event_frame_dt_ticks_by_serial[s])
            med_ticks = _percentile_nearest_rank([float(x) for x in xs_ticks], 0.50)
            abs_p95_ticks = _percentile_nearest_rank(sorted(abs(float(x)) for x in xs_ticks), 0.95)

            extra = ""
            xs_ms = event_frame_dt_ms_by_serial.get(s)
            if xs_ms:
                ms_sorted = sorted(xs_ms)
                med_ms = _percentile_nearest_rank(ms_sorted, 0.50)
                abs_p95_ms = _percentile_nearest_rank(sorted(abs(v) for v in ms_sorted), 0.95)
                extra = f" (ms: median={med_ms:.6f} abs_p95={abs_p95_ms:.6f})"

            print(
                f"- {s}: ticks median={med_ticks:.1f} abs_p95={abs_p95_ticks:.1f} n={len(xs_ticks)}{extra}"
            )

    else:
        print("\nExposureStart 事件匹配：未找到任何可用的 camera_event 记录（或匹配窗口过滤后为空）。")

    # 最后的“人话提示”
    if mappings is not None and master and master in delta_mapped_ms_by_serial:
        m_med = median_float(delta_mapped_ms_by_serial.get(master, []))
        if m_med is not None and abs(float(m_med)) >= 5.0:
            print(
                "\n提示：master 在映射后时间轴上的偏移仍然显著（>=5ms）。这通常不是字段缺失，而是触发链路/IO 输出模式导致的真实曝光时刻偏移。\n"
                "建议优先检查：LineMode(Output/Strobe)、StrobeDelay/StrobeDuration、LineSource/StrobeSource、LineInverter、TriggerActivation 边沿。"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
