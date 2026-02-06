# -*- coding: utf-8 -*-

"""生成采集时序统计报告（尽量细致）。

本脚本面向 `mvs.apps.quad_capture` 的采集输出目录（captures 目录），
从 `metadata.jsonl` 中提取并统计：

- 主机域（time.monotonic）：
  - soft_trigger_send 间隔（抖动）
        - send -> frame arrival 延迟（按 soft_trigger_send.seq 与 group.group_seq 对齐；若无法对齐则降级为仅统计各自分布）
  - send -> camera_event callback 延迟（先用设备 ticks 匹配事件，再用 host_monotonic 做差）
  - camera_event callback -> frame arrival（通知到图像到达的相对时序）

- 设备域（ticks）：
  - event_timestamp - frame.dev_timestamp（每台相机、每个事件）
  - ExposureStart/ExposureEnd 组合时的曝光时长（end-start）

- 组级：
  - arrival_spread：组内 max(arrival_monotonic) - min(arrival_monotonic)
  - （可选）host_timestamp 归一化到 ms_epoch 后的组内 spread
  - （可选）dev_timestamp 映射到 host_ms 后的组内 spread 与相对中位数偏差

说明与约束：
- `metadata.jsonl` 的写入顺序不保证物理时序，本脚本以字段语义为准做统计。
- send 与 group 的对齐：
    - 使用 soft_trigger_send.seq == group.group_seq。
    - 若 group_seq 缺失/不可用，会退化为“仅统计各自分布”，并在报告中提示。
- event 与 frame 的对齐使用同 serial + 最近邻（按设备 ticks 二分），并支持窗口过滤。
- 输出同时包含 JSON（机器可读）与 Markdown（人类可读）。

用法示例：
    uv run python tools/report_capture_timing.py \
      --captures-dir data/captures_master_slave/tennis_offline \
      --master-serial DA8199303 \
      --event-name ExposureStart --event-name ExposureEnd

"""

import argparse
import bisect
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mvs.session.metadata_io import iter_metadata_records
from mvs.session.time_mapping import LinearTimeMapping, load_time_mappings_json
from tennis3d.pipeline.time_utils import (
    delta_to_median_by_camera,
    host_timestamp_to_ms_epoch,
    spread_ms,
)


@dataclass(frozen=True, slots=True)
class _SendRec:
    """soft_trigger_send 记录。"""

    seq: int
    created_at: float | None
    host_monotonic: float | None
    targets: list[str]


@dataclass(frozen=True, slots=True)
class _EventRec:
    """camera_event 记录（从 metadata 抽取）。"""

    event_ts: int
    created_at: float | None
    host_monotonic: float | None


@dataclass
class _Reservoir:
    """固定容量水库抽样（用于近似分位数，避免爆内存）。"""

    max_samples: int
    seed: int

    seen: int = 0
    samples: list[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.samples = []

    def add(self, v: float) -> None:
        self.seen += 1
        if self.max_samples <= 0:
            return
        if len(self.samples) < self.max_samples:
            self.samples.append(float(v))
            return

        # 说明：这里用一个非常轻量的 LCG 伪随机来做“可复现”的抽样。
        # 目标不是加密随机，只要跨平台一致、成本低即可。
        j = _randint_0_n(self.seed, self.seen)
        if j < self.max_samples:
            self.samples[j] = float(v)


def _randint_0_n(seed: int, n: int) -> int:
    """返回 [0, n) 的伪随机整数（可复现）。"""

    if n <= 0:
        return 0
    # LCG: x_{k+1} = (a*x_k + c) mod m
    # 这里让 seed 与 seen 共同决定“当前状态”，避免维护全局状态。
    x = (int(seed) ^ (n * 1103515245 + 12345)) & 0x7FFFFFFF
    return int(x % int(n))


def _percentile_nearest_rank(sorted_values: list[float], q: float) -> float | None:
    """nearest-rank 分位数（不插值）。"""

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


def _summary_from_samples(samples: list[float]) -> dict[str, Any]:
    """把样本汇总成可序列化的统计信息。"""

    if not samples:
        return {
            "n": 0,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
            "mean": None,
            "abs_p95": None,
        }

    xs = sorted(float(x) for x in samples)
    mean = float(math.fsum(xs) / float(len(xs)))
    abs_p95 = _percentile_nearest_rank(sorted(abs(x) for x in xs), 0.95)

    return {
        "n": int(len(xs)),
        "p50": _percentile_nearest_rank(xs, 0.50),
        "p90": _percentile_nearest_rank(xs, 0.90),
        "p95": _percentile_nearest_rank(xs, 0.95),
        "p99": _percentile_nearest_rank(xs, 0.99),
        "min": float(xs[0]),
        "max": float(xs[-1]),
        "mean": float(mean),
        "abs_p95": float(abs_p95) if abs_p95 is not None else None,
    }


def _ticks_to_ms(mapping: LinearTimeMapping | None, dt_ticks: int) -> float | None:
    """用线性映射的斜率把 ticks 差换算成毫秒。"""

    if mapping is None:
        return None
    if mapping.a <= 0:
        return None
    return float(mapping.a) * float(int(dt_ticks))


def _find_nearest_event_by_ticks(
    *,
    evs_sorted: list[_EventRec],
    ev_ts_keys: list[int],
    dev_ts: int,
) -> tuple[_EventRec, int] | None:
    """在同一 (serial,event_name) 中按 event_timestamp 找最近邻。"""

    if not evs_sorted:
        return None

    i = bisect.bisect_left(ev_ts_keys, int(dev_ts))

    best: _EventRec | None = None
    best_abs: int | None = None
    for j in (i - 1, i, i + 1):
        if j < 0 or j >= len(evs_sorted):
            continue
        cand = evs_sorted[j]
        d = abs(int(cand.event_ts) - int(dev_ts))
        if best is None or best_abs is None or d < best_abs:
            best = cand
            best_abs = int(d)

    if best is None or best_abs is None:
        return None
    return best, int(best_abs)


def _write_markdown_report(*, out_md: Path, report: dict[str, Any]) -> None:
    """把 JSON 报告渲染成简洁的 Markdown。"""

    meta = report.get("meta", {}) if isinstance(report, dict) else {}
    counts = report.get("counts", {}) if isinstance(report, dict) else {}

    lines: list[str] = []
    lines.append("# 采集时序统计报告\n")

    lines.append("## 基本信息\n")
    lines.append(f"- captures_dir: `{meta.get('captures_dir', '')}`")
    lines.append(f"- metadata_path: `{meta.get('metadata_path', '')}`")
    lines.append(f"- time_mapping_path: `{meta.get('time_mapping_path', '')}`")
    lines.append(f"- master_serial: `{meta.get('master_serial', '')}`")
    lines.append(f"- event_names: `{', '.join(meta.get('event_names', []) or [])}`\n")
    lines.append(f"- send_alignment: `{meta.get('send_alignment', '')}`\n")

    lines.append("## 计数\n")
    for k in sorted(counts.keys()):
        lines.append(f"- {k}: {counts[k]}")
    lines.append("")

    def _md_stats(title: str, node: Any, unit: str) -> None:
        if not isinstance(node, dict):
            return
        s = node.get("stats")
        if not isinstance(s, dict) or not s.get("n"):
            lines.append(f"### {title}\n")
            lines.append("- 无可用数据\n")
            return

        lines.append(f"### {title}\n")
        lines.append(f"- n={s.get('n')} unit={unit}")
        lines.append(
            "- p50={p50:.3f} p90={p90:.3f} p95={p95:.3f} p99={p99:.3f} abs_p95={abs_p95:.3f}".format(
                p50=float(s.get("p50") or 0.0),
                p90=float(s.get("p90") or 0.0),
                p95=float(s.get("p95") or 0.0),
                p99=float(s.get("p99") or 0.0),
                abs_p95=float(s.get("abs_p95") or 0.0),
            )
        )
        lines.append("")

    host = report.get("host", {}) if isinstance(report, dict) else {}
    groups = report.get("groups", {}) if isinstance(report, dict) else {}

    lines.append("## 主机域（monotonic）\n")
    _md_stats("soft_trigger_send 间隔", host.get("soft_trigger_interval_ms"), "ms")
    _md_stats("send -> frame arrival", host.get("send_to_arrival_ms"), "ms")
    _md_stats("send -> event callback", host.get("send_to_event_cb_ms"), "ms")
    _md_stats("event callback -> frame arrival", host.get("event_cb_to_arrival_ms"), "ms")

    by_serial = host.get("send_to_arrival_ms_by_serial")
    if isinstance(by_serial, dict) and by_serial:
        lines.append("### send -> frame arrival（按相机）\n")
        lines.append("| serial | n | p50(ms) | p95(ms) | abs_p95(ms) |")
        lines.append("|---|---:|---:|---:|---:|")
        for serial in sorted(by_serial.keys()):
            node = by_serial.get(serial)
            if not isinstance(node, dict):
                continue
            s = node.get("stats")
            if not isinstance(s, dict) or not s.get("n"):
                continue
            lines.append(
                "| {serial} | {n} | {p50:.3f} | {p95:.3f} | {abs_p95:.3f} |".format(
                    serial=serial,
                    n=int(s.get("n", 0) or 0),
                    p50=float(s.get("p50") or 0.0),
                    p95=float(s.get("p95") or 0.0),
                    abs_p95=float(s.get("abs_p95") or 0.0),
                )
            )
        lines.append("")

    lines.append("## 组级\n")
    _md_stats("组内 arrival_spread", groups.get("arrival_spread_ms"), "ms")
    _md_stats("组内 host_timestamp spread（归一化 ms_epoch）", groups.get("host_ms_epoch_spread_ms"), "ms")
    _md_stats("组内 dev_timestamp 映射后 spread（ms_epoch）", groups.get("mapped_host_ms_spread_ms"), "ms")

    events = report.get("events", {}) if isinstance(report, dict) else {}
    by_se = events.get("by_serial_event", {}) if isinstance(events, dict) else {}

    def _fmt_opt_float(v: Any, fmt: str) -> str:
        """格式化可选数值。

        说明：
        - 当 v 为 None/不可转为 float 时，返回空字符串，用于 Markdown 表格空单元格。
        - 避免出现 "Unknown format code 'f' for object of type 'str'" 这类错误。
        """

        if v is None:
            return ""
        try:
            return format(float(v), fmt)
        except Exception:
            return ""

    lines.append("## 设备域（ticks / 可选 ms）\n")
    if not isinstance(by_se, dict) or not by_se:
        lines.append("- 未统计到事件-帧匹配数据（可能未订阅事件）。\n")
    else:
        lines.append("说明：dt = event_timestamp - frame.dev_timestamp；dt<0 表示事件发生在 dev_timestamp 之前（常见于最近邻匹配跨帧时）。\n")
        # 简洁输出：每个 serial/event 一行。
        lines.append("| serial | event | match_rate | dt_ticks_p50 | dt_ticks_min | dt_ticks_max | dt_ms_p50 | dt_ms_min | dt_ms_max | |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for serial in sorted(by_se.keys()):
            row = by_se.get(serial)
            if not isinstance(row, dict):
                continue
            for ev_name in sorted(row.keys()):
                node = row.get(ev_name)
                if not isinstance(node, dict):
                    continue
                n_frames = int(node.get("n_frames", 0) or 0)
                n_matched = int(node.get("n_matched", 0) or 0)
                rate = (float(n_matched) / float(n_frames)) if n_frames > 0 else 0.0

                dt_ticks = node.get("dt_ticks", {})
                dt_ms = node.get("dt_ms", {})
                p50_ticks = None
                min_ticks = None
                max_ticks = None
                p50_ms = None
                min_ms = None
                max_ms = None
                if isinstance(dt_ticks, dict):
                    st = dt_ticks.get("stats", {})
                    if isinstance(st, dict):
                        p50_ticks = st.get("p50")
                        min_ticks = st.get("min")
                        max_ticks = st.get("max")
                if isinstance(dt_ms, dict):
                    sm = dt_ms.get("stats", {})
                    if isinstance(sm, dict):
                        p50_ms = sm.get("p50")
                        min_ms = sm.get("min")
                        max_ms = sm.get("max")

                p50_ticks_s = _fmt_opt_float(p50_ticks, ".1f")
                min_ticks_s = _fmt_opt_float(min_ticks, ".1f")
                max_ticks_s = _fmt_opt_float(max_ticks, ".1f")
                p50_ms_s = _fmt_opt_float(p50_ms, ".6f")
                min_ms_s = _fmt_opt_float(min_ms, ".6f")
                max_ms_s = _fmt_opt_float(max_ms, ".6f")
                lines.append(
                    f"| {serial} | {ev_name} | {rate:.3f} | {p50_ticks_s} | {min_ticks_s} | {max_ticks_s} | {p50_ms_s} | {min_ms_s} | {max_ms_s} | |"
                )
        lines.append("")

    exposure = report.get("exposure", {}) if isinstance(report, dict) else {}
    if isinstance(exposure, dict) and exposure.get("by_serial"):
        lines.append("## 曝光时长（ExposureEnd - ExposureStart）\n")
        lines.append("| serial | n | duration_ticks_p50 | duration_ms_p50 | |")
        lines.append("|---|---:|---:|---:|---|")
        by_s = exposure.get("by_serial", {})
        if isinstance(by_s, dict):
            for serial in sorted(by_s.keys()):
                node = by_s.get(serial)
                if not isinstance(node, dict):
                    continue
                dt_ticks = node.get("duration_ticks", {}).get("stats", {})
                dt_ms = node.get("duration_ms", {}).get("stats", {})

                p50_ticks_s = _fmt_opt_float(dt_ticks.get("p50"), ".1f")
                p50_ms_s = _fmt_opt_float(dt_ms.get("p50"), ".6f")
                lines.append(
                    "| {serial} | {n} | {p50_ticks} | {p50_ms} | |".format(
                        serial=serial,
                        n=int(dt_ticks.get("n", 0) or 0),
                        p50_ticks=p50_ticks_s,
                        p50_ms=p50_ms_s,
                    )
                )
        lines.append("")

    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="生成 captures 的细致时序统计报告（JSON + Markdown）")
    p.add_argument("--captures-dir", required=True, help="captures 目录（包含 metadata.jsonl）")
    p.add_argument("--out-json", default="", help="输出 JSON 路径（默认：<captures-dir>/timing_report.json）")
    p.add_argument("--out-md", default="", help="输出 Markdown 路径（默认：<captures-dir>/timing_report.md）")
    p.add_argument(
        "--time-mapping",
        default="",
        help=(
            "time_mapping_dev_to_host_ms.json 路径（可选）。"
            "若为空且 <captures-dir>/time_mapping_dev_to_host_ms.json 存在，则自动使用。"
        ),
    )
    p.add_argument("--master-serial", default="", help="master 相机 serial（用于 send->arrival 的优先选择）")
    p.add_argument(
        "--event-name",
        action="append",
        default=[],
        help="要分析的 camera_event.event_name（可重复指定；默认=ExposureStart）",
    )
    p.add_argument("--max-groups", type=int, default=0, help="最多处理 N 个 group（0=不限制）")
    p.add_argument("--max-sends", type=int, default=0, help="最多处理 N 个 soft_trigger_send（0=不限制）")
    p.add_argument(
        "--event-match-window-ms",
        type=float,
        default=5.0,
        help="使用 time-mapping 时，过滤 |dt_ms| > window 的事件匹配（降低误配）",
    )
    p.add_argument(
        "--event-match-window-ticks",
        type=int,
        default=0,
        help="没有 time-mapping 时，过滤 |dt_ticks| > window 的事件匹配（0=不启用）",
    )
    p.add_argument(
        "--max-stats-samples",
        type=int,
        default=200000,
        help="每个指标最多保留的样本数（水库抽样上限）",
    )
    p.add_argument("--seed", type=int, default=0, help="抽样随机种子（可复现）")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    captures_dir = Path(str(args.captures_dir)).resolve()
    meta_path = captures_dir / "metadata.jsonl"
    if not meta_path.exists():
        raise SystemExit(f"metadata.jsonl not found: {meta_path}")

    out_json = Path(str(args.out_json)).resolve() if str(args.out_json).strip() else (captures_dir / "timing_report.json")
    out_md = Path(str(args.out_md)).resolve() if str(args.out_md).strip() else (captures_dir / "timing_report.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    master_serial = str(args.master_serial or "").strip()

    event_names = [str(x).strip() for x in (args.event_name or []) if str(x).strip()]
    if not event_names:
        event_names = ["ExposureStart"]

    wanted_event_names = set(event_names)

    # time mapping（可选）
    mapping_path: Path | None
    if str(args.time_mapping).strip():
        mapping_path = Path(str(args.time_mapping)).resolve()
    else:
        cand = captures_dir / "time_mapping_dev_to_host_ms.json"
        mapping_path = cand if cand.exists() else None

    mappings: dict[str, LinearTimeMapping] | None = None
    if mapping_path is not None:
        if not mapping_path.exists():
            raise SystemExit(f"time mapping json not found: {mapping_path}")
        mappings = load_time_mappings_json(mapping_path)

    # pass1: 扫描 send 与 events（顺便收集 group_seq 分布，用于判断能否用它对齐 send）
    sends_by_seq: dict[int, _SendRec] = {}
    sends_in_order: list[_SendRec] = []

    events_by_serial: dict[str, dict[str, list[_EventRec]]] = {}

    n_records = 0
    n_groups = 0
    n_events_total = 0
    n_sends_total = 0

    group_seqs_scanned: list[int] = []

    for rec in iter_metadata_records(meta_path):
        n_records += 1
        if not isinstance(rec, dict):
            continue

        rtype = str(rec.get("type", ""))

        if rtype == "soft_trigger_send":
            seq_raw = rec.get("seq")
            if seq_raw is None:
                continue
            try:
                seq = int(seq_raw)
            except Exception:
                continue

            created_at = rec.get("created_at")
            host_mono = rec.get("host_monotonic")
            targets_raw = rec.get("targets")
            targets: list[str] = []
            if isinstance(targets_raw, list):
                targets = [str(x).strip() for x in targets_raw if str(x).strip()]

            srec = _SendRec(
                seq=int(seq),
                created_at=float(created_at) if created_at is not None else None,
                host_monotonic=float(host_mono) if host_mono is not None else None,
                targets=targets,
            )

            # 说明：同一个 seq 理论上应唯一。若出现重复，这里保留最早出现的一条。
            if int(seq) not in sends_by_seq:
                sends_by_seq[int(seq)] = srec
            sends_in_order.append(srec)
            n_sends_total += 1
            if int(args.max_sends) > 0 and n_sends_total >= int(args.max_sends):
                # 只限制 send 的收集量，不影响后续 group 的扫描。
                pass
            continue

        if rtype == "camera_event":
            name = str(rec.get("event_name", "")).strip()
            if name not in wanted_event_names:
                continue

            serial = str(rec.get("serial", "")).strip()
            if not serial:
                continue

            ts_raw = rec.get("event_timestamp")
            if ts_raw is None:
                continue
            try:
                ts = int(ts_raw)
            except Exception:
                continue

            created_at = rec.get("created_at")
            host_mono = rec.get("host_monotonic")

            events_by_serial.setdefault(serial, {}).setdefault(name, []).append(
                _EventRec(
                    event_ts=int(ts),
                    created_at=float(created_at) if created_at is not None else None,
                    host_monotonic=float(host_mono) if host_mono is not None else None,
                )
            )
            n_events_total += 1
            continue

        # group 记录计数（frames）
        frames = rec.get("frames")
        if isinstance(frames, list) and frames:
            n_groups += 1

            gs_raw = rec.get("group_seq")
            if gs_raw is None:
                gs = -1
            else:
                try:
                    gs = int(gs_raw)
                except Exception:
                    gs = -1
            group_seqs_scanned.append(int(gs))
            continue

    # 排序事件，预计算 keys 便于 bisect
    ev_keys: dict[tuple[str, str], list[int]] = {}
    for serial, by_name in events_by_serial.items():
        for name, evs in by_name.items():
            evs.sort(key=lambda e: int(e.event_ts))
            ev_keys[(serial, name)] = [int(e.event_ts) for e in evs]

    # 准备统计桶
    max_samples = int(args.max_stats_samples)
    seed = int(args.seed)

    # 主机域指标（全局）
    soft_interval_ms = _Reservoir(max_samples=max_samples, seed=seed)
    send_to_arrival_ms = _Reservoir(max_samples=max_samples, seed=seed)
    send_to_event_cb_ms = _Reservoir(max_samples=max_samples, seed=seed)
    event_cb_to_arrival_ms = _Reservoir(max_samples=max_samples, seed=seed)

    # 主机域指标（按相机）
    send_to_arrival_ms_by_serial: dict[str, _Reservoir] = {}

    # 组级指标
    arrival_spread_ms = _Reservoir(max_samples=max_samples, seed=seed)
    host_ms_epoch_spread_ms = _Reservoir(max_samples=max_samples, seed=seed)
    mapped_host_ms_spread_ms = _Reservoir(max_samples=max_samples, seed=seed)

    # 设备域：按 serial/event 分桶
    by_serial_event: dict[str, dict[str, dict[str, Any]]] = {}

    def _bucket(serial: str, name: str) -> dict[str, Any]:
        bs = by_serial_event.setdefault(str(serial), {})
        b = bs.get(str(name))
        if b is None:
            b = {
                "serial": str(serial),
                "event_name": str(name),
                "n_frames": 0,
                "n_matched": 0,
                "n_filtered": 0,
                "n_no_event": 0,
                "n_no_mapping": 0,
                "dt_ticks": _Reservoir(max_samples=max_samples, seed=seed ^ 0xA5A5A5),
                "dt_ms": _Reservoir(max_samples=max_samples, seed=seed ^ 0x5A5A5A),
                # 主机域：仅在 send 与事件回调/到达时间都可用时才会填充。
                "send_to_event_cb_ms": _Reservoir(max_samples=max_samples, seed=seed ^ 0xBEEF),
                "event_cb_to_arrival_ms": _Reservoir(max_samples=max_samples, seed=seed ^ 0xFEED),
            }
            bs[str(name)] = b
        return b

    # 曝光时长：按 serial
    exposure_by_serial: dict[str, dict[str, Any]] = {}

    def _exp_bucket(serial: str) -> dict[str, Any]:
        b = exposure_by_serial.get(str(serial))
        if b is None:
            b = {
                "serial": str(serial),
                "duration_ticks": _Reservoir(max_samples=max_samples, seed=seed ^ 0x13579B),
                "duration_ms": _Reservoir(max_samples=max_samples, seed=seed ^ 0x2468AC),
                "n": 0,
                "n_missing": 0,
                "n_invalid": 0,
            }
            exposure_by_serial[str(serial)] = b
        return b

    # 统计 soft_trigger_send 间隔（按 host_monotonic 的出现顺序）
    prev_send_mono: float | None = None
    for srec in sends_in_order:
        if prev_send_mono is not None and srec.host_monotonic is not None:
            soft_interval_ms.add((float(srec.host_monotonic) - float(prev_send_mono)) * 1000.0)
        if srec.host_monotonic is not None:
            prev_send_mono = float(srec.host_monotonic)

    # 决定 send 对齐方式：当前 schema 只提供 group_seq，因此只尝试使用 group_seq。
    # 若 group_seq 缺失/不可用，则降级为不做 send 对齐。
    valid_group_seqs = [int(x) for x in group_seqs_scanned if int(x) >= 0]
    send_alignment = "group_seq" if (n_sends_total > 0 and len(valid_group_seqs) > 0) else "none"

    window_ms = float(args.event_match_window_ms)
    window_ticks = int(args.event_match_window_ticks)

    def _is_filtered(*, serial: str, dt_ticks: int) -> bool:
        """对单个事件匹配应用窗口过滤。

        规则：
        - 若该相机有 mapping：优先用 |dt_ms| > window_ms 过滤。
        - 若无 mapping：可选用 |dt_ticks| > window_ticks 过滤。
        """

        m = mappings.get(serial) if mappings is not None else None
        dt_ms = _ticks_to_ms(m, int(dt_ticks))
        if dt_ms is not None and window_ms > 0:
            return abs(float(dt_ms)) > float(window_ms)
        if window_ticks > 0:
            return abs(int(dt_ticks)) > int(window_ticks)
        return False

    # pass2: 再扫一遍 group，做更细统计（避免把所有 group 存内存）
    groups_seen = 0
    n_groups_with_send = 0
    n_frames_total = 0

    for rec in iter_metadata_records(meta_path):
        if not isinstance(rec, dict):
            continue

        frames = rec.get("frames")
        if not isinstance(frames, list) or not frames:
            continue

        groups_seen += 1
        if int(args.max_groups) > 0 and groups_seen > int(args.max_groups):
            break

        group_seq_raw = rec.get("group_seq")
        group_seq = -1
        if group_seq_raw is not None:
            try:
                group_seq = int(group_seq_raw)
            except Exception:
                group_seq = -1

        if send_alignment == "group_seq":
            send = sends_by_seq.get(int(group_seq)) if int(group_seq) >= 0 else None
        else:
            send = None
        if send is not None:
            n_groups_with_send += 1

        # group arrival spread
        arrivals: list[float] = []
        for fr in frames:
            if not isinstance(fr, dict):
                continue
            a = fr.get("arrival_monotonic")
            if a is None:
                continue
            try:
                arrivals.append(float(a))
            except Exception:
                continue

        if len(arrivals) >= 2:
            arrival_spread_ms.add((max(arrivals) - min(arrivals)) * 1000.0)

        # group host_timestamp spread（归一化到 ms_epoch）
        host_ms_vals: list[float] = []
        for fr in frames:
            if not isinstance(fr, dict):
                continue
            v = host_timestamp_to_ms_epoch(fr.get("host_timestamp"))
            if v is None:
                continue
            host_ms_vals.append(float(v))

        if len(host_ms_vals) >= 2:
            sp = spread_ms(host_ms_vals)
            if sp is not None:
                host_ms_epoch_spread_ms.add(float(sp))

        # group mapped host_ms spread（若有 mapping）
        mapped_by_cam: dict[str, float] = {}
        if mappings is not None:
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

                m = mappings.get(serial)
                if m is None:
                    continue
                mapped_by_cam[serial] = float(m.map_dev_to_host_ms(int(dev_ts)))

            if len(mapped_by_cam) >= 2:
                sp = spread_ms(list(mapped_by_cam.values()))
                if sp is not None:
                    mapped_host_ms_spread_ms.add(float(sp))

        # 每帧：send->arrival / event match / send->cb 等
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

            arrival_raw = fr.get("arrival_monotonic")
            arrival_mono: float | None
            if arrival_raw is None:
                arrival_mono = None
            else:
                try:
                    arrival_mono = float(arrival_raw)
                except Exception:
                    arrival_mono = None

            n_frames_total += 1

            # send -> arrival（按 seq 与 group_seq 对齐）
            if send is not None and send.host_monotonic is not None and arrival_mono is not None:
                dt = (float(arrival_mono) - float(send.host_monotonic)) * 1000.0
                send_to_arrival_ms.add(float(dt))
                send_to_arrival_ms_by_serial.setdefault(
                    str(serial), _Reservoir(max_samples=max_samples, seed=seed ^ 0x123456)
                ).add(float(dt))

            # 事件匹配（设备 ticks 最近邻）
            for name in event_names:
                b = _bucket(serial, name)
                b["n_frames"] += 1

                evs = events_by_serial.get(serial, {}).get(name, [])
                keys = ev_keys.get((serial, name), [])
                found = _find_nearest_event_by_ticks(evs_sorted=evs, ev_ts_keys=keys, dev_ts=int(dev_ts))
                if found is None:
                    b["n_no_event"] += 1
                    continue

                ev, abs_ticks = found
                dt_ticks = int(ev.event_ts) - int(dev_ts)

                m = mappings.get(serial) if mappings is not None else None
                dt_ms = _ticks_to_ms(m, dt_ticks)

                # 窗口过滤：优先用 ms（若有 mapping），否则用 ticks（可选）
                filtered = False
                if dt_ms is not None and window_ms > 0:
                    if abs(float(dt_ms)) > float(window_ms):
                        filtered = True
                elif dt_ms is None and mappings is not None:
                    # 有 mapping 文件但该相机缺项
                    b["n_no_mapping"] += 1
                    if window_ticks > 0 and abs(int(dt_ticks)) > int(window_ticks):
                        filtered = True
                else:
                    if window_ticks > 0 and abs(int(dt_ticks)) > int(window_ticks):
                        filtered = True

                if filtered:
                    b["n_filtered"] += 1
                    continue

                b["n_matched"] += 1
                b["dt_ticks"].add(float(dt_ticks))
                if dt_ms is not None:
                    b["dt_ms"].add(float(dt_ms))

                # send -> event callback（同一帧匹配到的事件，拿它的 host_monotonic 与 send 做差）
                if send is not None and send.host_monotonic is not None and ev.host_monotonic is not None:
                    dt_cb = (float(ev.host_monotonic) - float(send.host_monotonic)) * 1000.0
                    send_to_event_cb_ms.add(float(dt_cb))
                    b["send_to_event_cb_ms"].add(float(dt_cb))

                # event callback -> arrival
                if ev.host_monotonic is not None and arrival_mono is not None:
                    dt_ca = (float(arrival_mono) - float(ev.host_monotonic)) * 1000.0
                    event_cb_to_arrival_ms.add(float(dt_ca))
                    b["event_cb_to_arrival_ms"].add(float(dt_ca))

            # 曝光时长（若同时订阅了 ExposureStart/ExposureEnd）
            if "ExposureStart" in set(event_names) and "ExposureEnd" in set(event_names):
                evs_start = events_by_serial.get(serial, {}).get("ExposureStart", [])
                keys_start = ev_keys.get((serial, "ExposureStart"), [])
                evs_end = events_by_serial.get(serial, {}).get("ExposureEnd", [])
                keys_end = ev_keys.get((serial, "ExposureEnd"), [])

                f1 = _find_nearest_event_by_ticks(evs_sorted=evs_start, ev_ts_keys=keys_start, dev_ts=int(dev_ts))
                f2 = _find_nearest_event_by_ticks(evs_sorted=evs_end, ev_ts_keys=keys_end, dev_ts=int(dev_ts))

                expb = _exp_bucket(serial)
                if f1 is None or f2 is None:
                    expb["n_missing"] += 1
                else:
                    ev1, _abs1 = f1
                    ev2, _abs2 = f2

                    # 与上面的事件桶一致：若任一端明显错配，跳过时长统计。
                    dt1 = int(ev1.event_ts) - int(dev_ts)
                    dt2 = int(ev2.event_ts) - int(dev_ts)
                    if _is_filtered(serial=serial, dt_ticks=int(dt1)) or _is_filtered(serial=serial, dt_ticks=int(dt2)):
                        expb["n_invalid"] += 1
                        continue

                    dur_ticks = int(ev2.event_ts) - int(ev1.event_ts)
                    if dur_ticks <= 0:
                        expb["n_invalid"] += 1
                    else:
                        expb["n"] += 1
                        expb["duration_ticks"].add(float(dur_ticks))
                        m = mappings.get(serial) if mappings is not None else None
                        dur_ms = _ticks_to_ms(m, dur_ticks)
                        if dur_ms is not None:
                            expb["duration_ms"].add(float(dur_ms))

    # 组内相对中位数偏差（映射后的 host_ms）——按 serial 聚合
    # 说明：这个口径能直接复用 debug_master_slave_timing 的直觉，但这里做“全局统计”。
    mapped_delta_ms_by_serial: dict[str, _Reservoir] = {}
    if mappings is not None:
        groups_seen2 = 0
        for rec in iter_metadata_records(meta_path):
            if not isinstance(rec, dict):
                continue
            frames = rec.get("frames")
            if not isinstance(frames, list) or not frames:
                continue
            groups_seen2 += 1
            if int(args.max_groups) > 0 and groups_seen2 > int(args.max_groups):
                break

            mapped_ms_by_cam: dict[str, float] = {}
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
                m = mappings.get(serial)
                if m is None:
                    continue
                mapped_ms_by_cam[serial] = float(m.map_dev_to_host_ms(int(dev_ts)))

            deltas = delta_to_median_by_camera(mapped_ms_by_cam)
            if not isinstance(deltas, dict):
                continue
            for s, d in deltas.items():
                mapped_delta_ms_by_serial.setdefault(
                    str(s), _Reservoir(max_samples=max_samples, seed=seed ^ 0xCAFEBABE)
                ).add(float(d))

    # 组装报告
    report: dict[str, Any] = {
        "meta": {
            "created_at": float(time.time()),
            "captures_dir": str(captures_dir),
            "metadata_path": str(meta_path),
            "time_mapping_path": str(mapping_path) if mapping_path is not None else "",
            "has_time_mapping": bool(mappings is not None),
            "master_serial": master_serial,
            "event_names": event_names,
            "send_alignment": str(send_alignment),
            "args": {
                "max_groups": int(args.max_groups),
                "max_sends": int(args.max_sends),
                "event_match_window_ms": float(args.event_match_window_ms),
                "event_match_window_ticks": int(args.event_match_window_ticks),
                "max_stats_samples": int(args.max_stats_samples),
                "seed": int(args.seed),
            },
        },
        "counts": {
            "records_scanned": int(n_records),
            "groups_scanned": int(n_groups),
            "groups_analyzed": int(groups_seen),
            "frames_analyzed": int(n_frames_total),
            "soft_trigger_send_scanned": int(n_sends_total),
            "camera_event_scanned": int(n_events_total),
            "groups_with_send_match": int(n_groups_with_send),
            "send_match_rate": float(n_groups_with_send) / float(groups_seen) if groups_seen > 0 else 0.0,
            "send_match_rate_by_group_seq": float(n_groups_with_send) / float(groups_seen)
            if (groups_seen > 0 and send_alignment == "group_seq")
            else 0.0,
            "groups_with_valid_group_seq": int(sum(1 for x in valid_group_seqs if int(x) >= 0)),
        },
        "host": {
            "soft_trigger_interval_ms": {
                "stats": _summary_from_samples(soft_interval_ms.samples),
                "seen": int(soft_interval_ms.seen),
            },
            "send_to_arrival_ms": {
                "stats": _summary_from_samples(send_to_arrival_ms.samples),
                "seen": int(send_to_arrival_ms.seen),
            },
            "send_to_event_cb_ms": {
                "stats": _summary_from_samples(send_to_event_cb_ms.samples),
                "seen": int(send_to_event_cb_ms.seen),
            },
            "event_cb_to_arrival_ms": {
                "stats": _summary_from_samples(event_cb_to_arrival_ms.samples),
                "seen": int(event_cb_to_arrival_ms.seen),
            },
            "send_to_arrival_ms_by_serial": {
                s: {
                    "stats": _summary_from_samples(r.samples),
                    "seen": int(r.seen),
                }
                for s, r in sorted(send_to_arrival_ms_by_serial.items())
            },
        },
        "groups": {
            "arrival_spread_ms": {
                "stats": _summary_from_samples(arrival_spread_ms.samples),
                "seen": int(arrival_spread_ms.seen),
            },
            "host_ms_epoch_spread_ms": {
                "stats": _summary_from_samples(host_ms_epoch_spread_ms.samples),
                "seen": int(host_ms_epoch_spread_ms.seen),
            },
            "mapped_host_ms_spread_ms": {
                "stats": _summary_from_samples(mapped_host_ms_spread_ms.samples),
                "seen": int(mapped_host_ms_spread_ms.seen),
            },
            "mapped_delta_to_median_ms_by_serial": {
                s: {
                    "stats": _summary_from_samples(r.samples),
                    "seen": int(r.seen),
                }
                for s, r in sorted(mapped_delta_ms_by_serial.items())
            },
        },
        "events": {
            "by_serial_event": {},
        },
        "exposure": {
            "by_serial": {},
        },
    }

    # 序列化事件桶（把 Reservoir 转成 stats）
    for serial, by_name in sorted(by_serial_event.items()):
        out_ev: dict[str, Any] = {}
        for name, b in sorted(by_name.items()):
            dt_ticks_res: _Reservoir = b["dt_ticks"]
            dt_ms_res: _Reservoir = b["dt_ms"]
            out_ev[name] = {
                "serial": str(b.get("serial", serial)),
                "event_name": str(b.get("event_name", name)),
                "n_frames": int(b.get("n_frames", 0) or 0),
                "n_matched": int(b.get("n_matched", 0) or 0),
                "n_filtered": int(b.get("n_filtered", 0) or 0),
                "n_no_event": int(b.get("n_no_event", 0) or 0),
                "n_no_mapping": int(b.get("n_no_mapping", 0) or 0),
                "dt_ticks": {
                    "stats": _summary_from_samples(dt_ticks_res.samples),
                    "seen": int(dt_ticks_res.seen),
                },
                "dt_ms": {
                    "stats": _summary_from_samples(dt_ms_res.samples),
                    "seen": int(dt_ms_res.seen),
                },
                "send_to_event_cb_ms": {
                    "stats": _summary_from_samples((b["send_to_event_cb_ms"]).samples),
                    "seen": int((b["send_to_event_cb_ms"]).seen),
                },
                "event_cb_to_arrival_ms": {
                    "stats": _summary_from_samples((b["event_cb_to_arrival_ms"]).samples),
                    "seen": int((b["event_cb_to_arrival_ms"]).seen),
                },
                "match_rate": float(int(b.get("n_matched", 0) or 0)) / float(int(b.get("n_frames", 0) or 0))
                if int(b.get("n_frames", 0) or 0) > 0
                else 0.0,
            }
        report["events"]["by_serial_event"][serial] = out_ev

    for serial, b in sorted(exposure_by_serial.items()):
        dur_ticks_res: _Reservoir = b["duration_ticks"]
        dur_ms_res: _Reservoir = b["duration_ms"]
        report["exposure"]["by_serial"][serial] = {
            "serial": str(serial),
            "n": int(b.get("n", 0) or 0),
            "n_missing": int(b.get("n_missing", 0) or 0),
            "n_invalid": int(b.get("n_invalid", 0) or 0),
            "duration_ticks": {
                "stats": _summary_from_samples(dur_ticks_res.samples),
                "seen": int(dur_ticks_res.seen),
            },
            "duration_ms": {
                "stats": _summary_from_samples(dur_ms_res.samples),
                "seen": int(dur_ms_res.seen),
            },
        }

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown_report(out_md=out_md, report=report)

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")

    # 额外提示：若 send 匹配率过低，直接在终端提醒。
    rate = report.get("counts", {}).get("send_match_rate")
    try:
        rate_f = float(rate)
    except Exception:
        rate_f = 0.0

    if rate_f < 0.8:
        print(
            "提示：soft_trigger_send 与 group 的匹配率偏低（<0.8）。"
            "若你期望统计 send->arrival，请确认：你确实在用 Software 触发，且保存了 soft_trigger_send 记录；"
            "并确认 group_seq 是否能作为稳定的触发序号使用。"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
