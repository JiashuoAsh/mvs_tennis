"""在线模式：records 消费与输出循环。"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from typing import Any

from .jsonl_writer import _JsonlBufferedWriter
from .spec import OnlineRunSpec
from .terminal_format import _format_all_balls_lines, _format_best_ball_line


def run_output_loop(
    *,
    records: Iterable[dict[str, Any]],
    jsonl_writer: _JsonlBufferedWriter | None,
    spec: OnlineRunSpec,
    get_groups_done: Callable[[], int],
) -> tuple[int, int]:
    """消费 records 并执行：计数、状态输出、JSONL 写盘、逐组打印。"""

    def _format_timing_line(
        *,
        out_rec: dict[str, Any],
        loop_last_ms: float | None,
        out_write_ms: float | None,
        out_print_ms: float | None,
    ) -> str:
        """格式化单条 timing 行。

        说明：
            - 该输出用于在线排障/性能观察，默认关闭（--terminal-timing）。
            - pipeline 的分解耗时来自 out_rec['latency_host']。
            - output_loop 自己统计 write/print 的耗时，帮助区分“算得慢”还是“输出慢”。
        """

        gi = out_rec.get("group_index")
        balls = out_rec.get("balls") or []
        balls_n = len(balls) if isinstance(balls, list) else 0

        lat = out_rec.get("latency_host")
        align_ms = detect_ms = localize_ms = total_ms = None
        det_by_cam: dict[str, float] | None = None
        if isinstance(lat, dict):
            try:
                align_ms = float(lat.get("align_ms")) if lat.get("align_ms") is not None else None
            except Exception:
                align_ms = None
            try:
                detect_ms = float(lat.get("detect_ms")) if lat.get("detect_ms") is not None else None
            except Exception:
                detect_ms = None
            try:
                localize_ms = (
                    float(lat.get("localize_ms")) if lat.get("localize_ms") is not None else None
                )
            except Exception:
                localize_ms = None
            try:
                total_ms = float(lat.get("total_ms")) if lat.get("total_ms") is not None else None
            except Exception:
                total_ms = None

            dbc = lat.get("detect_ms_by_camera")
            if isinstance(dbc, dict) and dbc:
                det_by_cam = {}
                for k, v in dbc.items():
                    try:
                        det_by_cam[str(k)] = float(v)
                    except Exception:
                        continue

        parts: list[str] = [f"timing: group={gi} balls={balls_n}"]
        if loop_last_ms is not None:
            parts.append(f"loop_last~{float(loop_last_ms):.1f}ms")

        # pipeline 内部分解耗时
        pipe_parts: list[str] = []
        if align_ms is not None:
            pipe_parts.append(f"align={align_ms:.1f}")
        if detect_ms is not None:
            pipe_parts.append(f"det={detect_ms:.1f}")
        if localize_ms is not None:
            pipe_parts.append(f"loc={localize_ms:.1f}")
        if total_ms is not None:
            pipe_parts.append(f"total={total_ms:.1f}")
        if pipe_parts:
            parts.append("pipe_ms{" + ",".join(pipe_parts) + "}")

        # 输出耗时（写盘/打印）
        out_parts: list[str] = []
        if out_write_ms is not None:
            out_parts.append(f"write={float(out_write_ms):.1f}")
        if out_print_ms is not None:
            out_parts.append(f"print={float(out_print_ms):.1f}")
        if out_parts:
            parts.append("out_ms{" + ",".join(out_parts) + "}")

        if det_by_cam:
            # 说明：按 key 排序，保持输出稳定，便于 grep/对比。
            inner = ",".join(f"{k}:{det_by_cam[k]:.1f}" for k in sorted(det_by_cam.keys()))
            parts.append("det_cam_ms{" + inner + "}")

        return " ".join(parts)

    records_done = 0
    balls_done = 0

    if str(spec.terminal_print_mode) != "none" or float(spec.terminal_status_interval_s) > 0:
        print("Waiting for first ball observation...")

    last_status_t = time.monotonic()
    last_status_records = 0
    last_status_groups = 0
    last_status_capture_host_ms: int | None = None
    last_iter_t: float | None = None
    last_iter_dt_ms: float | None = None

    for out_rec in records:
        iter_now = time.monotonic()
        if last_iter_t is not None:
            dt_s = iter_now - last_iter_t
            if dt_s >= 0:
                last_iter_dt_ms = dt_s * 1000.0
        last_iter_t = iter_now

        records_done += 1
        balls = out_rec.get("balls") or []
        if isinstance(balls, list):
            balls_done += int(len(balls))

        if float(spec.terminal_status_interval_s) > 0:
            now = time.monotonic()
            if (now - last_status_t) >= float(spec.terminal_status_interval_s):
                dt_s = max(now - last_status_t, 1e-9)
                rec_delta = records_done - last_status_records

                groups_done = int(get_groups_done())
                grp_delta = groups_done - last_status_groups

                cap_host_ms = out_rec.get("capture_host_timestamp")
                cap_fps = None
                if cap_host_ms is not None:
                    try:
                        cap_host_ms_i = int(cap_host_ms)
                        if last_status_capture_host_ms is not None:
                            dms = cap_host_ms_i - last_status_capture_host_ms
                            if dms > 0:
                                cap_fps = 1000.0 * float(grp_delta) / float(dms)
                        last_status_capture_host_ms = cap_host_ms_i
                    except Exception:
                        pass

                proc_fps = float(rec_delta) / dt_s

                loop_ms_avg = None
                if rec_delta > 0:
                    loop_ms_avg = 1000.0 * dt_s / float(rec_delta)

                loop_ms_part = ""
                if loop_ms_avg is not None:
                    loop_ms_part += f" loop_avg~{loop_ms_avg:.1f}ms"
                if last_iter_dt_ms is not None:
                    loop_ms_part += f" loop_last~{last_iter_dt_ms:.1f}ms"

                lag_ms = None
                try:
                    ca = out_rec.get("created_at")
                    ta = out_rec.get("capture_t_abs")
                    if ca is not None and ta is not None:
                        lag_ms = (float(ca) - float(ta)) * 1000.0
                except Exception:
                    lag_ms = None

                cap_part = f" cap_fps~{cap_fps:.2f}" if cap_fps is not None else ""
                lag_part = f" lag~{lag_ms:.0f}ms" if lag_ms is not None else ""

                print(
                    f"status: groups={groups_done} records={records_done} balls={balls_done} "
                    f"proc_fps~{proc_fps:.2f}{cap_part}{lag_part}{loop_ms_part}"
                )

                last_status_t = now
                last_status_records = records_done
                last_status_groups = groups_done

        # 说明：若开启 --terminal-timing，则额外统计 output_loop 自身耗时。
        # - write_ms：jsonl_writer.write_line 的耗时
        # - print_ms：终端格式化 + print 的耗时
        out_write_ms: float | None = None
        out_print_ms: float | None = None

        if jsonl_writer is not None:
            if not (spec.out_jsonl_only_when_balls and (not isinstance(balls, list) or not balls)):
                t0 = time.monotonic() if bool(getattr(spec, "terminal_timing", False)) else None
                jsonl_writer.write_line(json.dumps(out_rec, ensure_ascii=False))
                if t0 is not None:
                    out_write_ms = 1000.0 * max(0.0, time.monotonic() - t0)

        if str(spec.terminal_print_mode) != "none":
            t0 = time.monotonic() if bool(getattr(spec, "terminal_timing", False)) else None
            if str(spec.terminal_print_mode) == "all":
                lines = _format_all_balls_lines(out_rec)
                if lines:
                    for ln in lines:
                        print(ln)
            else:
                line = _format_best_ball_line(out_rec)
                if line is not None:
                    print(line)
            if t0 is not None:
                out_print_ms = 1000.0 * max(0.0, time.monotonic() - t0)

        if bool(getattr(spec, "terminal_timing", False)):
            print(
                _format_timing_line(
                    out_rec=out_rec,
                    loop_last_ms=last_iter_dt_ms,
                    out_write_ms=out_write_ms,
                    out_print_ms=out_print_ms,
                )
            )

    return records_done, balls_done
