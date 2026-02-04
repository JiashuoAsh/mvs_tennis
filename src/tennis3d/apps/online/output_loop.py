"""在线模式：records 消费与输出循环。

职责：
- 消费 core pipeline 产出的 records（dict）。
- 计数（records/balls）、周期性状态输出（proc_fps/cap_fps/lag/loop_ms）。
- 可选写 JSONL。
- 可选把球结果打印到终端。

说明：
- 该模块属于 apps/entry 层（I/O 与人类可读输出），不应被 core 反向依赖。
"""

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
    """消费 records 并执行：计数、状态输出、JSONL 写盘、逐组打印。

    Returns:
        (records_done, balls_done)
    """

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

        if jsonl_writer is not None:
            if not (spec.out_jsonl_only_when_balls and (not isinstance(balls, list) or not balls)):
                jsonl_writer.write_line(json.dumps(out_rec, ensure_ascii=False))

        if str(spec.terminal_print_mode) == "none":
            continue
        if str(spec.terminal_print_mode) == "all":
            lines = _format_all_balls_lines(out_rec)
            if not lines:
                continue
            for ln in lines:
                print(ln)
        else:
            line = _format_best_ball_line(out_rec)
            if line is None:
                continue
            print(line)

    return records_done, balls_done
