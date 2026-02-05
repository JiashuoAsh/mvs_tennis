"""在线模式的 JSONL 写入工具。"""

from __future__ import annotations

import time


class _JsonlBufferedWriter:
    """JSONL 写入器：支持按条数/按时间间隔 flush。"""

    def __init__(
        self,
        *,
        f,
        flush_every_records: int,
        flush_interval_s: float,
    ) -> None:
        self._f = f
        self._flush_every_records = int(flush_every_records)
        self._flush_interval_s = float(flush_interval_s)
        self._records_since_flush = 0
        self._last_flush_t = time.monotonic()

    def write_line(self, line: str) -> None:
        self._f.write(line)
        self._f.write("\n")
        self._records_since_flush += 1

        need_flush_by_count = (
            self._flush_every_records > 0
            and self._records_since_flush >= self._flush_every_records
        )
        need_flush_by_time = False
        if self._flush_interval_s > 0:
            now = time.monotonic()
            need_flush_by_time = (now - self._last_flush_t) >= self._flush_interval_s

        if need_flush_by_count or need_flush_by_time:
            self.flush()

    def flush(self) -> None:
        self._f.flush()
        self._records_since_flush = 0
        self._last_flush_t = time.monotonic()
