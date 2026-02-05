"""在线模式：JSONL 输出资源管理。"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from .jsonl_writer import _JsonlBufferedWriter
from .spec import OnlineRunSpec


@contextmanager
def open_optional_jsonl_writer(spec: OnlineRunSpec) -> Iterator[_JsonlBufferedWriter | None]:
    """按 spec 打开 JSONL writer；未配置 out_path 时返回 None。"""

    f_out = None
    writer: _JsonlBufferedWriter | None = None

    if spec.out_path is not None:
        spec.out_path.parent.mkdir(parents=True, exist_ok=True)
        f_out = spec.out_path.open("w", encoding="utf-8")
        writer = _JsonlBufferedWriter(
            f=f_out,
            flush_every_records=int(spec.out_jsonl_flush_every_records),
            flush_interval_s=float(spec.out_jsonl_flush_interval_s),
        )

    try:
        yield writer
    finally:
        if f_out is not None:
            try:
                if writer is not None:
                    writer.flush()
            except Exception:
                pass
            try:
                f_out.close()
            except Exception:
                pass
