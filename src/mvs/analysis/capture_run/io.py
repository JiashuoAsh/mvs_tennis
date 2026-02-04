# -*- coding: utf-8 -*-

"""采集运行分析：I/O 辅助函数。

约定：
- 只负责读取 metadata.jsonl，不做统计口径。
- 读取失败要给出明确的行号/原因，便于用户定位损坏数据。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL 文件并返回记录列表。"""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    records: List[Dict[str, Any]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON decode failed at line {i}: {exc}") from exc

    if not records:
        raise ValueError(f"No records found in {path}")

    return records
