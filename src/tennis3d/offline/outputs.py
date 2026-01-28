"""输出写入（高内聚：只管落盘 JSON/CSV）。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)

    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def maybe_write_csv(path: Optional[Path], rows: list[dict[str, Any]]) -> None:
    if path is None:
        return
    write_csv(path, rows)
