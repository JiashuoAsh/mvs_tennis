# -*- coding: utf-8 -*-

"""生成离线可跑通的 sample_sequence（图片 + metadata.jsonl）。

说明：
- `tools/` 仅保留可执行入口；核心生成逻辑已下沉到 `src/tennis3d/io/sample_sequence.py`。
- 默认输出目录：data/captures/sample_sequence/
- 配合：data/calibration/sample_cams.yaml + detector=color

运行后可用：
- python -m tennis3d.apps.offline_localize_from_captures
"""

from __future__ import annotations

from pathlib import Path

from tennis3d.io.sample_sequence import ensure_sample_sequence


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    seq_dir = ensure_sample_sequence(captures_dir=root / "data" / "captures" / "sample_sequence")
    # Use ASCII to avoid Windows console encoding issues.
    print(f"Generated sample_sequence -> {seq_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
