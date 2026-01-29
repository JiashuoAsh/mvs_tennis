"""兼容入口：全量 tennis_test 可视化检测（已合并）。

说明：
- 原先这个脚本与 `tools/test_best_pt_ultralytics.py` 功能大量重复。
- 现在全量检测/画框功能已合并到 `tools/test_best_pt_ultralytics.py` 的 `--all` 模式。
- 这里保留一个“薄封装”，避免旧命令行入口失效。
"""

from __future__ import annotations

import sys

import test_best_pt_ultralytics


def main(argv: list[str] | None = None) -> int:
    """转发到合并后的脚本入口。"""

    if argv is None:
        argv = sys.argv[1:]

    # 说明：该脚本等价于：python tools/test_best_pt_ultralytics.py --all ...
    if "--all" not in argv:
        argv = ["--all", *argv]

    return int(test_best_pt_ultralytics.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
