# -*- coding: utf-8 -*-

"""mvs 路径相关的小工具。

这里集中放一些“定位仓库根目录/数据目录”的 best-effort 逻辑，避免在多个模块里重复实现。

说明：
- 本项目采用 src-layout（包位于 `src/`）。
- 运行时工作目录可能不是仓库根目录，因此不能仅依赖 `Path.cwd()`。
"""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """尽力定位仓库根目录。

    判定策略：
    1) 向上遍历父目录，找到包含 `pyproject.toml` 的目录；
    2) 找不到则使用固定层级兜底（src/mvs/paths.py -> 仓库根目录通常在上两级）。

    Returns:
        仓库根目录路径。
    """

    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p

    # 兜底：src/mvs/paths.py -> 仓库根目录通常在上两级。
    return here.parents[2]
