"""pytest 运行期配置。

该仓库采用 src-layout（包代码在 ./src 下）。
为了让开发者直接在仓库根目录执行 `python -m pytest` 时也能导入
`mvs` / `tennis3d` 等包，这里在测试收集阶段把 ./src 注入到 sys.path。

注意：这只是测试侧的便捷配置，不影响正式打包安装后的导入行为。
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_syspath() -> None:
    """将仓库的 ./src 目录加入 sys.path（若尚未存在）。"""

    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"

    # 使用绝对路径，避免因 cwd 不同导致的导入差异。
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)


_ensure_src_on_syspath()
