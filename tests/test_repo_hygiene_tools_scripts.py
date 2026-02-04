"""仓库约束：避免 tools/ 下脚本被误当作单元测试。

背景：tools/ 目录用于放置可执行脚本；单元测试统一放在 tests/ 下。
pytest 默认会收集 `test_*.py` 与 `*_test.py`，一旦 tools/ 下出现类似命名，
会导致“工具脚本 vs 单测”语义混淆（虽然本仓库通过 testpaths 限制了收集范围，
但仍建议从命名层面杜绝歧义）。
"""

from __future__ import annotations

import fnmatch
from pathlib import Path


def test_tools_scripts_do_not_look_like_pytest_tests() -> None:
    """确保 tools/ 下脚本不会匹配 pytest 默认发现模式。"""

    repo_root = Path(__file__).resolve().parents[1]
    tools_dir = repo_root / "tools"
    if not tools_dir.exists():
        return

    disallowed_patterns = ["test_*.py", "*_test.py"]

    bad: list[str] = []
    for p in tools_dir.rglob("*.py"):
        name = p.name
        if any(fnmatch.fnmatch(name, pat) for pat in disallowed_patterns):
            bad.append(str(p.relative_to(repo_root)).replace("\\", "/"))

    assert not bad, (
        "tools/ 下发现疑似单测命名的脚本（请重命名以避免歧义）：\n"
        + "\n".join(f"- {x}" for x in bad)
    )
