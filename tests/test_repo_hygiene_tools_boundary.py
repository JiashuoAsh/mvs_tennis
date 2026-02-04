"""单测：约束 tools/ 与 tests/ 的边界（结构层面）。

说明：
- 本仓库已在 `pyproject.toml` 中通过 `testpaths=["tests"]` 固定 pytest 的收集范围。
- 同时仓库也有多条“命名层面”的约束测试，用于禁止 `tools/test_*.py` 等命名。

本文件聚焦“结构层面”的边界：
- tools/ 仅用于放置可执行脚本，不应被当作可 import 的 Python 包（避免出现 __init__.py）。
- tools/ 不应承载 pytest 的配置/插件入口（避免出现 conftest.py）。
"""

from __future__ import annotations

from pathlib import Path


def test_tools_dir_is_not_a_python_package_or_pytest_plugin_root() -> None:
    """确保 tools/ 不会被误用为“包目录”或 pytest 插件目录。"""

    repo_root = Path(__file__).resolve().parents[1]
    tools_dir = repo_root / "tools"
    if not tools_dir.exists():
        # 说明：如果未来仓库移除了 tools/ 目录，这条约束自然失效。
        return

    bad: list[str] = []

    # 说明：tools/ 下出现 __init__.py 会让它变成可 import 的包目录，
    # 容易引导调用方写出 `import tools.xxx` 这种边界不清的用法。
    for p in tools_dir.rglob("__init__.py"):
        if "__pycache__" in p.parts:
            continue
        bad.append(p.relative_to(repo_root).as_posix())

    # 说明：conftest.py 会被 pytest 当成测试配置/插件入口文件；
    # tools/ 不应承担该角色。
    for p in tools_dir.rglob("conftest.py"):
        if "__pycache__" in p.parts:
            continue
        bad.append(p.relative_to(repo_root).as_posix())

    assert bad == [], (
        "tools/ 下不应出现包/pytest 插件入口文件（请移动到 src/ 或 tests/）：\n"
        + "\n".join(f"- {x}" for x in bad)
    )