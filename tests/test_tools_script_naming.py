"""单测：确保 tools/ 下工具脚本不会被误当成 pytest 用例。

背景：tools/ 目录的定位是“可执行工具脚本”，不属于单测。
为了避免未来有人在 tools/ 里新增 `test_*.py` 而导致语义混淆，
这里显式做一次命名约束检查。
"""

from __future__ import annotations

from pathlib import Path


def test_tools_dir_has_no_test_prefix_python_files() -> None:
    """tools/ 下禁止出现 test_*.py。

    说明：仓库已在 `pyproject.toml` 里固定 `testpaths=["tests"]`，
    正常运行 `pytest` 时不会误收集 tools/。
    但这个测试仍然有价值：它把“工具脚本不应叫 test_xxx.py”变成可执行约束。
    """

    repo_root = Path(__file__).resolve().parents[1]
    tools_dir = repo_root / "tools"
    assert tools_dir.exists(), "预期仓库根目录存在 tools/ 目录。"

    bad_files: list[str] = []
    for p in tools_dir.rglob("test_*.py"):
        # 跳过缓存目录（即使出现也不影响约束的表达清晰度）。
        if "__pycache__" in p.parts:
            continue
        bad_files.append(p.relative_to(repo_root).as_posix())

    assert (
        not bad_files
    ), f"tools/ 下发现以 test_ 开头的脚本（应改名以免与单测混淆）：{bad_files}"
