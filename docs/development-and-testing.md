# 开发与测试

本文档描述如何在不依赖相机硬件的情况下验证核心逻辑（几何/标定/重排），以及建议的工程化约束。

## 开发环境

- Python：>= 3.10（见 `pyproject.toml`）
- 推荐使用 uv 管理虚拟环境：

```bash
uv venv
uv sync
```

说明：本仓库采用 uv workspace（单仓多包）。根目录是依赖聚合与测试入口（virtual project），无需（也不应该）再对根目录执行 `pip install -e .`。

开发/测试依赖（例如 pytest）位于依赖组 `dev`，建议额外同步：

```bash
uv sync --group dev
```

## 构建/安装产物（egg-info）

当你执行 `pip install -e <某个包目录>`（或某些工具触发 setuptools editable 安装）时，setuptools 可能会在工作区生成 `*.egg-info/` 目录（例如旧结构遗留的 `src/mvs_deployment.egg-info/`），用于存放安装元数据（入口点、依赖列表等）。

- 这类目录属于构建/安装产物，不应作为源码的一部分长期维护。
- 本仓库的 `.gitignore` 已忽略 `*.egg-info/`，避免将其误提交入库。
- 如果你希望保持工作区整洁，可以直接删除；下次安装时会自动重新生成。

PowerShell 示例：

```powershell
Remove-Item -Recurse -Force mvs_deployment.egg-info, src/mvs_deployment.egg-info
```

验证标准：

- 不影响导入与运行。
- `pytest` 仍能全量通过。

## 运行测试

本仓库的单元测试位于 `tests/`，**同时包含**：

- `unittest` 风格（`unittest.TestCase`）
- `pytest` 风格（使用 `pytest` 的断言与夹具）

为了避免漏跑用例（例如 `tests/test_detector_pt.py`），推荐统一用 `pytest` 运行全部测试。

运行所有测试（推荐）：

```bash
uv run python -m pytest
```

你也可以只跑某个测试文件：

```bash
uv run python -m pytest -q tests/test_tennis_geometry.py
uv run python -m pytest -q tests/test_calibration_yaml.py
uv run python -m pytest -q tests/test_capture_relayout.py
```

如果你更习惯 `unittest`，也可以运行（但注意它不会执行 pytest 风格用例）：

```bash
python -m unittest
```

测试覆盖范围：
- `tests/test_tennis_geometry.py`：三角化与重投影误差
- `tests/test_calibration_yaml.py`：YAML 标定加载
- `tests/test_capture_relayout.py`：captures 按相机重排（并验证跳过事件记录）

注意：采集相关（MVS SDK、相机硬件）不适合在 CI 环境跑，建议保持为“手工/现场验证项”。

## tools/ 工具脚本与单测边界

- 工具脚本统一放在 `tools/`（见 `tools/README.md`），用于一次性分析、调试与诊断。
- 单元测试统一放在 `tests/`。
- pytest 已在 `pyproject.toml` 中设置 `testpaths = ["tests"]`，避免把工具脚本误收集成单测。
- 命名约定：不要在 `tools/` 下创建 `test_*.py` 或 `*_test.py`，避免与单测语义混淆。
- 该约束由单测 `tests/test_repo_hygiene_tools_scripts.py` 强制检查，避免未来回归。

## 日志与可观测性建议

当前代码中关键 CLI 已尽量统一 UTF-8 输出（避免 Windows 重定向乱码），但整体仍以 print 为主。

建议的工作方式：
- 采集/分析时，把 stdout/stderr 重定向到文件做留档
- 所有质量判定以 `metadata.jsonl` 与 `python -m mvs.apps.analyze_capture_run` 的报告为准

TODO（建议落地）：统一使用 `logging`，并支持 `--log-level/--log-file`。

## 代码风格（建议）

仓库当前以“可读、可调试、不过度设计”为主（见工程约束文件）。
建议保持：
- 模块职责明确（apps 只做 CLI，pipeline/core 保持框架无关）
- 关键数据结构使用 dataclass
- 入口脚本参数尽量向下传递，避免重复定义

## CI 建议（不改变核心功能的前提下可增量引入）

推荐引入 GitHub Actions（或同类 CI）做纯软件侧的质量门禁：

- 单元测试：`uv run python -m pytest -q`
- 代码风格/静态检查：
  - TODO：引入 ruff/pyright，并在 `pyproject.toml` 中配置

硬件相关项（相机采集）建议作为：
- 文档化的手工验收步骤
- 或在具备相机与 SDK 的专用 runner 上跑（成本较高）

