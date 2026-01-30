# 开发与测试

本文档描述如何在不依赖相机硬件的情况下验证核心逻辑（几何/标定/重排），以及建议的工程化约束。

## 开发环境

- Python：>= 3.10（见 `pyproject.toml`）
- 推荐使用 uv 管理虚拟环境：

```bash
uv venv
uv sync
uv pip install -e .
```

## 运行测试

本仓库的单元测试位于 `tests/`，**同时包含**：

- `unittest` 风格（`unittest.TestCase`）
- `pytest` 风格（使用 `pytest` 的断言与夹具）

为了避免漏跑用例（例如 `tests/test_detector_pt.py`），推荐统一用 `pytest` 运行全部测试。

运行所有测试（推荐）：

```bash
pytest -q
```

你也可以只跑某个测试文件：

```bash
pytest -q tests/test_tennis_geometry.py
pytest -q tests/test_calibration_yaml.py
pytest -q tests/test_capture_relayout.py
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

- 单元测试：`pytest -q`
- 代码风格/静态检查：
  - TODO：引入 ruff/pyright，并在 `pyproject.toml` 中配置

硬件相关项（相机采集）建议作为：
- 文档化的手工验收步骤
- 或在具备相机与 SDK 的专用 runner 上跑（成本较高）

