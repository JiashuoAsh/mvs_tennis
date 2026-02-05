# tools/ 工具脚本说明

本目录用于存放**可直接运行的工具脚本**（一次性分析、调试、数据迁移、诊断报告等）。

## 与单元测试的边界

- 单元测试统一放在 `tests/` 下。
- 本仓库的 pytest 配置在 `pyproject.toml` 中将测试发现范围固定为 `testpaths = ["tests"]`，避免将工具脚本误当作单测收集。

## 命名约定（重要）

为避免与单测语义混淆：

- 不要在 `tools/` 下创建 `test_*.py` 或 `*_test.py` 文件。
- 该约束由单测 `tests/test_repo_hygiene_tools_scripts.py` 强制检查，避免未来回归。
- 推荐使用明确的工具前缀或用途命名，例如 `mvs_*.py`、`time_mapping_*.py`、`generate_*.py`、`debug_*.py`。

## 输出位置约定（建议）

如果脚本需要写文件：

- 可复用/可对比的产物：优先写入 `data/tools_output/`
- 临时调试产物：优先写入 `temp/`

这样可以避免把一次性输出散落在仓库根目录，也便于后续清理与对比。

同时建议将脚本保持为“可执行脚本”而非“可 import 的库模块”：

- 不要在业务代码里 import `tools/` 下脚本来复用逻辑；如确实需要复用，请将逻辑下沉到可复用包中（本仓库当前为 uv workspace，多包源码位于 `packages/*/src/`）。

## 运行方式

多数脚本在文件末尾包含 `if __name__ == "__main__":`，可直接运行：

- `python tools/<script>.py`

说明：多数工具脚本会 import workspace 内的包（例如 `mvs`、`tennis3d`、`tennis3d_online`）。
为了避免误用系统 Python 导致导入失败，建议先按 `docs/development-and-testing.md`
完成虚拟环境与依赖同步，然后用 `uv run ...` 执行工具脚本，例如：

- `uv run python tools/<script>.py`

若脚本实现了参数解析，通常也支持 `--help` 查看用法。
