---
name: 040_defrag_modules
description: "反碎片化：识别可合并的小模块，合并成更少、更内聚的文件（保持行为不变）"
argument-hint: "scope=目录; threshold=行数阈值"
agent: "agent"
---

范围：${input:scope:例如 libs/apriltag_perf 或 src/pkg/core}
小文件阈值（行数）：${input:threshold:默认 150}

目标：减少过度碎片化，按“共同变化原因”合并模块，保持行为不变。

提示：如果当前碎片化是为了清晰边界/隔离依赖（例如避免循环依赖、降低 import 成本），允许不合并或只合并一部分。

流程：
1) 盘点 scope 内所有 .py 文件，找出 <threshold 行的文件列表（路径+行数+主要职责/符号）。
2) 基于“共同变化/依赖关系/调用链深度”提出合并分组建议，并解释理由。
3) 执行合并：合并后修复 imports，提供稳定 public API（避免深层 import）。
4) 自检：至少给 import 冒烟 + 关键入口 --help（若存在）。

约束：默认不新增依赖；当收益明显时允许（必须说明必要性/替代方案/影响面，并同步更新依赖声明如 pyproject.toml/锁文件与最小验证）；不改业务逻辑；中文注释/log/docstring；引用路径与符号名。
