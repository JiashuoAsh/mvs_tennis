---
name: 030_layout_cleanup
description: "破坏式结构治理：确定目标布局 -> 大搬家 -> 全库修引用 -> 清零旧路径（不保留兼容）"
argument-hint: "mode=plan|apply; profile=auto|lib|app|pipeline|research|mixed; scope=可选"
---

执行模式：${input:mode:plan 或 apply（plan只输出方案不改代码）}
布局档案：${input:profile:auto=自动判定；或 lib/app/pipeline/research/mixed}
范围（可选）：${input:scope:留空=全仓库；也可填某目录如 libs/ 或 step*}

目标：只做结构治理（搬家/重命名/修 imports/修路径引用/更新 README），允许破坏式变更：
- 不需要兼容旧 import / 旧脚本路径 / 旧 API
- 可以删除旧文件与旧目录
- 禁止保留 wrapper/兼容层/转发文件

硬约束：
1) 禁止臆测：所有结论基于仓库证据（README/pyproject/入口脚本/测试）。
2) 不改业务逻辑（除非为适配新结构的必要改动，例如 import 路径、模块拆分接口）。
3) 每个批次结束必须能验证：import 冒烟 + 入口 --help（如果存在）+ 有测试则跑最小测试。
4) 不新增第三方依赖；避免过度设计。

流程：
A) 判定仓库类型（只读）：根据证据选择 profile（auto 时必须给出理由）
B) 输出迁移映射表（必须可执行）：旧路径 -> 新路径（按目录分组）
C) 输出“全库修引用清单”：需要替换的 import 路径/模块名/配置路径（用 rg/搜索的关键字列出来）
D)（mode=apply）执行搬迁与全库修引用：搬家后立刻修 imports/路径引用/README
E) 清零检查（必须做）：
   - 用全库搜索确认旧路径/旧模块名引用为 0
   - 删除不再使用的旧文件/旧目录
F) 验证：输出 smoke checklist（命令+预期），并确保通过；无法验证则写清缺口
交付：docs/layout_migration.md（迁移映射表 + 清零清单 + smoke 清单）
