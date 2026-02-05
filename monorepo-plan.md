## 短期：把当前仓库重构成 monorepo（最小可行拆分）

短期目标不是“拆得越细越好”，而是：

1. **先把最容易变泥球的东西拆出去**：`online`（应用编排）与 `detectors`（重依赖、后端多样）。
2. **保留一个稳定的核心库**：几何/标定/融合/pipeline（不含具体推理后端）。
3. **保证全量测试仍然能跑通**（你现在 `pytest` 是绿的，这是最宝贵的资产）。

### 当前已落地（截至 2026-02-05）

说明：已完成 **uv workspace + `packages/` 物理拆包**。

- 根 `pyproject.toml` 为 workspace root（virtual project，用于聚合安装/测试，不产出可安装包）
- 各发行包（distribution）位于 `packages/*/`，并采用各自的 src-layout（`packages/<pkg>/src/...`）

- detectors：已拆到独立包 `packages/tennis3d_detectors/src/tennis3d_detectors/`
  - 统一工厂：`tennis3d_detectors.create_detector()`
  - 跨包契约：`Detector` 协议已下沉到 `tennis3d.models.Detector`
  - 旧入口 `tennis3d.detectors` 已硬失败（强制迁移）
- online：已拆到独立包 `packages/tennis3d_online/src/tennis3d_online/`
  - 运行命令：`python -m tennis3d_online --help`
  - console script：`tennis3d-online`（入口点已改为 `tennis3d_online.entry:main`）
  - 旧入口 `tennis3d.apps.online` 已硬失败（强制迁移）
- 物理拆包：已将 `mvs/curve/tennis3d（core）/tennis3d_detectors/tennis3d_online` 分别迁入对应 member。
- 验证：已在 Windows 下完成全量测试验收：
  - 环境：`uv sync --group dev`
  - 测试：`uv run python -m pytest`
  - 结果：`74 passed`

### 短期推荐的包划分（3～4 个就够）

我建议你短期先落地这 4 个发行包（distribution）：

1) **`mvs`（库包）**
- 内容：相机 SDK 绑定、采集、组包、落盘、captures 元数据
- 目标：尽量轻依赖、稳定 API

2) **`tennis3d_core`（库包）**
- 内容：标定 IO、几何、三角化、融合、通用 pipeline core、数据结构
- 目标：不包含推理后端（torch/rknn 不在这里）

> 注：当前代码层面 core 仍沿用导入名 `tennis3d`（未改名），短期先不做“改包名”这种高破坏操作。

3) **`tennis3d_detectors`（库包）**
- 内容：fake/color/pt/rknn detector 适配，负责把图像变成标准化 detections
- 目标：承接重依赖（torch/ultralytics 等），并把“推理世界”与“几何世界”隔离开

4) **`tennis3d_online`（应用包）**
- 内容：在线运行入口、配置装配、输出协议、运行时 wiring（mvs source + detector + core）
- 目标：把 apps 与库分离；CLI/console scripts 主要放这里

> 你现在 curve 可以短期保持不动：要么暂时仍属于 `tennis3d_core` 的内部模块（如果强耦合），要么等 `curve_v3` 出来时再正式独立成包（更稳）。

### 短期目录结构建议（落地形态）

把仓库变成“单仓多包”时，最常见也最易维护的结构是引入 `packages/`：

- 根目录：保留仓库资产（docs/configs/data/tools/tests），以及一个“聚合层”的开发/测试入口
- 每个包：有自己独立的 pyproject.toml + src + （可选）包内单测

示意结构（短期版）：

- `packages/`
  - `mvs/`
    - pyproject.toml
    - `src/mvs/...`
  - `tennis3d_core/`
    - pyproject.toml
    - `src/tennis3d_core/...`（或你保留 `tennis3d` 导入名也行，但短期我建议新导入名更“硬隔离”）
  - `tennis3d_detectors/`
    - pyproject.toml
    - `src/tennis3d_detectors/...`
  - `tennis3d_online/`
    - pyproject.toml
    - `src/tennis3d_online/...`
- tests（建议短期先保留：作为“集成测试层”，跑全链路更方便）
- configs、tools、docs、data 保持根目录不动

### 短期迁移步骤（推荐按顺序做，避免一口气大爆炸）

1) **先“逻辑拆分”再“物理搬家”**
   在不改打包结构的前提下，先把 `tennis3d` 内部模块按 core/detectors/online 做目录切分与 import 依赖梳理，确保依赖方向明确：
   - core 不 import detectors
   - core 不 import online
   - detectors 可以依赖 core 的数据结构/协议
   - online 依赖 mvs + core + detectors

2) **确定“跨包契约”**（这是拆包成败关键）
   比如：Detection 的数据结构、图像输入格式、pipeline 的输入输出记录结构、标定的加载 API。
   短期最靠谱做法：把这些契约放在 `tennis3d_core` 里，让 detectors/online 都依赖它。

3) **再做“物理拆包”**：把代码移动到 `packages/<pkg>/src/...`，各自建立 pyproject.toml
   同步把 console scripts 从根 pyproject.toml 迁到 `tennis3d_online`（或 `tennis3d_apps`）里。

4) **最后收口**：
   - 根目录保留一个“开发聚合环境”（用于一键装全包、跑全测试）
   - `pytest` 在根目录跑全量（保证 monorepo 的整体稳定性）

---

## 长期：最优 monorepo 结构（更符合工程实践）

长期你提出的目标是：新增 `curve_v3`、新增 `interception`、拆分 tennis3d（检测 vs 轨迹拟合）、online 独立——这意味着你最终会走向“分层清晰、依赖单向”的结构。

我建议长期以 **“领域库（算法） / 适配层（推理、IO、硬件） / 应用层（在线服务/CLI）”** 三层来组织。

### 长期推荐的“目标包”清单（示例）

**基础设施层：**
- `mvs`：采集与组包（硬件/SDK）
- （可选）`capture_io`：captures 读写/重排的纯软件能力（如果你希望非 MVS 输入也复用）

**领域核心层：**
- `tennis3d_geometry`：标定、投影、三角化、误差评估
- `curve_v3`：轨迹拟合（只做数学与模型，不关心检测来源）
- `interception`：落点/拦截点预测（依赖 `curve_v3`，可选依赖 `tennis3d_geometry` 的坐标约定/数据结构）

**推理/适配层：**
- `tennis_detection`（或 `tennis_detectors`）：各种 detector 后端（torch/rknn/颜色阈值/fake）
  - 只输出标准 detection 协议，不把几何与推理混在一起

**应用层：**
- `tennis3d_online`：在线 app（接 mvs source + detectors + geometry/fit/interception）
- `tennis3d_offline`：离线 app（读 captures + detectors + geometry/fit/interception）
- （可选）`tennis3d_tools`：可执行工具集合（如果 tools 的规模越来越大，且希望作为包发布）

> 你也可以把 `tennis3d_geometry + triangulation + localization` 合并成一个 `tennis3d_core`，长期再细分。工程上完全 OK；“最优”不等于“最细”。

### 长期结构里最重要的工程实践点

1) **依赖方向必须单向**
   领域核心层（geometry/curve/interception）不应依赖 detector 或 online。
   detector 层不应反向依赖 online。
   app 层可以依赖一切，但它只做“编排”，不沉淀业务核心算法。

2) **重依赖隔离**
   `torch/ultralytics` 等只应存在于 detector 包或 app 包，不要渗透到几何/拟合核心里。
   这样你才能在“只做几何/拟合”的 CI 环境里保持轻快、稳定。

3) **版本策略提前定**（两种都合理，选一种坚持）
   - **统一版本（recommended for internal monorepo）**：所有包同一个版本号，发布/回滚简单
   - **独立版本（recommended for public libraries）**：每个包独立发版，灵活但成本高
   如果你主要是内部使用，我倾向统一版本，减少维护负担。

4) **测试分层**
   - 每个包有自己的单测（快、聚焦）
   - 根目录保留集成测试（慢一点，但能覆盖跨包 wiring）
   你现在的 tests 很适合承担“集成层”的角色，建议保留并逐步补齐关键跨包场景。

---

## 给你的 Markdown 文档（可直接保存到 `docs/monorepo-plan.md`）

下面这段就是你要的“总结成一个 markdown 文档”的版本（你复制即可；如果你希望我直接在仓库里创建该文件，我也可以在下一步帮你落盘并配套实施迁移）。

---

# MVS_Deployment Monorepo 重构方案（短期可落地 + 长期最优形态）

## 背景与动机

当前仓库为单发行包（`mvs-deployment`），但已包含多个顶层导入包（`mvs/`、`tennis3d/`、`curve/`）。随着功能增长出现以下问题：

- `tennis3d` 边界逐渐模糊（apps、detectors、几何、工具链混杂）
- 未来会新增 `curve_v3` 与 `interception` 等新子域模块
- 希望将检测与轨迹拟合从 `tennis3d` 中独立出来
- `online` 作为应用层被糅进库包中，影响复用与依赖隔离

目标：将仓库升级为 Python monorepo（单仓多包），通过“安装边界 + 依赖方向”强制模块边界清晰，并支持未来扩展。

---

## 设计原则

1. 先保证正确性与可验证性（全量 `pytest` 绿）再逐步拆分
2. 依赖单向：核心算法层不依赖推理与应用层
3. 重依赖隔离：`torch/ultralytics` 等仅存在于 detector/app 包
4. 包数量控制：短期先 3～4 个包，避免过度拆分导致循环依赖

---

## 短期（最小可行拆分）目标结构

### 短期包划分

- `mvs`（库包）
  - 采集、组包、SDK 绑定、保存、captures 元数据
- `tennis3d_core`（库包）
  - 标定 IO、几何、三角化、融合、通用 pipeline core、数据结构（跨包契约）
- `tennis3d_detectors`（库包）
  - fake/color/pt/rknn 等 detector 适配与推理后端（承接重依赖）
- `tennis3d_online`（应用包）
  - 在线运行入口、配置装配、输出协议、运行时 wiring（依赖 mvs + core + detectors）

### 目录结构（建议）

- `packages/`
  - `mvs/`
    - pyproject.toml
    - `src/mvs/...`
  - `tennis3d_core/`
    - pyproject.toml
    - `src/tennis3d_core/...`
  - `tennis3d_detectors/`
    - pyproject.toml
    - `src/tennis3d_detectors/...`
  - `tennis3d_online/`
    - pyproject.toml
    - `src/tennis3d_online/...`
- tests（根目录集成测试层，短期建议保留）
- configs、tools、docs、data（仓库资产，保留根目录）

---

## 短期迁移步骤（建议顺序）

1. 逻辑拆分：先在现有代码中梳理依赖方向
   - core 不 import detectors
   - core 不 import online
   - online 仅做编排，不沉淀算法
2. 固化跨包契约：Detection 数据结构、pipeline 输入输出、标定加载 API 等放入 `tennis3d_core`
3. 物理拆包：移动代码到 `packages/<pkg>/src/...`，为每个包创建独立 pyproject.toml
4. 迁移入口点：把在线 CLI/console scripts 放入 `tennis3d_online`
5. 验收：根目录运行全量 `pytest`（保持现有测试约束与稳定性）

---

## 长期（最优工程化）演进结构

### 长期目标包（示例）

- 基础设施层
  - `mvs`
  - （可选）`capture_io`（captures 读写、重排等纯软件能力）
- 领域核心层
  - `tennis3d_geometry`（标定、投影、三角化、误差评估）
  - `curve_v3`（轨迹拟合）
  - `interception`（依赖 curve_v3，做落点/拦截点预测）
- 推理/适配层
  - `tennis_detection`（多后端 detector 适配，承接重依赖）
- 应用层
  - `tennis3d_online`
  - `tennis3d_offline`
  - （可选）`tennis3d_tools`

### 依赖方向（必须保持单向）

- 核心算法层（geometry/curve/interception）不依赖 detectors/online
- detectors 依赖核心算法层的协议/数据结构（可选）
- online/offline app 依赖所有下层并负责 wiring

---

## 版本与测试策略建议

- 版本策略（选一种并长期坚持）
  - 内部 monorepo 推荐：统一版本号（发布与回滚简单）
  - 对外发布库推荐：独立版本号（灵活但成本更高）
- 测试分层
  - 包内单测：快速验证单一职责
  - 根目录集成测试：验证跨包 wiring 与关键路径

---

## 风险与注意事项

- 拆包是破坏式变更：import 路径与 console scripts 会变，需要全仓库同步修改
- 避免过度拆分：短期先拆 detectors 与 online，控制包数量
- 严禁循环依赖：一旦出现循环依赖，说明包边界切错或契约未收敛

---

## 下一步行动（建议）

1. 明确短期目标：是否接受导入名变更（例如 `tennis3d_core` vs 继续使用 `tennis3d`）
2. 明确要优先独立的内容：online、detectors、curve_v3、interception 的优先级排序
3. 开始第一阶段迁移：在线 app 独立 + detectors 独立，并保持全量测试通过

---

## Todo 列表（delta 更新）

- [x] 输出短期/长期 monorepo 结构与迁移步骤的 Markdown 方案
- [x] 第一阶段落地：拆分 `tennis3d_online` + `tennis3d_detectors`，并保持全量 `pytest` 通过
- [x] 第二阶段落地：做“物理 monorepo”目录结构（`packages/` + 多 `pyproject.toml` + uv 工作区/安装方式），并再次全量 `pytest` 验收（`uv run python -m pytest` -> 74 passed）