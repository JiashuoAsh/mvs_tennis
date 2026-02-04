## 可直接使用的优质 Prompt（多球+误检抑制+3D世界坐标）

你是一个资深的计算机视觉/多相机几何工程师。请在现有代码库 MVS_Deployment 中，将“网球3D世界坐标定位”从**单球假设**升级为**多球鲁棒定位**，解决实机中“场地多球 + 模型误检”的可靠性问题。

### 1. 项目背景与当前流程（必须理解并对齐）
本仓库的端到端流程是：

- 采集与组包：`python -m mvs.apps.quad_capture` 输出 `<captures_dir>/metadata.jsonl` + 图像文件
- 离线从 captures 做 3D：入口 offline_localize_from_captures.py
  - 读取组：`tennis3d.pipeline.iter_capture_image_groups()`
  - 每组图像做检测：`tennis3d.detectors.create_detector()` 得到统一接口 `detect(img_bgr)->list[Detection]`
  - 几何定位主流程：`tennis3d.pipeline.run_localization_pipeline()`
  - 核心几何：`tennis3d.localization.localize_ball()`

当前关键限制（你必须在代码中精确定位并改掉）：
- `src/tennis3d/localization/localize.py::localize_ball()` 明确写了“每个相机只取 score 最大的一个检测框”，并通过 `_pick_best_detection()` 实现。
- `src/tennis3d/pipeline/core.py::run_localization_pipeline()` 也只产出**一个** `ball_3d_world` 记录（JSONL 每行最多一个球）。

这在实机完全不可靠：同一组图像里可能有多个球，且检测模型会误检。我们需要从“单输出”升级为“0..N 个球的输出”。

### 2. 改造目标（验收口径）
在 **每个同步组**（即 pipeline 的每次 `meta, images_by_camera`）里，系统应输出：
- 0..N 个球的 3D 世界坐标（N 可变）
- 每个 3D 球都要能解释为“来自至少 `require_views` 个相机的匹配检测”
- 要能有效抑制误检：只在跨视角几何一致时输出；单相机/不一致候选应被过滤
- 输出需要包含足够的可调试信息（每个球的 used_cameras、重投影误差、对应 bbox/center/score）
- 在多球情况下要避免“同一个相机的同一个检测框被多个 3D 球重复使用”（除非你给出明确的理由与实现）

### 3. 输出数据契约（必须落实到代码）
请将 pipeline 输出从单球字段：
- `ball_3d_world`, `ball_center_uv`, `detections`, `used_cameras`, `reprojection_errors`
升级为多球字段（建议结构，可在不降低可用性的前提下微调）：

- `balls`: list[dict]
  - 每个元素包含：
    - `ball_3d_world`: `[x,y,z]`
    - `ball_3d_camera`: `{camera_name: [x,y,z]}`
    - `used_cameras`: `[camera_name,...]`
    - `reprojection_errors`: 同现有结构（每相机 uv / uv_hat / error_px）
    - `detections`: `{camera_name: {bbox, score, cls, center}}`
    - `score` 或 `quality`：综合质量评分（例如视角数越多越好、重投影误差越小越好）
    - 可选：`ball_id`（组内排序编号即可）或 `track_id`（如果你实现跨帧跟踪）

并保证输出依然是“可 JSON 序列化 dict”，方便写 jsonl。

### 4. 具体任务（按优先级实现）
A) 多球几何候选生成（组内，跨相机）
- 输入：`detections_by_camera: Mapping[str, Sequence[Detection]]`（每相机多个候选）
- 输出：多个 3D 候选球（每个候选绑定各相机的一个检测）
- 你需要设计“跨视角匹配 + 三角化 + 几何一致性验证”的方法，至少满足：
  - 最少用两相机三角化（DLT 已有：`tennis3d.geometry.triangulation.triangulate_dlt`）
  - 对每个候选计算重投影误差（已有：`reprojection_errors`）
  - 通过阈值 gating 过滤（例如 `max_error_px`、`median_error_px`、正深度等）
  - 对其他相机做“投影一致性补全”：把 3D 点投影到该相机，在该相机的 detections 中找距离最近且在阈值内的 bbox center 作为匹配（避免组合爆炸）
  - 支持 `require_views`：视角数不足的候选不输出

B) 冲突消解与去重（组内多球）
- 如果两个 3D 候选使用了同一相机同一个 detection（同一 bbox/center），需要冲突处理：
  - 目标：选择一组互不冲突且整体质量最高的候选集合
  - 允许实现：
    - 贪心：按质量排序依次选择，冲突则跳过
    - 或更严谨：最大权重集合/匹配（如你要引入匈牙利算法请说明必要性；能不用新依赖就不用）
- 还需要 3D 级别去重：多个候选可能是同一个球的重复解（不同相机组合导致）。请做 3D-NMS（例如距离阈值 `merge_dist_m`）或按重投影误差合并。

C) 可配置参数（放到 config，并在 CLI 暴露或配置文件中可控）
在 config.py 与 `configs/*.yaml` 的离线/在线配置里加入（名称可调整，但必须具备能力）：
- `max_detections_per_camera`（每相机最多取 topK）
- `max_reproj_error_px`（候选有效阈值）
- `max_uv_match_dist_px`（投影到相机后匹配 bbox center 的阈值）
- `min_views`/`require_views`（已有，继续沿用）
- `merge_dist_m`（3D 去重/合并阈值）

D) 改造入口与文档
- 更新 `src/tennis3d/pipeline/core.py::run_localization_pipeline()`：输出多球结构
- 更新 offline_localize_from_captures.py、online/app.py：确保运行不报错，输出符合新契约
- 更新文档 architecture-and-dataflow.md 中 3D 输出格式小节（简短但准确）

E) 测试（必须写，且不能弱化现有测试）
- 运行并确保现有 tests 全通过
- 新增单元测试覆盖：
  - 单组内多球：构造两套跨相机一致的检测，必须输出两个 3D
  - 误检抑制：只有单相机/或跨相机不一致的假检测，不应输出
  - 冲突消解：同一相机同一 detection 不能被两个球同时使用（除非你给出策略并测试体现）

### 5. 工程约束（必须遵守）
- 不要写“兼容旧输出”的过渡层或 shim；请干净替换为新设计，让代码库保持简洁一致。
- 代码注释必须是中文，解释非显然的几何/匹配逻辑和阈值含义。
- 如需新增依赖，请使用 `uv` 并更新 pyproject.toml（不要直接用 pip 固化依赖）。

### 6. 交付物清单（你输出时必须包含）
1) 你修改/新增了哪些文件（路径列表 + 每个文件一句话目的）
2) 新的输出 JSONL 示例（展示同一组里 `balls` 有多个元素）
3) 新增/修改的配置项说明（默认值与含义）
4) 测试结果：完整测试套件通过（请给出你运行的方式与结果摘要）

开始执行：先定位 `localize_ball()` 的单球假设，设计并实现 `localize_balls()`（或等价结构），再改造 `run_localization_pipeline()` 输出多球，最后补齐配置、文档与测试。
