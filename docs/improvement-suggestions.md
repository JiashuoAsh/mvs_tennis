# 进一步建议（按优先级）

以下建议基于当前仓库代码与数据现状，目标是“可落地、可度量、不改变核心功能”。

每条包含：问题/风险、建议方案、预期收益、实现成本（S/M/L）、参考位置。

## P0（优先做）

### 1) 修复 sample_sequence 的“不可直接离线运行”问题

- 问题/风险：
  - `tools/generate_sample_sequence.py` 注释声称会生成 `metadata.jsonl`，但实际只写了图片。
  - `tennis3d.apps.offline_localize_from_captures` 默认 `--captures-dir data/captures/sample_sequence`，会因缺少 `metadata.jsonl` 直接失败。
- 建议方案：
  - 在 `tools/generate_sample_sequence.py` 同时生成一个最小 `metadata.jsonl`（frames 列表里写 cam_index、serial、file 等字段）。
  - 或调整离线入口默认参数为仓库现有数据（例如 `data/captures_master_slave/tennis_test`），并在 README 中明确说明。
- 预期收益：
  - 新用户无需相机即可一键跑通离线链路，减少排障成本。
- 实现成本：S
- 位置：`tools/generate_sample_sequence.py`、`src/tennis3d/apps/offline_localize_from_captures.py`、`src/tennis3d/pipeline/sources.py`

### 2) 明确并统一“标定相机名”的约定（serial vs cam0/cam1）

- 问题/风险：
  - 在线/离线(captures) pipeline 默认以相机 serial 作为 camera_name（`iter_*_image_groups` 的 key）。
  - 但仓库内 `data/calibration/sample_cams.yaml` 使用 cam0/cam1/cam2 命名；真实项目里也常见用“相机位号”命名。
  - 结果是：标定 key 不匹配时，相机会被静默跳过，导致没有 3D 输出或视角不足。
- 建议方案：
  - 增加“相机名映射”能力：例如在 config 里支持 `camera_aliases: {"DA...": "cam0"}`，或在标定文件中新增 `serial` 字段并在加载时建立映射。
  - 同时在 README/文档中把约定写死（并给出最小示例）。
- 预期收益：
  - 降低“有图有检测但没 3D”的隐性坑；适配更多真实标定数据格式。
- 实现成本：M
- 位置：`src/tennis3d/pipeline/sources.py`、`src/tennis3d/geometry/calibration.py`、`src/tennis3d/localization/localize.py`、`src/tennis3d/config.py`

### 3) 为 rknn detector 增加更清晰的依赖与错误提示

- 问题/风险：
  - `tennis3d_detectors.create_detector(name="rknn")` 会在运行时 import `tennis3d.offline_detect.detector.TennisDetector`。
  - 若缺少 RKNN 运行时依赖或平台不支持，错误可能不够聚焦，影响定位。
- 建议方案：
  - 在 `tennis3d.offline_detect.detector` 内对 RKNN 相关 import 做一次集中检查，并抛出包含“平台限制/安装方式/替代方案(fake/color)”的错误信息。
  - 在 `pyproject.toml` 增加可选依赖组（例如 `[project.optional-dependencies] rknn = [...]`）。
- 预期收益：
  - 更快定位环境问题；减少 Windows 用户误踩。
- 实现成本：S
- 位置：`src/tennis3d/detectors.py`、`src/tennis3d/offline_detect/detector.py`、`pyproject.toml`

## P1（建议做）

### 4) 把关键 CLI 的输出改为结构化日志（logging）

- 问题/风险：
  - 当前大量使用 print，长时间采集/现场排障不利于分级筛选与日志留档。
- 建议方案：
  - 引入 `logging`，提供 `--log-level/--log-file`；对关键指标（dropped_groups、lost_packet、fps、pending_groups）做定期汇报。
- 预期收益：
  - 更快定位瓶颈；日志可归档、可检索。
- 实现成本：M
- 位置：`src/mvs/apps/quad_capture.py`、`src/mvs/apps/analyze_capture_run.py`、`src/tennis3d/apps/*`

### 5) 增加一个“最小端到端 smoke test”（纯离线）

- 问题/风险：
  - 目前单测覆盖几何/标定/重排，但缺少“离线 pipeline 能跑”的集成级验证。
- 建议方案：
  - 基于 sample_sequence（修复后）或极小 synthetic 数据，新增测试：
    - 生成临时 captures_dir（含 metadata.jsonl + 3 张图片）
    - 跑 `iter_capture_image_groups` + `run_localization_pipeline(detector=fake)`
    - 断言输出至少 1 条、字段齐全
- 预期收益：
  - 防止未来改动导致 pipeline 悄悄失效。
- 实现成本：M
- 位置：`tests/`、`src/tennis3d/pipeline/*`

### 6) CI 建议：把“可在无硬件环境验证”的部分自动化

- 问题/风险：
  - 当前缺少 CI 门禁，文档/重构容易引入回归。
- 建议方案：
  - GitHub Actions：运行 `python -m unittest`。
  - TODO：后续引入 ruff/pyright 做风格与类型检查。
- 预期收益：
  - 低成本提升稳定性。
- 实现成本：S
- 位置：`.github/workflows/`（新增）

## P2（有时间再做）

### 7) 性能改进：降低 frame 拷贝与转换开销（需要谨慎）

- 问题/风险：
  - 多相机/大分辨率下，Python 侧数据拷贝与 `frame_to_bgr` 转换可能成为瓶颈。
- 建议方案：
  - 评估：抓流线程是否可以减少一次拷贝；是否可延迟转换或复用 buffer。
  - 需要配合实际硬件压测与内存安全检查。
- 预期收益：
  - 更高 FPS、更低延迟。
- 实现成本：L
- 位置：`src/mvs/grab.py`、`src/mvs/image.py`、`src/tennis3d/pipeline/sources.py`

### 8) 文档体系收敛与去重

- 问题/风险：
  - 根目录存在多份较长文档（`INDEX.md`、`QUICK_REFERENCE.md`、`docs/python-repository-overview.md`、`readme.md/README.md`），内容有部分重复与历史遗留差异。
- 建议方案：
  - 明确单一入口（README）与文档导航（docs/），其余文档改为链接与补充说明。
- 预期收益：
  - 降低维护成本；减少“看不同文档得出不同结论”。
- 实现成本：S
- 位置：根目录文档与 `docs/`

