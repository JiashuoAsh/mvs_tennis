# 迁移映射（旧 → 新）

本仓库已统一为 **src-layout**：

- 采集 SDK 封装：`src/mvs/`
- 网球检测 + 3D 定位业务库：`src/tennis3d/`
- 数据与输出：`data/`

> 说明：为避免出现“同一仓库两套入口/两套 import 路径”的混乱，旧的 `process/` 已删除（不再提供任何兼容转发）。

---

## 入口（Entry）迁移

| 旧入口/用途 | 新入口/用途 |
|---|---|
| （旧 process 入口） | `python -m tennis3d.apps.online_mvs_localize`（在线） |
| （旧 process 入口） | `python -m tennis3d.apps.offline_localize_from_captures`（离线） |
| 离线三相机检测/对齐（实验用） | `python -m tennis3d.offline.cli`（保留为调试工具） |

---

## 检测（RKNN）迁移

| 旧文件/概念 | 新文件/概念 |
|---|---|
| `process/ref_camera_processor.py`（RKNN 调用参考） | `examples/reference_rknn/ref_camera_processor.py`（仅保留为参考） |
| `process/ref_Inference_seg.py`（RKNN 推理参考） | `examples/reference_rknn/ref_Inference_seg.py`（仅保留为参考） |
| （自定义检测器） | `src/tennis3d/detectors.py`：`Detector` 协议 + `create_detector()` 工厂 |
| RKNN 检测实现 | `src/tennis3d/offline/detector.py`：`TennisDetector` |
| 离线无 RKNN 跑通 | `--detector color`（HSV 颜色阈值）或 `--detector fake` |

---

## 多相机 2D→3D（几何）迁移

| 旧文件/概念 | 新文件/概念 |
|---|---|
| （三角化/几何散落在脚本中） | `src/tennis3d/geometry/triangulation.py`：DLT 三角化 + 重投影误差 |
| （标定读取） | `src/tennis3d/geometry/calibration.py`：支持 JSON/YAML 标定加载 |
| （从 detections 做融合） | `src/tennis3d/localization/localize.py`：`localize_ball()` |

---

## Pipeline 拆分

| 旧做法 | 新做法 |
|---|---|
| 脚本串起来跑（难替换/难测试） | `src/tennis3d/pipeline/`：可复用流水线（source→align→detect→triangulate→output） |
| 在线取流与业务逻辑耦合 | `src/mvs/pipeline.py` 提供组包；`src/tennis3d/apps/online_mvs_localize.py` 只做 CLI 外壳 |
| 离线读取与业务逻辑耦合 | `src/tennis3d/pipeline/sources.py` 读取 `metadata.jsonl` 并产出统一 group |

---

## 数据目录迁移

| 旧目录 | 新目录 |
|---|---|
| `captures/` | `data/captures/legacy_root_captures/`（已迁入） |
| `tools_output/` | `data/tools_output/legacy_root_tools_output/`（已迁入） |
| `calibration/` | `data/calibration/`（统一存放标定） |

额外提供：

- 离线样例数据：`data/captures/sample_sequence/`（可直接跑通）
- 离线样例标定：`data/calibration/sample_cams.yaml`
