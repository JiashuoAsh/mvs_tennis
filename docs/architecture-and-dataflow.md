# 架构与数据流（mvs + tennis3d）

本文档聚焦“模块职责、调用关系、输入输出”，以便你快速定位入口并理解 online/offline 的数据流。

## 一句话架构

- `mvs` 负责采集与同步组包（硬件保证同曝光，软件负责正确配对与落盘）。
- `tennis3d` 负责检测与几何（检测 bbox → 取中心点 → 多视角三角化 → 评估重投影误差 → 输出 3D）。

## 入口与调用关系

### 采集与诊断（mvs）

- 采集主入口：`tools/mvs_quad_capture.py`
  - 调用：`mvs.load_mvs_binding()` → `mvs.pipeline.open_quad_capture()` → 循环 `QuadCapture.get_next_group()`
  - 输出：
    - 图片（可选）：`--save-mode sdk-bmp|raw|none`
    - 元数据：`<output_dir>/metadata.jsonl`

- 采集分析：`tools/mvs_analyze_capture_run.py`
  - 输入：`<output_dir>/metadata.jsonl`
  - 输出：报告文本（stdout）+ 可选 JSON 汇总 `--write-json`

- 按相机重排：`tools/mvs_relayout_by_camera.py`
  - 输入：包含 `metadata.jsonl` 的 captures 目录
  - 输出：`<captures_dir>_by_camera/`（`cam{i}_{serial}/...`）

### 网球 3D 定位（tennis3d）

- 在线定位：`python -m tennis3d.apps.online_mvs_localize`
  - 调用链：
    - `mvs.pipeline.open_quad_capture()` 打开相机与组包
    - `tennis3d.pipeline.iter_mvs_image_groups()` 把 `FramePacket` 转为 OpenCV BGR 图
    - `tennis3d.detectors.create_detector()` 选择 fake/color/rknn
    - `tennis3d.pipeline.run_localization_pipeline()`：detect → `tennis3d.localization.localize_ball()` → 输出 JSONL

- 离线定位（从 captures）：`python -m tennis3d.apps.offline_localize_from_captures`
  - 调用链：
    - `tennis3d.pipeline.iter_capture_image_groups()` 读取 `captures_dir/metadata.jsonl` + 图片
    - 后续与在线相同：detector → run_localization_pipeline

- 三相机离线检测（时间对齐 + 推理）：`python -m tennis3d.offline.cli`
  - 调用：`tennis3d.offline.pipeline.run_pipeline()`
  - 输出：`data/tools_output/tennis_detections.json`（可选 CSV/可视化）

- 仅几何三角化（已有 detections.json）：`tools/tennis_localize_from_detections.py`
  - 输入：检测结果 JSON + 标定文件
  - 输出：`data/tools_output/tennis_positions_3d.json`（可配）

## 数据流图

```mermaid
flowchart TD
    A[tools/mvs_quad_capture.py] --> B[output_dir/metadata.jsonl]
    A --> C[output_dir/group_*/cam*.bmp]

    B --> D[tennis3d.pipeline.iter_capture_image_groups]
    C --> D

    E[open_quad_capture (online)] --> F[tennis3d.pipeline.iter_mvs_image_groups]

    D --> G[tennis3d.pipeline.run_localization_pipeline]
    F --> G

    H[Detector: fake|color|rknn] --> G
    I[Calibration: load_calibration] --> G

    G --> J[offline_positions_3d.jsonl / out_jsonl]
```

## 关键输入/输出格式

### 1) captures：`metadata.jsonl`

`tools/mvs_quad_capture.py` 在 `output_dir/metadata.jsonl` 中混合写入两类记录：

- 事件记录：`{"type": "camera_event", ...}`、`{"type": "soft_trigger_send", ...}`
- 组记录：包含 `frames` 字段（`tennis3d.pipeline.iter_capture_image_groups()` 会自动跳过事件记录）

典型组记录结构（字段会更多，这里只列关键）：

```json
{
  "group_seq": 0,
  "group_by": "frame_num",
  "trigger_index": 0,
  "frames": [
    {
      "cam_index": 0,
      "serial": "DA8199303",
      "frame_num": 1,
      "dev_timestamp": 123,
      "lost_packet": 0,
      "file": "data\\captures_master_slave\\...\\cam0_seq000000_f1.bmp"
    }
  ]
}
```

注意：
- `file` 可能是相对路径；离线读取会用 `captures_dir` 进行补全。
- 分组键 `group_by` 推荐以采集时选择为准：`trigger_index|frame_num|sequence`。

### 2) 标定文件（JSON/YAML）

`tennis3d.geometry.calibration.load_calibration()` 支持 `.json/.yaml/.yml`。
外参约定为 world→camera：

$$X_c = R_{wc} X_w + t_{wc}$$

投影矩阵：

$$P = K [R_{wc} | t_{wc}]$$

关键点：标定中 `cameras` 的 key 必须与 pipeline 使用的“相机名”一致。
- 在线/离线（captures）模式下，默认相机名为 **serial 字符串**（见 `tennis3d.pipeline.sources`）。
- 如果你的标定用 cam0/cam1 命名，会导致相机被跳过（见改进建议）。

### 3) 3D 输出（JSONL）

在线/离线定位输出为 JSON Lines，每行一个成功定位记录（至少 `require_views` 个视角）：

- `ball_3d_world`: 世界坐标系 3D 点 `[x,y,z]`
- `used_cameras`: 实际参与三角化的相机名列表
- `reprojection_errors`: 每相机重投影误差（像素）
- `detections`: 每相机选中的 bbox/score/center（可选）

