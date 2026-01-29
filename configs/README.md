# config YAML 模板（tennis3d online/offline）

这里的文件是“模板”：字段名与 `src/tennis3d/config.py` 的 loader 保持一致，适合直接复制一份再改成你的参数。

## 快速用法

- 离线入口：`src/tennis3d/apps/offline_localize_from_captures.py`
  - 支持：`--config path/to/offline.yaml`
- 在线入口：`src/tennis3d/apps/online_mvs_localize.py`
  - 支持：`--config path/to/online.yaml`

## 模板清单

### 离线（offline）

- `offline_color_minimal.yaml`
  - 最小可用模板：不依赖模型文件，适合先验证“读 captures -> 分组 -> 三角化 -> 输出”的链路。
- `offline_fake_smoke.yaml`
  - 冒烟模板：更偏“程序能否跑通”的连通性测试。
- `offline_pt_ultralytics.yaml`
  - `.pt` 模型模板：Windows/CPU 上常用（Ultralytics YOLOv8）。

### 在线（online）

- `online_software_trigger_minimal.yaml`
  - 最小可用模板：纯软件触发（所有相机 Software，按 `soft_trigger_fps` 发软触发）。
- `online_master_slave_line1_template.yaml`
  - 主从触发拓扑模板：只对 master 发软触发，master 通过 LineOut 触发 slave。

### 标定（calibration）

- `calibration_multi_camera.yaml`
  - 标定文件结构模板（`tennis3d.geometry.calibration.load_calibration` 可读取）。

## 字段要点（最容易踩坑的部分）

1) **标定 cameras 的 key 要匹配 camera_name**

- 在线/离线（captures）pipeline 默认用“相机 serial 字符串”作为 camera_name（见 `src/tennis3d/pipeline/sources.py`）。
- 因此标定文件里 `cameras:` 的 key 推荐直接用 serial：
  - 正例：`"DA8199285": { ... }`
  - 反例：`cam0: { ... }`（除非你的输入里 camera_name 也叫 cam0）

2) `model` / `dll_dir` 这类字段允许空字符串

- `model: ""` 或 `dll_dir: ""` 会被 loader 当作 `None`。

3) `max_groups: 0` 表示不限

- 离线与在线模板都遵循这个约定。
