# config YAML 模板（tennis3d online/offline）

这里的文件是“模板”：字段名与 `src/tennis3d/config.py` 的 loader 保持一致。

约定：

- **有默认值的字段**：尽量写出默认值（便于你理解默认行为）
- **必填但没有默认值的字段**：使用占位符（例如 `<PATH_TO_CALIB_JSON_OR_YAML>`、`<CAM1_SERIAL>`），需要你手动替换

建议用法：从这里复制一份到 `configs/`，再改成你的真实参数。

## 快速用法

- 离线入口：`src/tennis3d/apps/offline_localize_from_captures.py`
  - 支持：`--config path/to/offline.yaml`
- 在线入口：`src/tennis3d/apps/online/entry.py`
  - 支持：`--config path/to/online.yaml`

## 模板清单

### 离线（offline）

- `offline_color_minimal.yaml`
  - 颜色检测模板：不依赖模型文件，适合先验证“读 captures -> 分组 -> 三角化 -> 输出”的链路。
- `offline_pt_ultralytics.yaml`
  - `.pt` 模型模板：Windows/CPU 上常用（Ultralytics YOLOv8）。
- `offline_rknn_board_or_linux.yaml`
  - RKNN 模板：通常用于 Rockchip 或 Linux 工具链（Windows 上一般不可用）。

- `offline_fake_smoke.yaml`（可选）
  - 冒烟/连通性模板：使用 fake 检测器，主要用于验证程序链路是否可跑通。

### 在线（online）

- `online_software_trigger_minimal.yaml`
  - 软件触发模板：所有相机 Software，按 `soft_trigger_fps` 发软触发。
- `online_master_slave_template.yaml`
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

4) 多球鲁棒定位的 4 个关键参数

这些字段来自 `src/tennis3d/config.py`，模板中会显式写出（即使你暂时不需要调参也建议保留）：

- `max_detections_per_camera`：每相机最多取 topK 个候选，避免组合爆炸
- `max_reproj_error_px`：最大重投影误差阈值（像素），越小越严格
- `max_uv_match_dist_px`：投影补全匹配阈值（像素）
- `merge_dist_m`：3D 去重阈值（米）

验证标准（最小）：

- 运行 offline/online 后，输出 jsonl 中能看到 `balls` 字段；当检测到球时 `balls` 非空。

5) 相机侧 ROI（硬件裁剪）与像素格式

在线（online）配置支持相机图像参数（字段来自 `src/tennis3d/config.py`，并会下发到 MVS SDK）：

- `pixel_format`：留空表示不设置（沿用相机当前配置）。常见可填：`Mono8`、`BayerRG8`、`BayerBG8` 等。
- `image_width` + `image_height`：相机输出 ROI 的宽高（必须同时设置；只写一个会报错）。
- `image_offset_x` + `image_offset_y`：ROI 左上角偏移。

注意：

- 这里的 ROI 是“裁剪”，不是缩放；能显著降低带宽与 CPU 压力，但视场会变小。
- 多数机型对 Width/Height/Offset 有步进（Inc）限制，不对齐时 SDK 会失败或被自动对齐。

6) 相机侧 AOI（运行中动态 OffsetX/OffsetY）

在线（online）支持启用“相机侧 AOI 运行中平移”，配置字段为：

- `camera_aoi_runtime: true`：开启动态 AOI。
- `camera_aoi_update_every_groups`：每隔 N 个 group 才尝试更新一次（>=1）。
- `camera_aoi_min_move_px`：小于该像素变化不更新（减少抖动与写节点频率）。
- `camera_aoi_smooth_alpha`：平滑系数 $\in[0,1]$，越大越稳但跟随更慢。
- `camera_aoi_max_step_px`：单次最大移动像素（限速）。
- `camera_aoi_recenter_after_missed`：连续无球后逐步回到初始 offset（0 表示禁用）。

使用建议：

1) 先设置一个“足够大、不易丢球”的固定相机 ROI（`image_width/image_height`），用于降带宽。
2) 再启用 `camera_aoi_runtime`，让窗口跟随球移动，进一步降带宽。
3) 如需进一步提速，可叠加 `detector_crop_size` 做 AOI 内的软件裁剪（降推理算力）。

注意：

- 不是所有机型都支持在取流中写 OffsetX/OffsetY；不支持时写入会失败，窗口不会移动。
- 启用动态 AOI 时，不应做一次性的标定主点平移；本仓库会在 `camera_aoi_runtime=true` 时自动跳过该步骤。

