# 运行模式与调试（online/offline/采集诊断）

本文档给出三类“可落地”的运行模式，并提供排障路径（优先从数据与指标出发）。

## 模式一：只做采集（推荐先把采集链路跑稳）

### A) 枚举设备

```bash
python -m mvs.apps.quad_capture --list
```

如果此步失败，优先处理 DLL（见下方“常见问题”）。

### B) 软触发验证链路

目标：验证“打开相机→取流→组包→写 metadata”是否正常。

```bash
python -m mvs.apps.quad_capture \
  --serial <SN0> <SN1> <SN2> <SN3> \
  --trigger-source Software \
  --soft-trigger-fps 5 \
  --save-mode none \
  --max-groups 50
```

建议先 `--save-mode none`，避免写盘造成瓶颈与误判。

### C) master/slave（常见硬件拓扑）

目标：master 用软件触发驱动自身曝光，同时通过输出线把曝光信号打到 slave 的输入线上。

关键参数：
- `--master-serial`：指定 master 相机序列号（该相机会被强制为 Software 触发）
- `--master-line-source ExposureStartActive`：配置输出线信号源
- `--trigger-source Line0`：slave 的触发输入线（要与接线一致）

```bash
python -m mvs.apps.quad_capture \
  --serial <SN_MASTER> <SN1> <SN2> <SN3> \
  --master-serial <SN_MASTER> \
  --master-line-source ExposureStartActive \
  --soft-trigger-fps 18 \
  --trigger-source Line0 \
  --trigger-activation FallingEdge \
  --save-mode sdk-bmp \
  --max-groups 10 \
  --max-wait-seconds 10
```

脚本启动时会打印 `trigger_sources=serial->source` 映射，请务必核对。

## 模式二：采集后离线复盘（强烈推荐作为验收标准）

### A) 运行分析脚本

```bash
python -m mvs.apps.analyze_capture_run \
  --output-dir data/captures_master_slave/tennis_test \
  --write-json data/captures_master_slave/tennis_test/analysis_summary.json
```

### B) 如何看报告（最小关注集）

- 组包完整性：`groups_complete == records` 且 `groups_incomplete == 0`
- 网络丢包：`lost_packet_total == 0`
- 出图频率：优先看 `arrival_fps_median`（比 `created_at` 更接近真实出图节拍）
- 分组键诊断：
  - 如果 `trigger_index_all_same=true`（例如恒为 0），不要使用 `--group-by trigger_index`
  - 建议保持采集默认 `frame_num` 分组

### C) 常见现象对照

- `dropped_groups` 持续增加：通常是某台相机慢/丢帧/触发不一致 → 优先看 `lost_packet` 与接线
- `qsize` 持续增长：下游处理/保存跟不上 → 先用 `--save-mode none` 排除写盘瓶颈

## 模式三：网球 3D 定位

### A) 离线（从 captures）

推荐先用 `fake` 检测器跑通“读数据→几何”链路：

```bash
python -m tennis3d.apps.offline_localize_from_captures \
  --captures-dir data/captures_master_slave/tennis_test \
  --calib data/calibration/example_triple_camera_calib.json \
  --detector fake \
  --require-views 2 \
  --max-groups 5
```

如果你想在 Windows 上做简单检测调试，可尝试 `--detector color`（阈值检测，适合绿色球/高对比场景）。

### B) 在线（连接相机）

```bash
python -m tennis3d.apps.online \
  --serial <SN0> <SN1> <SN2> <SN3> \
  --calib data/calibration/example_triple_camera_calib.json \
  --detector fake \
  --require-views 2
```

```bash
python -m tennis3d.apps.online \
  --config configs/online/pt_windows_cpu_software_trigger.yaml
```

### C) 两级 ROI（推荐：先带宽，再推理）

适用场景：
- 满幅 2448×2048 带宽/解码压力大，导致采集 FPS 上不去；
- 但直接把相机 AOI 缩得过小又容易丢球；
- 你希望把“带宽瓶颈”和“推理瓶颈”拆开分别优化。

做法：
1) **相机侧 AOI（Sensor ROI）**：设置 `camera.roi.width/camera.roi.height/camera.roi.offset_*`，让相机只输出一个较大的窗口（比如 1280×1080），主要用于降带宽。
2) **主机软件裁剪（detector.crop.*）**：在 detector 前再裁成 640×640（或类似大小）来提升推理吞吐，并把 bbox 坐标回写到 AOI 坐标系下，保证几何一致。

本仓库已内置两级 ROI 的配置样例：

```bash
python -m tennis3d.apps.online \
  --config configs/online/pt_windows_cpu_two_level_roi_4cam.yaml
```

要点/常见坑：
- MVS 的 ROI 是**裁剪不是缩放**。标定通常按满幅做的，因此在线入口会在你配置 `camera.roi.width/camera.roi.height` 时自动调用
  `tennis3d.geometry.calibration.apply_sensor_roi_to_calibration()` 做主点平移，确保像素坐标与标定一致。
- `camera.roi.offset_x/offset_y` 往往有步进（Inc，例如 4）。配置里尽量用对齐后的值（例如 584/484）。
- 如果你的相机是“固定 AOI（Width/Height 不可写）但 Offset 可写”，只要当前输出尺寸确实等于你填的 `camera.roi.width/height`，
  仍建议保留这两个字段用于标定修正。

预期输出（最小）：

- 启动后会先提示 `Waiting for first ball observation...`
- 当观测到球（跨视角几何一致，`balls` 非空）时，会在终端持续打印类似：
  - `t=<capture_t_abs> group=<group_index> xyz_w=(x=..., y=..., z=...) err_mean=...px ...`
  - 其中 `xyz_w` 即 `balls[0].ball_3d_world`（世界坐标系 3D）

### D) 如何确定“时间映射后各相机时间差是多少”

前置条件：
- 使用 `--time-sync-mode dev_timestamp_mapping`（或在 config 中设置 `time.sync_mode: dev_timestamp_mapping`）
- 建议同时开启 `--out-jsonl`（便于离线统计）

在线 source（`src/tennis3d/pipeline/sources.py::iter_mvs_image_groups`）会在每条输出记录中透传以下字段（均为毫秒 ms）：

- `time_mapping_mapped_host_ms_by_camera`：映射后的每相机 host 时间戳（ms_epoch）
- `time_mapping_mapped_host_ms_spread_ms`：同一 group 内各相机时间戳跨度（max-min）
- `time_mapping_mapped_host_ms_delta_to_median_by_camera`：每相机相对组内“中位数”的偏差（正=更晚，负=更早）

对照用（不经映射、直接看 host_timestamp 的离散程度）：

- `time_mapping_host_ms_by_camera`
- `time_mapping_host_ms_spread_ms`
- `time_mapping_host_ms_delta_to_median_by_camera`

终端打印：当上述字段存在时，online 的“有球时输出”会追加：

- `dt_raw=<...>ms`：原始 host_timestamp 组内跨度
- `dt_map=<...>ms`：映射后组内跨度
- `dt_map_by_cam={<serial:+/-...>}`：每相机相对组内中位数的偏差

离线汇总（推荐看分布，而不是只看某一帧）：

```bash
python tools/time_mapping_report.py --jsonl <你的输出.jsonl>
```

预期输出/验证标准（最小）：

- 能看到“映射后组内跨度（ms）”的 p50/p95/max
- 能看到每台相机“相对组内中位数偏差（ms）”的 median 与 abs_p95

### C) 可选：轨迹拟合（落点 + 落地时间 + 置信走廊）

前置条件：
- 你已经能稳定输出 3D（`balls[*].ball_3d_world`）
- 你的输入 meta 能提供 `capture_t_abs`（本仓库的 online/offline source 已默认注入；见 `src/tennis3d/pipeline/sources.py`）

启用方式：推荐通过 config 文件开启。在 config 中增加：

```yaml
curve:
  enabled: true
```

然后使用 `--config` 运行（示例）：

```bash
python -m tennis3d.apps.offline_localize_from_captures --config configs/offline/pt_windows_cpu.yaml
```

预期输出/验证标准（最小）：
- 输出 jsonl 中每条记录新增 `curve` 字段
- `curve.track_updates[*].v3` 下包含：
  - `predicted_land_point`（落点）
  - `predicted_land_time_abs` / `predicted_land_time_rel`（落地时间）
  - `corridor_on_planes_y`（走廊统计；在模型达到可预测状态后逐步变为非空）

## 常见问题

### 1) 找不到 MvCameraControl.dll

处理顺序：
1) 安装 MVS
2) 设置 `MVS_DLL_DIR` 指向 DLL 所在目录（目录，不是具体文件）
3) 或使用 `--dll-dir` 显式指定

### 2) 一直等不到组包

优先排查：
- 触发是否真的到达了每台相机（硬件接线与 `--trigger-source` 必须一致）
- master/slave 是否设置了 `--master-line-source` 并正确接线
- 带宽是否超限（先 `--save-mode none`、再 ROI/降低像素格式）

### 3) 离线定位没有输出 3D

常见原因：
- 标定相机 key 与相机名不一致：captures 模式默认用 serial 作为相机名；标定 key 不匹配会被跳过
- `--require-views` 过大：当可用相机不足时会没有输出

