# 实时延迟测量方案（采集/曝光/传输/算法）

本仓库的在线系统要“既实时又可复盘”，最稳妥的做法是：**把延迟拆成多段，各段都记录原始时间戳与可解释的统计量**，而不是试图用一个时间映射把所有问题一次性解决。

本文给出一套在当前仓库结构下可直接落地的口径：

- 采集侧（MVS）：帧时间戳、到达主机时刻、相机事件（ExposureStart/ExposureEnd）
- 算法侧（tennis3d pipeline）：每个阶段的主机耗时（detect/localize/打包等）
- 输出侧：写盘/打印带来的额外抖动

> 重要提醒：**不同时间轴不要直接相减**。
>
> - `*_t_abs` / `created_at`：epoch（墙上时钟）时间轴
> - `*_monotonic`：主机单调时间轴（推荐用于延迟/抖动/耗时）
> - `dev_timestamp` / `event_timestamp`：相机设备 tick 时间轴

---

## 1. 你能“可靠测量”的延迟分段

### 1.1 算法处理耗时（最可靠）
在线输出记录（JSONL）现在包含 `latency_host` 字段（主机单调时钟）：

- `latency_host.align_ms`：对齐阶段耗时（通常很小）
- `latency_host.detect_ms`：检测阶段总耗时
- `latency_host.detect_ms_by_camera`：每相机 detect 耗时
- `latency_host.localize_ms`：多视角匹配+三角化耗时
- `latency_host.total_ms`：本组从进入 pipeline 到产出记录的总耗时

这些值用于回答：

- 算法瓶颈在哪里（GPU/CPU detect 还是 localize）
- 抖动来自哪里（某一路相机 detect 偶发变慢、还是整体偶发变慢）

### 1.2 采集/组包相关耗时（主机侧可解释）
在线 `iter_mvs_image_groups()` 透传了采集侧诊断字段（同样是主机单调时钟）：

- `capture_arrival_monotonic_by_camera`：每台相机帧到达主机（应用层）时刻
- `capture_arrival_monotonic_median`：组内到达时刻中位数
- `capture_arrival_monotonic_spread_ms`：组内到达抖动（max-min）
- `capture_group_ready_monotonic`：组包器凑齐 N 帧、返回这一组的时刻
- `capture_group_ready_minus_arrival_median_ms`：粗略估计的“主机侧组包/排队”开销

这些值用于回答：

- 网络/SDK/线程调度导致的到达抖动是否变大
- 组包器是否在积压（`group_ready_minus_arrival_median_ms` 变大通常意味着队列/组包等待）

### 1.3 端到端（epoch）“观测延迟”（可用但要理解含义）
在线输出里已有：

- `created_at`（epoch 秒）
- `capture_t_abs`（epoch 秒，来自 `frame_host_timestamp` 或 `dev_timestamp_mapping`）

你可以计算：

$$\text{lag\_epoch\_ms} = (created\_at - capture\_t\_abs) \times 1000$$

注意：当 `capture_t_source=dev_timestamp_mapping` 时，`capture_t_abs` 是通过“帧 dev_timestamp 与帧 host_timestamp”拟合得到的映射输出，它**会吸收链路延迟的平均项**，因此它更像“把设备 tick 映射到主机看到帧的时间”，不是严格的“曝光发生时刻”。

---

## 2. 曝光延迟与传输延迟：现实与可落地做法

### 2.1 曝光时长（ExposureStart→ExposureEnd）：推荐用设备 tick
采集工具 `mvs.apps.quad_capture` 支持订阅并落盘相机事件（写入 captures/metadata.jsonl）：

- `type=camera_event`，字段包含：`event_name`、`event_timestamp`（设备 tick）、`host_monotonic`（主机接收回调时刻）

当你同时订阅 `ExposureStart` 与 `ExposureEnd` 后，可在设备 tick 上计算曝光时长：

$$\Delta_{exp,ticks} = ts(ExposureEnd) - ts(ExposureStart)$$

若你能从相机/SDK 文档得到 tick 频率 $f$，则

$$\Delta_{exp,ms} = 1000 \times \frac{\Delta_{exp,ticks}}{f}$$

### 2.2 “触发→曝光开始”的曝光延迟：需要触发时刻
要严格得到“触发边沿→曝光开始”，你必须能记录触发时刻（设备侧或统一时钟侧）。

- 硬触发：通常要靠硬件测量（示波器/光电）或相机提供 Trigger 相关事件时间戳
- 软触发：仓库会记录 `soft_trigger_send`（主机 monotonic），但这只代表“主机下发命令”，不等价于“触发边沿到达相机”

因此，在没有额外触发时刻的前提下，**触发→曝光开始延迟无法严格拆分**。

### 2.3 “曝光结束→帧到达主机”的传输延迟：需要统一时间轴
要严格拆分“曝光结束→到达主机”，需要把 `event_timestamp`（设备 tick）与主机时间轴对齐：

- 最推荐：PTP/IEEE1588（相机与主机共享时钟）
- 次优：NIC 硬件时间戳（需要系统/驱动/网卡配合）
- 工程近似：使用 dev->host 的线性拟合，但它会吸收平均链路延迟，只能做趋势/抖动分析

---

## 3. 在线与离线的组合建议（最小改动，最大收益）

1) **在线运行（实时性为主）**：看 `latency_host.*` 与 `capture_*_monotonic*`，定位瓶颈与抖动来源。

2) **定期做一次采集 Profiling（真相为主）**：用 `mvs.apps.quad_capture` 采几十秒，订阅 `ExposureStart/ExposureEnd/FrameStart`，用 captures/metadata.jsonl 做曝光节拍、抖动、丢包与（在设备 tick 上）曝光时长统计。

3) 如果业务需要“严格拆分曝光延迟与传输延迟”，优先投入 PTP/硬件测量链路；否则建议把指标定义为：

- `exposure_duration_ms`（能做）
- `arrival_jitter_ms`（能做）
- `pipeline_compute_ms`（能做）
- `end_to_end_epoch_lag_ms`（可做但要注明口径）

---

## 4. 字段速查（在线 JSONL）

- 采集侧诊断（来自 `tennis3d.pipeline.sources_online.iter_mvs_image_groups()`）
  - `capture_arrival_monotonic_by_camera`
  - `capture_arrival_monotonic_spread_ms`
  - `capture_group_ready_monotonic`
  - `capture_group_ready_minus_arrival_median_ms`
  - `capture_frame_dev_timestamp_by_camera`
  - `capture_frame_host_timestamp_by_camera`

- 算法侧诊断（来自 `tennis3d.pipeline.core.run_localization_pipeline()`）
  - `latency_host.align_ms/detect_ms/localize_ms/total_ms`
  - `latency_host.detect_ms_by_camera`

- 时间映射质量（已存在）
  - `time_mapping_worst_p95_ms/time_mapping_worst_rms_ms`
  - `time_mapping_*_spread_ms` 与 `*_delta_to_median_by_camera`
