# 主从触发时间偏移排查与修复报告（slaves TriggerActivation: FallingEdge）

日期：2026-02-03

## 背景

- 硬件拓扑：4 台相机。
  - master：`DA8199303`（主机软件触发自身采集）。
  - slaves：3 台（由 master 的输出线触发，硬触发采集）。
- 目标：多相机在同一触发下“曝光时刻”尽可能同步，用于后续融合/定位。

## 现象与影响

- 现象：组内时间对齐时，master 的时间戳相对 slaves 存在稳定偏移（约 10ms），且非常规律。
- 影响：如果把该时间戳当作“曝光开始时刻”参与多相机对齐，会导致跨相机时刻不一致，影响三维定位与时序分析。

## 关键证据（定位思路）

1) 偏移在“映射之前”就存在

- 直接看采集写出的 `metadata.jsonl` 中每帧的原始 `host_timestamp`（主机侧时间戳），组内 master 与 slaves 已存在稳定差值。
- 这表明问题不是 dev→host 的映射算法引入，而是采集链路/触发链路导致的真实曝光时刻差异。

2) 偏移与曝光时间强相关

- 将曝光从约 10ms 调到约 2ms 后，master 与 slaves 的稳定偏移也从约 10ms 变为约 2ms。
- 该“偏移≈曝光时长”的指纹非常典型：说明 slaves 很可能是被配置为在错误的触发沿上启动曝光（例如在曝光结束沿触发，而不是曝光开始沿触发）。

3) 改触发沿后偏移归零

- 将 slaves 的 `TriggerActivation` 从 `RisingEdge` 改为 `FallingEdge` 后，四台相机在组内的时间偏移归零（host_ms 维度可观测到 0ms spread）。

## 根因结论

slaves 的外触发沿配置不匹配 master 输出信号的极性/波形语义。

更具体地说：master 输出线（例如 `ExposureStartActive`）在一次曝光周期内通常会维持一个“有效电平”（高或低），其脉宽往往与曝光时长相关。

- 如果 slaves 监听 `RisingEdge`，但 master 的“有效电平”是低有效（或波形在曝光开始时拉低、曝光结束时拉高），则 `RisingEdge` 可能对应“曝光结束”的时刻。
- 因此会出现：slaves 的曝光启动时刻相对 master 偏移一个曝光时长。

## 修复措施

将 slaves 的触发沿设置为下降沿：

- `TriggerActivation = FallingEdge`

在本仓库命令参数层面，对应：

- `--trigger-activation FallingEdge`

说明：该修复仅改变 slaves 的触发沿选择，不涉及时间映射代码，也不依赖额外字段。

## 验证方法与验收标准

### 采集与验证命令

- 用主从模式采集一段数据（master 软件触发 + slaves 硬触发）。
- 然后运行诊断脚本：`tools/debug_master_slave_timing.py`

示例（按你的目录与序列号替换）：

- 仅看原始 host_timestamp 的组内偏移：
  - `uv run python tools/debug_master_slave_timing.py --captures-dir data/captures_master_slave/timer_debug --master DA8199303`

- 若你同时拟合了时间映射，也可带上映射文件进一步确认：
  - `uv run python tools/debug_master_slave_timing.py --captures-dir data/captures_master_slave/tennis_offline --master DA8199303 --time-mapping data/captures_master_slave/tennis_offline/time_mapping_dev_to_host_ms.json`

### 验收标准

- 组内 `host_ms_epoch spread` 应接近 0（在 host_timestamp 量化为 1ms 的情况下，理想表现常见为 0ms）。
- 每台相机相对组内中位数的偏移分布应集中在 0 附近（中位数为 0，P95/P99 不应出现“≈曝光时长”的系统性偏差）。

## 经验总结（快速排查口诀）

- “偏移稳定”且“偏移≈曝光时长” → 首先检查 **外触发沿（Rising/Falling）** 是否选反。
- 如果改曝光时间，偏移跟着等比例变化 → 更进一步支持“触发沿/极性不匹配”的结论。

