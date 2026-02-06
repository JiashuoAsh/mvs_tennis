# MVS（海康工业相机）Python 采集包

这个包把海康 MVS SDK 的取流、触发配置、多相机“组包”与诊断指标封装成可复用的 Python 组件。

你用硬件（外触发/主从触发/触发分配器）保证“同一时刻曝光”，而本包负责：

- 稳定取流（多线程 Grabber）
- 把多台相机的帧按同一次触发正确配对成一组（grouping）
- 记录可复盘的元信息（`metadata.jsonl`）
- 给出足够的诊断指标（`lost_packet` / `dropped_groups` / 队列深度等）

---

## TL;DR：先用脚本跑起来

项目推荐直接用 `python -m mvs.apps.quad_capture` 做采集，它会把参数、触发映射与诊断数据都记录下来。

入口位置：`packages/mvs/src/mvs/apps/quad_capture.py`。

建议的运行方式（避免误用系统 Python）：

- `uv run python -m mvs.apps.quad_capture --help`
- `uv run python -m mvs.apps.quad_capture --list`

### 1) 确保能找到 `MvCameraControl.dll`

三种方式任选其一：

**方式 A：系统 PATH 中可找到 DLL**

安装海康 MVS 软件后通常可用。

**方式 B：命令行显式指定 DLL 目录**

```bash
python -m mvs.apps.quad_capture --dll-dir "C:\\path\\to\\mvs\\bin" --list
```

**方式 C：通过环境变量指定 DLL 目录**

```bash
set MVS_DLL_DIR=C:\\path\\to\\mvs\\bin
python -m mvs.apps.quad_capture --list
```

### 2) 枚举设备

```bash
python -m mvs.apps.quad_capture --list
```

### 3) 采集（推荐先 `--save-mode none` 测上限）

#### 先验证链路：纯软件触发

```bash
python -m mvs.apps.quad_capture \
  --serial DA8199285 DA8199303 DA8199402 \
  --trigger-source Software \
  --soft-trigger-fps 15 \
  --save-mode none \
  --output-dir ./captures \
  --max-groups 50
```

#### 生产常用：硬件外触发（严格同步靠硬件保证）

```bash
python -m mvs.apps.quad_capture \
  --serial DA8199285 DA8199303 DA8199402 \
  --trigger-source Line0 \
  --trigger-activation FallingEdge \
  --save-mode sdk-bmp \
  --output-dir ./captures \
  --max-groups 100
```

#### master/slave：master 软件触发 + slaves 硬触发（常见于“master 输出曝光脉冲”）

```bash
python -m mvs.apps.quad_capture \
  --serial DA8199303 DA8199285 DA8199402 \
  --master-serial DA8199303 \
  --master-line-source ExposureStartActive \
  --soft-trigger-fps 15 \
  --trigger-source Line0 \
  --trigger-activation FallingEdge \
  --save-mode sdk-bmp \
  --output-dir ./captures_master_slave \
  --max-groups 20 \
  --max-wait-seconds 10
```

> 重要：脚本启动时会打印 `trigger_sources=...` 映射。一定要确认 master 是 `->Software`，slave 是 `->Line0`，避免“以为在硬触发，实际全是软件触发”。

---

## 代码结构（读代码从这里开始）

```
mvs/
├── __init__.py           # 对外导出 API
├── analysis/             # 采集输出分析（analyze_capture_run）
├── apps/                 # CLI 入口（quad_capture / analyze_capture_run 等）
├── capture/              # 抓流/分组/保存/软触发等采集流水线
├── core/                 # 纯工具与公共结构（文本/ROI/事件/容错等）
├── sdk/                  # MVS SDK 绑定加载、设备枚举、相机生命周期、运行时 ROI best-effort
└── session/              # captures 会话与离线处理（metadata/time_mapping/relayout 等）
```

### 数据流（最重要的 30 秒）

```text
MVS SDK -> Grabber(每台相机一个线程) -> frame_queue -> QuadCapture.get_next_group()
                                              |
                                              +-> TriggerGroupAssembler(grouping) -> 输出一组 frames
```

抓到的每一帧都会被封装为 `FramePacket`（纯数据，便于跨线程传递/落盘/分析）。

---

## 分组（同步“配对”）到底怎么做？

### 1) 你要分清：同步曝光 vs 同步配对

- **同步曝光**：靠硬件触发链路（你已经做对了）
- **同步配对**：靠软件把多台相机的“同一次触发产生的帧”放进同一个 group

本包的分组器是 `mvs/grouping.py::TriggerGroupAssembler`。

### 2) `--group-by` 的选择（强烈建议读懂）

采集入口（`mvs.apps.quad_capture`）当前支持两种分组键：

1. `frame_num`（脚本默认）：使用 `nFrameNum`，并对每台相机做“基准归一化”
  - **优点**：兼容性最好；在触发频率一致且不丢帧时很好用
  - **坑点**：一旦某台相机丢帧/断流，帧号会错位，组包会超时并产生 `dropped_groups`

2. `sequence`：按“进入分组器的顺序编号”
  - **用途**：极简兜底；要求更严格（不能丢帧、不能乱序）

说明：历史上曾尝试过使用设备端某些触发相关字段作为分组键，但在部分机型/固件/配置下可能恒为常数或不递增，容易制造误判，因此已从本仓库移除。

### 3) “基准归一化”为什么重要

同一个字段（例如 `frame_num`）在不同相机上可能不是从同一个起点开始。
分组器会记录每台相机首次看到的值作为 base，然后用：

$$group\_key = (value - base) \& 0xFFFFFFFF$$

把每台相机的序列都对齐到从 0 开始。

---

## 诊断指标：别把它们混着用

这些字段经常被误解，建议固定口径：

| 指标 | 来自哪里 | 含义 | “正常”是什么样 |
|---|---|---|---|
| `lost_packet` | SDK `nLostPacket`（每帧） | GigE 丢包数（会导致画面损坏/延迟） | 0 |
| `dropped_groups` | 分组器 `_prune()` | 组包超时/缓存过多导致丢组 | 0 或极小 |
| `qsize` | 线程队列深度 | 下游处理是否跟不上 | 越小越好（持续增长就是瓶颈） |

---

## 输出文件：`metadata.jsonl` 里有什么

采集入口（`mvs.apps.quad_capture`）会在输出目录写 `metadata.jsonl`，里面混合两类记录：

1) **事件记录**（例如 `ExposureStart`）：`type=camera_event`

2) **组记录**：包含 `group_seq/group_by/frames[]`，每个 `frames[]` 中有：
- `frame_num`
- `dev_timestamp`（设备时间戳）
- `host_timestamp` / `arrival_monotonic`（诊断用）
- `lost_packet`
- 保存文件路径（若 `--save-mode` 非 none）

---

## 排坑清单（你大概率会踩的那种）

### 1) `frame_num` 错位 / 组包超时（最常见）

**原因**：某台相机丢帧/断流/触发不一致，导致 `frame_num` 进度不一致。

**怎么确认**：
- 优先看 `lost_packet` 是否为 0
- 看 `dropped_groups` 是否持续增长

**建议做法**：
- 先用默认 `--group-by frame_num`
- 在你确认“不丢帧且节拍稳定”时，再尝试 `--group-by sequence`

### 2) 你以为是 master/slave，实际全软件触发

采集入口会把 master 改成 `Software` 触发，其它相机用 `--trigger-source`。
如果你输出目录里出现大量 `type=soft_trigger_send`，说明你当次采集确实在下发软件触发。

**建议**：每次启动先看脚本打印的 `trigger_sources=serial->source`。

### 3) master 的输出线模式不叫 Output（叫 Strobe）

不同机型的 `LineMode` 枚举值可能不同：有的机型 `Output` 会失败，需要 `Strobe`。
本包在 `mvs/sdk/camera.py::configure_line_output()` 里做了 Output/Strobe 候选尝试，但你仍需要：

- 在 MVS Client 里确认可选值
- 必要时命令行传 `--master-line-mode Strobe`

### 4) “没有出图 / 凑不齐组包”的 80% 根因

按优先级排查：

1. 触发链路没通（硬件线没接/接错口/边沿不一致/电平不一致）
2. master 没配置输出源（缺了 `--master-line-source ExposureStartActive` 或相机不支持）
3. 程序启动前就开始触发，导致各相机“看到的触发序列”不同步
4. 带宽不足/网络丢包（看 `lost_packet`）

### 5) 保存 BMP 会把 FPS 拉崩（尤其是大分辨率）

`sdk-bmp` 需要做像素转换/写盘，瓶颈往往不在取流而在保存。

**推荐排查顺序**：
1) 先 `--save-mode none` 测最大可达 FPS
2) 再 `--save-mode raw`（更轻）
3) 最后才 `--save-mode sdk-bmp`

---

## 作为库使用：核心 API 入口

你可以直接用 `mvs.open_quad_capture()` 得到 `QuadCapture`：

```python
from mvs import open_quad_capture, load_mvs_binding

binding = load_mvs_binding()

with open_quad_capture(
    binding=binding,
    serials=["DA8199285", "DA8199303", "DA8199402"],
    trigger_sources=["Line0", "Line0", "Line0"],
  trigger_activation="FallingEdge",
    trigger_cache_enable=False,
    timeout_ms=1000,
    group_timeout_ms=1000,
    max_pending_groups=256,
    group_by="frame_num",
    enable_soft_trigger_fps=0.0,
    exposure_auto="Off",
    exposure_time_us=5000.0,
    gain_auto="Off",
    gain=12.0,
) as cap:
    group = cap.get_next_group(timeout_s=1.0)
    if group:
        print([f"cam{f.cam_index}: frame_num={f.frame_num}" for f in group])
```

---

## 开发建议（稳妥的验证路径）

1) `--list` 确认枚举 OK

2) 纯软件触发低频（例如 5fps）跑通保存/写盘/分析链路

3) 再切到硬触发 / master-slave

4) 以 `metadata.jsonl` 为准做判定：
- `lost_packet == 0`
- `dropped_groups == 0`
- 每组帧数等于相机数

5) 需要自动出报告可用：`python -m mvs.apps.analyze_capture_run`

