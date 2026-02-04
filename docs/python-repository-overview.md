# MVS_Deployment 代码库总结（mvs + tennis3d）

**最后更新**：2026 年 1 月（已升级为多球鲁棒 3D 输出）
**Python 版本**：3.10+（见 `pyproject.toml` 的 `requires-python`）

---

## 目录

1. [介绍与背景](#介绍与背景)
2. [系统架构](#系统架构)
3. [mvs：采集与组包核心模块](#mvs采集与组包核心模块)
4. [tennis3d：检测 + 多球 3D 定位核心模块](#tennis3d检测--多球-3d-定位核心模块)
5. [数据流与工作流](#数据流与工作流)
6. [输出数据契约（JSONL）](#输出数据契约jsonl)
7. [配置与运行方式](#配置与运行方式)
8. [测试与开发](#测试与开发)
9. [性能与优化](#性能与优化)
10. [常见问题](#常见问题)
11. [附录](#附录)

---

## 介绍与背景

### 项目目标

本仓库包含两条主线能力：

1) 在 Python 环境下实现海康工业相机的**多相机同步采集与组包**（常见 2~4 路）：

- 同一时刻 4 张图片（基于相同的触发事件）
- 30fps 或更低帧率（取决于网络带宽）
- 保存为 BMP/RAW，并记录精确时间戳
- 用于后续的推理、处理或分析

2) 在采集到的多相机图像上完成**网球检测 + 多球鲁棒三角化**，输出网球 3D 世界坐标（在线/离线均支持）：

- 在线：`python -m tennis3d.apps.online`（实时取流 → 检测 → 多球 3D → JSONL）
- 离线：`python -m tennis3d.apps.offline_localize_from_captures`（读 captures/metadata.jsonl → 检测 → 多球 3D → JSONL）

### 核心挑战

| 挑战 | 解决方案 |
|------|--------|
| **带宽限制** | 4×2448×2048@30fps 在 1GbE 下不可行 → 建议降帧率/ROI/多网卡 |
| **时间同步** | 不用主机到达时间，而是用相机的 `nTriggerIndex` 字段分组 |
| **触发同步** | 优先硬件外触发（保证同一曝光时刻），次选 PTP + Scheduled Action |
| **SDK 可用性** | 工作区内缺失 dll → 延迟加载绑定，清晰报错与重定向输出 |

### 技术栈

- **语言**：Python 3.10+（见 `pyproject.toml` 的 `requires-python`；类型提示、dataclass 等特性）
- **SDK**：海康 MVS Python ctypes 示例绑定（MvImport）
- **线程**：`threading` 进行异步取流与软触发
- **时间**：`time.monotonic()` 用于性能统计，相机的 `dev_timestamp` 用于帧对齐

---

## 系统架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      应用层（CLI）                           │
│  采集：python -m mvs.apps.quad_capture                         │
│  在线 3D：python -m tennis3d.apps.online                       │
│  离线 3D：python -m tennis3d.apps.offline_localize_from_captures│
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    mvs：采集与同步组包                         │
│  • load_mvs_binding() → 延迟加载 MVS ctypes 绑定               │
│  • open_quad_capture() → QuadCapture（取流 + 分组）            │
│  • save_frame_as_bmp() → SDK 保存 BMP                         │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   ┌─────────┐    ┌──────────┐    ┌────────────┐
   │ pipeline │    │ grouping │    │   grab     │
   │   模块    │    │   模块    │    │   模块     │
   └────┬────┘    └────┬─────┘    └────┬───────┘
        │             │               │
        └─────────────┼───────────────┘
                      ▼
        ┌─────────────────────────────┐
        │   相机控制核心（camera.py）    │
        │ • MvsCamera 生命周期管理      │
        │ • configure_trigger()       │
        │ • GigE 网络优化             │
        └────────┬────────────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
   ┌─────────┐      ┌──────────┐
   │binding. │      │devices.py│
   │  py     │      │  枚举    │
   └────┬────┘      └────┬─────┘
        │                │
        └────────┬───────┘
                 ▼
      ┌─────────────────────────┐
      │  MVS ctypes 绑定        │
      │  (MvImport 目录)        │
      └────────┬────────────────┘
               │
               ▼
      ┌─────────────────────────┐
      │  MvCameraControl.dll    │
      │  (硬件驱动库)            │
      └─────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │               tennis3d：检测 + 多相机几何 + 多球定位            │
    │  • pipeline.sources: 在线/离线统一产出 (meta, images_by_camera) │
    │  • detectors: fake/color/pt/rknn → detect(img)->list[Detection] │
    │  • geometry: 标定/投影/DLT 三角化/重投影误差                     │
    │  • localization: localize_balls（多球鲁棒匹配/去重/冲突消解）      │
    │  • pipeline.core: run_localization_pipeline → 输出 JSONL         │
    └─────────────────────────────────────────────────────────────┘
```

### 执行流（时序图）

```
应用              binding   SDK    camera    grabber   assembler
  │                 │       │        │          │          │
  ├─load_binding──→ │       │        │          │          │
  │                 ├─init──│──→    │          │          │
  │                 │       │ enum  │          │          │
  ├─open_quad_cap──→│       ├──→    │          │          │
  │                 │       │    CreateHandle │          │
  │                 │       │    OpenDevice   │          │
  │                 │       │    StartGrab─→  ├─start──→ │
  │                 │       │       │         │ pull  │  │
  ├─get_next_group──│───────│───────│─────────┤ frames├─→│
  │                 │       │       │         │    │  │  add()
  │ (per frame)    │       │       │         │  queue   │
  │                 │       │       │         │    ├─────┤
  │ (waits for)    │       │       │         │    │  ├─group ready
  │ (4 frames)     │       │       │         │    │  │
  ├─(get_next_    │       │       │         │    │  │
  │   group ok)    │       │       │         │    │  │
  │ apps/tools save│      │       │         │    │  │
  │                 │       │       │         │    │  │
  ├─close()────────│───────│───────├─StopGrab│    │  │
  │                 │       │   CloseDevice  │    │  │
  │                 │       │   DestroyHandle│    │  │
  │                 │       ├─finalize       │    │  │
```

---

## mvs：采集与组包核心模块

### 1. binding.py - MVS 绑定加载

**职责**：处理 DLL 搜索路径，延迟加载 MvImport 绑定。

**关键类**：
- `MvsBinding`：已加载的绑定符号集合（只读数据类）
- `MvsDllNotFoundError`：DLL 未找到异常

**关键函数**：
- `load_mvs_binding(dll_dir=None) → MvsBinding`
  - 尝试多个 DLL 搜索位置（用户指定 > 环境变量 > 工作区候选）
  - 若失败，抛出清晰的中英文提示信息

**设计亮点**：
- 延迟 import，让包即使缺 DLL 也能被 import（只在调用时才报错）
- 自动探测系统架构（win32/win64），无需用户手工指定

---

### 2. devices.py - 设备枚举

**职责**：调用 `MV_CC_EnumDevices`，返回简化的设备描述。

**关键类**：
- `DeviceDesc`：设备信息（index, model, serial, ip, tlayer_type）

**关键函数**：
- `enumerate_devices(binding) → (st_dev_list, List[DeviceDesc])`
  - 返回 SDK 的原始 device list（供后续 CreateHandle 使用）
  - 以及简化后的描述列表（供用户阅读）

---

### 3. camera.py - 相机生命周期

**职责**：相机句柄创建、打开、触发配置、取流开关、关闭回收。

**关键类**：
- `MvsCamera`：封装已打开的相机（不直接构造，使用 `open_from_device_list`）
- `MvsSdk`：SDK 全局初始化/反初始化（上下文管理器）

**关键函数**：
- `configure_trigger(binding, cam, trigger_source, trigger_activation, trigger_cache_enable)`
  - 配置 TriggerMode/Source/Activation/Delay/CacheEnable
  - 失败时部分忽略（如 CacheEnable 某些机型不支持）

- `_best_effort_gige_network_tuning(binding, cam)`
  - 自动调优 GigE 参数（最优 packet size、重发策略）

**工作流**：
```python
sdk = MvsSdk(binding)
sdk.initialize()  # MV_CC_Initialize
try:
    cam = MvsCamera.open_from_device_list(...)  # CreateHandle → Open → Configure → StartGrab
    # ... 取流 ...
finally:
    cam.close()  # StopGrab → Close → Destroy
    sdk.finalize()  # MV_CC_Finalize
```

---

### 4. grab.py - 异步取流

**职责**：启动后台线程拉取图像，放入队列供上层处理。

**关键类**：
- `FramePacket`：单帧信息（cam_index, serial, trigger_index, dev_timestamp, host_timestamp, width, height, pixel_type, frame_len, lost_packet, arrival_monotonic, data）
- `Grabber`：单相机取流线程

**关键方法**：
- `Grabber.run()`（后台线程）
  - 循环调用 `MV_CC_GetImageBuffer(st_frame, timeout_ms)`
  - 提取帧元数据（宽、高、像素类型、时间戳、trigger_index）
  - 复制数据到 Python bytes
  - 调用 `MV_CC_FreeImageBuffer` 回收 SDK 缓存
  - 将 FramePacket 放入队列（超时/满时丢弃，不中断取流）

**特殊处理**：
- 如果取流失败（超时/无数据），继续循环（不抛异常）
- 如果队列满，丢弃当前帧（避免队列堆积导致延迟）

---

### 5. grouping.py - trigger_index 分组

**职责**：接收单个 FramePacket，按 trigger_index 聚合成 4 张一组。

**关键类**：
- `TriggerGroupAssembler`

**关键方法**：
- `add(pkt: FramePacket) → Optional[List[FramePacket]]`
  - 若这个 pkt 凑齐了一组（4 台相机），返回该组并清理
  - 否则返回 None，继续等待
  - 自动清理过期/超额的 pending 组（并统计 `dropped_groups`）

**参数**：
- `group_timeout_s`：单组等待超时（默认 0.2s）
- `max_pending_groups`：最多缓存多少个未凑齐的组（防止内存溢出）

**统计指标**：
- `dropped_groups`：因超时或内存限制而丢弃的组数量

---

### 6. soft_trigger.py - 软触发循环

**职责**：当 `trigger_source=Software` 时，按固定频率下发 `TriggerSoftware` 命令。

**关键类**：
- `SoftwareTriggerLoop`：后台线程

**用途**：
- 链路验证（不保证 4 台同一曝光时刻，只用于测试）
- 生产环境**不推荐**，应使用硬件外触发

---

### 7. save.py - 图像保存

**职责**：使用 SDK 提供的 `MV_CC_SaveImageToFileEx` 保存 BMP。

**关键函数**：
- `save_frame_as_bmp(binding, cam, out_path, frame, bayer_method=2)`
  - 创建 `MV_SAVE_IMAGE_TO_FILE_PARAM_EX` 结构
  - 复制图像数据到 ctypes buffer
  - 调用 SDK 保存（支持 Bayer→RGB 插值）
  - Windows 路径使用 mbcs 编码，避免乱码

**异常处理**：
- 若 SDK 返回错误，上层通常会捕获并降级为保存 RAW

---

### 8. pipeline.py - 四相机采集管线

**职责**：整合前述模块，提供最小可用的"打开 4 台相机→抓取→分组"完整流程。

**关键类**：
- `QuadCapture`：数据类，包含 binding/sdk/cameras/grabbers/assembler/stop_event/frame_queue
  - 方法 `get_next_group(timeout_s)`：获取下一组 4 张图
  - 方法 `close()`：清理资源
  - 支持 `with` 语句

**关键函数**：
- `open_quad_capture(...) → QuadCapture`
  - 接收 binding、4 个 serial、触发配置等参数
  - 枚举设备、打开 4 台相机、启动 Grabber 线程、创建 Assembler
  - 若指定了 `enable_soft_trigger_fps > 0`，启动软触发线程
  - 返回 QuadCapture 对象（上下文管理器）

---

## tennis3d：检测 + 多球 3D 定位核心模块

`tennis3d` 是业务侧库：把每个相机的 2D 检测结果融合为 3D 世界坐标，并提供在线/离线两种入口。

### 入口（apps）

- 在线：`src/tennis3d/apps/online/entry.py`
  - 从 `mvs.open_quad_capture()` 实时取流并组包
  - 调用 `tennis3d.pipeline.iter_mvs_image_groups()` 转为 `images_by_camera`
  - 调用 `tennis3d.pipeline.run_localization_pipeline()` 输出 JSONL

- 离线：`src/tennis3d/apps/offline_localize_from_captures.py`
  - 读取 `captures_dir/metadata.jsonl` + 图像
  - 调用 `tennis3d.pipeline.iter_capture_image_groups()` 产出同步组
  - 其余流程与在线一致

### 配置（config）

配置模型位于 `src/tennis3d/config.py`，支持 `.json/.yaml/.yml`：

- `OfflineAppConfig`：离线入口参数
- `OnlineAppConfig`：在线入口参数（含 trigger 子配置）

多球鲁棒定位相关的关键参数（会在本文后面的“配置与运行方式”与“输出契约”中再次说明）：

- `require_views`：候选球至少需要多少个相机视角参与
- `max_detections_per_camera`：每相机最多取 topK 个候选（抑制组合爆炸）
- `max_reproj_error_px`：最大重投影误差阈值（像素）
- `max_uv_match_dist_px`：投影补全匹配的最大像素距离阈值
- `merge_dist_m`：3D 去重阈值（米）

### 检测器适配（detectors）

`src/tennis3d/detectors.py` 提供统一接口：

- `create_detector(...)`：根据配置创建后端（fake/color/pt/rknn）
- 统一方法：`detect(img_bgr) -> list[Detection]`

检测输出数据结构：`src/tennis3d/offline/models.py::Detection`，包含 `bbox/score/cls` 与 `center` 计算属性。

### 几何与标定（geometry）

- `src/tennis3d/geometry/calibration.py`
  - `load_calibration()`：读取 JSON/YAML 标定
  - 外参约定：world→camera，$X_c = R_{wc} X_w + t_{wc}$
  - 投影矩阵：$P = K [R_{wc}|t_{wc}]$

- `src/tennis3d/geometry/triangulation.py`
  - `triangulate_dlt()`：2~N 视角 DLT 三角化
  - `reprojection_errors()`：重投影误差评估（像素）
  - `project_point()`：用于投影补全匹配

### 多球定位融合（localization）

核心入口：`src/tennis3d/localization/localize.py::localize_balls()`。

设计目标是解决实机常见问题：同一同步组里可能有多个球，同时检测器可能误检。为此采用“几何一致性优先”的策略：

1) **两视角生成种子**：枚举两相机的 topK 检测组合，先用 DLT 得到 3D 种子。
2) **投影补全匹配**：把 3D 点投影到其它相机，在其检测中心中找最近邻，且距离小于 `max_uv_match_dist_px` 才算匹配。
3) **几何 gating**：对使用的视角计算重投影误差，若最大误差大于 `max_reproj_error_px` 则丢弃；并要求各视角深度为正。
4) **3D 去重**：不同两视角组合可能得到同一球的重复解，用 `merge_dist_m` 做 3D-NMS 去重。
5) **冲突消解**：同一相机同一 detection 不允许被多个 3D 球复用，按 `quality` 从高到低贪心选择。

该过程最终输出 0..N 个 `BallLocalization`（按 `quality` 排序）。

### 流水线编排（pipeline）

- `src/tennis3d/pipeline/sources.py`
  - 在线/离线统一产出 `(meta, images_by_camera)`
- `src/tennis3d/pipeline/core.py`
  - `run_localization_pipeline()`：每组做检测并调用 `localize_balls()`，输出可 JSON 序列化记录

---

## 数据流与工作流

### 典型采集循环

```python
from mvs import open_quad_capture

with open_quad_capture(...) as cap:
    while True:
        group = cap.get_next_group(timeout_s=1.0)

        if group is None:
            # 超时：可能某台相机卡了，或网络拥塞
            print("timeout")
            continue

        # group 是 List[FramePacket]，长度为 4，都有相同的 trigger_index
        for frame in group:
            print(f"cam{frame.cam_index}: {frame.width}x{frame.height}")
            # 处理图像数据（frame.data 是 bytes）
```

### 内部数据流

```
相机硬件
  │ MV_CC_GetImageBuffer
  ▼
Grabber 线程
  │ 每帧 → FramePacket
  ▼
frame_queue (Queue[FramePacket])
  │ 生产者：Grabber
  │ 消费者：应用层 get_next_group()
  ▼
TriggerGroupAssembler
  │ 按 trigger_index 分组
  ▼
应用层
  │ 处理 4 张同步图
  ▼
保存/推理
```

### 时间戳对齐策略

| 时间戳来源 | 精度 | 用途 |
|----------|------|------|
| `dev_timestamp` | 微秒级（设备端） | **主要用途**：帧对齐、触发时刻判断 |
| `host_timestamp` | 毫秒级（主机端） | 诊断用：检测网络/程序延迟 |
| `trigger_index` | 无单位（计数） | **分组键**：确保 4 张图来自同一触发 |
| `arrival_monotonic` | 秒（主机单调时钟） | 性能统计 |

**最佳实践**：
```python
for frame in group:
    # 用 dev_timestamp 做精确对齐
    precise_time = frame.dev_timestamp  # 微秒

    # 用 trigger_index 确认是同一次触发
    assert frame.trigger_index == group[0].trigger_index

    # host_timestamp 仅用于调试
    host_delay = frame.host_timestamp - frame.dev_timestamp
    # 若 host_delay 很大，说明有网络/程序延迟
```

---

## 输出数据契约（JSONL）

`tennis3d.pipeline.run_localization_pipeline()` 的输出建议以 **JSONL** 落盘：

- **一行 = 一个同步组**（同一次触发/同一组序列）
- 每行都包含 `balls: [] | [ ... ]`
  - `[]` 表示该组没有任何跨视角一致的 3D 解（可能误检过多、视角不足或不满足阈值）
  - 列表长度为 N 表示该同步组里存在 N 个“几何一致”的球（多球场景）

### 顶层字段

- `created_at`: `float`，Unix epoch 秒（用于落盘时间，不等价于相机曝光时间）
- `meta...`: 来自 source 的可序列化元信息
  - 在线常见：`group_index`
  - 离线常见：`group_seq` / `group_by` / `trigger_index`
- `balls`: `list[dict]`

### balls[*] 字段

每个 ball dict 目前包含：

- `ball_id`: `int`（在该同步组内的排序编号，按 `quality` 从高到低）
- `ball_3d_world`: `[x, y, z]`（世界坐标，纯 Python float）
- `ball_3d_camera`: `{serial: [x, y, z]}`（各相机坐标系下的 3D 点）
- `used_cameras`: `list[str]`（参与三角化/验证的相机 serial 列表）
- `quality`: `float`（综合质量评分，用于排序与冲突消解；值越大越好）
- `num_views`: `int`（等价于 `len(used_cameras)`）
- `median_reproj_error_px`: `float`（中位数重投影误差）
- `max_reproj_error_px`: `float`（最大重投影误差，用于阈值 gating）
- `reprojection_errors`: `list[dict]`
  - 每项包含：`camera` / `uv` / `uv_hat` / `error_px`
- `detections`（可选）：`{serial: {bbox, score, cls, center}}`
  - 是否输出由 `include_detection_details` 控制

### 示例（单行 JSONL，省略部分字段）

```json
{
  "created_at": 1738212345.12,
  "group_index": 12,
  "balls": [
    {
      "ball_id": 0,
      "ball_3d_world": [1.234, 0.567, 2.345],
      "used_cameras": ["DA8199285", "DA8199303", "DA8199402"],
      "quality": 0.91,
      "num_views": 3,
      "median_reproj_error_px": 1.8,
      "max_reproj_error_px": 3.2
    }
  ]
}
```

---

## 配置与运行方式

### 环境配置（MVS SDK 与 Python 环境）

**Step 1：安装 MVS SDK**

下载海康 MVS 软件（Windows 版），安装后确保 `MvCameraControl.dll` 在系统 PATH 中。或指定 DLL 路径：

```bash
set MVS_DLL_DIR=C:\Program Files\Hikvision\MVS\bin\win64
```

**Step 1.5：创建 Python 环境（推荐 uv）**

仓库包含 `pyproject.toml` 与 `uv.lock`，推荐使用 uv 保持依赖一致。

```bash
uv venv
uv sync
```

**Step 2：配置网络**

将 4 台相机连接到网络交换机，为每台配置 IP 地址（通常 192.168.x.x）。

**Step 3：查找相机序列号**

```bash
python -m mvs.apps.quad_capture --list
```

输出：
```
[0] model=MV-CS050-10GC serial=DA8199285 name= ip=192.168.0.10 tlayer=0x00000001
[1] model=MV-CS050-10GC serial=DA8199303 name= ip=192.168.0.11 tlayer=0x00000001
...
```

### 第一次采集（验证链路）

使用软触发 15fps，仅采集 10 组：

```bash
python -m mvs.apps.quad_capture \
  --serial DA8199285 DA8199303 DA8199402 DA8199??? \
  --trigger-source Software \
  --soft-trigger-fps 15 \
  --save-mode raw \
  --max-groups 10
```

检查 `data/captures/<run>/metadata.jsonl` 中的 `dropped_groups`，若为 0，说明链路正常。

### 生产采集（硬件外触发）

连接外部脉冲信号到相机的 Line0，执行：

```bash
python -m mvs.apps.quad_capture \
  --serial DA8199285 DA8199303 DA8199402 DA8199??? \
  --trigger-source Line0 \
  --trigger-activation FallingEdge \
  --save-mode sdk-bmp \
  --max-groups 1000
```

---

### 3D 定位（tennis3d）

配置文件与示例位于：

- `configs/`：项目内更偏“可直接运行/更贴近真实部署”的配置（常绑定你的真实标定与 serial）。
- `examples/configs/`：仅保留模板目录（不再放可运行示例配置）。
- `examples/configs/templates/`：模板（有占位符，需要你填 serial/标定/路径等）。

#### 离线：从 captures/metadata.jsonl 输出多球 3D（JSONL）

```bash
python -m tennis3d.apps.offline_localize_from_captures --config configs/offline_pt_windows_cpu.yaml
```

常用输出文件会写到 `data/tools_output/`（可在 config 中修改 `out_jsonl`）。

#### 在线：实时取流 → 检测 → 多球定位 → JSONL

```bash
python -m tennis3d.apps.online --config configs/online_pt_windows_cpu_software_trigger.yaml
```

在线模式会通过 `mvs` 打开相机、分组，然后对每个同步组输出一行 JSONL（包含 `balls`）。

---

## 测试与开发

### 运行测试

本仓库使用 pytest，测试用例在 `tests/`：

```bash
pytest -q
```

覆盖重点包括：

- 标定文件解析（YAML/JSON）
- capture 元数据重排与路径稳健解析
- DLT 三角化与几何一致性
- **多球定位**：多球输出、误检抑制、冲突消解（同一 detection 不复用）

### API 速查（mvs）

### mvs.load_mvs_binding

```python
def load_mvs_binding(*, dll_dir: Optional[str] = None) -> MvsBinding:
    """加载 MVS 绑定。

    Args:
        dll_dir: 可选，包含 MvCameraControl.dll 的目录

    Returns:
        MvsBinding: 已加载的绑定对象

    Raises:
        MvsDllNotFoundError: 若 DLL 无法找到
    """
```

### mvs.enumerate_devices

```python
def enumerate_devices(binding: Any) -> Tuple[Any, List[DeviceDesc]]:
    """枚举所有可用设备。

    Args:
        binding: MVS 绑定对象

    Returns:
        (st_dev_list, descs)：
        - st_dev_list: SDK 原始设备列表（供 CreateHandle 使用）
        - descs: DeviceDesc 列表（用户可读）
    """
```

### mvs.MvsCamera / mvs.MvsSdk

```python
class MvsCamera:
    @classmethod
    def open_from_device_list(
        cls,
        *,
        binding: Any,
        st_dev_list: Any,
        dev_index: int,
        serial: str,
        tlayer_type: int,
        trigger_source: str,
        trigger_activation: str,
        trigger_cache_enable: bool,
    ) -> "MvsCamera":
        """打开一台相机。"""

    def close(self) -> None:
        """关闭相机。"""

class MvsSdk:
    def initialize(self) -> None:
        """SDK 初始化。"""

    def finalize(self) -> None:
        """SDK 反初始化。"""
```

### mvs.open_quad_capture / mvs.QuadCapture

```python
def open_quad_capture(
    *,
    binding: MvsBinding,
    serials: Sequence[str],
    trigger_source: str,
    trigger_activation: str,
    trigger_cache_enable: bool,
    timeout_ms: int = 1000,
    group_timeout_ms: int = 200,
    max_pending_groups: int = 256,
    enable_soft_trigger_fps: float = 0.0,
) -> QuadCapture:
    """打开四相机采集。"""

class QuadCapture:
    def get_next_group(self, timeout_s: float = 0.5) -> Optional[List[FramePacket]]:
        """获取下一组 4 张图。"""

    def close(self) -> None:
        """关闭采集。"""

    def __enter__(self) -> "QuadCapture": ...
    def __exit__(self, exc_type, exc, tb) -> None: ...
```

### mvs.save_frame_as_bmp

```python
def save_frame_as_bmp(
    *,
    binding: Any,
    cam: Any,
    out_path: Path,
    frame: FramePacket,
    bayer_method: int = 2,
) -> None:
    """使用 SDK 保存 BMP。

    Args:
        binding: MVS 绑定
        cam: 相机对象
        out_path: 输出文件路径
        frame: FramePacket 对象
        bayer_method: Bayer 插值方法 (0-3)
    """
```

---

## 性能与优化

### 带宽分析

4 台 2448×2048 Bayer8 相机：

- **单帧大小**：2448 × 2048 × 1 byte = 5,012,480 bytes ≈ 4.78 MB
- **4 台单帧**：≈ 19.1 MB
- **30fps**：19.1 × 30 ≈ 573 MB/s = **4.58 Gbps**（网络流量）

在 **1 GbE**（约 125 MB/s = 1 Gbps 实际）下：

- 可支持帧率：1000 Mbps ÷ 4.58 Gbps ≈ **0.2 fps**（全分辨率时几乎无法取流）

### 可行性方案

| 方案 | 帧率 | 带宽占用 | 可行性 |
|------|------|--------|--------|
| 全分辨率 30fps | 30 | 4.58 Gbps | ❌ 1GbE 不支持 |
| 全分辨率 5fps | 5 | 0.76 Gbps | ✅ 1GbE 可用 |
| ROI 50% (1224×2048) 15fps | 15 | 0.72 Gbps | ✅ 1GbE 勉强 |
| 多网卡 2×1GbE | 30 | 2.29 Gbps/网卡 | ✅ 理论可行 |
| 10GbE | 30 | 0.46 Gbps | ✅ 充裕 |

### 优化建议

1. **网络层**
   - 启用 GigE 最优 packet size（包已自动处理）
   - 考虑多网卡或 10GbE 升级
   - 减少网络丢包（交换机配置、网线质量）

2. **相机配置**
   - ROI：仅拉取感兴趣区域
   - 像素格式：从 Bayer12 改为 Bayer8
   - 帧率：从 30fps 降到 15fps 或更低

3. **主机程序**
   - 增加 `queue.maxsize` 避免丢帧
   - 异步存图/推理，不阻塞取流
   - 监控 `lost_packet` 指标

4. **硬件外触发**
   - 相比软触发，硬件外触发减少上位机往返延迟
   - 同时保证 4 台相机曝光时刻一致

---

## 常见问题

### Q：启动时报 "MvCameraControl.dll not found"

**A：** 检查以下几点：

1. 海康 MVS 是否已安装
2. 若已安装，`MvCameraControl.dll` 是否在 `C:\Program Files\Hikvision\MVS\Bin\win64` 中
3. 若在非标准位置，使用 `--dll-dir` 或 `MVS_DLL_DIR` 环境变量

例：
```bash
set MVS_DLL_DIR=D:\MVS\Bin\win64
python -m mvs.apps.quad_capture --list
```

### Q：相机在列表中，但采集时没有出图

**A：** 检查触发配置：

1. 若使用 `--trigger-source Software`，确保设置了 `--soft-trigger-fps`（如 15）
2. 若使用 `--trigger-source Line0`，确保外部设备已连接并在发送脉冲
3. 查看打印的心跳日志，若 `qsize` 一直为 0，说明相机未出图

### Q：采集时 `dropped_groups` 不为 0

**A：** 原因和解决方案：

| 现象 | 可能原因 | 解决方案 |
|------|--------|--------|
| `dropped_groups` 逐渐增加 | 某台相机慢/丢帧，4 台无法凑齐 | 增加 `--group-timeout-ms`；检查网络/相机 |
| `dropped_groups` 突增 | 缓存满了（`max_pending_groups`） | 增加 `--max-pending-groups` |
| 同时看到 `lost_packet > 0` | GigE 网络丢包 | 检查交换机、网线、GigE 配置 |

### Q：内存占用逐渐增长

**A：** 可能原因：

1. 队列堆积（Grabber 生产快，处理慢）→ 减少其他工作负载，或加快处理速度
2. 保存文件很慢 → 考虑异步保存或降低采集帧率
3. 内存泄漏 → 确保相机/SDK 正确清理（调用 `close()` 和 `finalize()`）

---

## 附录

### A. 文件树

```
MVS_Deployment/
├── pyproject.toml                      # Python 项目元数据与依赖声明
├── uv.lock                             # uv 锁定文件（推荐用 uv sync）
├── src/
│   ├── mvs/                            # 多相机采集与同步组包（ctypes 绑定 MVS SDK）
│   └── tennis3d/                       # 检测 + 多相机几何 + 多球 3D 定位
├── tools/                              # 工具脚本（采集/分析/可视化等）
├── configs/                            # 在线/离线配置模板（yaml）
├── data/
│   ├── calibration/                    # 示例标定文件
│   └── captures_master_slave/          # 示例采集数据（如存在）
├── models/                             # 推理模型（.pt/.rknn）
├── tests/                              # pytest 测试
├── docs/                               # 文档
├── SDK_Development/                    # 海康 SDK 随附文件/示例（供应商目录）
└── 工业相机Windows SDK开发指南V4.7.0（Python）.chm
```

### B. 关键常数与字段

#### TriggerSource（触发源）

- `Line0/Line1/Line2/Line3`：外部输入（推荐严格同步）
- `Software`：上位机下发命令
- `FrequencyConverter`：频率转换器
- `PTP`：精密时间协议（高端相机支持）

#### TriggerActivation（触发沿）

- `FallingEdge`：下降沿（本仓库默认）
- `RisingEdge`：上升沿
- `LevelHigh`/`LevelLow`：电平

#### PixelType（像素格式，示例）

- `0x01080001`：Mono8
- `0x010c0004`：Bayer8-GR（常用）
- `0x010c0003`：Bayer8-RG

### C. 时间戳精度

| 时间戳 | 精度 | 范围 |
|-------|------|------|
| `dev_timestamp` | 微秒 | 64 位（约 584,942 年） |
| `host_timestamp` | 毫秒 | 64 位（约 584,942,417 年） |
| `arrival_monotonic` | 秒 | float64（约 270 万年） |

### D. 错误代码

SDK 返回的错误码通常为 0x 开头的十六进制。常见：

- `0x00000000`：MV_OK（成功）
- `0xA0000001`：MV_E_HANDLE（句柄无效）
- `0xA0000004`：MV_E_NOTINITLIZED（未初始化）
- `0xA0000201`：MV_E_UNKNOWERR（未知错误）

详细列表见 `MvErrorDefine_const.py`。

### E. 相关文档链接

- 海康 MVS 官方文档：工作区 `SDK_Development/Documentations/` 中的 CHM 文件
- Python 官方 threading：https://docs.python.org/3/library/threading.html
- ctypes 绑定：https://docs.python.org/3/library/ctypes.html

---

**文档结束**

若有问题或建议，欢迎反馈。

