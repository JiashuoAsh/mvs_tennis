# MVS 采集包 - 快速参考卡

## 安装与初始化

```bash
# 1. 设置 DLL 路径（选一个）
set MVS_DLL_DIR=C:\path\to\mvs\bin  # Windows

# 注意：MVS_DLL_DIR 填“目录”，不要填到具体的 MvCameraControl.dll 文件路径；
# 且需要与 Python 位数匹配（64 位 Python → Win64_x64 / win64）。
# 示例：set MVS_DLL_DIR=C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64

# 2. 列举相机
python -m mvs.apps.quad_capture --list

# 3. 采集数据
python -m mvs.apps.quad_capture --serial SN0 SN1 SN2 SN3 [options]
```

---

## CLI 常用命令

### 验证链路（15fps 软触发，仅 10 组）

```bash
python -m mvs.apps.quad_capture \
  --serial DA8199285 DA8199303 DA8199402 DA8199??? \
  --trigger-source Software --soft-trigger-fps 15 \
  --save-mode raw --max-groups 10
```

### 生产采集（硬件外触发，保存 BMP）

```bash
python -m mvs.apps.quad_capture \
  --serial DA8199285 DA8199303 DA8199402 DA8199??? \
  --trigger-source Line0 --trigger-activation RisingEdge \
  --save-mode sdk-bmp --max-groups 1000
```

### 仅获取元数据（无图片保存）

```bash
python -m mvs.apps.quad_capture \
  --serial DA8199285 DA8199303 DA8199402 DA8199??? \
  --trigger-source Software --soft-trigger-fps 30 \
  --save-mode none --max-groups 100
```

---

## Python 代码示例

### 最小示例

```python
from mvs import open_quad_capture, load_mvs_binding

binding = load_mvs_binding()
with open_quad_capture(
    binding=binding,
    serials=["SN0", "SN1", "SN2", "SN3"],
    trigger_source="Software",
    trigger_activation="RisingEdge",
    enable_soft_trigger_fps=15,
) as cap:
    group = cap.get_next_group(timeout_s=1.0)
    if group:
        for frame in group:
            print(f"cam{frame.cam_index}: {frame.width}x{frame.height}")
```

### 自定义处理

```python
from mvs import open_quad_capture, save_frame_as_bmp, load_mvs_binding
from pathlib import Path

binding = load_mvs_binding()
with open_quad_capture(binding, serials=[...]) as cap:
    for _ in range(100):
        group = cap.get_next_group()
        if not group:
            continue

        # 处理 4 张图
        for frame in group:
            # 自定义处理
            img = process_raw_data(frame.data, frame.width, frame.height)

            # 保存 BMP
            bmp_path = Path(f"cam{frame.cam_index}.bmp")
            save_frame_as_bmp(binding, cam=cap.cameras[frame.cam_index].cam,
                            out_path=bmp_path, frame=frame)
```

---

## 关键 API

| 模块 | 函数/类 | 说明 |
|------|--------|------|
| **binding** | `load_mvs_binding(dll_dir)` | 加载 MVS 绑定 |
| **devices** | `enumerate_devices(binding)` | 枚举设备 |
| **camera** | `MvsSdk.initialize()` | SDK 初始化 |
| | `MvsCamera.open_from_device_list()` | 打开相机 |
| | `configure_trigger(...)` | 配置触发 |
| **pipeline** | `open_quad_capture(...)` | 打开四机采集（推荐） |
| | `QuadCapture.get_next_group()` | 获取下一组 |
| **save** | `save_frame_as_bmp(...)` | 保存 BMP |

---

## 数据结构

### FramePacket

```python
@dataclass
class FramePacket:
    cam_index: int           # 相机索引 (0-3)
    trigger_index: int       # 触发计数（用于分组）⭐
    dev_timestamp: int       # 设备时间戳（微秒）⭐
    host_timestamp: int      # 主机时间戳（毫秒）
    width: int               # 图像宽度
    height: int              # 图像高度
    frame_len: int           # 数据长度（字节）
    lost_packet: int         # 丢包计数
    data: bytes              # 图像数据
```

---

## 性能指标

| 指标 | 含义 | 正常范围 |
|------|------|--------|
| `dropped_groups` | 无法凑齐的组数 | **0 或非常小** |
| `lost_packet` | 每帧丢包数 | **0** |
| `trigger_index` | 触发计数 | 连续递增 |
| `qsize` | 队列深度 | < 100（监测程序处理延迟） |

---

## 故障排查

| 问题 | 可能原因 | 解决方案 |
|------|--------|--------|
| 找不到 DLL | 未安装 MVS 或 DLL 路径错误 | `set MVS_DLL_DIR=...` |
| 相机枚举成功但无出图 | 触发源配置错误或未发送触发信号 | 检查 `--trigger-source` 与硬件连接 |
| `dropped_groups > 0` | 某台相机慢/丢帧/网络拥塞 | 增加 `--group-timeout-ms` 或检查网络 |
| `lost_packet > 0` | GigE 网络丢包 | 检查交换机配置、网线质量 |
| 内存占用增长 | 程序处理慢或队列堆积 | 加快处理速度或降低采集帧率 |

---

## 触发配置速查表

| 场景 | trigger_source | trigger_activation | 备注 |
|------|----------------|------------------|------|
| 硬件外触发（推荐） | Line0 | RisingEdge | 相机收到外部脉冲，最准确 |
| 软件测试 | Software | - | 用 `--soft-trigger-fps` 控制频率 |
| PTP 同步 | PTP | - | 仅高端相机支持 |

---

## 目录结构

```
mvs/
├── __init__.py          ← 对外 API
├── binding.py           ← DLL 加载
├── camera.py            ← 相机生命周期
├── devices.py           ← 设备枚举
├── grab.py              ← 取流线程
├── grouping.py          ← 分组器
├── pipeline.py          ← 管线 ⭐
├── save.py              ← 保存 BMP
├── soft_trigger.py      ← 软触发
└── README.md            ← 包文档

src/mvs/apps/
├── quad_capture.py           ← CLI ⭐（python -m mvs.apps.quad_capture）
└── analyze_capture_run.py    ← CLI ⭐（python -m mvs.apps.analyze_capture_run）

examples/
└── quad_capture_demo.py ← 示例 ⭐

docs/
├── python-repository-overview.md     ← 完整文档 ⭐
└── PROJECT_COMPLETION_SUMMARY.md     ← 项目总结
```

---

## 环境变量

| 变量名 | 含义 | 示例 |
|--------|------|------|
| `MVS_DLL_DIR` | MvCameraControl.dll 所在目录 | `C:\Program Files\Hikvision\MVS\Bin\win64` |
| `PATH` | 系统路径（包含 DLL 目录） | 自动追加 |

---

## 时间戳对齐建议

```python
for frame in group:
    # ✅ 用这个做精确对齐
    precise_time = frame.dev_timestamp  # 微秒，设备端

    # ✅ 用这个做分组确认
    group_key = frame.trigger_index  # 触发计数

    # ⚠️  这个仅用于调试
    network_delay = frame.host_timestamp - frame.dev_timestamp
```

---

## 常用参数

```bash
--serial SN0 SN1 SN2 SN3      # 4 个相机序列号
--trigger-source Line0        # 触发源：Line0/Line1/Software
--soft-trigger-fps 15         # 软触发频率（仅 Software 时生效）
--save-mode raw/sdk-bmp/none  # 保存模式
--max-groups 100              # 采集多少个组后退出（0=无限）
--output-dir ./captures       # 输出目录
--group-timeout-ms 200        # 等待凑齐 1 组的超时（毫秒）
--dll-dir PATH                # DLL 目录
```

---

## 更多信息

- 📖 完整文档：`docs/python-repository-overview.md`
- 📝 包文档：`mvs/README.md`
- 💻 示例代码：`examples/quad_capture_demo.py`
- 📋 项目总结：`docs/PROJECT_COMPLETION_SUMMARY.md`

