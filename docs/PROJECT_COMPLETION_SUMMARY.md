# MVS 四相机采集项目 - 完成总结

**完成日期**：2026 年 1 月 21 日
**状态**：✅ 可工程化部署

---

## 项目成果概览

### 核心交付物

| 交付物 | 说明 | 状态 |
|--------|------|------|
| **mvs 包** | 子包拆分（core/sdk/capture/session/analysis/apps + `mvs.__init__` 公共导出） | ✅ 完成 |
| **CLI 工具** | `python -m mvs.apps.quad_capture` 采集入口（位于 `src/mvs/apps/quad_capture.py`） | ✅ 完成 |
| **文档** | docs/ 下的完整项目文档 + mvs/README.md 包文档 | ✅ 完成 |
| **示例** | examples/quad_capture_demo.py 最小可运行示例 | ✅ 完成 |

### 功能覆盖

- ✅ **相机初始化**：SDK init/finalize、设备枚举、打开/关闭
- ✅ **触发配置**：硬件外触发（Line0~3）、软触发（用于验证）、参数化配置
- ✅ **严格同步**：按 `nTriggerIndex` 分组，确保 4 台同一时刻
- ✅ **取流与处理**：异步 Grabber 线程、队列缓冲、FramePacket 数据结构
- ✅ **时间戳**：设备 devtimestamp + 主机 hosttimestamp + 单调时钟 arrival
- ✅ **保存与降级**：SDK BMP 保存（支持 Bayer→RGB）、失败自动降 RAW
- ✅ **网络优化**：GigE packet size 自动调优、重发策略
- ✅ **错误处理**：清晰的 DLL 缺失提示、中英文错误信息、Windows 编码处理

---

## 工程化架构亮点

### 1. 关注点分离（SoC）

- **binding.py**：专职 DLL 加载与路径搜索
- **devices.py**：单一职责枚举设备
- **camera.py**：相机生命周期与配置
- **grab.py**：取流线程隔离
- **grouping.py**：分组逻辑不混杂其他
- **pipeline.py**：只负责流程编排，依赖其他模块

### 2. 模块化与复用

```python
# 业务层可直接调用（推荐：稳定公共导出）
from mvs import (
   FramePacket,
   Grabber,
   MvsCamera,
   MvsSdk,
   configure_trigger,
   enumerate_devices,
   load_mvs_binding,
   open_quad_capture,
   save_frame_as_bmp,
)
```

### 3. 上下文管理

```python
with open_quad_capture(...) as cap:
    # 自动初始化、启动线程、配置相机
    group = cap.get_next_group()
    # ...
# 自动关闭相机、停止线程、反初始化 SDK
```

### 4. 清晰的错误报告

缺失 DLL 时的输出示例：
```
MVS DLL not found: MvCameraControl.dll (or dependency).
找不到 MvCameraControl.dll（或其依赖）。

解决方法：
1) 安装海康 MVS（Machine Vision Software），确保系统 PATH 可找到 MvCameraControl.dll；
2) 或使用参数 dll_dir / 环境变量 MVS_DLL_DIR 指向 DLL 目录；
3) 提示：MvCameraControl.dll 通常位于 MVS 安装目录的 Runtime/Bin 之类路径。
```

---

## 使用案例

### 案例 1：快速验证链路（15fps 软触发）

```bash
python -m mvs.apps.quad_capture \
  --serial DA8199285 DA8199303 DA8199402 DA8199??? \
  --trigger-source Software --soft-trigger-fps 15 \
  --save-mode raw --max-groups 20
```

**预期**：metadata.jsonl 中 `dropped_groups=0`，验证网络/程序正常。

### 案例 2：生产采集（硬件外触发）

```bash
python -m mvs.apps.quad_capture \
  --serial DA8199285 DA8199303 DA8199402 DA8199??? \
   --trigger-source Line0 --trigger-activation FallingEdge \
  --save-mode sdk-bmp --max-groups 1000
```

**预期**：采集 1000 个同步组，每组 4 张 BMP，trigger_index 不重复。

### 案例 3：自定义处理（推理流程）

```python
from mvs import open_quad_capture, load_mvs_binding

binding = load_mvs_binding()

with open_quad_capture(binding, serials=[...]) as cap:
    for _ in range(100):
        group = cap.get_next_group()
        if group is None:
            continue

        # 自定义处理（如推理）
        images = [frame.data for frame in group]
        results = inference_model(images)

        # 保存结果
        save_results(results, trigger_index=group[0].trigger_index)
```

---

## 已知限制与建议

### 限制 1：带宽约束

**现状**：4×2448×2048@30fps = 4.58 Gbps > 1 GbE 容量

**建议方案**：
- ROI：仅采集感兴趣区域，减少数据量
- 降帧：从 30fps 改 15fps
- 升级网络：10GbE 或多 1GbE 网卡
- 像素格式：Bayer12 → Bayer8

### 限制 2：软触发精度

**现状**：软触发下逐个下发命令，4 台相机曝光时刻非严格同步

**建议**：
- 生产环境使用硬件外触发（推荐）
- 或升级相机支持 PTP（精密时间协议）

### 限制 3：Python ctypes 绑定维护

**现状**：依赖海康官方的 MvImport 示例，若 SDK 版本更新可能需适配

**建议**：
- 定期检查 SDK 更新
- 若版本差异大，可考虑 CFFI/pybind11 升级

---

## 项目文件清单

```
c:\Users\woan\Desktop\MVS_Deployment\
│
├── src/mvs/                            # 核心包 ⭐
│   ├── __init__.py                    # 对外 API 导出
│   ├── binding.py                     # DLL 加载
│   ├── devices.py                     # 设备枚举
│   ├── camera.py                      # 相机生命周期
│   ├── grab.py                        # 取流线程
│   ├── grouping.py                    # 分组器
│   ├── soft_trigger.py                # 软触发
│   ├── save.py                        # 保存 BMP
│   ├── pipeline.py                    # 四机管线
│   ├── apps/                           # CLI 入口（python -m mvs.apps.*）
│   │   ├── quad_capture.py             # 四相机采集
│   │   └── analyze_capture_run.py      # captures 离线分析与报告
│   └── README.md                      # 包使用文档
│
├── examples/
│   └── quad_capture_demo.py           # 使用示例
│
└── docs/
    └── python-repository-overview.md  # 完整项目文档
```

---

## 下一步建议

### 短期（立即可做）

1. **测试与验证**
   - 若有 4 台实际相机，运行 CLI 验证端到端流程
   - 检查 metadata.jsonl 的 trigger_index 连续性和时间戳准确度

2. **性能调优**
   - 根据实际网络环境调整 `group_timeout_ms` 和 `max_pending_groups`
   - 监控 `lost_packet` 和 `dropped_groups` 指标

3. **部署文档**
   - 撰写部署指南（相机网络配置、触发信号接线等）
   - 制作故障排查表

### 中期（1-2 周）

4. **单元测试**
   - 为 binding/devices/grouping 等模块编写测试
   - Mock SDK 调用，验证逻辑正确性

5. **性能基准**
   - 在目标硬件上测试吞吐量、延迟、内存占用
   - 优化关键路径（如 Grabber 队列大小）

6. **集成推理**
   - 将推理模型集成到采集管线
   - 测试同步采集 + 实时推理的可行性

### 长期（1 个月+）

7. **高级功能**
   - PTP 时间同步支持（相机间时钟对齐）
   - 多组相机支持（>4 台）
   - ROI 动态调整

8. **监控与日志**
   - 添加结构化日志（如 logging 模块）
   - 性能指标收集（FPS、队列深度、网络 KPI）

9. **开源与社区**
   - 考虑在 GitHub 开源
   - 撰写论文或博客分享实现细节

---

## 技术债清单

| 项 | 优先级 | 说明 |
|----|--------|------|
| 单元测试覆盖 | 中 | 尚无自动化测试，建议补充 |
| 类型提示完整性 | 低 | 已使用基本类型提示，可进一步精化 |
| 文档示例运行 | 中 | docs 中示例需在有 DLL 的环境实际验证 |
| 日志系统 | 低 | 目前仅用 print，可升级为 logging 模块 |

---

## 联系与反馈

若在使用中遇到问题或有改进建议，建议：

1. 检查 docs/python-repository-overview.md 中的"常见问题"章节
2. 查看 mvs/README.md 的快速开始和故障排查
3. 运行 examples/quad_capture_demo.py 验证环境配置
4. 如持续遇到问题，建议记录错误日志并提出反馈

---

**项目完成。祝采集顺利！** 🎉

