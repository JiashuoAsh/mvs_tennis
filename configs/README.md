# configs：可直接运行的 YAML 配置

本目录下的 YAML 是“真正要用/可直接运行”的配置文件（online/offline）。

如果你想从模板开始写配置：请从 `examples/configs/templates/` 复制一份到本目录，再替换占位符。

## 前置条件

1) Python 环境

- 推荐使用 `uv`（仓库根目录下执行）：
  - 安装依赖：`uv sync`
  - 运行命令：`uv run python -m ...`

2) MVS SDK（仅 online 需要）

- 你需要让程序能找到 MVS SDK 的 DLL 以及 Python 绑定（MvImport）。二选一即可：
  - 配置环境变量：
    - `MVS_DLL_DIR=<包含 MvCameraControl.dll 的目录>`
    - `MVS_MVIMPORT_DIR=<包含 MvCameraControl_class.py 等文件的目录>`
  - 或在 YAML 中填写 `dll_dir` / `mvimport_dir`。

## 配置清单（怎么选）

### 在线（online：采集 + 定位）

- `online_pt_windows_cpu_software_trigger.yaml`
  - 纯软件触发（所有相机 Software），适合先把“取流 -> 检测 -> 三角化 -> 终端打印/写 jsonl”跑通。
- `online_pt_windows_cpu_software_trigger_params_4cam.yaml`
  - 四相机版本（serials + 标定均为 4cam），并启用 `time_sync_mode: dev_timestamp_mapping`。
- `online_master_slave_line0.yaml`
  - 主从触发示例（只对 master 发软触发；slaves 使用 Line0 作为触发输入）。

### 离线（offline：读 captures 定位）

- `offline_pt_windows_cpu.yaml`
  - 常用离线配置：Windows/CPU + YOLOv8 `.pt`。
- `offline_pt_windows_cpu_params_4cam.yaml`
  - 四相机离线配置（serials 显式列出，且启用 `dev_timestamp_mapping`）。
- `offline_color_debug.yaml`
  - 颜色阈值调试配置：不依赖模型文件，适合验证三角化链路。
- `offline_rknn_board_or_linux.yaml`
  - RKNN 模板（通常用于 Rockchip 或 Linux 工具链；Windows 上一般不可用）。

## 运行命令与验证标准

### 在线：软件触发（推荐先跑这个）

命令：

- `uv run python -m tennis3d.apps.online_mvs_localize --config configs/online_pt_windows_cpu_software_trigger.yaml`

验证标准：

- 终端在检测到球时会打印包含 `xyz_w=(x=..., y=..., z=...)` 的行。
- 若 `out_jsonl` 非空：对应的 jsonl 文件会持续追加记录，且每行包含 `balls` 字段。

### 在线：主从触发（Line0）

命令：

- `uv run python -m tennis3d.apps.online_mvs_localize --config configs/online_master_slave_line0.yaml`

验证标准：

- 同上（能打印、能写 jsonl）。
- 若你启用了 `time_sync_mode: dev_timestamp_mapping`：输出记录中会包含 `time_mapping_mapped_host_ms_*` 一类字段（用于离线统计时间映射质量）。

### 离线：从 captures 输出 3D jsonl

命令：

- `uv run python -m tennis3d.apps.offline_localize_from_captures --config configs/offline_pt_windows_cpu.yaml`

验证标准：

- `out_jsonl` 指向的文件存在且非空。
- 任意一行（JSON）中包含 `balls`，且当检测到球时 `balls` 非空。
