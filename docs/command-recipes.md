# 命令配方（Command Recipes）

本页把仓库中常用的“长命令组合”整理成可复用的配方，避免根 `readme.md` 被命令淹没。

## 使用约定

- 建议在仓库根目录执行（`c:/Users/woan/Desktop/MVS_Deployment`）。
- 若你用 uv（推荐）：先 `uv venv`、`uv sync`，再用 `uv run ...` 执行命令。
- 采集相关命令依赖海康 MVS SDK（Windows）与相机可用；并确保 MVS Client 等软件未占用相机。

## 目录

- [命令配方（Command Recipes）](#命令配方command-recipes)
  - [使用约定](#使用约定)
  - [目录](#目录)
  - [采集与分析（mvs）](#采集与分析mvs)
    - [列出设备](#列出设备)
    - [分析一次采集输出](#分析一次采集输出)
    - [纯软件触发（Software）](#纯软件触发software)
      - [低分辨率 + offset demo（单相机）](#低分辨率--offset-demo单相机)
      - [四相机软件触发（常用参数组](#四相机软件触发常用参数组)
    - [主从触发（master 软件 + slaves 硬触发）](#主从触发master-软件--slaves-硬触发)
      - [低帧率校准：master/slave 同步抓图，用于相机矫正](#低帧率校准masterslave-同步抓图用于相机矫正)
      - [offline 网球抓取（持续采集）](#offline-网球抓取持续采集)
  - [探测：运行中能否高频改 ROI](#探测运行中能否高频改-roi)
    - [探测脚本](#探测脚本)
    - [如何解读输出](#如何解读输出)
    - [推荐工程策略](#推荐工程策略)
      - [常用指令（变体）](#常用指令变体)
  - [离线工具链（tools）](#离线工具链tools)
    - [按相机重排 captures](#按相机重排-captures)
    - [生成 4 相机标定参数 JSON](#生成-4-相机标定参数-json)
    - [拟合时间映射](#拟合时间映射)
  - [新加的](#新加的)
    - [curve2 vs curve3 比较](#curve2-vs-curve3-比较)
    - [离线：Ultralytics 叠框 + 三角化 3D（两脚本联动）](#离线ultralytics-叠框--三角化-3d两脚本联动)
      - [1) 离线检测 + 保存叠框可视化](#1-离线检测--保存叠框可视化)
      - [2) 用检测结果三角化得到 3D](#2-用检测结果三角化得到-3d)

---

## 采集与分析（mvs）

### 列出设备

前置条件：

- 已安装 MVS SDK。
- DLL 路径可被加载：参考根 `readme.md` 的 “设置 MVS DLL”。

命令：

```bash
python -m mvs.apps.quad_capture --list
```

验证标准：

- 终端能列出可用相机（序列号等信息）。
- 若报 `MV_E_ACCESS_DENIED`，通常是相机被占用（关闭 MVS Client 后重试）。

### 分析一次采集输出

用途：对 `quad_capture` 写出的某次输出目录（含 `metadata.jsonl` 与 `group_*/`）做完整性/FPS/时间差分析。

前置条件：

- 已完成一次采集，并得到 `output-dir`。

命令示例：

```bash
python -m mvs.apps.analyze_capture_run --output-dir ./data/captures/offset640x640 \
	--write-json ./data/captures/offset640x640/analysis_summary.json

python -m mvs.apps.analyze_capture_run --output-dir ./data/captures \
	--write-json ./data/captures/analysis_summary.json

python -m mvs.apps.analyze_capture_run --output-dir ./data/captures_master_slave \
	--write-json ./data/captures_master_slave/analysis_summary.json
```

验证标准：

- 能生成 `analysis_summary.json`（若提供了 `--write-json`）。
- 报告里可关注：
  - `arrival fps (median)`
  - `per-camera arrival fps (min/median/max)`
  - 组包完整性/丢帧统计

### 纯软件触发（Software）

#### 低分辨率 + offset demo（单相机）

前置条件：

- 相机可用，且触发源配置为 Software（脚本会设置）。

命令：

```bash
python -m mvs.apps.quad_capture \
	--serial DA8199303 \
    --trigger-cache-enable \
	--trigger-source Software \
	--soft-trigger-fps 90 \
	--exposure-auto Off --exposure-us 5000 \
	--gain-auto Off --gain 12 \
	--save-mode sdk-bmp \
	--max-groups 20 \
	--image-width 1280 \
	--image-height 1080 \
    --image-offset-x 640 \
    --image-offset-y 640 \
	--pixel-format BayerRG8 \
    --output-dir data/captures/offset640x640
```

- `output-dir` 下生成 `metadata.jsonl` 与若干 `group_*/` 子目录。
- 每个 `group_*/` 下能看到保存的图片文件。

#### 四相机软件触发（常用参数组

命令（示例：1920×1080 + offset=0）：

```bash
python -m mvs.apps.quad_capture \
	--serial DA8199303 DA8199402 DA8199243 DA8199285 \
    --trigger-cache-enable \
	--trigger-source Software \
	--soft-trigger-fps 18 \
	--exposure-auto Off --exposure-us 5000 \
	--gain-auto Off --gain 12 \
	--save-mode sdk-bmp \
	--max-groups 20 \
	--image-width 1920 \
	--image-height 1080 \
    --image-offset-x 0 \
    --image-offset-y 0 \
	--pixel-format BayerRG8
```

命令（示例：2448×2048 + fps=15）：

```bash
python -m mvs.apps.quad_capture \
	--serial DA8199303 DA8199402 DA8199243 DA8199285 \
    --trigger-cache-enable \
	--trigger-source Software \
	--soft-trigger-fps 15 \
	--exposure-auto Off --exposure-us 10000 \
	--gain-auto Off --gain 15 \
	--max-groups 20 \
    --image-width 2448 \
	--image-height 2048 \
    --image-offset-x 0 \
    --image-offset-y 0 \
	--pixel-format BayerRG8 \
	--save-mode sdk-bmp \
    --output-dir data/captures_software_trigger
```

验证标准：

- 能持续产生组包目录，且 `analyze_capture_run` 能读到合理的 fps/组包统计。
- 若出现“请求 2448×2048 但实际变小”，参考根 `readme.md` 的 Troubleshooting：通常是历史 `OffsetX/OffsetY` 残留。

### 主从触发（master 软件 + slaves 硬触发）

#### 低帧率校准：master/slave 同步抓图，用于相机矫正

前置条件：

- 确认物理接线：master 输出线 → 每台 slave 的触发输入线。
- master 的输出线信号源已配置（例如 `ExposureStartActive`）。

命令：

```bash
python -m mvs.apps.quad_capture \
	--serial DA8199303 DA8199402 DA8199243 DA8199285 \
    --trigger-cache-enable \
	--master-serial DA8199303 \
	--master-line-source ExposureStartActive \
	--trigger-source Line0 \
	--trigger-activation FallingEdge \
	--exposure-auto Off --exposure-us 10000 \
	--gain-auto Off --gain 15 \
	--save-mode sdk-bmp \
	--soft-trigger-fps 5 \
	--max-groups 10 \
	--max-wait-seconds 10 \
    --image-width 2448 \
	--image-height 2048 \
    --image-offset-x 0 \
    --image-offset-y 0 \
	--pixel-format BayerRG8 \
	--output-dir data/captures_master_slave/for_calib
```

验证标准：

- 目录中持续产生完整组包（多相机同一触发的帧能凑齐）。
- 若只有 master 出图、一直凑不齐组包：优先排查接线与 `--master-line-source`。

补充排查（时间戳对不齐但组包完整）：

- 如果你观察到 **master 与 slaves 的稳定时间差约等于曝光时间**（例如曝光 10ms 时差约 10ms；改成 2ms 时差约 2ms），通常是 **slaves 触发沿选反**。
	- 处理方式：使用 `--trigger-activation FallingEdge` 再采集验证（如果你之前使用的是 `RisingEdge`，改为 `FallingEdge`）。
	- 验证工具：`uv run python tools/debug_master_slave_timing.py --captures-dir <DIR> --master <MASTER_SERIAL>`
	- 留档报告：`docs/reports/2026-02-03-master-slave-trigger-falling-edge.md`

#### offline 网球抓取（持续采集）

命令：

```bash
python -m mvs.apps.quad_capture \
	--serial DA8199303 DA8199402 DA8199243 DA8199285 \
    --trigger-cache-enable \
	--master-serial DA8199303 \
	--master-line-source ExposureStartActive \
	--trigger-source Line0 \
	--trigger-activation FallingEdge \
	--exposure-auto Off --exposure-us 10000 \
	--gain-auto Off --gain 15 \
	--save-mode sdk-bmp \
	--soft-trigger-fps 18 \
	--max-groups 10 \
	--max-wait-seconds 10 \
    --image-width 2448 \
	--image-height 2048 \
    --image-offset-x 0 \
    --image-offset-y 0 \
	--pixel-format BayerRG8 \
	--output-dir data/captures_master_slave/tennis_offline

```

---

## 探测：运行中能否高频改 ROI

你关心的问题是：**MVS 相机驱动是否允许在采集开始后（`StartGrabbing` 之后）高频修改 ROI（尤其是 `OffsetX/OffsetY`）？**

结论先行：

- 在本仓库的实现里，我们明确把 ROI 设置写在 `StartGrabbing` 之前（见 `mvs.sdk.camera`，文件位置：`packages/mvs/src/mvs/sdk/camera.py`），原因是：
	**很多机型在开始取流后会把 `Width/Height/OffsetX/OffsetY` 这类节点锁定为不可写**。
- 但是“很多”不等于“全部”：是否可写依赖你的机型/固件/当前模式（触发/连续采集/像素格式等）。

因此最靠谱的方法是：**在你的现场机上做一次实测**。

### 探测脚本

仓库提供了一个轻量探测脚本：

- `tools/mvs_runtime_roi_probe.py`

运行示例（单相机）：

```bash
uv run python tools/mvs_runtime_roi_probe.py --serial DA8199303
```

更贴近在线的示例（边出图边测写入；把写入频率设成 30Hz）：

```bash
uv run python tools/mvs_runtime_roi_probe.py \
	--serial DA8199303 \
	--set-hz 30 \
	--iters 300 \
	--enable-soft-trigger-fps 30
```

它会：

1. 打开指定相机并启动取流（`StartGrabbing`）。
2. 在运行中循环调用 `MV_CC_SetIntValue("OffsetX")/MV_CC_SetIntValue("OffsetY")`。
3. 统计成功率、返回码直方图和调用耗时。

### 如何解读输出

- 若输出里出现大量 `ret=0x...`，并且命中 `MV_E_ACCESS_DENIED`：
	- 通常表示运行中被锁定，不支持你想要的“每帧改 ROI”。
	- 即便偶尔成功，也可能伴随丢帧/延迟抖动。
- 若 `ret==MV_OK` 占比接近 100%，并且调用耗时稳定：
	- 说明至少从“写节点”角度看是可行的。
	- 此时可进一步关注脚本输出的 `readback: OffsetX/OffsetY cur=... (last_set=...)`：
		- 若读回值能跟上最后一次写入，说明设置不仅返回成功，而且节点值确实生效。
	- 但仍建议你进一步观察：
		- ROI 改动是否真正反映到后续帧（Width/Height/Offset 是否生效）
		- 是否出现明显丢帧/卡顿
		- 多相机 master/slave 同步是否被破坏

补充说明：

- 脚本会尝试用 `MV_XML_GetNodeAccessMode` 查询节点访问模式（AM_RW/AM_RO 等）。
  若你的环境里该接口返回非 0（脚本会打印 ret），就会显示 `(unknown)`。
  这不影响 `SetIntValue` 压测结论：**当 `SetIntValue` 持续返回 `MV_OK`，并且读回值正确时，可以认为“运行中写 Offset 是可行的”。**

### 推荐工程策略

- 如果你的目标是“跟踪球，同时冲 30fps 以上”，推荐按优先级分两步做：
	1) **主机侧软件裁剪（默认优先）**：相机保持固定输出；在 detector 前裁剪小窗（`--detector-crop-size 640`）。
		- 优点：不依赖相机运行中可写性；对多相机同步影响最小；几何坐标系更稳定。
	2) **两级 ROI（进阶）**：当你的机型实测支持运行中写 `OffsetX/OffsetY`，可开启“相机侧 AOI 平移”先降带宽，再叠加软件裁剪降算力。

在线两级 ROI 示例（相机 AOI=1280×1080，AOI 内再 software crop=640×640）：

```bash
uv run python -m tennis3d_online \
	--serial DA8199303 DA8199402 DA8199243 DA8199285 \
	--trigger-source Software --soft-trigger-fps 30 \
	--pixel-format BayerRG8 \
	--image-width 1280 --image-height 1080 --image-offset-x 0 --image-offset-y 0 \
	--camera-aoi-runtime \
	--camera-aoi-update-every-groups 2 \
	--camera-aoi-min-move-px 8 \
	--camera-aoi-smooth-alpha 0.3 \
	--camera-aoi-max-step-px 160 \
	--camera-aoi-recenter-after-missed 30 \
	--detector pt --model data/models/best.pt --pt-device cpu \
	--detector-crop-size 640
```

提示：

- 若你发现启用后偶发写入失败或丢帧抖动，可先把 `--camera-aoi-update-every-groups` 提大（例如 3~5），
  或把 `--camera-aoi-min-move-px` 提大（例如 16~32）降低写入频率。

验证标准：

- `--max-groups 0` 表示持续采集；可用 Ctrl+C 停止。
- 停止后使用 `analyze_capture_run` 进行组包/FPS/时间差复盘。

#### 常用指令（变体）

命令：

```bash
python -m mvs.apps.quad_capture \
	--serial DA8199303 DA8199402 DA8199243 DA8199285 \
    --trigger-cache-enable \
	--master-serial DA8199303 \
	--master-line-source ExposureStartActive \
	--soft-trigger-fps 18 \
	--trigger-source Line0 \
	--trigger-activation FallingEdge \
	--exposure-auto Off --exposure-us 5000 \
	--gain-auto Off --gain 12 \
	--save-mode sdk-bmp \
	--output-dir data/captures_master_slave \
	--max-groups 20 \
	--max-wait-seconds 10 \
	--image-width 1920 \
	--image-height 1080 \
    --image-offset-x 0 \
    --image-offset-y 0 \
	--pixel-format BayerRG8

```

---

## 离线工具链（tools）

### 按相机重排 captures

用途：把采集输出从“按 group”改为“按 camera”组织，便于标定/质检/挑帧。

前置条件：

- 输入目录包含 `metadata.jsonl` 与 `group_*/`。

命令：

```bash
uv run python tools/mvs_relayout_by_camera.py \
  --captures-dir data/captures_master_slave/for_calib \
  --output-dir data/captures_master_slave/for_calib_by_camera
```

验证标准：

- 输出目录下生成 `cam0_<SERIAL>/`、`cam1_<SERIAL>/` 等目录。
- 对应图片文件被 hardlink/copy/symlink 到新结构（取决于脚本参数与系统能力）。

### 生成 4 相机标定参数 JSON

前置条件：

- 已准备内参目录与外参文件（见命令参数）。

命令：

```bash
uv run python tools/generate_camera_extrinsics.py \
  --intrinsics-dir data/calibration/inputs/2026-02-03 \
  --extrinsics-file data/calibration/base_to_camera_extrinsics.json \
  --out data/calibration/camera_extrinsics_C_T_B.json \
  --map cam0=DA8199303 \
  --map cam1=DA8199402 \
  --map cam2=DA8199243 \
  --map cam3=DA8199285
```

验证标准：

- 输出文件 `data/calibration/camera_extrinsics_C_T_B.json` 生成成功。

### 拟合时间映射

命令：

```bash
uv run python tools/mvs_fit_time_mapping.py --captures-dir data/captures_master_slave/tennis_offline
```

验证标准：

- 命令执行完成，并在终端/输出文件中给出拟合结果（以工具实际输出为准）。



## 新加的

### curve2 vs curve3 比较

```bash
python examples/scratch/curve2_curve3_compare.py --disable-curve3 --png-all-post-n --out-png ./temp
```

### 离线：Ultralytics 叠框 + 三角化 3D（两脚本联动）

用途：

- 从 `data/captures_master_slave/tennis_offline` 的采集结果中，先用 `best.pt` 离线检测并保存叠框图。
- 再把检测输出喂给 `tools/tennis_localize_from_detections.py` 做多目几何融合，得到每个 group 的 `balls` 列表（0..N 个 3D 点）。

前置条件：

- 已有离线采集目录：`data/captures_master_slave/tennis_offline`（包含 `metadata.jsonl` 与 `group_*/`）。
- 已有 YOLO 模型：`data/models/best.pt`。
- 已有相机标定：`data/calibration/camera_extrinsics_C_T_B.json`（相机 key 为序列号）。

#### 1) 离线检测 + 保存叠框可视化

命令：

```bash
uv run python tools/ultralytics_best_pt_smoketest.py \
	--captures-dir data/captures_master_slave/tennis_offline \
	--model data/models/best.pt \
	--all \
	--out-vis-dir data/tools_output/tennis_ultralytics_vis
```

预期输出 / 验证标准：

- 生成检测 JSONL：`data/tools_output/tennis_ultralytics_detections.jsonl`
- 生成叠框图片目录：`data/tools_output/tennis_ultralytics_vis/group_*/cam*_*.jpg`
- 终端输出类似：`Done. groups=... images=... images_with_ball=...`

#### 2) 用检测结果三角化得到 3D

命令（无参默认读取上一步的 JSONL）：

```bash
uv run python tools/tennis_localize_from_detections.py
```

预期输出 / 验证标准：

- 输出文件：`data/tools_output/tennis_positions_3d.json`
- 终端输出类似：`Done. groups: N` 且 N>0
- 若画面里有球且至少两路检出：`Done. balls: M` 且 M>0

提示：

- 若你只想先快速试跑：`uv run python tools/tennis_localize_from_detections.py --max-frames 50`