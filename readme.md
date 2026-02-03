
# MVS_Deployment：海康 MVS 多相机采集 + 网球 3D 定位（online/offline）

## 仓库总结（10~20 行）

- 本仓库用于海康工业相机（MVS SDK）多相机同步采集，并在采集图像上完成网球检测与多视角三角化，输出网球 3D 位置。
- `src/mvs/` 是采集封装：DLL 绑定加载、设备枚举、相机配置、取流线程、按分组键组包、保存 BMP/RAW、事件记录、带宽估算。
- 采集主入口：`python -m mvs.apps.quad_capture`：支持 software/硬触发、master/slave、ROI、像素格式、曝光/增益，并写出 `metadata.jsonl` 供离线复盘。
- 采集分析入口：`python -m mvs.apps.analyze_capture_run`：用于分析一次采集输出：组包完整性、丢包、FPS、时间差等。
- `src/tennis3d/` 是业务库：检测器适配（fake/color/rknn）、标定读写、三角化与重投影误差、在线/离线共用 pipeline。
- 在线入口：`python -m tennis3d.apps.online_mvs_localize`（实时取流→检测→三角化→输出 JSONL）。
- 离线入口：`python -m tennis3d.apps.offline_localize_from_captures`（读 captures/metadata.jsonl→检测→三角化→输出 JSONL）。
- 样例采集数据位于 `data/captures_master_slave/tennis_test/`（包含图片与 `metadata.jsonl`）。
- 关键依赖：Python >= 3.10、numpy、opencv-python、pyyaml；RKNN 推理依赖运行时环境（通常不在 Windows 上直接可用）。

## 快速开始

### 1) 安装（推荐 uv）

本仓库采用 src-layout（代码在 `src/` 下），推荐在虚拟环境里使用 editable 安装。

```bash
uv venv
uv sync
uv pip install -e .
```

### 2) 设置 MVS DLL（采集相关必需）

三种方式任选其一：

- 把 DLL 目录加入系统 PATH（取决于你的 MVS 安装位置）
- 使用环境变量：

```bash
set MVS_DLL_DIR=C:\\path\\to\\MVS\\Runtime\\Win64_x64
```

- 或命令行指定：`python -m mvs.apps.quad_capture --dll-dir ...`

### 3) 离线跑通（无需相机）

使用仓库内现成数据 + fake 检测器，先验证“读取 captures→几何输出”链路：

```bash
python -m tennis3d.apps.offline_localize_from_captures \
	--captures-dir data/captures_master_slave/tennis_test \
	--calib data/calibration/example_triple_camera_calib.json \
	--detector fake \
	--require-views 2 \
	--max-groups 5 \
	--out-jsonl data/tools_output/offline_positions_3d.jsonl
```

### 4) 采集 + 分析（连接相机）

```bash
python -m mvs.apps.quad_capture --list
python -m mvs.apps.analyze_capture_run --output-dir data/captures_master_slave/tennis_test
```

## 文档导航

- 架构与数据流：`docs/architecture-and-dataflow.md`
- 运行模式与调试：`docs/run-modes-and-debugging.md`
- 命令配方（采集/工具链）：`docs/command-recipes.md`
- 开发与测试：`docs/development-and-testing.md`
- 改进建议清单：`docs/improvement-suggestions.md`
- MVS 采集包详细说明：`src/mvs/README.md`

---

以下为更细的说明与命令收集（包含大量采集经验与历史内容），需要深挖时再看。

## 你最关心的入口（Entry）

### 在线 3D 定位

- 入口：`src/tennis3d/apps/online_mvs_localize.py`
- 推荐运行方式：

```bash
python -m tennis3d.apps.online_mvs_localize --help
```

也支持通过配置文件启动（YAML/JSON）：

```bash
python -m tennis3d.apps.online_mvs_localize --config path/to/online.yaml
```

核心参数：

- `--serial`：相机序列号列表（2 台及以上，顺序无所谓但建议固定顺序）
- 触发拓扑二选一：
  - 纯软件：`--trigger-source Software --soft-trigger-fps <Hz>`
  - 主从：`--master-serial <SERIAL_MASTER> --trigger-source Line0 --soft-trigger-fps <Hz>`
- `--calib`：标定文件（支持 `.json/.yaml/.yml`）
- `--detector`：`fake` / `color` / `rknn`
- `--require-views`：三角化所需的最少视角数（建议保持 >=2；当 4 台里只有 1 台看到球时，本帧不会输出 3D）
- `--out-jsonl`：可选，输出 JSONL（不传则在终端输出）

终端输出说明：

- 在线模式默认**仅在观测到球**（`balls` 非空）时打印一行，包含 `group_index` 与 `ball_3d_world`（世界坐标系 3D）。
- 这样可以在“无球阶段”避免刷屏；一旦开始有球，就会持续打印每次观测到的球位置。

### 离线 3D 定位

- 入口：`src/tennis3d/apps/offline_localize_from_captures.py`

```bash
python -m tennis3d.apps.offline_localize_from_captures --help
```

也支持通过配置文件启动（YAML/JSON）：

```bash
python -m tennis3d.apps.offline_localize_from_captures --config path/to/offline.yaml
```

离线输入是一个 captures 目录：

- `data/captures/<run>/metadata.jsonl`
- 若干帧图片文件（由采集工具保存）

#### 最小可运行示例（无需相机）

仓库内已带一份真实采集样例：`data/captures_master_slave/tennis_test/`（包含图片与 `metadata.jsonl`）。直接运行：

```bash
python -m tennis3d.apps.offline_localize_from_captures \
	--captures-dir data/captures_master_slave/tennis_test \
	--calib data/calibration/example_triple_camera_calib.json \
	--detector fake \
	--require-views 2 \
	--max-groups 5
```

输出默认在：`data/tools_output/offline_positions_3d.jsonl`（也可用 `--out-jsonl` 自定义）。

补充：`tools/generate_sample_sequence.py` 目前用于生成若干 BMP 图片，尚未生成可被离线 pipeline 直接读取的 `metadata.jsonl`（见 `docs/improvement-suggestions.md`）。

### 生成“假标定”（没有真实标定也能先跑通链路）

- 入口：`src/tennis3d/apps/fake_calibration.py`

```bash
python -m tennis3d.apps.fake_calibration --help
```

也支持 console script：

```bash
tennis3d-fake-calib --help
```

### 仅几何：从已有 bbox JSON 做三角化（不跑检测）

- 入口：`tools/tennis_localize_from_detections.py`

```bash
python tools/tennis_localize_from_detections.py --help
```

---

## 代码结构（高内聚/低耦合）

旧结构到新结构的对照表见：`docs/MIGRATION_MAP.md`。

> `mvs` 视为底层 SDK 封装层（src-layout：`src/mvs`），不建议随意重构；业务逻辑集中在 `tennis3d`（`src/tennis3d`）。

- `src/mvs/`：工业相机采集封装（打开设备、抓流、分组、软触发、保存、像素格式转换等）
- `src/tennis3d/apps/`：**应用入口层（CLI）**，尽量薄
  - `online_mvs_localize.py`：在线取流并定位
  - `offline_localize_from_captures.py`：离线读取 captures 并定位
  - `detectors.py`：检测器适配（fake/rknn）
- `src/tennis3d/pipeline/`：**在线/离线共享的流水线**
  - `sources.py`：两种 source（在线/离线）统一产出 `(meta, images_by_camera)`
  - `core.py`：统一执行 detect → triangulate/localize → 产出 JSON 记录
- `src/tennis3d/geometry/`：几何与标定 IO（投影矩阵、DLT 三角化、误差等）
- `src/tennis3d/localization/`：融合逻辑（选每相机最佳 bbox、要求至少 N 视角等）

### 参考文件（不会被 pipeline 使用）

`examples/` 下包含一些参考代码（例如 RKNN 的原始示例/实验脚本），不参与当前在线/离线 pipeline。

---

## 需要哪些数据？

### 1) 多相机标定（必需）

路径示例：`data/calibration/example_triple_camera_calib.json`

样例标定（YAML）：`data/calibration/sample_cams.yaml`

格式要点（见 `src/tennis3d/geometry/calibration.py`）：

- 顶层必须有 `cameras` 字典
- 每个相机项必须包含：
  - `image_size`: `[width, height]`
  - `K`: 3×3 内参矩阵
  - `R_wc`: 3×3 外参旋转（world → camera）
  - `t_wc`: 长度 3 外参平移（world → camera）
  - `dist`: 可选（当前 pipeline 不用畸变做反投影，先忽略）

外参约定：

$$X_c = R_{wc} X_w + t_{wc}$$

投影矩阵：

$$P = K [R_{wc}|t_{wc}]$$

> 注意：如果你手里是 camera→world（$R_{cw}, t_{cw}$），需要先转换成 world→camera。

### 2) 检测结果（bbox）

在线/离线入口默认会跑 detector 并生成 bbox。

- `--detector fake`：永远在图像中心给一个 bbox（用于没有 RKNN/没有模型时跑通链路）
- `--detector color`：HSV 颜色阈值找球（适合绿色球/高对比场景，Windows 上可用于先跑通“检测+几何”链路）
- `--detector pt`：Ultralytics YOLOv8 `.pt`（CPU 推理，适合 Windows 上验证真实模型）
- `--detector rknn`：RKNN 模型推理（需要 RKNN 运行时支持；由 `src/tennis3d/detectors.py` 统一适配）

> Windows 通常无法直接跑 RKNNLite；建议：Windows 上先用 fake 跑通采集/同步/几何链路，然后在支持 RKNN 的环境（通常是 Linux + Rockchip）切换为 rknn。

### 3) 离线 captures 目录（离线模式必需）

离线入口读取 `data/captures/<run>/metadata.jsonl`。该文件由 `python -m mvs.apps.quad_capture` 生成（默认输出到 `data/captures/`），包含两类记录：

- **事件记录**（相机事件/软触发发送记录等）：会被离线 pipeline 自动跳过
- **组包记录**：包含 `frames: [...]`，每个 frame 至少包含 `serial` 与 `file`

并且离线 pipeline 支持：

- `file` 为相对路径：会自动按 `captures_dir` 进行 resolve
- `file` 为绝对路径：直接读取

---

## 输出是什么？（JSONL）

在线与离线都输出 JSON Lines，**每行对应一个同步组**（不论该组是否成功定位）。

- 顶层字段：
	- `created_at`: 处理时间（Unix epoch 秒）
	- `meta...`: source 产出的组信息（在线常见 `group_index`；离线常见 `group_seq/group_by/trigger_index`）
	- `balls`: 该同步组的 0..N 个球（跨视角几何一致才会输出）

其中 `balls` 是列表，每个元素是一个 ball dict（按质量从高到低排序），常见字段：

- `ball_id`: 组内编号
- `used_cameras`: 实际参与该球三角化/验证的相机列表
- `ball_3d_world`: 世界坐标系 3D 点 `[x,y,z]`
- `ball_3d_camera`: 各相机坐标系下 3D 点 `{serial: [x,y,z]}`
- `quality`: 质量评分（用于排序与冲突消解）
- `num_views`: 参与视角数（等价于 `len(used_cameras)`）
- `median_reproj_error_px` / `max_reproj_error_px`: 重投影误差统计
- `reprojection_errors`: 按相机列出 `uv/uv_hat/error_px`
- `detections`:（默认包含）本次每相机选中的 bbox/score/cls/center

meta 字段因 source 不同而不同：

- 在线：包含 `group_index`
- 离线：包含 `group_seq` / `group_by` / `trigger_index`（取决于 captures 的记录）

---

## 在线同步怎么做？（很关键）

在线入口 `tennis3d.apps.online_mvs_localize` 通过 `mvs.pipeline.open_quad_capture` 获取“同步组”（按 `--group-by` 组包）。你需要选择合适的触发拓扑：

### A) 纯 Software 触发

- 所有相机 `trigger_source=Software`
- 程序内部按 `--soft-trigger-fps` 对所有相机下发软触发

优点：接线简单，调试方便。

缺点：严格意义上同步一致性一般不如硬触发；并且多相机/高帧率容易受带宽/主机负载影响。

### B) master(Software) + slave(硬触发)

常见做法：

- master：Software 触发（程序只对 master 下发软触发）
- master 输出线：配置为 `ExposureStartActive`（或机型支持的等价信号）
- slave：`trigger_source=Line0/Line1/...`（与实际接线一致）

最重要的注意点：

1) **一定要接线**：master 输出线 → slave 触发输入线（例如 Line1 → Line0）。
2) **一定要配置 master 的输出线信号源**：否则 slave 永远收不到触发。
3) **组包键选择**：
   - 如果相机支持且递增正常：优先 `--group-by trigger_index`
   - 如果发现 `nTriggerIndex` 恒为 0：用 `--group-by frame_num`

---

## 先跑通链路（推荐步骤）

1) 用 `python -m mvs.apps.quad_capture` 先采一段数据到 `data/captures/`（确认同步组包正常）。
2) 没有真实标定时，用 `python -m tennis3d.apps.fake_calibration`（或安装后用 `tennis3d-fake-calib`）生成一个临时标定 JSON。
3) 用离线入口跑一遍（debug 体验最好）。
4) 再切在线入口接入实时流。

---

## 采集工具：mvs_quad_capture（保存图片 + metadata.jsonl）

采集/分析/工具链的常用命令配方已整理到：`docs/command-recipes.md`（包含前置条件与验证标准）。

快速查看帮助与列举设备：

```bash
python -m mvs.apps.quad_capture --help
python -m mvs.apps.quad_capture --list
python -m mvs.apps.analyze_capture_run --help
```

---

## 把 captures 图片按相机重排（从按 group 变成按 camera）

默认情况下，采集输出目录是按 `group_*/` 分组的（每个 group 里包含同一次触发的多相机帧）。

在标定/质检/挑帧等场景中，经常希望把同一台相机的所有帧放在一起。本仓库提供脚本：

- 脚本：`tools/mvs_relayout_by_camera.py`
- 核心逻辑：`src/mvs/capture_relayout.py`

### 输入/输出结构

输入目录必须包含：

- `metadata.jsonl`
- `group_XXXXXXXXXX/` 子目录（里面是 `cam*_*.bmp` 之类的图片）

输出目录会创建按相机划分的子目录，例如：

- `cam0_<SERIAL>/`
- `cam1_<SERIAL>/`
- `cam2_<SERIAL>/`

并把对应图片放进去。文件名保持不变（例如 `cam0_seq000123_f124.bmp`）。

### 运行方式

推荐在仓库根目录运行（如果你没有做 `pip install -e .`，请确保 `PYTHONPATH=src`）：

```bash
# 例：把 for_calib 输出重排到 for_calib_by_camera
PYTHONPATH=src python tools/mvs_relayout_by_camera.py \
  --captures-dir data/captures_master_slave/for_calib \
  --output-dir data/captures_master_slave/for_calib_by_camera \
  --mode hardlink
```

常用参数：

- `--mode hardlink|copy|symlink`：落盘方式。
  - `hardlink`（默认）：省空间；要求源文件与输出目录在同一磁盘分区。
  - `symlink`：Windows 上可能需要管理员权限或开启开发者模式。
  - `copy`：最稳妥，但会占用额外磁盘空间。
  - 注意：当 `hardlink/symlink` 失败时，脚本会自动回退为 `copy`（确保流程不中断）。
- `--overwrite`：目标文件存在时是否覆盖。
- `--max-groups N`：只处理前 N 个组（调试用）。
- `--dry-run`：演练，不实际写文件。

### 如何正确统计 FPS（重要）

很多人会用 `created_at`（写入 metadata 的时间）去估计 FPS，但它更像“端到端吞吐”，会被保存/写盘/推理等流程拖慢。

更贴近“相机实际出图/有效触发频率”的指标是：

- **arrival fps**：基于每帧的 `arrival_monotonic`（Grabber 线程拿到帧时记录的单调时间）估计。
- **per-camera arrival fps**：每台相机分别估计帧率（能看出是否某一路明显拖后腿）。

运行分析脚本后，请优先看报告里的：

- `arrival fps (median)`
- `per-camera arrival fps (min/median/max)`

### 带宽上限估算（为什么多相机跑不满 15fps）

以本项目常见的 2448×2048 Mono8 为例，单帧约 5MB（metadata 里的 `frame_len` 也能看到）。

- 单相机 8fps 大约需要 5MB × 8 = 40MB/s（≈320Mbps），在 1GbE 上通常还能跑。
- 三相机 15fps 则是 5MB × 3 × 15 = 225MB/s（≈1.8Gbps），**单口 1GbE 物理上不可能**。

如果你的网络/主机吞吐基本固定，那么多相机情况下的可达 FPS 往往近似按相机数反比下降，例如：

- 单相机能到 8fps，三相机理论上限大约是 8/3 ≈ 2.6fps（再叠加协议/交换机/驱动开销后会更低）。

#### 想提高 FPS 的常用手段

1) **减数据量**：ROI / 降分辨率 / 降位深 / 合理像素格式。
2) **增带宽**：多网卡分流、每台相机独立网口/独立交换机、升级到 10GbE。
3) **减主机开销**：先用 `--save-mode none` 测上限；保存时尽量用 SSD、避免 BMP 转码成为瓶颈。

---

## Troubleshooting（常见问题）

### 1) 传了 `--image-width/--image-height`，但实际分辨率变小（例如 2448×2048 变成 2248×1648）

现象：

- CLI 打印的采集配置里显示：`roi=2448x2048 offset=(0,0)`
- 但启动后“带宽估算”里每台相机显示的是更小的分辨率，例如：`2248x1648`

原因（最常见）：

- 相机上一次运行/在 MVS Client 里被设置过非零 ROI 偏移（`OffsetX/OffsetY`）。
- 许多机型的 ROI 约束是：
	- $Width \le SensorWidth - OffsetX$
	- $Height \le SensorHeight - OffsetY$
	因此当历史 `OffsetX/OffsetY` 不为 0 时，你请求的宽高会被相机按最大允许值“卡小”。

如何确认：

- 看采集启动后的“带宽估算”行：它会读取相机当前的 `Width/Height/PayloadSize/PixelFormat` 并打印。
- 如果你看到的宽高差值很“整齐”（例如少 200、少 400），基本就是 offset 残留导致的上限变化。

解决步骤：

1) 确认你确实传了你想要的 ROI：
	 - `--image-width 2448 --image-height 2048 --image-offset-x 0 --image-offset-y 0`
2) 用最新代码重试（本仓库已修复一个“读取 Width/Height 范围前未可靠归零 Offset”的顺序问题）。
3) 如果仍然被卡小：打开 MVS Client，手动把该相机的 ROI 参数恢复到全幅：
	 - `OffsetX=0, OffsetY=0`
	 - `Width/Height` 调到最大（或你目标值）
	 然后关闭 MVS Client（避免占用相机），再运行采集脚本。

预期验证标准：

- 采集启动后，“带宽估算”里每台相机都应显示期望的 `WidthxHeight`（例如 2448×2048）。

### 2) `OpenDevice ... MV_E_ACCESS_DENIED` / 相机被占用

现象：

- 启动采集时报错无访问权限，常见原因是相机正在被 MVS Client 或另一个脚本占用。

解决步骤：

- 关闭所有可能占用相机的软件（尤其是 MVS Client）。
- 等 2~5 秒后重试；必要时给相机断电重启。

### 3) master/slave 硬触发模式下只有 master 出图、一直凑不齐组包

现象：

- master 使用 Software 触发能出图，但 slaves 配置为 `Line0/Line1...` 后一直没有完整组包。

排查要点：

- 确认物理接线：master 输出线（例如 Line1）必须接到每台 slave 的触发输入线（例如 Line0）。
- 确认 master 的输出信号源已配置：例如 `--master-line-source ExposureStartActive`。
- 想先排除硬触发链路问题，可临时切到纯软件触发验证采集链路：
	- `--trigger-source Software --soft-trigger-fps 5`
