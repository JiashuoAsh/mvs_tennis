你现在在一个 Python 工程里做一次“破坏性重构 + 功能落地”，目标是把多相机网球检测与定位做成可上线（online）与可离线调试（offline debug）两套流水线，并把代码结构一次性整理到位。

背景与参考：

* `process/ref_camera_processor.py`、`process/ref_Inference_seg.py` 仅作为参考示例：展示如何使用 `process/tennis_20241227.rknn` 做网球检测。请阅读并复用其中的关键调用方式/前后处理思路，但不要把它们当成最终架构。
* `msv` 是上游/SDK相关模块（可能是相机采集、数据封装等）。约束：不要改变 `msv` 的对外接口行为（可以在新代码里适配它，但不要破坏它）。

核心需求（必须全部满足）：

1. 多相机网球检测与“像素框 -> 3D位置”

* 输入来自 msv：多相机捕获到的图片流（online）或图片序列（offline）。
* 每个相机对网球做检测，得到 bbox（以及置信度等）。
* 基于相机内参/外参 + bbox 推出网球在世界坐标系的位置（你可以先实现一个合理的几何管线：例如 bbox center 反投影成视线 + 多目三角化；或用观测权重的最小二乘）。
* 目前没有真实内外参与真实检测数据：请先“编造一份可运行的样例数据”，包括：

  * 2~4 个相机的内参/外参（合理的数值与相对位置）
  * 一套示例图片输入方式（可以是随机图/空白图 + mock 检测输出，或提供一个可插拔的 Detector 接口，默认走 mock）
  * 一个清晰的数据格式（JSON/YAML/npz 均可），保证后续换成真实数据时改动最小

2. 两种模式 + 各自独立 entry（可直接运行）

* Online（实际应用）：从多相机实时采集进入流水线，要求考虑“相机同步问题”
* Offline（调试）：从磁盘读取多相机图片序列/录制数据进入流水线
* 两种模式都必须是“pipeline”形式，结构一致但 source 不同
* 两种模式都要支持“2 个以上相机”，相机数量可配置，不能写死

3. Online 模式触发/同步必须支持两类启动方式

* 纯 software 同步（软触发/软同步策略）
* master(software)-slave(硬触发) 同步方式，参考 `mvs_quad_capture` 的行为与概念（不要求一字不差复刻，但要提供同等级别的能力入口与配置）
* 输出的定位结果要携带时间戳/帧号，并提供同步策略（例如：按时间戳对齐、容忍窗口、丢帧策略）

4. 代码工程化与重构要求（一次性到位）

* 模块化：各模块要能独立运行与测试，做成高内聚低耦合的 Python 包（带清晰接口、可单测）
* 允许你大胆重构不合理的文件结构（除了 msv）
* 我对 `process/*` 这个名字不满意：请评估并重命名为更语义化的包名（例如 `perception/`、`vision/`、`ball_tracking/` 等），并完成全量迁移
* “破坏性更大”的一次性重构方案（必须执行）：

  * 把 `mvs/`、旧的 `process/`（以及你新建的相关包）全部迁入 `src/`
  * 修正所有 import、入口、默认路径、README
  * 把 `captures/`、`calibration/`、`tools_output/` 迁入 `data/`
  * 目标是一次干净利落：不要留兼容层/临时 shim（除非确实必要且你能解释原因）

实现建议（你可以按更优方式做，但需满足验收）：

* 采用“可插拔接口”分层：Source(online/offline) -> Sync/Align -> Detector(RKNN/mock) -> Geometry(Triangulation) -> Output(可视化/日志)
* 用 dataclass/typing 明确数据结构（Frame、CameraIntrinsics、Extrinsics、Detection、Track/Estimate）
* 提供基本单元测试（至少覆盖：配置加载、mock pipeline 运行、三角化/最小二乘输出稳定）
* 给出一个最小可运行 demo：不依赖真实相机也能跑通 offline（用 mock 输入/或 data 下样例）

最终交付（你输出给我的是代码变更方案/补丁级实现方向）：

* 新的目录结构与关键文件清单
* 两个 entry（online/offline）可运行
* pipeline 可跑通（至少 mock 数据）
* README 完整
* 说明你对“process 包重命名”的选择与理由

开始执行时请按这个顺序：

1. 先给出重构后的目录结构树（src/data/tests、包命名、entry 放哪）
2. 再实现最小可跑通的 offline pipeline（mock 数据 + mock detector + 几何定位）
3. 再接入 RKNN detector（复用参考文件的调用方式）
4. 最后补 online pipeline 框架与同步/触发配置入口（即使没有真实相机，也要保证结构正确、可扩展）

不要泛泛而谈，直接落到“工程可提交”的结构与实现。
