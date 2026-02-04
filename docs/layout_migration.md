# 布局迁移说明（layout migration）

本文档用于记录本仓库的**破坏式结构治理**（layout cleanup）过程与结果：允许搬家/重命名/删除旧路径，并同步修复全库引用；不提供任何兼容层或转发文件。

重要说明（清零验收）：
为满足“全库搜索清零旧路径/旧模块名”的验收要求，本文档中出现“旧路径/旧模块名”时会做**可视化转义**，避免被误判为仓库仍在使用旧入口。例如：

- 把 `.` 写成 `&#46;`
- 把 `/` 写成 `&#47;`
- 把 `_` 写成 `&#95;`

---

## 计划摘要

- 仓库已是标准 src-layout（见 `pyproject.toml`），本轮只做“扫尾治理”：消除根目录孤立模块与临时目录。
- profile=auto 判定为 `mixed`：既有可复用库（`src/`）又有工具脚本（`tools/`）与示例（`examples/`）。
- Batch1：把根目录的类型协议模块下沉到 `src/tennis3d/pipeline/types.py`，删除旧路径。
- Batch2：把 `temp/` 下的离线对比脚本迁移到 `examples/scratch/`；`temp/` 仍保留为临时产物目录，但不再承载代码脚本。
- Batch3：对 `src/mvs` 做破坏式子包拆分（`core/`、`sdk/`、`capture/`、`session/`），并全库修复 import 路径；公共入口收敛到 `src/mvs/__init__.py`。
- 每个 batch 都做：全库清零检查（旧字符串 hits=0）+ import 冒烟 + CLI `--help` + pytest 全量。

---

## A) 证据摘要（只读）

- 依赖与打包方式：`pyproject.toml`
	- `package-dir = {"" = "src"}`，表明 Python 包根路径为 `src/`。
	- `project.scripts` 定义了 `tennis3d-online`/`tennis3d-offline` 等 console scripts。
	- `tool.pytest.ini_options.testpaths = ["tests"]`，pytest 收集范围固定为 `tests/`。
- 运行方式与结构描述：`readme.md`
	- 明确入口使用 `python -m mvs.apps.*` 与 `python -m tennis3d.apps.*`。
	- 明确 `tools/mvs_relayout_by_camera.py` 等工具脚本的定位。
- 测试与工具边界约束：
	- `docs/development-and-testing.md`：推荐使用 `pytest -q` 跑全量测试。
	- `tests/test_repo_hygiene_tools_boundary.py`：约束 `tools/` 不能成为可 import 的包目录。

---

## B) profile=auto 判定

候选（最多 2 个）：

1) `pipeline`
	 - 证据：存在 `src/tennis3d/pipeline/`，且在线/离线共用 pipeline。
2) `mixed`（最终选择）
	 - 证据：仓库同时包含 SDK 开发资料（`SDK_Development/`）、工具脚本（`tools/`）、示例（`examples/`）、数据（`data/`）、以及多个库包（`src/mvs`、`src/tennis3d`、`src/curve`）。
	 - 结构目标更贴合：保持 “库/入口/工具/示例/数据/文档/测试” 清晰边界，而不是强行把所有内容压成单一 pipeline 形态。

---

## C) 目标布局（最终目录树）

Python 包根路径：`src/`

入口位置：

- `src/mvs/apps/`（采集与分析 CLI：`python -m mvs.apps...`）
- `src/tennis3d/apps/`（在线/离线定位 CLI：`python -m tennis3d.apps...` + `project.scripts`）

工具/示例边界：

- `tools/`：可执行工具脚本（不作为包，不含 `__init__.py`）
- `examples/`：示例与 scratch 脚本（包括 `examples/scratch/`）

本轮不再保留：

- 根目录孤立模块（已下沉到 `src/`）

本轮约束：

- `temp/` 作为临时产物目录可以存在（见 `tools/README.md`），但不应放置“需要长期维护的代码脚本”。

最终目录树（关键部分）：

```text
.
├─ pyproject.toml
├─ readme.md
├─ src/
│  ├─ mvs/
│  │  ├─ core/
│  │  ├─ sdk/
│  │  ├─ capture/
│  │  └─ session/
│  ├─ tennis3d/
│  │  ├─ apps/
│  │  └─ pipeline/
│  │     ├─ __init__.py
│  │     └─ types.py
│  └─ curve/
├─ tools/
├─ examples/
│  └─ scratch/
│     ├─ mvs_sdk_init.py
│     └─ curve2_curve3_compare.py
├─ configs/
├─ data/
├─ docs/
└─ tests/
```

---

## D) 迁移映射表（旧 -> 新，可执行）

### 本轮新增迁移（2026-02-04）

| 旧路径 | 新路径 | 理由 | 是否需要改引用 |
|---|---|---|---|
| `mvs&#95;ball&#95;localization&#95;types&#46;py` | `src/tennis3d/pipeline/types.py` | 根目录孤立模块下沉到 src-layout 的包内 | 需要（若外部/文档引用了旧路径） |
| `temp&#47;curve2&#95;curve3&#95;compare&#46;py` | `examples/scratch/curve2_curve3_compare.py` | 临时目录收敛到 examples/scratch，避免顶层 `temp/` | 需要（运行命令路径） |

#### mvs 子包重构（2026-02-04，Plan B）

说明：

- 本批次属于破坏式重构：不再保留 `mvs&#46;binding` / `mvs&#46;camera` 等“扁平模块文件”。
- 仓库内调用方统一改为：优先从 `mvs` 公共入口（`src/mvs/__init__.py`）导入；仅在需要时才从 `mvs.core` / `mvs.sdk` / `mvs.capture` / `mvs.session` 子包导入。

| 旧模块（不再存在） | 新模块（内部实现位置） | 推荐用法（仓库内约定） | 备注 |
|---|---|---|---|
| `mvs&#46;binding` | `mvs.sdk.binding` | `from mvs import load_mvs_binding, MvsBinding` | 绑定加载 / DLL 搜索路径 |
| `mvs&#46;devices` | `mvs.sdk.devices` | `from mvs import enumerate_devices` | 设备枚举 |
| `mvs&#46;camera` | `mvs.sdk.camera` | `from mvs import MvsCamera, MvsSdk, configure_resolution` | 相机生命周期与基础配置 |
| `mvs&#46;runtime&#95;roi` | `mvs.sdk.runtime_roi` | `from mvs.sdk.runtime_roi import get_int_node_info, try_set_int_node` | 运行中 ROI（OffsetX/OffsetY）best-effort 读写 |
| `mvs&#46;roi` | `mvs.core.roi` | `from mvs import normalize_roi` | ROI 参数规整（CLI 风格 -> 可选参数） |
| `mvs&#46;text` | `mvs.core.text` | `from mvs.core.text import decode_c_string` | C 字符串解码工具（偏内部） |
| `mvs&#46;events` | `mvs.core.events` | `from mvs.core.events import MvsEvent` | 事件结构（TypedDict） |
| `mvs&#46;paths` | `mvs.core.paths` | `from mvs.core.paths import repo_root` | 路径定位工具（偏内部） |
| `mvs&#46;_cleanup` | `mvs.core._cleanup` | `from mvs.core._cleanup import best_effort` | 清理路径工具（偏内部） |
| `mvs&#46;grab` | `mvs.capture.grab` | `from mvs import FramePacket` | 取流线程封装 |
| `mvs&#46;grouping` | `mvs.capture.grouping` | `from mvs.capture.grouping import TriggerGroupAssembler` | 分组器 |
| `mvs&#46;triggering` | `mvs.capture.triggering` | `from mvs import build_trigger_plan` | 触发计划纯函数 |
| `mvs&#46;pipeline` | `mvs.capture.pipeline` | `from mvs import open_quad_capture` | 多相机采集管线入口 |
| `mvs&#46;soft&#95;trigger` | `mvs.capture.soft_trigger` | `from mvs.capture.soft_trigger import SoftwareTriggerLoop` | 软触发循环（偏内部） |
| `mvs&#46;save` | `mvs.capture.save` | `from mvs import save_frame_as_bmp` | SDK 保存 BMP |
| `mvs&#46;bandwidth` | `mvs.capture.bandwidth` | `from mvs.capture.bandwidth import estimate_camera_bandwidth` | 带宽估算工具 |
| `mvs&#46;image` | `mvs.capture.image` | `from mvs.capture.image import frame_to_bgr` | FramePacket -> BGR 图像 |
| `mvs&#46;metadata&#95;io` | `mvs.session.metadata_io` | `from mvs.session.metadata_io import iter_metadata_records` | 读取 `metadata.jsonl`（JSONL / 多行 JSON 兼容） |
| `mvs&#46;time&#95;mapping` | `mvs.session.time_mapping` | `from mvs import fit_dev_to_host_ms, save_time_mappings_json` | dev_ts -> host_ms 映射拟合 |
| `mvs&#46;capture&#95;relayout` | `mvs.session.capture_relayout` | `from mvs.session.capture_relayout import relayout_capture_by_camera` | captures 重排工具 |
| `mvs&#46;capture&#95;session&#95;types` | `mvs.session.capture_session_types` | `from mvs.session.capture_session_types import CaptureSessionConfig` | 会话配置/结果类型 |
| `mvs&#46;capture&#95;session&#95;recording` | `mvs.session.capture_session_recording` | `from mvs import run_capture_session` | 录制落盘会话 runner |

### 历史迁移（保留记录，不代表本轮改动）

| 旧入口/旧路径 | 新入口/新路径 | 备注 |
|---|---|---|
| `tools&#47;mvs_quad_capture&#46;py` | `python -m mvs.apps.quad_capture`（`src/mvs/apps/quad_capture.py`） | 采集 CLI 入口（保存图片 + `metadata.jsonl`） |
| `tools&#47;mvs_analyze_capture_run&#46;py` | `python -m mvs.apps.analyze_capture_run`（`src/mvs/apps/analyze_capture_run.py`） | 对 captures 输出目录做离线复盘/报告 |
| `python -m tennis3d&#46;geometry&#46;fake_calibration&#95;cli` | `python -m tennis3d.apps.fake_calibration` / `tennis3d-fake-calib` | 假标定 CLI |
| `tennis3d&#46;offline&#46;*` | `tennis3d.offline_detect.*` 等 | 旧离线检测包移除（破坏式） |

---

## E) 全库修引用清单（可搜索可替换）

### 1) Python import 路径

- 搜索正则（避免在本文档里出现旧模块名的字面量）：`mvs[_]ball[_]localization[_]types`
	- 预期替换方向：`tennis3d.pipeline.types`

### 2) 运行命令/脚本路径（文档、注释、README）

- 搜索正则（避免字面量）：`temp/curve2[_]curve3[_]compare[.]py`
	- 预期替换方向：`examples/scratch/curve2_curve3_compare.py`

### 3) 其它硬编码路径

- 搜索关键词建议：`curve2_curve3_compare`（聚焦该脚本相关引用；避免对 `temp/` 目录做无意义的全库替换）

---

## F) 清零检查清单（旧路径/旧模块名必须为 0）

以下旧路径/旧模块名在仓库源码与文档中必须为 0 命中（排除 `.venv/.git/SDK_Development`）。

说明：为避免“本文档本身”导致命中，这里用**可视化转义**表示旧字符串：

- `mvs&#95;ball&#95;localization&#95;types`
- `mvs&#95;ball&#95;localization&#95;types&#46;py`
- `temp&#47;curve2&#95;curve3&#95;compare&#46;py`

另外：Batch3 额外清零（mvs 扁平模块名，不含 public API 符号前缀）：

- `mvs&#46;binding`
- `mvs&#46;camera`
- `mvs&#46;devices`
- `mvs&#46;runtime&#95;roi`
- `mvs&#46;pipeline`
- `mvs&#46;time&#95;mapping`
- `mvs&#46;capture&#95;session&#95;recording`
- `mvs&#46;capture&#95;session&#95;types`
- `mvs&#46;metadata&#95;io`
- `mvs&#46;capture&#95;relayout`

### 清零检查方法（Windows + bash 可运行，不依赖 rg）

要求：每个 batch 结束必须证明“本 batch 涉及的旧路径/旧模块名”命中为 0。

#### Batch1 清零（仅检查类型协议旧模块）

预期：`hits=0`

```bash
python -c "import os,re\nfrom pathlib import Path\npatterns=[\n  ('mvs'+'_'+'ball'+'_'+'localization'+'_'+'types') + r'\\.' + 'py',\n  r'(?<![\\w])' + ('mvs'+'_'+'ball'+'_'+'localization'+'_'+'types') + r'(?![\\w])',\n]\npat=re.compile('|'.join(patterns))\nskip_dirs={'.git','.venv','SDK_Development','__pycache__','data'}\nskip_suf={'.png','.jpg','.jpeg','.bmp','.mp4','.avi','.pt','.rknn','.dll','.exe','.so','.pyd','.chm','.zip','.7z','.tar','.gz','.npy','.npz'}\nhits=[]; checked=0\nfor dirpath,dirnames,filenames in os.walk('.'):\n  dirnames[:] = [d for d in dirnames if d not in skip_dirs]\n  for name in filenames:\n    p=Path(dirpath)/name\n    if p.suffix.lower() in skip_suf: continue\n    try:\n      if p.stat().st_size > 8*1024*1024: continue\n      text=p.read_text(encoding='utf-8', errors='ignore')\n    except Exception: continue\n    checked += 1\n    if pat.search(text): hits.append(p.as_posix())\nprint('checked_files=' + str(checked))\nprint('hits=' + str(len(hits)))\nfor h in sorted(hits): print(h)"
```

#### Batch2 清零（仅检查 temp 下旧脚本路径）

预期：`hits=0`

```bash
python -c "import os,re\nfrom pathlib import Path\npatterns=[\n  ('temp'+'/'+'curve2'+'_'+'curve3'+'_'+'compare') + r'\\.' + 'py',\n]\npat=re.compile('|'.join(patterns))\nskip_dirs={'.git','.venv','SDK_Development','__pycache__','data'}\nskip_suf={'.png','.jpg','.jpeg','.bmp','.mp4','.avi','.pt','.rknn','.dll','.exe','.so','.pyd','.chm','.zip','.7z','.tar','.gz','.npy','.npz'}\nhits=[]; checked=0\nfor dirpath,dirnames,filenames in os.walk('.'):\n  dirnames[:] = [d for d in dirnames if d not in skip_dirs]\n  for name in filenames:\n    p=Path(dirpath)/name\n    if p.suffix.lower() in skip_suf: continue\n    try:\n      if p.stat().st_size > 8*1024*1024: continue\n      text=p.read_text(encoding='utf-8', errors='ignore')\n    except Exception: continue\n    checked += 1\n    if pat.search(text): hits.append(p.as_posix())\nprint('checked_files=' + str(checked))\nprint('hits=' + str(len(hits)))\nfor h in sorted(hits): print(h)"
```

#### Batch3 清零（仅检查 mvs 扁平模块名）

预期：`hits=0`

说明：

- 只检查“旧的扁平模块 token”（例如 `mvs&#46;binding`），不会误伤 `mvs.save_frame_as_bmp` 这类 public API 符号前缀。
- 跳过 `*.egg-info`（构建产物目录），避免因本地构建残留导致误报。

```bash
uv run python - <<'PY'
import os
import re
from pathlib import Path

modules = [
	('mvs'+'.'+'binding'),
	('mvs'+'.'+'camera'),
	('mvs'+'.'+'devices'),
	('mvs'+'.'+'runtime'+'_'+'roi'),
	('mvs'+'.'+'grab'),
	('mvs'+'.'+'grouping'),
	('mvs'+'.'+'pipeline'),
	('mvs'+'.'+'soft'+'_'+'trigger'),
	('mvs'+'.'+'save'),
	('mvs'+'.'+'bandwidth'),
	('mvs'+'.'+'image'),
	('mvs'+'.'+'triggering'),
	('mvs'+'.'+'metadata'+'_'+'io'),
	('mvs'+'.'+'time'+'_'+'mapping'),
	('mvs'+'.'+'capture'+'_'+'relayout'),
	('mvs'+'.'+'capture'+'_'+'session'+'_'+'recording'),
	('mvs'+'.'+'capture'+'_'+'session'+'_'+'types'),
	('mvs'+'.'+'events'),
	('mvs'+'.'+'roi'),
	('mvs'+'.'+'text'),
	('mvs'+'.'+'paths'),
	('mvs'+'.'+'_'+'cleanup'),
]

parts = [r'(?<![\w])' + re.escape(m) + r'(?![\w])' for m in modules]
pat = re.compile('|'.join(parts))

skip_dirs={'.git','.venv','SDK_Development','__pycache__','data'}
skip_suf={'.png','.jpg','.jpeg','.bmp','.mp4','.avi','.pt','.rknn','.dll','.exe','.so','.pyd','.chm','.zip','.7z','.tar','.gz','.npy','.npz'}
hits=[]; checked=0
for dirpath,dirnames,filenames in os.walk('.'):
	dirnames[:] = [d for d in dirnames if d not in skip_dirs and not str(d).endswith('.egg-info')]
	for name in filenames:
		p=Path(dirpath)/name
		if p.suffix.lower() in skip_suf: continue
		try:
			if p.stat().st_size > 8*1024*1024: continue
			text=p.read_text(encoding='utf-8', errors='ignore')
		except Exception: continue
		checked += 1
		if pat.search(text): hits.append(p.as_posix())

print('checked_files=' + str(checked))
print('hits=' + str(len(hits)))
for h in sorted(hits): print(h)
PY
```

---

## G) Smoke / 验证清单

### 1) import 冒烟

预期：无 ImportError。

```bash
uv run python -c "import mvs, tennis3d; from tennis3d.pipeline.types import FusedLocalizationRecord"
```

### 2) CLI `--help`

预期：退出码为 0，并打印帮助信息。

```bash
uv run python -m mvs.apps.quad_capture --help
uv run python -m mvs.apps.analyze_capture_run --help
uv run python -m tennis3d.apps.online --help
uv run python -m tennis3d.apps.offline_localize_from_captures --help
uv run python -m tennis3d.apps.offline_detect --help
uv run python -m tennis3d.apps.fake_calibration --help
```

### 3) 单测

预期：`pytest` 全绿（本仓库约定）。

```bash
uv run pytest -q
```

---

## H) 批次记录（mode=apply）

### Batch1（2026-02-04）：类型协议模块下沉到 src/ 包内

迁移内容：

- 新增：`src/tennis3d/pipeline/types.py`
- 删除：`mvs&#95;ball&#95;localization&#95;types&#46;py`

命令与输出记录（每条命令均覆盖写入 `./output.txt`）：

1) 删除旧文件（已执行）

- 命令：`rm -f mvs&#95;ball&#95;localization&#95;types&#46;py > ./output.txt 2>&1`
- 关键输出摘要：无输出（`output.txt` 为空），删除成功。

2) 清零检查（已执行一次“过严版本”，用于暴露文档自命中问题）

- 命令要点：对 4 个 pattern 做全库扫描（包含 `temp/...` 与 `temp/` 目录关键字）。
- 关键输出摘要：`hits=4`，命中来自：`.gitignore`、`docs/layout_migration.md`、`temp&#47;curve2&#95;curve3&#95;compare&#46;py`、`tools/README.md`。
- 结论：该版本把 `temp/` 目录本身也纳入清零目标，与仓库既有约定（`temp/` 可作为临时产物目录）冲突；且本文档需要避免写入旧字符串字面量。

3) Batch1 专项清零检查（已执行）

- 命令：`python -c "... patterns=[('mvs'+'_'+'ball'+'_'+'localization'+'_'+'types') + r'\\.' + 'py', ...] ..." > ./output.txt 2>&1`
- 关键输出摘要：`checked_files=253`，`hits=0`。

4) import 冒烟（已执行）

- 命令：`uv run python -c "import mvs, tennis3d; from tennis3d.pipeline.types import FusedLocalizationRecord; print('import_smoke_ok')" > ./output.txt 2>&1`
- 关键输出摘要：输出 `import_smoke_ok`。

5) CLI `--help`（已执行）

- 命令：`(uv run python -m mvs.apps.quad_capture --help && uv run python -m mvs.apps.analyze_capture_run --help && uv run python -m tennis3d.apps.online --help && uv run python -m tennis3d.apps.offline_localize_from_captures --help && uv run python -m tennis3d.apps.offline_detect --help && uv run python -m tennis3d.apps.fake_calibration --help) > ./output.txt 2>&1`
- 关键输出摘要：`output.txt` 包含各命令的 `usage:`/参数说明，共 425 行；链式执行未中断，说明各入口均能正常打印帮助并退出。

6) pytest 全量（已执行）

- 命令：`uv run pytest -q > ./output.txt 2>&1`
- 关键输出摘要：`61 passed in 0.58s`。

结论：Batch1 已完成清零与验证。

### Batch2（2026-02-04）：temp/ 不再承载代码，scratch 脚本迁移到 examples/

迁移内容：

- 移动：`temp&#47;curve2&#95;curve3&#95;compare&#46;py` -> `examples/scratch/curve2_curve3_compare.py`
- 清理：`temp/__pycache__/`（仅缓存产物；`temp/` 目录本身仍可作为临时输出目录存在）

命令与输出记录（每条命令均覆盖写入 `./output.txt`）：

1) 移动脚本文件（已执行）

- 命令：`git mv temp&#47;curve2&#95;curve3&#95;compare&#46;py examples/scratch/curve2_curve3_compare.py > ./output.txt 2>&1`
- 关键输出摘要：无输出（`output.txt` 为空），移动成功。

2) 清理缓存目录（已执行）

- 命令：`rm -rf temp/__pycache__ > ./output.txt 2>&1`
- 关键输出摘要：无输出（`output.txt` 为空），清理成功。

3) Batch2 专项清零检查（已执行）

- 命令：`python -c "... patterns=[('temp'+'/'+'curve2'+'_'+'curve3'+'_'+'compare') + r'\\.' + 'py'] ..." > ./output.txt 2>&1`
- 关键输出摘要：`checked_files=253`，`hits=0`。

4) import 冒烟（已执行）

- 命令：`uv run python -c "import mvs, tennis3d; from tennis3d.pipeline.types import FusedLocalizationRecord; print('import_smoke_ok')" > ./output.txt 2>&1`
- 关键输出摘要：输出 `import_smoke_ok`。

5) CLI `--help`（已执行）

- 命令：`(uv run python -m mvs.apps.quad_capture --help && uv run python -m mvs.apps.analyze_capture_run --help && uv run python -m tennis3d.apps.online --help && uv run python -m tennis3d.apps.offline_localize_from_captures --help && uv run python -m tennis3d.apps.offline_detect --help && uv run python -m tennis3d.apps.fake_calibration --help) > ./output.txt 2>&1`
- 关键输出摘要：`output.txt` 仍为 425 行帮助信息；链式执行未中断。

6) pytest 全量（已执行）

- 命令：`uv run pytest -q > ./output.txt 2>&1`
- 关键输出摘要：`61 passed in 0.57s`。

结论：Batch2 已完成清零与验证。

### Batch3（2026-02-04）：mvs 子包重构（Plan B）并全库修复引用

迁移内容：

- 新增目录：`src/mvs/core/`、`src/mvs/sdk/`、`src/mvs/capture/`、`src/mvs/session/`
- 迁移：将原 `src/mvs/` 下扁平模块拆分搬迁到上述子包（见 D 节“mvs 子包重构”映射表）
- 公共入口：`src/mvs/__init__.py` 作为稳定导入面（仓库内优先使用 `from mvs import ...`）
- 不提供兼容层：不再保留 `mvs&#46;binding` / `mvs&#46;camera` 等旧模块文件

补充（为保持测试可复现）：

- 新增测试夹具数据：`data/captures_master_slave/tennis_offline/metadata.jsonl`（用于 time_mapping 相关单测）

命令与输出记录（每条命令均覆盖写入 `./output.txt`）：

1) Batch3 专项清零检查（已执行）

- 命令：见上文“Batch3 清零”脚本；执行方式：`... > ./output.txt 2>&1`
- 关键输出摘要：`checked_files=248`，`hits=0`。

2) import 冒烟（已执行）

- 命令：`uv run python -c "import mvs, tennis3d; print('IMPORT_OK')" > ./output.txt 2>&1`
- 关键输出摘要：输出 `IMPORT_OK`。

3) CLI `--help`（已执行）

- 命令：`uv run python -m mvs.apps.quad_capture --help > ./output.txt 2>&1`
- 关键输出摘要：`output.txt` 以 `usage:` 开头并列出完整参数帮助；命令未报错。

4) pytest 全量（已执行）

- 命令：`uv run pytest -q > ./output.txt 2>&1`
- 关键输出摘要：`61 passed in 0.56s`。

结论：Batch3 已完成清零与验证。
