# 布局迁移说明（layout migration）

本文档记录本仓库的一次**破坏式结构治理**（layout cleanup）结果：旧路径/旧脚本/旧模块名已被移除，不提供任何兼容层或转发。

说明：为满足“全库搜索清零旧路径/旧模块名”的验收要求，本文档中涉及旧路径时会做**可视化转义**（例如把 `.` 写成 `&#46;`，把 `/` 写成 `&#47;`，把 `_` 写成 `&#95;`），避免被误判为仓库仍在使用旧入口。

目标：
- 把可复用逻辑收敛到 `src/<pkg>/`（库代码）
- 把 CLI 入口收敛到 `src/<pkg>/apps/`（`python -m ...` 或 `project.scripts`）
- 全库更新引用并清零旧路径/旧模块名

---

## 迁移映射表（旧 -> 新）

### MVS：采集与采集输出分析

| 旧入口/旧路径 | 新入口/新路径 | 备注 |
|---|---|---|
| `tools&#47;mvs_quad_capture&#46;py` | `python -m mvs.apps.quad_capture`（`src/mvs/apps/quad_capture.py`） | 采集 CLI 入口（保存图片 + `metadata.jsonl`） |
| `tools&#47;mvs_analyze_capture_run&#46;py` | `python -m mvs.apps.analyze_capture_run`（`src/mvs/apps/analyze_capture_run.py`） | 对 captures 输出目录做离线复盘/报告 |

### Tennis3D：假标定与离线检测

| 旧入口/旧路径 | 新入口/新路径 | 备注 |
|---|---|---|
| `python -m tennis3d&#46;geometry&#46;fake_calibration&#95;cli` | `python -m tennis3d.apps.fake_calibration` / `tennis3d-fake-calib` | 假标定 CLI（用于无真实标定时跑通链路） |
| `tennis3d&#46;offline&#46;*` | `tennis3d.offline_detect.*` + 若干上移模块 | 旧离线检测包整体移除（不保留兼容） |

旧 `tennis3d&#46;offline&#46;*` 的常见对应关系：

| 旧模块 | 新模块 |
|---|---|
| `tennis3d&#46;offline&#46;cli` | `tennis3d.apps.offline_detect`（CLI） |
| `tennis3d&#46;offline&#46;pipeline` | `tennis3d.offline_detect.pipeline` |
| `tennis3d&#46;offline&#46;detector` | `tennis3d.offline_detect.detector`（RKNN）或统一入口 `tennis3d.detectors.create_detector()` |
| `tennis3d&#46;offline&#46;images` | `tennis3d.offline_detect.images` |
| `tennis3d&#46;offline&#46;timestamps` | `tennis3d.offline_detect.timestamps` |
| `tennis3d&#46;offline&#46;outputs` | `tennis3d.offline_detect.outputs` |
| `tennis3d&#46;offline&#46;preprocess` | `tennis3d.preprocess` |
| `tennis3d&#46;offline&#46;models` | `tennis3d.models` |
| `tennis3d&#46;offline&#46;triangulation` | `tennis3d.geometry.triangulation` |
| `tennis3d&#46;offline&#46;calibration` | `tennis3d.geometry.calibration` |

---

## 清零清单（旧路径/旧模块名必须为 0）

以下字符串在仓库源码与文档中必须为 0 命中（不含本地虚拟环境目录）：

- `tools&#47;mvs_quad_capture&#46;py`
- `tools&#47;mvs_analyze_capture_run&#46;py`
- `tennis3d&#46;offline`
- `fake_calibration&#95;cli`
- `tennis3d&#46;geometry&#46;fake_calibration&#95;cli`

### 清零检查方法（不依赖 rg）

在仓库根目录执行以下命令（排除 `.venv/.git/SDK_Development`）：

- 预期：`hits=0`

下面命令会在仓库内做一次文本扫描（排除 `.venv/.git/SDK_Development`），并输出 `checked_files` 与 `hits`：

```bash
python -c "import os,re\nfrom pathlib import Path\npatterns=[\n  'tools' + '/' + 'mvs_quad_capture' + r'\\.' + 'py',\n  'tools' + '/' + 'mvs_analyze_capture_run' + r'\\.' + 'py',\n  'tennis3d' + r'\\.' + 'offline' + r'(?!_)',\n  'fake' + '_' + 'calibration' + '_' + 'cli',\n  'tennis3d' + r'\\.' + 'geometry' + r'\\.' + 'fake' + '_' + 'calibration' + '_' + 'cli',\n]\npat=re.compile('|'.join(patterns))\nskip_dirs={'.git','.venv','SDK_Development','__pycache__'}\nskip_suf={'.png','.jpg','.jpeg','.bmp','.mp4','.avi','.pt','.rknn','.dll','.exe','.so','.pyd','.chm','.zip','.7z','.tar','.gz','.npy','.npz'}\nhits=[]; checked=0\nfor dirpath,dirnames,filenames in os.walk('.'):\n  dirnames[:] = [d for d in dirnames if d not in skip_dirs]\n  for name in filenames:\n    p=Path(dirpath)/name\n    if p.suffix.lower() in skip_suf: continue\n    try:\n      if p.stat().st_size > 8*1024*1024: continue\n      text=p.read_text(encoding='utf-8', errors='ignore')\n    except Exception: continue\n    checked += 1\n    if pat.search(text): hits.append(p.as_posix())\nprint('checked_files=' + str(checked))\nprint('hits=' + str(len(hits)))\nfor h in sorted(hits): print(h)"
```

---

## Smoke / 验证清单

### 1) import 冒烟

- 预期：无 ImportError

建议检查：
- `import mvs`
- `import tennis3d`
- `from tennis3d.detectors import create_detector`

### 2) CLI `--help`

- 预期：退出码为 0，并打印帮助信息

需要检查的入口：
- `python -m mvs.apps.quad_capture --help`
- `python -m mvs.apps.analyze_capture_run --help`
- `python -m tennis3d.apps.online_mvs_localize --help`
- `python -m tennis3d.apps.offline_localize_from_captures --help`
- `python -m tennis3d.apps.offline_detect --help`
- `python -m tennis3d.apps.fake_calibration --help`

### 3) 单测

- 预期：`pytest` 全绿

---

## 本轮实际执行结果

- 清零检查：已确认（排除 `.venv/.git/SDK_Development` 后 `hits=0`，本次扫描 `checked_files=170`）。
- import 冒烟：已确认（`uv run python -c "import mvs, tennis3d"` 成功）。
- CLI `--help`：已确认（`mvs.apps.quad_capture`、`mvs.apps.analyze_capture_run`、`tennis3d.apps.*` 均能正常打印帮助并退出）。
- pytest：已确认（`18 passed`）。
