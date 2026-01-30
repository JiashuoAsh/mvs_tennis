# -*- coding: utf-8 -*-

"""MVS ctypes 绑定加载与 DLL 搜索路径处理。

该模块的职责仅限于：
1) 把 MVS 官方 Python 示例的 `MvImport/` 放进 `sys.path`（让其可被 import）；
2) 在 Windows 上把 DLL 目录加入搜索路径（让 `MvCameraControl.dll` 可被加载）；
3) 把官方示例里分散的符号收拢成一个 `MvsBinding`，便于其它模块注入依赖。

设计要点：
- 延迟 import MvImport（因为 import 时会立即尝试加载 MvCameraControl.dll）。
- 在无 DLL 的环境下也能 import 本包，并输出清晰错误信息。
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from mvs.paths import repo_root


class MvsDllNotFoundError(RuntimeError):
    pass


# Windows 下 `os.add_dll_directory()` 返回的句柄需要保活；
# 否则对象被 GC 回收后，目录会自动从 DLL 搜索路径中移除。
_DLL_DIR_HANDLES: list[Any] = []


def _ensure_dll_dir(dll_dir: Path) -> None:
    """把 DLL 目录加入搜索路径。

    Windows：优先用 os.add_dll_directory（Python 3.8+），并同步追加到 PATH。
    """

    if not dll_dir.exists():
        return

    try:
        if hasattr(os, "add_dll_directory"):
            handle = os.add_dll_directory(str(dll_dir))
            # 句柄保活，避免目录被悄悄移除。
            _DLL_DIR_HANDLES.append(handle)
    except OSError:
        # 某些环境可能不允许添加目录；兜底走 PATH。
        pass

    os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")


def _ensure_mvimport_on_syspath(mvimport_dir: Path) -> None:
    # MvImport 目录内部使用同目录 import（不是标准包结构），必须把目录加到 sys.path。
    if str(mvimport_dir) not in sys.path:
        sys.path.insert(0, str(mvimport_dir))


@dataclass(frozen=True, slots=True)
class MvsBinding:
    """已加载的 MVS 绑定符号集合。"""

    MvCamera: Any
    params: Any
    err: Any

    @property
    def MV_OK(self) -> int:
        return int(self.err.MV_OK)

    @property
    def MV_GIGE_DEVICE(self) -> int:
        return int(self.params.MV_GIGE_DEVICE)

    @property
    def MV_USB_DEVICE(self) -> int:
        return int(self.params.MV_USB_DEVICE)


def load_mvs_binding(*, dll_dir: Optional[str] = None) -> MvsBinding:
    """加载 MVS Python 示例绑定。

    Args:
        dll_dir: 包含 MvCameraControl.dll 的目录（可选）。

    Returns:
        MvsBinding: MVS 绑定对象。

    Raises:
        MvsDllNotFoundError: 找不到 MvCameraControl.dll 或其依赖。
    """

    root = repo_root()
    mvimport_dir = root / "SDK_Development" / "Samples" / "Python" / "MvImport"
    if not mvimport_dir.exists():
        raise MvsDllNotFoundError(
            "MVS python bindings (MvImport) not found in this repo layout.\n"
            "找不到 MVS 官方 Python 示例绑定目录（MvImport）。\n"
            "\n"
            f"Expected: {mvimport_dir}\n"
            "\n"
            "解决方法：\n"
            "1) 确认本仓库包含 SDK_Development/Samples/Python/MvImport；\n"
            "2) 如果你移动了 SDK_Development 目录，请把它放回仓库根目录；\n"
            "3) 或者改用已安装的 MVS Python 包（若你的环境提供）。"
        )
    _ensure_mvimport_on_syspath(mvimport_dir)

    # 1) 用户指定目录
    if dll_dir:
        _ensure_dll_dir(Path(dll_dir))

    # 2) 环境变量
    env_dir = os.environ.get("MVS_DLL_DIR")
    if env_dir:
        _ensure_dll_dir(Path(env_dir))

    # 3) 工作区内候选目录（当前仓库通常没有 dll，但保留逻辑）
    _ensure_dll_dir(root / "SDK_Development" / "Bin" / "win64")
    _ensure_dll_dir(root / "SDK_Development" / "Bin" / "win32")

    try:
        mv = importlib.import_module("MvCameraControl_class")
        params = importlib.import_module("CameraParams_header")
        err = importlib.import_module("MvErrorDefine_const")
    except (FileNotFoundError, ImportError, OSError) as exc:
        raise MvsDllNotFoundError(
            "MVS DLL not found: MvCameraControl.dll (or dependency).\n"
            "找不到 MvCameraControl.dll（或其依赖）。\n"
            "\n"
            "解决方法：\n"
            "1) 安装海康 MVS（Machine Vision Software），确保系统 PATH 可找到 MvCameraControl.dll；\n"
            "2) 或使用参数 dll_dir / 环境变量 MVS_DLL_DIR 指向 DLL 目录；\n"
            "3) 注意：本仓库的 SDK_Development/Bin 下通常只有示例 exe，没有 dll。"
        ) from exc

    return MvsBinding(MvCamera=mv.MvCamera, params=params, err=err)
