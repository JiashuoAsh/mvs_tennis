"""RKNN 运行时加载（适配器层）。

说明：
- 该模块只负责“找到并初始化可用的 RKNN 运行时”。
- 通过函数内延迟 import，避免在 Windows/CI 环境里因缺少 SDK 而 import 失败。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class _RKNNRuntime(Protocol):
    """RKNN 运行时对象的最小协议。

    说明：
    - 这里不强依赖具体 SDK 类型，仅约束本仓库会用到的最小方法集合。
    - 该协议仅用于类型提示，不会影响运行时行为。
    """

    def load_rknn(self, model_path: str) -> int:  # noqa: D401 - 协议方法无需完整文档
        ...

    def init_runtime(self) -> int:  # noqa: D401 - 协议方法无需完整文档
        ...

    def inference(self, inputs: list[Any]) -> list[Any]:  # noqa: D401 - 协议方法无需完整文档
        ...


def _load_with_runtime_api(rknn_obj: _RKNNRuntime, model_path: Path) -> _RKNNRuntime:
    """以统一方式调用 load/init。"""

    ret = rknn_obj.load_rknn(str(model_path))
    if ret != 0:
        raise RuntimeError(f"load_rknn failed, ret={ret}")

    ret = rknn_obj.init_runtime()
    if ret != 0:
        raise RuntimeError(f"init_runtime failed, ret={ret}")

    return rknn_obj


def _try_load_rknnlite(model_path: Path) -> _RKNNRuntime | None:
    """尝试用 rknnlite 初始化运行时，失败则返回 None。"""

    try:
        from rknnlite.api import RKNNLite  # type: ignore

        rknn = RKNNLite()
        return _load_with_runtime_api(rknn, model_path)
    except Exception:
        return None


def _try_load_rknn_api(model_path: Path) -> tuple[_RKNNRuntime | None, Exception | None]:
    """尝试用 rknn.api 初始化运行时。"""

    try:
        from rknn.api import RKNN  # type: ignore

        rknn = RKNN()
        return _load_with_runtime_api(rknn, model_path), None
    except Exception as e:
        return None, e


def load_rknn_runtime(model_path: Path) -> Any:
    """加载并初始化 RKNN 运行时。"""

    model_path = Path(model_path)

    # 优先尝试 rknnlite（板端），其次 rknn.api（PC/Linux 工具链）
    rknn = _try_load_rknnlite(model_path)
    if rknn is not None:
        return rknn

    rknn, err = _try_load_rknn_api(model_path)
    if rknn is not None:
        return rknn

    if err is None:
        err = RuntimeError("未知错误")

    raise RuntimeError(
        "未找到可用的 RKNN 运行时。\n"
        "- 若在 Rockchip 设备上运行：请安装 rknnlite\n"
        "- 若在 Linux/x86 上做工具链推理：请安装 rknn-toolkit2（通常不支持 Windows）\n"
        f"原始错误: {err}"
    )
