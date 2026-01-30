"""RKNN 运行时加载（适配器层）。

说明：
- 该模块只负责“找到并初始化可用的 RKNN 运行时”。
- 通过函数内延迟 import，避免在 Windows/CI 环境里因缺少 SDK 而 import 失败。

注意：
- 这里的错误信息与旧实现保持一致，避免改变既有行为。
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
    """以统一方式调用 load/init。

    该函数是纯粹的去重封装，不改变外部行为：
    - load_rknn / init_runtime 的返回码为 0 表示成功，否则抛出 RuntimeError。
    """

    ret = rknn_obj.load_rknn(str(model_path))
    if ret != 0:
        raise RuntimeError(f"load_rknn failed, ret={ret}")

    ret = rknn_obj.init_runtime()
    if ret != 0:
        raise RuntimeError(f"init_runtime failed, ret={ret}")

    return rknn_obj


def _try_load_rknnlite(model_path: Path) -> _RKNNRuntime | None:
    """尝试用 rknnlite 初始化运行时，失败则返回 None。

    注意：
    - 与旧实现保持一致：任何异常都会被吞掉并继续尝试 rknn.api。
    """

    try:
        from rknnlite.api import RKNNLite  # type: ignore

        rknn = RKNNLite()
        return _load_with_runtime_api(rknn, model_path)
    except Exception:
        return None


def _try_load_rknn_api(model_path: Path) -> tuple[_RKNNRuntime | None, Exception | None]:
    """尝试用 rknn.api 初始化运行时。

    Returns:
        (rknn_obj, err)
        - 成功时 err 为 None
        - 失败时 rknn_obj 为 None，err 为捕获到的异常（用于拼接原始错误信息）
    """

    try:
        from rknn.api import RKNN  # type: ignore

        rknn = RKNN()
        return _load_with_runtime_api(rknn, model_path), None
    except Exception as e:
        return None, e


def load_rknn_runtime(model_path: Path) -> Any:
    """加载并初始化 RKNN 运行时。

    优先级：
    1) rknnlite（板端运行时）
    2) rknn.api（PC/Linux 工具链）

    Args:
        model_path: .rknn 模型路径。

    Returns:
        已初始化的 RKNN/RKNNLite 对象（具体类型取决于运行环境）。

    Raises:
        RuntimeError: 当无法导入/初始化任何 RKNN 运行时。
    """

    model_path = Path(model_path)

    # 优先尝试 rknnlite（板端），其次 rknn.api（PC/Linux 工具链）
    rknn = _try_load_rknnlite(model_path)
    if rknn is not None:
        return rknn

    rknn, err = _try_load_rknn_api(model_path)
    if rknn is not None:
        return rknn

    # err 理论上必然存在；保留显式防御，避免未来维护误改导致 None 泄露。
    if err is None:
        err = RuntimeError("未知错误")

    raise RuntimeError(
        "未找到可用的 RKNN 运行时。\n"
        "- 若在 Rockchip 设备上运行：请安装 rknnlite\n"
        "- 若在 Linux/x86 上做工具链推理：请安装 rknn-toolkit2（通常不支持 Windows）\n"
        f"原始错误: {err}"
    )
