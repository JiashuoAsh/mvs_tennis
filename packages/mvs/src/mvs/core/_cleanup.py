# -*- coding: utf-8 -*-

"""清理/收尾相关的小工具。

设计目标：
- 只服务于 close()/finally 等“资源释放路径”；
- 清理过程中尽量不抛异常，避免遮蔽真正的业务异常；
- 保持极简，避免引入复杂的日志/重试策略。
"""

from __future__ import annotations

import threading
from typing import Any, Callable


def best_effort(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """尽力调用函数，忽略所有异常。

    Notes:
        该函数只应该用于清理路径：例如 close()、except/finally。
        主流程不要使用它，以免把真实错误“吞掉”。

    Args:
        fn: 待调用的函数/可调用对象。
        *args: 位置参数。
        **kwargs: 关键字参数。
    """

    try:
        fn(*args, **kwargs)
    except Exception:
        return


def join_quietly(t: threading.Thread, timeout_s: float) -> None:
    """尽力 join 线程，避免清理路径被异常打断。

    Args:
        t: 线程对象。
        timeout_s: join 超时（秒）。
    """

    try:
        t.join(timeout=float(timeout_s))
    except KeyboardInterrupt:
        # Ctrl+C：让上层有机会更快退出，这里不做额外处理。
        return
    except Exception:
        return
