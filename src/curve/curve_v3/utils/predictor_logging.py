"""curve_v3 预测器相关的日志工具。

说明：
    curve_v3 作为轻依赖模块，不强制要求外部提供特定的日志框架。
    这里提供一个“可用即可”的默认 logger，避免在脚本/单测环境中出现
    无 handler 导致的静默。
"""

from __future__ import annotations

import logging


def default_logger() -> logging.Logger:
    """获取 curve_v3 的默认 logger。

    Returns:
        标准库 `logging.Logger` 实例。
    """

    logger = logging.getLogger("curve_v3")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
