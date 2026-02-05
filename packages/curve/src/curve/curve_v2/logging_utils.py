"""curve_v2 内部日志工具。

legacy 代码原本使用外部工程的 get_logger（带缓冲/节流等能力）。
为了不引入额外依赖，这里实现一个最小版本：
    - 支持控制台输出
    - 可选文件输出
    - 避免重复添加 handler

注意：
    - 为了保持 legacy 调用点不变，本函数签名兼容常见参数；未用到的参数会被忽略。
"""

from __future__ import annotations

import logging
from pathlib import Path


def get_logger(
    name: str,
    *,
    console_output: bool = True,
    file_output: bool = False,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    time_interval: float | None = None,  # noqa: ARG001
    use_buffering: bool = False,  # noqa: ARG001
    buffer_capacity: int = 0,  # noqa: ARG001
    flush_interval: float | None = None,  # noqa: ARG001
    log_file: str | None = None,
) -> logging.Logger:
    """创建或获取一个 logger。

    Args:
        name: logger 名称。
        console_output: 是否输出到控制台。
        file_output: 是否输出到文件。
        console_level: 控制台日志级别（字符串）。
        file_level: 文件日志级别（字符串）。
        log_file: 日志文件路径；为 None 时默认写到当前工作目录下的 `<name>.log`。

    Returns:
        logging.Logger: 配置完成的 logger。
    """

    logger = logging.getLogger(name)

    # 防止同名 logger 在多次构造时重复叠加 handler。
    if getattr(logger, "_curve_v2_configured", False):
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if console_output:
        ch = logging.StreamHandler()
        ch.setLevel(_parse_level(console_level))
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if file_output:
        path = Path(log_file or f"{name}.log")
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setLevel(_parse_level(file_level))
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    setattr(logger, "_curve_v2_configured", True)
    return logger


def _parse_level(level: str) -> int:
    """解析日志级别字符串。"""

    value = logging.getLevelName(level.upper())
    if isinstance(value, int):
        return value
    return logging.INFO
