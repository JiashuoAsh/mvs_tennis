"""在线入口（CLI / python -m）。

该模块是在线模式的“薄入口层”，只负责：
- 解析 CLI 参数
- （Optional）加载配置文件
- 构建运行规格并校验
- 调用运行循环 `tennis3d_online.runtime.run_online`
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

from tennis3d.config import load_online_app_config

from .cli import build_arg_parser
from .runtime import run_online
from .spec import build_spec_from_args, build_spec_from_config


def main(argv: Optional[Sequence[str]] = None) -> int:
    """在线模式主入口。"""

    # 尽量固定 UTF-8 输出，避免在重定向到文件时出现乱码。
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)

    try:
        config_raw = str(getattr(args, "config", "") or "").strip()
        if config_raw:
            cfg = load_online_app_config(Path(config_raw).resolve())
            spec = build_spec_from_config(cfg)
        else:
            spec = build_spec_from_args(args)
    except ValueError as exc:
        print(str(exc))
        return 2
    except RuntimeError as exc:
        print(str(exc))
        return 2

    return int(run_online(spec))
