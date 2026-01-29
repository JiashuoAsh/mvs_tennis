"""MVS SDK 初始化/反初始化最小示例（实验脚本）。

说明：
- 这是对供应商示例的最小裁剪，用于验证本机 MVS SDK 是否可被 Python 导入与初始化。
- 该脚本不属于仓库核心库代码，不参与单测，也不保证跨机器可用。
"""

from __future__ import annotations

import importlib
import sys


def main() -> int:
    # 说明：供应商 Python 示例通常要求把 MvImport 目录加入 sys.path。
    # 你需要按本机 MVS 安装路径调整该目录。
    sys.path.append("C:/Program Files (x86)/MVS/Development/Sample/python/MvImport")

    mv = importlib.import_module("MvCameraControl_class")
    # 说明：这里通过动态属性获取，避免静态分析器因找不到供应商模块而报错。
    MvCamera = getattr(mv, "MvCamera")

    _ = importlib.import_module("CameraParams_header")
    _ = importlib.import_module("MvErrorDefine_const")

    # 1. 初始化SDK
    MvCamera.MV_CC_Initialize()

    # 2. 进行设备发现，控制，图像采集等操作

    # 3. 反初始化SDK
    MvCamera.MV_CC_Finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
