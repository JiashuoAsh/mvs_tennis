# -*- coding: utf-8 -*-

"""运行中 ROI（OffsetX/OffsetY）读写辅助。

设计目标：
- 该模块只封装“GenICam int 节点”的读写与对齐逻辑；不涉及 Start/StopGrabbing。
- 在部分机型上，OffsetX/OffsetY 在 StartGrabbing 后依然可写（已实测）。
  但也可能失败（节点锁定/访问权限变化/固件差异）。因此这里提供 best-effort API：
  - 读取失败返回 None
  - 写入失败返回 (False, ret)

注意：
- 不在此模块做任何 DLL/绑定加载；调用方需要传入已加载的 binding 与 cam 句柄。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from .binding import MvsBinding


@dataclass(frozen=True)
class IntNodeInfo:
    """int 节点信息（与 MVCC_INTVALUE 对齐）。"""

    cur: int
    vmin: int
    vmax: int
    inc: int


def get_int_node_info(*, binding: MvsBinding, cam: Any, key: str) -> Optional[IntNodeInfo]:
    """读取 int 节点的当前值/最小/最大/步进。

    Args:
        binding: 已加载的 MVS binding。
        cam: SDK 相机句柄（MvCamera 实例）。
        key: 节点名，例如 "OffsetX" / "Width"。

    Returns:
        IntNodeInfo；读取失败返回 None。
    """

    try:
        st = binding.params.MVCC_INTVALUE()
        ret = cam.MV_CC_GetIntValue(str(key), st)
        if int(ret) != int(binding.MV_OK):
            return None
        return IntNodeInfo(
            cur=int(getattr(st, "nCurValue")),
            vmin=int(getattr(st, "nMin")),
            vmax=int(getattr(st, "nMax")),
            inc=int(getattr(st, "nInc")),
        )
    except Exception:
        return None


def clamp_and_align(
    value: int,
    *,
    info: IntNodeInfo,
    mode: Literal["nearest", "down"] = "nearest",
) -> int:
    """把目标值裁剪到 [min,max] 并按 inc 对齐。

    说明：
    - OffsetX/OffsetY 常见 Inc=4；不对齐会导致 SetIntValue 失败。
    - mode="nearest" 更适合 offset（减少系统性偏差）；mode="down" 更适合 Width/Height（避免变大）。
    """

    inc = int(info.inc) if int(info.inc) > 0 else 1
    vmin = int(info.vmin)
    vmax = int(info.vmax)

    v = int(value)
    if v < vmin:
        v = vmin
    if v > vmax:
        v = vmax

    if inc <= 1:
        return int(v)

    # 对齐到以 vmin 为起点的步进格。
    rel = int(v - vmin)
    if rel < 0:
        rel = 0

    if str(mode) == "down":
        steps = rel // inc
    else:
        # nearest：使用整数四舍五入（0.5 向上），避免 Python round 的 banker's rounding。
        steps = (rel + inc // 2) // inc

    out = int(vmin + steps * inc)
    if out < vmin:
        out = vmin
    if out > vmax:
        out = vmax
    return int(out)


def try_set_int_node(
    *,
    binding: MvsBinding,
    cam: Any,
    key: str,
    value: int,
) -> tuple[bool, int]:
    """best-effort 写入 int 节点。

    Returns:
        (ok, ret_code)
    """

    try:
        ret = int(cam.MV_CC_SetIntValue(str(key), int(value)))
    except Exception:
        # 异常通常意味着节点不存在/SDK 状态异常。
        return False, -1

    return (int(ret) == int(binding.MV_OK)), int(ret)
