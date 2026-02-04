# -*- coding: utf-8 -*-

"""ROI（相机侧裁剪）参数规整。

动机：
- 同一套 ROI 约束在多个 CLI/入口里重复实现，容易随时间分叉。
- 本模块作为“单一真相”，供 apps 层把 CLI 风格参数规整为 pipeline 可直接使用的参数。

约定：
- CLI 风格：0 表示“不设置”。
- 规整后：width/height 用 None 表示“不设置”；offset 始终是 int（默认 0）。
"""

from __future__ import annotations


def normalize_roi(
    *,
    image_width: int,
    image_height: int,
    image_offset_x: int,
    image_offset_y: int,
) -> tuple[int | None, int | None, int, int]:
    """把 ROI 参数从 CLI 风格（0 表示不设置）规整为 pipeline 使用的可选参数。

    规则：
    - 宽高必须同时设置（>0）或同时不设置（<=0）。
    - offset 始终返回 int；当 width/height 不设置时，offset 仍保留（由下游决定是否使用）。

    Raises:
        ValueError: 当只设置了宽或高其中一个时。
    """

    w = int(image_width or 0)
    h = int(image_height or 0)
    ox = int(image_offset_x or 0)
    oy = int(image_offset_y or 0)

    # 约束：宽高必须同时设置，避免出现“只裁宽不裁高”的歧义。
    if (w > 0) ^ (h > 0):
        raise ValueError("ROI 参数错误：image_width 与 image_height 必须同时设置，或同时为 0（不设置）。")

    if w <= 0 and h <= 0:
        return None, None, ox, oy

    return w, h, ox, oy
