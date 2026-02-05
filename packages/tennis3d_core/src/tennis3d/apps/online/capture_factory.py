"""在线模式：打开多相机采集（open_quad_capture 的 spec 适配层）。

职责：
- 把 `OnlineRunSpec` + `TriggerPlan` 映射为 `mvs.open_quad_capture` 的参数。

说明：
- 该模块属于 entry 层：负责把“运行规格”翻译成“硬件调用参数”。
- 这里不做业务逻辑（不跑 pipeline、不写输出）。
"""

from __future__ import annotations

from typing import Any

from mvs import TriggerPlan, open_quad_capture

from .spec import OnlineRunSpec


def open_online_quad_capture(*, binding: Any, spec: OnlineRunSpec, plan: TriggerPlan):
    """按在线 spec 打开多相机采集（返回上下文管理器对象）。"""

    return open_quad_capture(
        binding=binding,
        serials=spec.serials,
        trigger_sources=plan.trigger_sources,
        trigger_activation=str(spec.trigger_activation),
        trigger_cache_enable=bool(spec.trigger_cache_enable),
        timeout_ms=int(spec.timeout_ms),
        group_timeout_ms=int(spec.group_timeout_ms),
        max_pending_groups=int(spec.max_pending_groups),
        group_by=spec.group_by,
        enable_soft_trigger_fps=float(plan.enable_soft_trigger_fps),
        soft_trigger_serials=plan.soft_trigger_serials,
        camera_event_names=(),
        master_serial=str(spec.master_serial),
        master_line_output=str(spec.master_line_out) if spec.master_serial else "",
        master_line_source=str(spec.master_line_source) if spec.master_serial else "",
        master_line_mode=str(spec.master_line_mode) if spec.master_serial else "",
        pixel_format=str(spec.pixel_format),
        image_width=(int(spec.image_width) if spec.image_width is not None else None),
        image_height=(int(spec.image_height) if spec.image_height is not None else None),
        image_offset_x=int(spec.image_offset_x),
        image_offset_y=int(spec.image_offset_y),
        exposure_auto=str(spec.exposure_auto),
        exposure_time_us=(
            float(spec.exposure_time_us) if spec.exposure_time_us is not None else None
        ),
        gain_auto=str(spec.gain_auto),
        gain=(float(spec.gain) if spec.gain is not None else None),
    )
