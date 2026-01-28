"""触发拓扑（software / master-slave）计算。

目标：
- 把 CLI 参数（serials、trigger_source、master_serial、soft_trigger_fps）转换成 open_quad_capture 所需的：
  - trigger_sources: list[str]
  - soft_trigger_serials: list[str]
  - enable_soft_trigger_fps: float

说明：
- 该模块不直接操作 SDK，只负责“配置计算”，便于后续扩展更多触发拓扑。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TriggerPlan:
    trigger_sources: list[str]
    soft_trigger_serials: list[str]
    enable_soft_trigger_fps: float


def build_trigger_plan(
    *,
    serials: list[str],
    trigger_source: str,
    master_serial: str,
    soft_trigger_fps: float,
) -> TriggerPlan:
    """根据用户参数构造触发计划。

    规则：
    - 纯软件触发：trigger_source=Software 且 master_serial 为空 → 所有相机 Software；对全部相机下发软触发。
    - 主从触发：master_serial 非空 → master 固定 Software；slaves 用 trigger_source（例如 Line0）；只对 master 下发软触发。

    Args:
        serials: 相机序列号列表（>=1）。
        trigger_source: slave 触发源（Software/Line0/Line1/...）。
        master_serial: master 序列号（为空表示不启用 master/slave）。
        soft_trigger_fps: 软触发频率。

    Returns:
        TriggerPlan。
    """

    serials = [str(s).strip() for s in (serials or []) if str(s).strip()]
    if not serials:
        raise ValueError("serials is empty")

    master_serial = str(master_serial or "").strip()
    trigger_source = str(trigger_source or "").strip()

    if master_serial:
        if master_serial not in serials:
            raise ValueError("master_serial must be in serials")
        trigger_sources = ["Software" if s == master_serial else trigger_source for s in serials]
        soft_trigger_serials = [master_serial]
    else:
        trigger_sources = [trigger_source] * len(serials)
        soft_trigger_serials = serials if trigger_source.lower() == "software" else []

    enable = float(soft_trigger_fps) if soft_trigger_serials else 0.0

    return TriggerPlan(
        trigger_sources=trigger_sources,
        soft_trigger_serials=soft_trigger_serials,
        enable_soft_trigger_fps=enable,
    )
