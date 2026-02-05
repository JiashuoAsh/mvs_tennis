from __future__ import annotations

import math
from typing import Any

from tennis3d.trajectory.curve_stage.config import CurveStageConfig
from tennis3d.trajectory.curve_stage.math_fit import _estimate_const_accel_y
from tennis3d.trajectory.curve_stage.models import _RecentObs, _Track


def _episode_trim_recent(cfg: CurveStageConfig, tr: _Track, *, now_t_abs: float) -> None:
    if not tr.recent:
        return
    if float(cfg.episode_buffer_s) <= 0:
        tr.recent.clear()
        return
    t_min = float(now_t_abs) - float(cfg.episode_buffer_s)
    while tr.recent and float(tr.recent[0].t_abs) < t_min:
        tr.recent.popleft()


def _episode_try_start(cfg: CurveStageConfig, tr: _Track, *, events: list[dict[str, Any]]) -> None:
    """尝试从 recent 缓冲中判定 episode 开始。"""

    if not bool(cfg.episode_enabled):
        return
    if bool(tr.episode_active):
        return
    if int(cfg.episode_min_obs) < 3:
        return
    if len(tr.recent) < int(cfg.episode_min_obs):
        return

    k = int(cfg.episode_min_obs)
    window = list(tr.recent)[-k:]
    t0 = float(window[0].t_abs)
    t1 = float(window[-1].t_abs)
    if t1 <= t0:
        return

    p0 = window[0].pos
    p1 = window[-1].pos
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    dz = float(p1[2] - p0[2])
    dt = float(t1 - t0)
    if dt <= 0:
        return

    disp3 = float(math.sqrt(dx * dx + dy * dy + dz * dz))
    avg_speed3 = float(disp3 / dt)

    abs_dz = float(abs(dz))
    abs_vz = float(abs_dz / dt)

    if float(abs_dz) < float(cfg.episode_min_abs_dz_m):
        return
    if float(abs_vz) < float(cfg.episode_min_abs_vz_mps):
        return
    z_dir = int(cfg.episode_z_dir)
    if z_dir not in {-1, 0, 1}:
        z_dir = 0
    if z_dir != 0:
        if float(dz) * float(z_dir) <= 0:
            return

    ay = _estimate_const_accel_y(window)
    if ay is None or not math.isfinite(float(ay)):
        return
    g = float(cfg.episode_gravity_mps2)
    tol = float(cfg.episode_gravity_tol_mps2)
    if abs(abs(float(ay)) - abs(g)) > float(tol):
        return

    tr.episode_id += 1
    tr.episode_active = True
    tr.episode_start_t_abs = float(t0)
    tr.episode_end_t_abs = None
    tr.episode_end_reason = None
    tr.last_motion_t_abs = float(t1)

    events.append(
        {
            "event": "episode_start",
            "track_id": int(tr.track_id),
            "episode_id": int(tr.episode_id),
            "t_abs": float(t0),
            "avg_speed3_mps": float(avg_speed3),
            "displacement3_m": float(disp3),
            "dz_m": float(dz),
            "abs_vz_mps": float(abs_vz),
            "ay_mps2": float(ay),
        }
    )


def _episode_try_end(cfg: CurveStageConfig, tr: _Track, *, now_t_abs: float, events: list[dict[str, Any]]) -> bool:
    """尝试判定 episode 结束。

    Returns:
        若发生 episode_end 且调用方应该做“结束后的清理动作”（如 reset predictor），返回 True。
    """

    if not bool(cfg.episode_enabled):
        return False
    if not bool(tr.episode_active):
        return False

    # 说明：
    # - predicted_land_time_abs：第一段落地/反弹锚点（用于判定何时“应该已经有 second land”）。
    # - predicted_second_land_time_abs：第二段（反弹后）再次落地时刻（episode 应覆盖两段）。
    #
    # 新约束：
    # - episode 的“基于落地的自动结束”只允许使用 second land。
    # - 若已经到了 first land + buffer 的时间窗口，但 second land 仍不可用，则直接报错，
    #   不允许回退到 predicted_land_time_abs（避免向下兼容掩盖问题）。

    if tr.predicted_land_time_abs is not None:
        t_gate = float(tr.predicted_land_time_abs) + float(cfg.episode_end_after_predicted_land_s)

        if float(now_t_abs) >= float(t_gate) and tr.predicted_second_land_time_abs is None:
            raise RuntimeError(
                "episode_end 需要 predicted_second_land_time_abs，但当前为 None；"
                "已进入 first_land+buffer 窗口，禁止回退到 predicted_land_time_abs。"
            )

    if tr.predicted_second_land_time_abs is not None:
        if float(now_t_abs) >= float(tr.predicted_second_land_time_abs) + float(cfg.episode_end_after_predicted_land_s):
            tr.episode_active = False
            tr.episode_end_t_abs = float(tr.predicted_second_land_time_abs)
            tr.episode_end_reason = "after_predicted_second_land"
            events.append(
                {
                    "event": "episode_end",
                    "track_id": int(tr.track_id),
                    "episode_id": int(tr.episode_id),
                    "t_abs": float(tr.episode_end_t_abs),
                    "reason": str(tr.episode_end_reason),
                }
            )
            return True

    if tr.last_motion_t_abs is not None:
        if float(now_t_abs - float(tr.last_motion_t_abs)) >= float(cfg.episode_end_if_stationary_s):
            tr.episode_active = False
            tr.episode_end_t_abs = float(now_t_abs)
            tr.episode_end_reason = "stationary"
            events.append(
                {
                    "event": "episode_end",
                    "track_id": int(tr.track_id),
                    "episode_id": int(tr.episode_id),
                    "t_abs": float(tr.episode_end_t_abs),
                    "reason": str(tr.episode_end_reason),
                }
            )
            return True

    return False


def _episode_maybe_release_lock(locked_track_id: int | None, tr: _Track) -> int | None:
    """当锁定 track 的 episode 结束时，自动解除锁定。"""

    if locked_track_id is None:
        return None
    if int(tr.track_id) != int(locked_track_id):
        return locked_track_id
    if bool(tr.episode_active):
        return locked_track_id
    return None


def _episode_should_feed_curve(cfg: CurveStageConfig, tr: _Track) -> bool:
    """决定当前观测是否喂给 curve predictor。"""

    if bool(cfg.episode_enabled) and bool(cfg.feed_curve_only_when_episode_active):
        return bool(tr.episode_active)
    return True


def _episode_stationary_motion_update(cfg: CurveStageConfig, tr: _Track, *, speed_mps: float, t_abs: float) -> None:
    """更新“最近一次明显运动”的时刻，用于 stationary 结束判定。"""

    if (not bool(cfg.episode_enabled)) or float(speed_mps) >= float(cfg.episode_stationary_speed_mps):
        tr.last_motion_t_abs = float(t_abs)


def _episode_replay_window(window: list[_RecentObs]) -> list[_RecentObs]:
    # 说明：显式复制，避免上层修改 window 的 list 影响 episode recent 缓冲。
    return [o for o in window]
