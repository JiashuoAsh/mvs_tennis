from __future__ import annotations

import pytest

from tennis3d.trajectory.curve_stage.config import CurveStageConfig
from tennis3d.trajectory.curve_stage.episode import _episode_try_end
from tennis3d.trajectory.curve_stage.models import _Track


def test_episode_end_prefers_predicted_second_land_even_with_feed_gate() -> None:
    cfg = CurveStageConfig(
        episode_enabled=True,
        feed_curve_only_when_episode_active=True,
        episode_end_after_predicted_land_s=0.2,
    )

    tr = _Track(track_id=1)
    tr.episode_id = 1
    tr.episode_active = True

    tr.predicted_land_time_abs = 10.0
    tr.predicted_second_land_time_abs = 20.0

    events: list[dict[str, object]] = []
    ended = _episode_try_end(cfg, tr, now_t_abs=20.3, events=events)

    assert ended is True
    assert tr.episode_active is False
    assert tr.episode_end_t_abs == 20.0
    assert tr.episode_end_reason == "after_predicted_second_land"


def test_episode_end_raises_if_second_land_missing_when_window_reached() -> None:
    cfg = CurveStageConfig(
        episode_enabled=True,
        feed_curve_only_when_episode_active=True,
        episode_end_after_predicted_land_s=0.2,
    )

    tr = _Track(track_id=1)
    tr.episode_id = 1
    tr.episode_active = True

    # 只有 bounce/触地时刻（v3 的 bounce_event），second land 不可用。
    tr.predicted_land_time_abs = 10.0
    tr.predicted_second_land_time_abs = None

    events: list[dict[str, object]] = []
    with pytest.raises(RuntimeError):
        _episode_try_end(cfg, tr, now_t_abs=10.3, events=events)
