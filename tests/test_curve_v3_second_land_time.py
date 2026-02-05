from __future__ import annotations

import numpy as np
import pytest

from curve.curve_v3.config import CurveV3Config
from curve.curve_v3.core import CurvePredictorV3
from curve.curve_v3.types import BounceEvent, Candidate, PosteriorState


def _bounce_event(*, t_rel: float = 1.0) -> BounceEvent:
    # v_minus 在本测试中不参与 second land 计算，但 BounceEvent 要求该字段存在。
    return BounceEvent(
        t_rel=float(t_rel),
        x=0.0,
        z=0.0,
        v_minus=np.array([0.0, -1.0, 0.0], dtype=float),
    )


def test_predicted_second_land_time_rel_prefers_posterior() -> None:
    cfg = CurveV3Config(gravity=10.0)
    p = CurvePredictorV3(config=cfg)

    # 直接注入内部状态：本测试只验证 second land 的解析逻辑，而不依赖完整的在线拟合流程。
    p._bounce_event = _bounce_event(t_rel=1.0)  # type: ignore[attr-defined]
    p._candidates = [
        Candidate(
            e=0.7,
            kt=0.6,
            weight=1.0,
            v_plus=np.array([0.0, 1.0, 0.0], dtype=float),
        )
    ]  # type: ignore[attr-defined]
    p._posterior_state = PosteriorState(
        t_b_rel=1.2,
        x_b=0.0,
        z_b=0.0,
        vx=0.0,
        vy=5.0,
        vz=0.0,
    )  # type: ignore[attr-defined]

    expected = 1.2 + 2.0 * 5.0 / 10.0
    assert p.predicted_second_land_time_rel() == pytest.approx(expected)


def test_predicted_second_land_time_rel_falls_back_to_prior_nominal() -> None:
    cfg = CurveV3Config(gravity=12.0)
    p = CurvePredictorV3(config=cfg)

    p._bounce_event = _bounce_event(t_rel=2.0)  # type: ignore[attr-defined]
    p._posterior_state = None  # type: ignore[attr-defined]

    # 名义 vy = 0.25*3 + 0.75*7 = 6.0，tau = 2*6/12 = 1.0
    p._candidates = [
        Candidate(
            e=0.7,
            kt=0.6,
            weight=0.25,
            v_plus=np.array([0.0, 3.0, 0.0], dtype=float),
        ),
        Candidate(
            e=0.7,
            kt=0.6,
            weight=0.75,
            v_plus=np.array([0.0, 7.0, 0.0], dtype=float),
        ),
    ]  # type: ignore[attr-defined]

    assert p.predicted_second_land_time_rel() == pytest.approx(3.0)


def test_predicted_second_land_time_rel_returns_none_when_vy_non_positive() -> None:
    cfg = CurveV3Config(gravity=9.8)
    p = CurvePredictorV3(config=cfg)

    p._bounce_event = _bounce_event(t_rel=1.0)  # type: ignore[attr-defined]
    p._posterior_state = PosteriorState(
        t_b_rel=1.0,
        x_b=0.0,
        z_b=0.0,
        vx=0.0,
        vy=0.0,
        vz=0.0,
    )  # type: ignore[attr-defined]

    assert p.predicted_second_land_time_rel() is None
