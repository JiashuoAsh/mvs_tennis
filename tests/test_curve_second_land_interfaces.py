from __future__ import annotations

import pytest

from curve.curve_v2.curve2 import Curve as CurveV2
from curve.curve_v3.legacy import Curve as CurveV3Legacy


def test_curve_v2_has_predicted_second_land_time_interfaces() -> None:
    c = CurveV2()

    # 直接注入必要状态：接口应当只是读取 land_point[1] 与 time_base，不依赖完整拟合流程。
    c.time_base = 100.0
    # 说明：curve_v2 内部的 land_point 默认元素为 None，这里用最小状态注入验证接口行为。
    setattr(c, "land_point", [None, [0.0, 0.0, 0.0, 1.23], None])  # type: ignore[attr-defined]

    assert c.predicted_second_land_time_rel() == pytest.approx(1.23)
    assert c.predicted_second_land_time_abs() == pytest.approx(101.23)


def test_curve_v3_legacy_has_predicted_second_land_time_interfaces() -> None:
    c = CurveV3Legacy()

    c.time_base = 200.0
    setattr(c, "land_point", [None, [0.0, 0.0, 0.0, 2.5], None])

    assert c.predicted_second_land_time_rel() == pytest.approx(2.5)
    assert c.predicted_second_land_time_abs() == pytest.approx(202.5)
