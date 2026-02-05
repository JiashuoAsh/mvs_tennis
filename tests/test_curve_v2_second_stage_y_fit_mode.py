import numpy as np
import pytest


def test_curve_v2_fit_second_stage_y_only_smoke():
    """确保“只用第二段点拟合”的方法存在且可调用。

    这个测试的目标很简单：避免因为缩进/位置错误导致方法变成“嵌套函数”或不可访问。
    """

    from curve.curve_v2 import Curve

    c = Curve()

    # 选择一组简单的点：并不要求严格符合抛体，只要能拟合出系数即可。
    x = np.array([0.10, 0.12, 0.14, 0.16], dtype=float)
    y = np.array([0.20, 0.23, 0.21, 0.18], dtype=float)
    w = np.ones_like(x)

    coeffs, mse = c.fit_second_stage_y_only(x, y, w)

    assert isinstance(coeffs, np.ndarray)
    assert coeffs.shape == (3,)
    assert isinstance(mse, float)


def test_curve_v2_second_stage_y_fit_mode_validation():
    """second_stage_y_fit_mode 必须是已知枚举值。"""

    from curve.curve_v2 import Curve

    c = Curve()

    c.set_second_stage_y_fit_mode("stage_only")
    c.set_second_stage_y_fit_mode("joint")

    with pytest.raises(ValueError):
        c.set_second_stage_y_fit_mode("unknown")


def test_curve_v2_fit_two_curves_still_available():
    """旧的联合拟合逻辑仍然保留（未删除），并且可直接调用。"""

    from curve.curve_v2 import Curve

    c = Curve()

    # fit_two_curves 会读取 self.land_point[0][-1] 作为枚举反弹时刻的基准。
    c.land_point[0] = [0.0, 0.0, 0.0, 0.10]

    # 构造一组跨反弹的“形状合理”的数据：
    # - 段1（反弹前）在接近 t0 时下降
    # - 段2（反弹后）在离开 t0 时上升
    x1 = np.array([0.06, 0.08, 0.09], dtype=float)
    y1 = np.array([0.20, 0.10, 0.03], dtype=float)
    x2 = np.array([0.11, 0.13, 0.15], dtype=float)
    y2 = np.array([0.03, 0.10, 0.18], dtype=float)

    out = c.fit_two_curves(x1, y1, x2, y2)

    assert isinstance(out, dict)
    assert "coeffs" in out
    assert isinstance(out["coeffs"], np.ndarray)
    assert out["coeffs"].shape == (3,)
