from __future__ import annotations

from tennis3d.trajectory.curve_stage.models import _RecentObs


def _solve_3x3(A: list[list[float]], b: list[float]) -> tuple[float, float, float] | None:
    """解 3x3 线性方程组 A x = b（高斯消元）。

    说明：
        - 不依赖 numpy，避免引入额外依赖。
        - 对数值病态情况返回 None。
    """

    # 复制，避免原地修改。
    M = [[float(A[i][j]) for j in range(3)] + [float(b[i])] for i in range(3)]

    for col in range(3):
        # 选主元
        piv = col
        piv_abs = abs(M[col][col])
        for r in range(col + 1, 3):
            v = abs(M[r][col])
            if v > piv_abs:
                piv_abs = v
                piv = r
        if piv_abs < 1e-12:
            return None
        if piv != col:
            M[col], M[piv] = M[piv], M[col]

        # 归一化
        div = float(M[col][col])
        for j in range(col, 4):
            M[col][j] = float(M[col][j]) / div

        # 消元
        for r in range(3):
            if r == col:
                continue
            factor = float(M[r][col])
            if abs(factor) < 1e-12:
                continue
            for j in range(col, 4):
                M[r][j] = float(M[r][j]) - factor * float(M[col][j])

    return (float(M[0][3]), float(M[1][3]), float(M[2][3]))


def _estimate_const_accel_y(window: list[_RecentObs]) -> float | None:
    """用二次拟合估计窗口内 y 轴常加速度。

    Returns:
        ay（m/s^2）或 None。

    说明：
        - 拟合模型：y = a t^2 + b t + c（t 相对窗口起点）。
        - 常加速度 ay = 2a。
    """

    if len(window) < 3:
        return None

    t0 = float(window[0].t_abs)
    ts: list[float] = []
    ys: list[float] = []
    for o in window:
        ts.append(float(o.t_abs) - t0)
        ys.append(float(o.y))

    # 构造正规方程 (X^T X) beta = X^T y，其中 X=[t^2, t, 1]
    s_t4 = 0.0
    s_t3 = 0.0
    s_t2 = 0.0
    s_t1 = 0.0
    s_1 = float(len(ts))
    s_y_t2 = 0.0
    s_y_t1 = 0.0
    s_y = 0.0
    for t, y in zip(ts, ys):
        t2 = float(t * t)
        t3 = float(t2 * t)
        t4 = float(t2 * t2)
        s_t1 += float(t)
        s_t2 += float(t2)
        s_t3 += float(t3)
        s_t4 += float(t4)
        s_y += float(y)
        s_y_t1 += float(y) * float(t)
        s_y_t2 += float(y) * float(t2)

    A = [
        [s_t4, s_t3, s_t2],
        [s_t3, s_t2, s_t1],
        [s_t2, s_t1, s_1],
    ]
    bb = [s_y_t2, s_y_t1, s_y]
    sol = _solve_3x3(A, bb)
    if sol is None:
        return None
    a, _b, _c = sol
    return float(2.0 * float(a))
