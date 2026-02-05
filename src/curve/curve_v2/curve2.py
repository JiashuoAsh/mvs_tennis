"""curve2：旧版网球轨迹拟合/预测（LEGACY）。

说明：
        - 本文件历史较久且体量较大（>1000 行），包含多段曲线拟合、反弹后曲线推断、以及
            legacy 输出格式（不同高度的接球点）。
        - 若你在做“反弹后第二段轨迹预测 + 少点快速校正”的新需求，优先使用 `curve_v3/`
            中的两阶段实现（参见 `docs/curve.md` 与 `curve_v3/core.py::CurvePredictorV3`）。
        - 该文件主要用于保持历史业务接口与行为，后续如要继续维护，建议按职责拆分为更小的
            模块（prefit / prior candidates / posterior / legacy outputs）。
"""

import numpy as np
import math
import time
import logging
from curve.curve_v2.bot_motion_config import BotMotionConfig
from curve.curve_v2.logging_utils import get_logger
try:
    # SciPy 在本仓库环境里不一定存在；仅少数辅助拟合函数需要。
    from scipy.optimize import curve_fit  # type: ignore
except Exception:  # pragma: no cover
    curve_fit = None



np.set_printoptions(suppress=True, precision=3)


def quadratic_func(x, a, b, c):
    return a * x**2 + b * x + c


class Curve:
    G = 9.8

    # 定义空气阻力系数
    AIR_DRAG_K = 0.02

    # 球拍出球无旋转，瓷砖条件下，落点误差约为在0.1m之内
    # 观察0425的测试情况，弹跳速度普遍会更高，增加0.072的系数
    BOUNCE_G_K = 0.670 + 0.025 + 0.072
    BOUNCE_G_B = 0.388 + 0.012 + 0.025  # 0.08

    # update 0812, 稍微调高反弹，使得z轴更好跑到位
    BOUNCE_VX_K = 0.55 + 0.09
    BOUNCE_VX_B = -0.26  # + 0.1 + 0.15 #+0.1 #+ 0.25

    # 反弹速度预测为 vx' = vx / (1 + a*vx)
    BOUNCE_VX_A = BotMotionConfig.BOUNCE_VX_A

    # 反弹速度预测为 vx' = a * e^(-b*x) * x , a是最大反弹系数，b是衰减率
    # BOUNCE_VX_MAX_RATIO = 0.88
    # BOUNCE_VX_DEDUCTION_RATIO = -0.075

    FIT_Y_WEIGHT = 1.05
    FIT_X_WEIGHT = 1.01
    # FIXME: 调整z轴的权重增长，避免过拟合
    FIT_Z_WEIGHT = 1.1

    MAX_LEN_TO_FIT = 15  # 最多只拟合个数，避免积分误差

    # --------------用于判断第一曲线的是否成立（确认是不是网球轨迹）--------------
    # Z_SPEED_RANGE = [1.5, 27]  # z方向的速度范围
    Z_SPEED_RANGE = [0.5, 27]  # z方向的速度范围

    # 第二段（id==1）Y 拟合策略：
    # - stage_only：只使用第二段观测点拟合（当前默认）
    # - joint：旧实现，跨反弹边界联合拟合两段
    SECOND_STAGE_Y_FIT_STAGE_ONLY = "stage_only"
    SECOND_STAGE_Y_FIT_JOINT = "joint"

    def __init__(self) -> None:
        self.is_debug = False
        self.reset()
        # 为此类定义记录器
        self.logger = get_logger(
            "ball_curve",
            console_output=True,
            file_output=False,
            # console_level="INFO",
            console_level="DEBUG",
            file_level="DEBUG",
            time_interval=-0.1,
            use_buffering=True,
            buffer_capacity=100,
            flush_interval=2.0,
        )

    def reset(self, is_bot_serve=True):
        """
        xs: 球的 x 坐标序列（横向位置）
        ys: 球的 y 坐标序列（垂直高度）
        zs: 球的 z 坐标序列（纵向深度）
        ts: 对应的时间戳序列

        x_ws: x 坐标拟合时的权重
        y_ws: y 坐标拟合时的权重
        z_ws: z 坐标拟合时的权重
        """
        self.xs = []
        self.ys = []
        self.zs = []
        self.ts = []

        self.x_ws = []
        self.y_ws = []
        self.z_ws = []

        # update 0614: 不再在curve中寻找起点，交由ball_tracer判断球的状态

        self.time_base = time.perf_counter()

        # 采用数组方式进行各个曲线的存储

        self.move_frames_threshold = [3, 4, 4]
        self.curve_samples_cnt = [0, 0, 0]  # 每条曲线对应有几帧
        self.ball_start_cnt = []

        self.land_point = [None, None, None]
        self.land_speed = [None, None, None]

        self.x_coeff = [None, None, None]
        self.y_coeff = [None, None, None]
        self.y_view_coeff = [None, None, None]
        self.z_coeff = [None, None, None]

        self.x_error_rate = [0, 0, 0]
        self.y_error_rate = [0, 0, 0]
        self.y_view_error_rate = [0, 0, 0]
        self.z_error_rate = [0, 0, 0]

        self.is_curve_valid = True  # 用来帮助外层业务判断当前曲线是否有效，

        self.is_first_ball_removed = False  # 是否已经移除第一个球

        # 第二段 Y 拟合策略开关：默认只用第二段点拟合。
        # 如需对比旧算法，可在外部设置：curve.second_stage_y_fit_mode = "joint"。
        self.second_stage_y_fit_mode = Curve.SECOND_STAGE_Y_FIT_STAGE_ONLY

    # 直接取g的加速度，不考虑空气阻力和旋转。返回系数和误差
    # TODO：未来可能需要优化
    def constrained_polyfit(self, x, y, w, fixed_a=-4.9):
        """
        拟合一个二次多项式，并确保二次项系数为固定值 -4.9，
        通过线性拟合求解剩余的b和c系数。

        参数：
        x (array-like): 自变量数据
        y (array-like): 因变量数据
        w (array-like): 权重

        返回：
        coefficients (np.ndarray): 多项式系数 [a, b, c]
        """
        x = np.array(x)
        y = np.array(y)
        w = np.array(w)

        # 调整因变量
        y_adjusted = y - fixed_a * x**2

        # 使用np.polyfit进行加权线性拟合，拟合一次多项式 b*x + c
        coeffs = np.polyfit(x, y_adjusted, 1, w=w)
        fit_values = np.polyval(coeffs, x)
        mse = np.mean((fit_values - y_adjusted) ** 2)  # 均方误差

        b, c = coeffs

        # 返回最终的多项式系数
        final_coefficients = np.array([fixed_a, b, c])

        return final_coefficients, mse

    def set_second_stage_y_fit_mode(self, mode: str) -> None:
        """设置第二段（id==1）Y 拟合策略。

        Args:
            mode: "stage_only" 或 "joint"
        """

        if mode not in (
            Curve.SECOND_STAGE_Y_FIT_STAGE_ONLY,
            Curve.SECOND_STAGE_Y_FIT_JOINT,
        ):
            raise ValueError(
                f"未知的 second_stage_y_fit_mode={mode!r}；"
                f"只支持 {Curve.SECOND_STAGE_Y_FIT_STAGE_ONLY!r} / {Curve.SECOND_STAGE_Y_FIT_JOINT!r}"
            )
        self.second_stage_y_fit_mode = mode

    def fit_second_stage_y_only(self, x, y, w, fixed_a=-4.9):
        """仅使用第二段观测点拟合 y(t)（忽略第一段与反弹点约束）。

        说明：
            - 旧实现会跨反弹边界联合拟合两段（依赖第一段下降点 + 第二段上升点），并显式
              枚举反弹时刻以增强稳定性。
            - 按当前需求，这里默认使用“只用第二段点”的带重力约束二次拟合：
              y(t) = a·t² + b·t + c，其中 a 固定为 -4.9。
            - 该拟合完全不依赖第一段数据，也不强制反弹点连续性；适合做快速校正，但在第二段
              点数很少或噪声较大时，落点时间可能更不稳定。

        Returns:
            tuple[np.ndarray, float]: (coeffs, mse)
        """

        return self.constrained_polyfit(x, y, w, fixed_a=fixed_a)

    def fit_piecewise_quadratic(self, x_data, y_data):
        """
        对数据 (x_data, y_data) 用分段函数拟合：
        f(x) = (t - x)*v1 - 4.9*(t - x)**2    if x < t
            = (t - x)*v2 - 4.9*(t - x)**2    if x >= t

        使用显式雅可比并对参数提供边界。
        返回：
        A, B, C，对应 x>=t 的二次展开系数
        """
        import numpy as np
        try:
            from scipy.optimize import least_squares  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "fit_piecewise_quadratic 需要 scipy；当前环境未安装。"
            ) from e

        x = np.asarray(x_data)
        y = np.asarray(y_data)

        # 模型函数
        def piecewise_model(params, x):
            t, v1, v2 = params
            dx = t - x
            base = -4.9 * dx**2
            return np.where(x < t, v1 * dx + base, v2 * dx + base)

        # 残差
        def residuals(params, x, y):
            return piecewise_model(params, x) - y

        # 显式雅可比（Jacobian）
        def jacobian(params, x, y):
            t, v1, v2 = params
            dx = t - x
            # ∂res/∂t = ∂model/∂t
            dm_dt = np.where(x < t, dx - 9.8 * dx, dx - 9.8 * dx)
            # ∂res/∂v1
            dm_dv1 = np.where(x < t, dx, 0.0)
            # ∂res/∂v2
            dm_dv2 = np.where(x >= t, dx, 0.0)
            # residual = model - y, so same derivatives
            return np.vstack((dm_dt, dm_dv1, dm_dv2)).T

        t0 = self.land_point[0][-1]  # 使用第一个曲线的落点时间作为初始猜测
        v1 = -self.land_speed[0][1]
        # 初始猜测: t 设置为最后着地点, v1,v2 默认零
        init = np.array([t0, v1, 0.0])

        # 参数边界: t 在 [min(x), max(x)]，v1, v2 在 [-1000, 1000]（可根据数据调整）
        lower = np.array([t0 - 0.020, v1 - 1.0, -20.0])
        upper = np.array([t0 + 0.020, v1 + 1.0, 20.0])

        # 调用 least_squares
        res = least_squares(
            residuals,
            init,
            jac=jacobian,
            args=(x, y),
            bounds=(lower, upper),
            method="trf",
            max_nfev=100,
        )

        t_fit, v1_fit, v2_fit = res.x

        # 展开 x>=t 的二次系数
        A = -4.9
        B = 2 * 4.9 * t_fit - v2_fit
        C = -4.9 * t_fit**2 + v2_fit * t_fit

        # print(f"land point 0 speed: {self.land_speed[0]}, t_fit: {t_fit}, v1_fit: {v1_fit}, v2_fit: {v2_fit}, land point 0 time: {self.land_point[0][-1]}")
        return np.array([A, B, C])

    def fit_two_curves(self, x1, y1, x2, y2):
        """
        同时对两条曲线做拟合：
        f1(x1) = v1*(x1 - t0) + A*(x1 - t0)**2
        f2(x2) = v2*(x2 - t0) + A*(x2 - t0)**2
        A 固定为 -4.9，枚举多个 t0，选取使 SSE = Σ(f1-y1)^2 + Σ(f2-y2)^2 最小的参数。

        参数
        ----
        x1, y1 : array_like
            第一条曲线的数据点
        x2, y2 : array_like
            第二条曲线的数据点

        返回
        ----
        result : dict
            {
                't0'     : 最优的 t0,
                'v1'     : 第一条曲线的线性系数,
                'v2'     : 第二条曲线的线性系数,
                'coeffs' : np.array([A, B, C])，对应 f2 展开后的二次多项式系数,
                'error'  : 最小的 SSE,
                't_shift': t0 - t0_base
            }
        """
        """
        联合拟合共享反弹时间的两个轨迹段。

        通过枚举落地时间并求解最小化总平方误差的速度，
        跨反弹边界拟合分段二次函数。

        模型：
            段1（反弹前）：f1(t) = v1·(t - t0) + A·(t - t0)²
            段2（反弹后）：f2(t) = v2·(t - t0) + A·(t - t0)²
            其中 A = -4.9 m/s²（重力），t0 = 反弹时间

        参数：
            x1, y1 (array_like): 下降阶段的时间和高度样本
            x2, y2 (array_like): 上升阶段的时间和高度样本

        返回：
            dict: 最优拟合参数
                {
                    't0': float         # 估计的反弹时间（相对于 time_base）
                    'v1': float         # 反弹前垂直速度 (m/s, 负值)
                    'v2': float         # 反弹后垂直速度 (m/s, 正值)
                    'coeffs': [A,B,C]   # 段2的标准形式系数
                    'error': float      # 两段的总平方误差
                    'avg_error': float  # 每个样本的均方误差
                    't_shift': float    # 相对于预测落地时间的偏差 (s)
                }

        算法：
            1. 使用段1的预测落地时间作为初始猜测 (t0_base)
            2. 在 ±15ms 窗口内以 3ms 步长枚举 t0 候选（11个候选）
            3. 对每个 t0，通过最小二乘计算闭式最优 v1, v2
            4. 计算总 SSE = Σ(f1-y1)² + Σ(f2-y2)²
            5. 返回最小化 SSE 的参数

        闭式解（对每个t0）：
            对于段 i: v_i = Σ[dx·(y - A·dx²)] / Σ[dx²]
            其中 dx = time - t0（相对于反弹）

        为什么联合拟合：
            - 在反弹点强制连续性
            - 减少从段1到段2的误差传播
            - 更好地处理反弹附近的测量噪声
            - 提供更稳定的落地时间估计

        其中：
            - 假设瞬时反弹（位置连续，速度不连续）
            - 搜索窗口可能需要针对高速回球调整
            - 返回段2的标准多项式形式系数 [a, b, c]
        """
        # 转成 numpy 数组
        x1 = np.asarray(x1)
        y1 = np.asarray(y1)
        x2 = np.asarray(x2)
        y2 = np.asarray(y2)

        if self.is_debug:
            print(f"x1: {x1}")
            print(f"y1: {y1}")
            print(f"x2: {x2}")
            print(f"y2: {y2}")

        A = -4.9
        t0_base = self.land_point[0][-1]  # 你已有的基准落点时间

        # 在基准附近枚举 t0：这里 ±0.01，步长 0.001
        t0_candidates = [t0_base + i * 0.003 for i in range(-5, 6)]

        best = {"error": np.inf}
        for t0 in t0_candidates:
            # 一、分别计算 dx
            dx1 = x1 - t0
            dx2 = x2 - t0

            # 二、闭式求 v1, v2
            #    v = ∑[dx*(y - A dx^2)] / ∑[dx^2]
            denom1 = np.dot(dx1, dx1)
            denom2 = np.dot(dx2, dx2)
            if denom1 == 0 or denom2 == 0:
                continue

            v1 = np.dot(dx1, (y1 - A * dx1**2)) / denom1
            v2 = np.dot(dx2, (y2 - A * dx2**2)) / denom2

            # 三、计算总 SSE
            y1_pred = v1 * dx1 + A * dx1**2
            y2_pred = v2 * dx2 + A * dx2**2
            error = np.sum((y1_pred - y1) ** 2) + np.sum((y2_pred - y2) ** 2)

            # 四、保留最优解
            if error < best["error"]:
                # 对第二条曲线 f2 展开： A·x² + B·x + C
                B = v2 - 2 * A * t0
                C = A * t0**2 - v2 * t0

                best = {
                    "t0": t0,
                    "v1": v1,
                    "v2": v2,
                    "coeffs": np.array([A, B, C]),
                    "error": error,  # 平均误差
                    "avg_error": error / (len(x1) + len(x2)),  # 平均误差
                    "t_shift": t0 - t0_base,
                }

        if self.is_debug:
            print("最佳拟合结果：", best)
        return best

    def fit_two_curves_v2(self, x1, y1, x2, y2):
        """
        同时对两条曲线做拟合：
        f1(x1) = v1*(x1 - t0) + A*(x1 - t0)**2
        f2(x2) = v2*(x2 - t0) + A*(x2 - t0)**2
        A 固定为 -4.9，枚举多个 t0，选取使 SSE = Σ(f1-y1)^2 + Σ(f2-y2)^2 最小的参数。

        参数
        ----
        x1, y1 : array_like
            第一条曲线的数据点
        x2, y2 : array_like
            第二条曲线的数据点

        返回
        ----
        result : dict
            {
                't0'     : 最优的 t0,
                'v1'     : 第一条曲线的线性系数,
                'v2'     : 第二条曲线的线性系数,
                'coeffs' : np.array([A, B, C])，对应 f2 展开后的二次多项式系数,
                'error'  : 最小的 SSE,
                't_shift': t0 - t0_base
            }
        """
        """
        联合拟合共享反弹时间的两个轨迹段。

        通过枚举落地时间并求解最小化总平方误差的速度，
        跨反弹边界拟合分段二次函数。

        模型：
            段1（反弹前）：f1(t) = v1·(t - t0) + A·(t - t0)²
            段2（反弹后）：f2(t) = v2·(t - t0) + A·(t - t0)²
            其中 A = -4.9 m/s²（重力），t0 = 反弹时间

        参数：
            x1, y1 (array_like): 下降阶段的时间和高度样本
            x2, y2 (array_like): 上升阶段的时间和高度样本

        返回：
            dict: 最优拟合参数
                {
                    't0': float         # 估计的反弹时间（相对于 time_base）
                    'v1': float         # 反弹前垂直速度 (m/s, 负值)
                    'v2': float         # 反弹后垂直速度 (m/s, 正值)
                    'coeffs': [A,B,C]   # 段2的标准形式系数
                    'error': float      # 两段的总平方误差
                    'avg_error': float  # 每个样本的均方误差
                    't_shift': float    # 相对于预测落地时间的偏差 (s)
                }

        算法：
            1. 使用段1的预测落地时间作为初始猜测 (t0_base)
            2. 在 ±15ms 窗口内以 3ms 步长枚举 t0 候选（11个候选）
            3. 对每个 t0，通过最小二乘计算闭式最优 v1, v2
            4. 计算总 SSE = Σ(f1-y1)² + Σ(f2-y2)²
            5. 返回最小化 SSE 的参数

        闭式解（对每个t0）：
            对于段 i: v_i = Σ[dx·(y - A·dx²)] / Σ[dx²]
            其中 dx = time - t0（相对于反弹）

        为什么联合拟合：
            - 在反弹点强制连续性
            - 减少从段1到段2的误差传播
            - 更好地处理反弹附近的测量噪声
            - 提供更稳定的落地时间估计

        其中：
            - 假设瞬时反弹（位置连续，速度不连续）
            - 搜索窗口可能需要针对高速回球调整
            - 返回段2的标准多项式形式系数 [a, b, c]
        """
        # 转成 numpy 数组
        x1 = np.asarray(x1)
        y1 = np.asarray(y1)
        x2 = np.asarray(x2)
        y2 = np.asarray(y2)

        if self.is_debug:
            print(f"x1: {x1}")
            print(f"y1: {y1}")
            print(f"x2: {x2}")
            print(f"y2: {y2}")

        # 归一化权重
        n1, n2 = len(x1), len(x2)
        w1 = 1.0 / max(n1, 1)
        w2 = 1.0 / max(n2, 1)
        total_w = w1 + w2
        w1, w2 = w1 / total_w, w2 / total_w

        A = -4.9
        t0_base = self.land_point[0][-1]  # 你已有的基准落点时间
        # 在基准附近枚举 t0：这里 ±0.01，步长 0.001
        t0_candidates = [t0_base + i * 0.003 for i in range(-5, 6)]

        best = {"error": np.inf}
        for t0 in t0_candidates:
            # 一、分别计算 dx
            dx1 = x1 - t0
            dx2 = x2 - t0

            # 二、闭式求 v1, v2
            #    v = ∑[dx*(y - A dx^2)] / ∑[dx^2]
            denom1 = np.dot(dx1, dx1)
            denom2 = np.dot(dx2, dx2)
            if denom1 == 0 or denom2 == 0:
                continue

            v1 = np.dot(dx1, (y1 - A * dx1**2)) / denom1
            v2 = np.dot(dx2, (y2 - A * dx2**2)) / denom2

            # 三、计算总 SSE
            y1_pred = v1 * dx1 + A * dx1**2
            y2_pred = v2 * dx2 + A * dx2**2
            error = w1 * np.sum((y1_pred - y1) ** 2) + w2 * np.sum((y2_pred - y2) ** 2)

            # 四、保留最优解
            if error < best["error"]:
                # 对第二条曲线 f2 展开： A·x² + B·x + C
                B = v2 - 2 * A * t0
                C = A * t0**2 - v2 * t0

                best = {
                    "t0": t0,
                    "v1": v1,
                    "v2": v2,
                    "coeffs": np.array([A, B, C]),
                    "error": error,  # 平均误差
                    "avg_error": error / (len(x1) + len(x2)),  # 平均误差
                    "t_shift": t0 - t0_base,
                }

        if self.is_debug:
            print("最佳拟合结果：", best)
        return best

    # 返回一次曲线的拟合系数，以及误差
    def linear_polyfit(self, x, y, w):
        y = np.array(y)
        coeffs = np.polyfit(x, y, 1, w=w)
        fit_values = np.polyval(coeffs, x)
        mse = np.mean((fit_values - y) ** 2)  # 均方误差

        return coeffs, mse

    def polyfit_quadratic_2(self, x, y, w):
        # legacy: 使用有界的二次拟合。SciPy 可用时走 curve_fit；否则使用 numpy 回退。
        if curve_fit is not None:
            bounds = ([-6.4, -np.inf, -np.inf], [-3.4, np.inf, np.inf])
            params, _ = curve_fit(quadratic_func, x, y, bounds=bounds)
            a, b, c = params
        else:
            # 回退：先做普通二次拟合，再把 a 裁剪到边界，并用最小二乘重求 b,c。
            a0, b0, c0 = np.polyfit(x, y, 2)
            a = float(np.clip(a0, -6.4, -3.4))
            y_adj = np.asarray(y, dtype=float) - a * (np.asarray(x, dtype=float) ** 2)
            b, c = np.polyfit(x, y_adj, 1)

        coeffs = np.array([a, b, c])
        fit_values = np.polyval(coeffs, x)
        mse = np.mean((fit_values - y) ** 2)
        return coeffs, mse

    def calc_land_point_and_speed_rk4(
        self, x_coeff, y_coeff, z_coeff, start_t, final_t
    ):
        """
        使用4阶龙格库塔方法计算考虑空气阻力的落地点和速度（更高精度）

        相比前向欧拉方法，RK4提供O(dt⁴)的精度，误差更小

        参数：
            x_coeff, y_coeff, z_coeff: 多项式系数
            start_t: 起始时间
            final_t: 目标时间

        返回：
            (landing_point, landing_speed)
        """
        # 初始状态
        vx = np.polyval(np.polyder(x_coeff), start_t)
        vz = np.polyval(np.polyder(z_coeff), start_t)
        cx = np.polyval(x_coeff, start_t)
        cz = np.polyval(z_coeff, start_t)
        ct = start_t

        # 自适应时间步长：高速时用小步长，低速时用大步长
        v_horizontal = math.sqrt(vx**2 + vz**2)
        if v_horizontal > 15:
            dt = 0.025  # 高速：25ms
        elif v_horizontal < 5:
            dt = 0.100  # 低速：100ms
        else:
            dt = 0.050  # 中速：50ms（默认）

        def calc_drag_accel(vx, vz):
            """计算空气阻力加速度"""
            v = math.sqrt(vx**2 + vz**2)
            if v < 1e-6:
                return 0, 0
            a_magnitude = v * v * Curve.AIR_DRAG_K
            ax = -a_magnitude * (vx / v)
            az = -a_magnitude * (vz / v)
            return ax, az

        # RK4积分循环
        while ct < final_t:
            # 调整最后一步的时间步长
            if ct + dt > final_t:
                dt = final_t - ct

            # RK4四个斜率
            # k1
            ax1, az1 = calc_drag_accel(vx, vz)
            k1_vx = ax1
            k1_vz = az1
            k1_x = vx
            k1_z = vz

            # k2
            vx_temp = vx + 0.5 * dt * k1_vx
            vz_temp = vz + 0.5 * dt * k1_vz
            ax2, az2 = calc_drag_accel(vx_temp, vz_temp)
            k2_vx = ax2
            k2_vz = az2
            k2_x = vx_temp
            k2_z = vz_temp

            # k3
            vx_temp = vx + 0.5 * dt * k2_vx
            vz_temp = vz + 0.5 * dt * k2_vz
            ax3, az3 = calc_drag_accel(vx_temp, vz_temp)
            k3_vx = ax3
            k3_vz = az3
            k3_x = vx_temp
            k3_z = vz_temp

            # k4
            vx_temp = vx + dt * k3_vx
            vz_temp = vz + dt * k3_vz
            ax4, az4 = calc_drag_accel(vx_temp, vz_temp)
            k4_vx = ax4
            k4_vz = az4
            k4_x = vx_temp
            k4_z = vz_temp

            # 更新速度和位置
            vx += (dt / 6.0) * (k1_vx + 2 * k2_vx + 2 * k3_vx + k4_vx)
            vz += (dt / 6.0) * (k1_vz + 2 * k2_vz + 2 * k3_vz + k4_vz)
            cx += (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
            cz += (dt / 6.0) * (k1_z + 2 * k2_z + 2 * k3_z + k4_z)
            ct += dt

        # 计算最终Y速度
        vy = np.polyval(np.polyder(y_coeff), final_t)
        final_speed = [vx, vy, vz, math.sqrt(vx**2 + vz**2)]
        final_point = [cx, 0, cz, final_t]

        return final_point, final_speed

    def calc_land_point_and_speed(self, x_coeff, y_coeff, z_coeff, start_t, final_t):
        """
        数值积分模拟：
        1. 50ms步长forward simulation
        2. 每步更新速度：v -= drag_force * dt
        3. 每步更新位置：pos += v * dt
        4. 阻力模型：F = k * v²
        """
        """
        计算考虑空气阻力的落地点和速度。

        使用前向欧拉积分模拟从 start_t 到 final_t 的球轨迹，
        对水平速度分量应用二次空气阻力。

        参数：
            x_coeff (array): x(t) 的多项式系数
            y_coeff (array): y(t) 的多项式系数
            z_coeff (array): z(t) 的多项式系数
            start_t (float): 模拟的初始时间（相对于 time_base）
            final_t (float): 目标落地时间（相对于 time_base）

        返回：
            tuple: (landing_point, landing_speed)
                - landing_point: [x, 0, z, final_t] - 落地位置（y=0在地面）
                - landing_speed: [vx, vy, vz, v_horizontal] - 速度分量 (m/s)

        数值积分模拟：
            - 时间步长：50ms（最后一步自适应）
            - 空气阻力：F = k·v²，其中 k = AIR_DRAG_K (0.02)
            - 阻力仅作用于水平面 (x, z)
            - 垂直速度 (y) 直接从多项式导数计算
            - 位置更新：p(t+dt) = p(t) + v(t)·dt
            - 速度更新：v(t+dt) = v(t) - drag_force·dt

        其中：
            - Y速度不受阻力影响（重力主导垂直运动）
            - 用于预测反弹轨迹和验证击球可行性
        """
        # 曲线初始的x、z速度和位置
        vx = np.polyval(np.polyder(x_coeff), start_t)
        vz = np.polyval(np.polyder(z_coeff), start_t)
        cx = np.polyval(x_coeff, start_t)
        cz = np.polyval(z_coeff, start_t)
        ct = start_t

        # 以50ms forward 模拟, 最后一段时间则有多少算多少
        dt = 0.05
        is_final_dt_calculated = False
        while not is_final_dt_calculated:
            # 更新时间
            if ct + dt > final_t:
                dt = final_t - ct
                is_final_dt_calculated = True
            ct += dt

            # 计算位置
            cx += vx * dt
            cz += vz * dt

            # 更新速度
            a = (vx**2 + vz**2) * Curve.AIR_DRAG_K
            x_ratio = vx / math.sqrt(vx**2 + vz**2)
            z_ratio = vz / math.sqrt(vx**2 + vz**2)
            vx -= a * x_ratio * dt
            vz -= a * z_ratio * dt

        vy = np.polyval(np.polyder(y_coeff), final_t)
        final_speed = [vx, vy, vz, math.sqrt(vx**2 + vz**2)]
        final_point = [cx, 0, cz, final_t]

        return final_point, final_speed

    # 保存当前帧的数据和权重
    def append_loc(self, ball_loc):
        self.xs.append(ball_loc[0])
        self.ys.append(ball_loc[1])
        self.zs.append(ball_loc[2])
        self.ts.append(ball_loc[3] - self.time_base)
        if len(self.y_ws) == 0:
            self.x_ws.append(1.0)
            self.y_ws.append(1.0)
            self.z_ws.append(1.0)
        else:
            self.x_ws.append(self.x_ws[-1] * Curve.FIT_X_WEIGHT)
            self.y_ws.append(self.y_ws[-1] * Curve.FIT_Y_WEIGHT)
            self.z_ws.append(self.z_ws[-1] * Curve.FIT_Z_WEIGHT)

    def predict_next_curve_v3(self, land_point, land_speed, y_coeff=None):
        """
        反弹物理模型：
        1. XZ方向：new_v = old_v * k1 + k2 + spin_effect
        2. Y方向： new_vy = -(old_vy * k3 + k4) + spin_effect
        3. 考虑上下旋对反弹系数的影响

        输出：下一段轨迹的多项式系数
        """
        """
        使用经验反弹模型预测反弹后的轨迹。

        计算球从地面反弹后下一轨迹段的多项式系数，
        应用速度衰减和旋转效应。

        参数：
            land_point (array): 落地位置 [x, 0, z, t_land]
            land_speed (array): 落地速度 [vx, vy, vz, v_horizontal]
            y_coeff (array, optional): 前一段的Y多项式，用于旋转检测

        返回：
            tuple: (x2_coeff, y2_coeff, z2_coeff) - 下一段的线性/二次系数
                - x2_coeff: [vx_bounce, c] 用于 x(t) = vx_bounce·t + c
                - z2_coeff: [vz_bounce, c] 用于 z(t) = vz_bounce·t + c
                - y2_coeff: [a, b, c] 用于 y(t) = a·t² + b·t + c（带重力的抛物线）

        反弹物理模型：
            水平方向（XZ平面）：
                - v_new = v_old · k1 + k2 + spin_correction
                - X轴系数来自 BotMotionConfig.BOUNCE_X_COEFF
                - Z轴系数来自 BotMotionConfig.BOUNCE_Z_COEFF
                - 速度方向保持（比例维持）

            垂直方向（Y方向）：
                - vy_new = -(vy_old · k3 + k4) + spin_correction
                - 负号：下落后向上反弹
                - 系数来自 BotMotionConfig.BOUNCE_Y_COEFF

            旋转检测（可选）：
                - 分析 y_coeff[0]（相对于 -4.9 m/s² 的加速度偏差）
                - 正偏差 → 上旋 → 减少水平反弹
                - 负偏差 → 下旋 → 增加水平反弹
                - 当前禁用（乘以0）等待校准

        其中：
            - 假设瞬时反弹（不建模接触持续时间）
            - 经验系数针对特定表面/球组合调优
            - Y轨迹在落地时间初始化为向上速度
            - 第二次反弹后模型精度下降（误差累积）
        """
        t1 = land_point[-1]  # 当前曲线的落地时刻
        # speed 格式为 [vx, vy, vz, sqrt(vx**2+vz**2)]
        # 查看a项系数，判断上下旋，对反弹的b项系数进行调整
        xz_b_affect = 0.0
        y_b_affect = 0.0
        if y_coeff is not None:
            a = y_coeff[0]
            bias = a - (-0.5 * Curve.G)
            affect_ratio = -max(min(bias, 1.5), -1.5) / 1.5
            xz_b_affect = 0 * affect_ratio
            y_b_affect = 0 * affect_ratio

        # 分别计算X轴和Z轴的反弹速度
        vx = (
            land_speed[0] * BotMotionConfig.BOUNCE_X_COEFF[0]
            + BotMotionConfig.BOUNCE_X_COEFF[1]
            + xz_b_affect
        )
        vz = (
            land_speed[2] * BotMotionConfig.BOUNCE_Z_COEFF[0]
            + BotMotionConfig.BOUNCE_Z_COEFF[1]
            + xz_b_affect
        )
        x2_coeff = np.array([vx, -t1 * vx + land_point[0]])
        z2_coeff = np.array([vz, -t1 * vz + land_point[2]])

        vy = (
            -(
                land_speed[1] * BotMotionConfig.BOUNCE_Y_COEFF[0]
                + BotMotionConfig.BOUNCE_Y_COEFF[1]
            )
            + y_b_affect
        )
        y2_coeff = np.array(
            [-0.5 * Curve.G, Curve.G * t1 + vy, -0.5 * Curve.G * t1 * t1 - t1 * vy]
        )

        return x2_coeff, y2_coeff, z2_coeff

    # calculate the time when the ball is at the target height
    def calc_t_at_height(self, y_coeff, target_y):
        y_coeff = y_coeff.copy()
        y_coeff[2] -= target_y
        if y_coeff[1] ** 2 - 4 * y_coeff[0] * y_coeff[2] <= 0:
            return None
        else:
            ts = np.roots(y_coeff)

        return [min(ts), max(ts)]

    def add_receive_locs_at_height(self, id, score0, score1):
        """
        为给定轨迹段在网高范围内生成击球点候选。

        1. 找出球在最小/最大击球高度的时间区间
        2. 沿下降阶段采样点（从高到低）
        3. 计算每个采样时间的球位置和速度
        4. 将候选点追加到 self.loc_results

        参数：
            id (int): 轨迹段索引（0=首次飞行，1=第一次反弹，2=第二次反弹）
            score0 (float): 上升期优先级分数（当前未使用）
            score1 (float): 下降期优先级分数（当前未使用）

        返回：
            None: 方法将结果追加到 self.loc_results 而不是返回

        其中：
            - NET_HEIGHT_1: 最小击球高度（下网间隙）
            - NET_HEIGHT_2: 最大击球高度（最佳击球区）
            - 时间步长：候选点之间间隔 0.02s（20ms）
            - 仅采样下降阶段（峰值后、低网高度前）
            - 如果球永远达不到最小高度，返回None（不可达）
            - 如果球达不到最大高度，使用轨迹峰值代替

        dict 结构：
            添加到 loc_results 的每个点包含：
            - point: [x, y, z, global_timestamp] 世界坐标系
            - score: 相对时间 (t_down) 用于优先级排序
            - x_speed, z_speed: 球速度分量 (m/s)
            - fit_samples: 轨迹拟合使用的总观测数
            - curve_1_samples: 反弹段的观测数（质量指标）
        """
        low_net_receive_ts = self.calc_t_at_height(
            self.y_coeff[id], BotMotionConfig.NET_HEIGHT_1
        )
        if low_net_receive_ts is None:  # 最低接球高度也无法碰到球，返回None
            return None

        high_net_receive_ts = self.calc_t_at_height(
            self.y_coeff[id], BotMotionConfig.NET_HEIGHT_2
        )
        if high_net_receive_ts is None:
            mid_t = (low_net_receive_ts[0] + low_net_receive_ts[1]) / 2
            high_net_receive_ts = [mid_t, mid_t]

        # 依次从最高点开始列举可选的击球点
        t_down = high_net_receive_ts[1]
        while t_down <= low_net_receive_ts[1]:
            p_down = [
                np.polyval(self.x_coeff[id], t_down),
                np.polyval(self.y_coeff[id], t_down),
                np.polyval(self.z_coeff[id], t_down),
                t_down + self.time_base,
            ]

            # 球飞行方向的角度
            z_speed = np.polyval(np.polyder(self.z_coeff[id]), t_down)
            x_speed = np.polyval(np.polyder(self.x_coeff[id]), t_down)
            z_tolerance = 0
            self.loc_results.append(
                {
                    "point": p_down,
                    "score": t_down,
                    "x_speed": x_speed,
                    "z_speed": z_speed,
                    "fit_samples": self.curve_samples_cnt[0]
                    + self.curve_samples_cnt[1],
                    "curve_1_samples": self.curve_samples_cnt[1],
                }
            )

            t_down += 0.02

    # 早期阶段未移出第一个顶点，只假设1.5s后接球，让车先跑
    def add_early_target(self):
        t = 1.5  # 从第一个球往后1.5s
        p = [
            np.polyval(self.x_coeff[0], t),
            np.polyval(self.y_coeff[0], t),
            np.polyval(self.z_coeff[0], t),
            t + self.time_base,
        ]
        z_speed = np.polyval(np.polyder(self.z_coeff[0]), t)
        x_speed = np.polyval(np.polyder(self.x_coeff[0]), t)
        self.loc_results.append(
            {
                "point": p,
                "score": 0,
                "curve_id": "1_guess",
                "z_tolerance": 0,
                "x_speed": x_speed,
                "z_speed": z_speed,
                "fit_samples": self.curve_samples_cnt[0] + self.curve_samples_cnt[1],
            }
        )

    # 移出第一个球, 不需要修改time_base, 只要把起始点和总的sample数记对即可
    def remove_first_ball(self):
        if (not self.is_first_ball_removed) and self.curve_samples_cnt[
            0
        ] == 5:  # 第5个球时已出第一个
            self.is_first_ball_removed = True
            self.curve_samples_cnt[0] -= 1
            self.ball_start_cnt[0] += 1

    # 如果是用户回球， 默认情况下是用户回球，也即is_bot_fire=-1， 当机器发球时，vz为正
    def add_frame(self, ball_loc, is_bot_fire=-1):
        """
        添加新的球位观测并计算最优击球点。

        1. 数据采集：存储球的位置 (x, y, z) 和时间戳
        2. 曲线分段：检测反弹转换点（最多3段）
        3. 多项式拟合：对当前段拟合 x(t), y(t), z(t)
        4. 轨迹预测：使用物理模型预测未来段
        5. 击球点生成：在网高范围内计算可行的击球位置

        参数：
            ball_loc (array-like): 球位观测 [x, y, z, timestamp]
                - x: 横向位置 (m)
                - y: 垂直高度 (m)
                - z: 纵向深度/前向距离 (m)
                - timestamp: 绝对时间 (秒)
            is_bot_fire (int, optional): 方向指示器
                - -1: 用户回球（默认，球靠近机器人）
                - 1: 机器人发球（球远离机器人）

        返回：
            list | None | int:
                - list: 击球点候选数组，每个元素结构为：
                    {
                        "point": [x, y, z, t],      # 击球位置和时间
                        "score": float,              # 基于时间的优先级分数
                        "x_speed": float,            # x 方向球速
                        "z_speed": float,            # z 方向球速
                        "fit_samples": int,          # 拟合使用的总样本数
                        "curve_1_samples": int       # 反弹段的样本数
                    }
                - None: 数据不足或接近落地转换期
                - -1: 拟合错误（无效轨迹，例如速度超出范围）

        Attributes:
            - self.ball_start_cnt: 记录每段曲线的起始样本索引
            - self.curve_samples_cnt: 记录每段曲线的样本数量

        状态转换：
            - 当球超过预测落地时间时检测反弹点
            - 分段：0（初始飞行）→ 1（第一次反弹）→ 2（第二次反弹）
            - 在采集到5个观测后移除早期样本以提高拟合质量

        拟合策略：
            - X, Z: 线性拟合（考虑空气阻力修正的恒定速度）
            - Y（段0）：约束二次拟合（固定重力加速度 -4.9 m/s²）
            - Y（段1+）：仅使用第二段观测点做约束二次拟合（忽略第一段/反弹点）
            - 融合：反弹早期样本混合预测值和观测值

        质量检查：
            - 拟合至少需要5个样本
            - Z方向速度必须在 [1.5, 27] m/s 范围内（真实球速）
            - Y多项式必须有实根（落地时间存在）
            - 落地前后30ms返回None以避免转换噪声

        物理模型：
            - 空气阻力：F = k·v²，通过前向模拟应用（50ms步长）
            - 反弹模型：XZ速度 = f(落地速度)，Y速度 = -g(落地速度)
            - 旋转效应：基于y加速度偏差调整反弹系数

        示例：
            >>> curve = Curve()
            >>> result = curve.add_frame([0.1, 1.2, 3.5, time.time()])
            >>> if isinstance(result, list) and result:
            >>>     best_hit = min(result, key=lambda x: x['score'])
            >>>     print(f"Hit at {best_hit['point']}")
        """
        self.loc_results = []

        if is_bot_fire == -1:  # 判断是否需要移除第一个点,  只在用户回球下判断
            self.remove_first_ball()

        # id表示 当前第几条曲线
        id = len(self.ball_start_cnt) - 1
        # 如果接近当前曲线落地点，为了防止落地反弹阶段不对的误差，不予计算而退出
        if (self.land_point[id] is not None) and abs(
            self.land_point[id][-1] - (ball_loc[-1] - self.time_base)
        ) < 0.03:
            # self.logger.info(f"Too close to land point, return None. Time: {self.ts[-1]} id:{id} Predict landing ball {self.land_point[id]}")
            return None

        # 往ball_loc中添加点数
        if len(self.ts) == 0:
            self.time_base = ball_loc[-1]
            self.ball_start_cnt.append(0)
        self.append_loc(ball_loc)

        # 如果超过了当前曲线的落点，则进入下一条曲线阶段
        if (self.land_point[id] is not None) and self.ts[-1] > self.land_point[id][-1]:
            if id == 2:
                return None
            self.ball_start_cnt.append(len(self.ts) - 1)
            id += 1
            self.logger.info(f"start to fit curve {id}")

        self.curve_samples_cnt[id] += 1

        # 开始对当前曲线进行拟合， 如果采样点不足，直接返回。
        start_cnt = self.ball_start_cnt[id]
        start_xz_cnt = max(
            len(self.ts) - 12, self.ball_start_cnt[id]
        )  # x和z最多只拟合10帧，即可，避免过度
        n_sample = len(self.ts) - self.ball_start_cnt[id]
        # self.logger.info(f"start_cnt: {start_cnt}, n_sample: {n_sample}")
        if self.is_debug:
            print(f"start_cnt: {start_cnt}, n_sample: {n_sample}")

        if n_sample < 2 or (id == 0 and n_sample < self.move_frames_threshold[0]):
            return None

        if n_sample > 25:  # 拟合最多的帧数，保证曲线0的质量
            self.ball_start_cnt[id] += 1

        # 采样点足够，进行当前数据拟合
        self.x_coeff[id], self.x_error_rate[id] = self.linear_polyfit(
            self.ts[start_xz_cnt:], self.xs[start_xz_cnt:], w=self.x_ws[start_xz_cnt:]
        )
        self.z_coeff[id], self.z_error_rate[id] = self.linear_polyfit(
            self.ts[start_xz_cnt:], self.zs[start_xz_cnt:], w=self.z_ws[start_xz_cnt:]
        )

        if id == 0:
            self.y_coeff[id], self.y_error_rate[id] = self.constrained_polyfit(
                self.ts[start_cnt:], self.ys[start_cnt:], self.y_ws[start_cnt:]
            )
        elif id == 1:
            test_start_t = time.perf_counter()

            if self.second_stage_y_fit_mode == Curve.SECOND_STAGE_Y_FIT_JOINT:
                # 旧实现：跨反弹边界联合拟合两段。
                curve0_y_start_cnt = max(
                    self.ball_start_cnt[0], self.ball_start_cnt[1] - 15
                )
                curve0_y_end_cnt = self.ball_start_cnt[1]
                self.y_coeff[id] = self.fit_two_curves(
                    self.ts[curve0_y_start_cnt:curve0_y_end_cnt],
                    self.ys[curve0_y_start_cnt:curve0_y_end_cnt],
                    self.ts[start_cnt:],
                    self.ys[start_cnt:],
                )["coeffs"]
                # legacy: 旧路径未返回 mse，这里保持历史行为（不更新 y_error_rate）。
            else:
                # 默认：仅使用第二段观测点拟合（忽略第一段/反弹点）。
                self.y_coeff[id], self.y_error_rate[id] = self.fit_second_stage_y_only(
                    self.ts[start_cnt:],
                    self.ys[start_cnt:],
                    self.y_ws[start_cnt:],
                )
            if self.is_debug:
                mode = self.second_stage_y_fit_mode
                print(f"second_stage_y_fit({mode}) time: {time.perf_counter() - test_start_t}")

        # 如果没有移除第一个点，直接按照猜测进行返回
        # if not self.is_first_ball_removed and is_bot_fire == -1:
        # return None
        #     self.add_early_target()
        # return self.loc_results

        if is_bot_fire == 1 and n_sample == 6:
            self.inital_vxz = math.sqrt(
                self.z_coeff[0][0] ** 2 + self.x_coeff[0][0] ** 2
            )

        # 如果是第二或第三曲线，前期采样点不够多，需要与之前曲线的反弹预测进行融合
        if id >= 1 and n_sample < self.move_frames_threshold[id]:
            x2_coeff, _, z2_coeff = self.predict_next_curve_v3(
                self.land_point[id - 1], self.land_speed[id - 1], self.y_coeff[id - 1]
            )
            self.z_coeff[id] = (
                self.z_coeff[id] * n_sample
                + z2_coeff * (self.move_frames_threshold[id] - n_sample)
            ) / self.move_frames_threshold[id]
            self.x_coeff[id] = (
                self.x_coeff[id] * n_sample
                + x2_coeff * (self.move_frames_threshold[id] - n_sample)
            ) / self.move_frames_threshold[id]

        # TODO: 质检待优化， 目前只判断下网的情况，也就是当最近8帧z的速度过小，判定为下网。转而放弃接球。 如果不放弃，且速度很慢的情况导致车持续跑的情况，已经在recalc loc中修复。
        # 其他情况的异常值，会有mc 放弃球做处理。 如果无法放弃也会有位置限制保护。
        # 此处暂时不做放弃也是ok的。
        if n_sample >= 10:
            z_sample_cnt = 9
            # recent_z_coeff, recent_z_error_rate = self.linear_polyfit(self.ts[-z_sample_cnt:], self.zs[-z_sample_cnt:], w=self.z_ws[-z_sample_cnt:])
            vz = np.polyval(np.polyder(self.z_coeff[id]), self.ts[-1])
            if (
                vz * is_bot_fire < Curve.Z_SPEED_RANGE[0]
                or vz * is_bot_fire > Curve.Z_SPEED_RANGE[1]
            ):
                if self.is_debug:
                    print(
                    f"  Z speed is not qualified! return None. Time: {self.ts[-1]} Predict landing ball {self.land_point[id]}, speed: {self.land_speed[id]}, z_coeff: {self.z_coeff[id]} is not good, zs is {self.zs[start_cnt:]}"
                )
                self.logger.error(
                    f"Z速度不合格！拒绝曲线 | z_coeff={self.z_coeff[id]} | 最近10帧z坐标={self.zs[-10:]}"
                )
                self.logger.error(
                    f"Z速度检查 | vz={vz:.3f} m/s | is_bot_fire={is_bot_fire} | vz*is_bot_fire={vz * is_bot_fire:.3f} | 范围:[{Curve.Z_SPEED_RANGE[0]}, {Curve.Z_SPEED_RANGE[1]}]"
                )
                self.is_curve_valid = False
                return -1

        ts_at_ground = self.calc_t_at_height(self.y_coeff[id], 0)
        if ts_at_ground is None:
            self.logger.error(
                f"  Predict curve has no root, Return None!  y_coeff: {self.y_coeff[id]}"
            )
            return -1

        # 计算当前曲线落地点与落地速度，并预测接下来曲线的落点和速度
        self.land_point[id], self.land_speed[id] = self.calc_land_point_and_speed(
            self.x_coeff[id],
            self.y_coeff[id],
            self.z_coeff[id],
            self.ts[-1],
            max(np.roots(self.y_coeff[id])),
        )

        for i in range(id, 2):
            self.x_coeff[i + 1], self.y_coeff[i + 1], self.z_coeff[i + 1] = (
                self.predict_next_curve_v3(
                    self.land_point[i], self.land_speed[i], self.y_coeff[i]
                )
            )
            self.land_point[i + 1], self.land_speed[i + 1] = (
                self.calc_land_point_and_speed(
                    self.x_coeff[i + 1],
                    self.y_coeff[i + 1],
                    self.z_coeff[i + 1],
                    self.land_point[i][-1],
                    max(np.roots(self.y_coeff[i + 1])),
                )
            )

        # 曲线全部预测完毕，开始计算接球点. 依次计算第二曲线上升、下降， 第三曲线上升、下降的接球点。

        # 打球模型，先只接下降期
        self.add_receive_locs_at_height(1, 1.0, 1.5)

        return self.loc_results

    # 获得t时刻，抛物线的点
    def get_point_at_time(self, t):
        for i in range(3):
            if self.land_point[i] is None:
                return None
            if t < self.land_point[i][-1]:  # 表示在第i曲线内
                return [
                    np.polyval(self.x_coeff[i], t),
                    np.polyval(self.y_coeff[i], t),
                    np.polyval(self.z_coeff[i], t),
                ]
        return None

    """用于logger打印信息的函数
    """

    # 返回当前曲线id，曲线0和1的落点速度
    def get_current_curve_land_speed(self):
        id = len(self.ball_start_cnt) - 1
        return id, self.land_speed[0], self.land_speed[1]

    def get_current_curve_y_view_fit(self):
        id = len(self.ball_start_cnt) - 1
        if self.y_view_coeff[id] is None:
            return None, None, None, None
        return id, self.y_view_coeff[id], self.y_view_error_rate[id], self.ts[-1]

    # update: 曲线1修改为起跳速度, 返回格式是id0的v
    def get_bounce_speed(self):
        id = len(self.ball_start_cnt) - 1
        # return id, self.land_speed[0], self.land_speed[1]
        land_t = self.land_point[0][-1]

        # 计算从落地点到当前点的位移
        dx = self.xs[-1] - self.land_point[0][0]
        dz = self.zs[-1] - self.land_point[0][2]
        dt = self.ts[-1] - land_t

        # 计算水平方向总速度
        bounce_vxz = math.sqrt(dx**2 + dz**2) / dt

        # 根据位移方向分解到 X 和 Z 轴
        # 如果水平位移太小，使用曲线1的斜率
        if bounce_vxz > 0.1:  # 避免除零
            bounce_vx = dx / dt
            bounce_vz = dz / dt
        else:
            # 水平速度太小，使用拟合曲线的速度
            bounce_vx = self.x_coeff[1][0] if self.x_coeff[1] is not None else 0
            bounce_vz = self.z_coeff[1][0] if self.z_coeff[1] is not None else 0

        # Y方向速度（从曲线1的导数）
        bounce_vy = np.polyval(np.polyder(self.y_coeff[1]), land_t)

        if self.is_debug:
            print(
            f"t gap: {dt:.3f}s, bounce_vxz: {bounce_vxz:.3f}, vx: {bounce_vx:.3f}, vz: {bounce_vz:.3f}"
        )
        return id, self.land_speed[0], [bounce_vx, -bounce_vy, bounce_vz, bounce_vxz]

    # 计算网球在过网时的高度
    def calc_net_clearance(self):
        # 根据z系数，计算网球位于z=12时的时间（即过网时刻）. 然后根据y系数，得到过网高度
        t_at_net = (12 - self.z_coeff[0][1]) / self.z_coeff[0][0]

        return np.polyval(self.y_coeff[0], t_at_net)

    def predicted_second_land_time_rel(self):
        """预测第二段（第一次反弹后）的落地相对时刻。

        说明：
            - 该接口用于与 curve_v3 对齐：返回“反弹后再次落地”的时刻。
            - curve_v2 内部已预测多段曲线，并维护 land_point[i]。
              其中 land_point[1][-1] 即第一次反弹后的下一次落地时刻（相对 time_base）。

        Returns:
            float | None：可用时返回相对时刻（秒），否则返回 None。
        """

        lp = self.land_point[1]
        if lp is None:
            return None
        try:
            return float(lp[-1])
        except Exception:
            return None

    def predicted_second_land_time_abs(self):
        """预测第二段（第一次反弹后）的落地绝对时刻。"""

        t_rel = self.predicted_second_land_time_rel()
        if t_rel is None:
            return None
        try:
            return float(self.time_base + float(t_rel))
        except Exception:
            return None
