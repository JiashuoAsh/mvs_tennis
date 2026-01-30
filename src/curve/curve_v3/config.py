"""curve_v3 的配置定义。

该包实现两阶段的反弹后轨迹预测：
    - prior：多候选 + 走廊（corridor）。
    - posterior：反弹后少点（N<=5）快速校正。

说明：
    legacy 兼容接口由 `curve_v3.legacy` 单独提供。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence


def _try_get_bot_motion_constant(name: str, default: float) -> float:
    """尝试从 `BotMotionConfig` 读取常量，不可用时回退到默认值。"""

    try:
        from hit.cfgs.bot_motion_config import BotMotionConfig  # type: ignore

        return float(getattr(BotMotionConfig, name))
    except Exception:
        return float(default)


@dataclass(frozen=True)
class CurveV3Config:
    """CurvePredictorV3 的配置项。

    属性:
        gravity: 重力加速度标量（m/s^2），方向沿 -y。
        ground_normal: 地面法向（世界系），默认 (0,1,0)。
            - `docs/curve.md` 用一般形式的法向/切向分解描述反弹。
            - 绝大多数网球场景可视为水平地面，因此默认固定为 y-up。
            - 若向量长度过小，会在使用处回退到 (0,1,0)。
        e_bins: 恢复系数候选集合（必须 >0）。
        kt_bins: 切向映射系数候选集合（必须 >=0）。
        e_range: e 的裁剪范围（用于兜底防止非物理/数值爆炸）。
        fit_mode: 后验拟合模式：
            - 当前实现仅保留 "rls"（信息形式递推）这一条求解路径。
            - 当 λ=1 时与批量最小二乘/正则最小二乘等价。
            - 历史兼容：若传入 "ls"，会被当作等价情形处理（"rls" 且 λ=1）。
        fit_params: 后验校正参数化（v_only 或 v+axz）。
        max_post_points: 后验最多使用的反弹后点数。
            - 工程建议（docs/curve.md）：第二阶段只使用少量 post 点（典型 N<=5）。
            - 本配置默认值设为较大值，表示“不在此处强制截断”，由上层/回放策略决定。
        weight_sigma_m: 候选打分的残差尺度（m），用于似然权重。
        posterior_obs_sigma_m: 后验观测噪声尺度（m）。
            - 为 None 时回退到 weight_sigma_m。
            - 该尺度同时影响 MAP 求解与 J_post 评分（与 `docs/curve.md` 的 W/R 对齐）。
        posterior_prior_strength: 后验融合中高斯先验强度（0 关闭；>0 启用 MAP 锚定）。
        posterior_prior_sigma_v: 速度先验标准差（m/s）。
        posterior_prior_sigma_a: 水平加速度先验标准差（m/s^2）。
        posterior_anchor_weight: 若 >0，则将“后验锚点”作为合成候选注入走廊混合。
        posterior_rls_lambda: RLS 遗忘因子 λ（0<λ<=1，1 表示不遗忘）。
            - 仅在 fit_mode=="rls" 时生效。
        corridor_dt: corridor_by_time 的采样步长（s）。
        corridor_horizon_s: corridor_by_time 的预测时域（s）。
        net_height_1: legacy 接球下限高度（m）。
        net_height_2: legacy 接球上限高度（m）。
        legacy_receive_dt: legacy 接球点采样步长（s）。
        legacy_too_close_to_land_s: 距离落地过近时抑制输出（s）。
        z_speed_range: legacy 基本合法性检查的 z 速度范围（m/s）。

        bounce_contact_y_m: 反弹事件的球心高度（m）。
            - 观测 y 若为球心高度：建议设为球半径 ball_radius_m。
            - 若为 None：默认使用 ball_radius_m。

        candidate_beta_warmup_points: 候选权重更新的退火预热点数。
            - 0 表示关闭退火（beta 恒为 1.0）。
            - >0 表示在前 N 个反弹后点中用较小 beta，避免过早锁定错误分支。
        candidate_beta_min: 退火的最小 beta（0<beta<=1）。
        prefit_xz_window_points: prefit 阶段用于 x/z 拟合的短窗口点数。
            - 同时用于选取 τ=t-t_ref 的参考时刻（t_ref 取该窗口起点）。
            - 窗口越大：更平滑但响应慢；窗口越小：更敏感但更容易被噪声/离群点带跑。
        prefit_robust_iters: 第一段拟合的鲁棒重加权迭代次数（0 表示关闭）。
        prefit_robust_delta_m: 鲁棒重加权的残差阈值（m），用于抑制离群点。
        prefit_min_inlier_points: 触发离群点剔除/重加权后，至少保留的点数。

        kt_angle_bins_rad: 切向映射的“偏转角”离散集合（弧度，绕地面法向旋转）。
            - 对应 `docs/curve.md` 的 φ（切平面内偏转）。
            - 设为 (0,) 可关闭偏转项，候选数从 27 退化为 9。
        kt_range: kt 的裁剪范围（允许为负）。

        ball_radius_m: 球半径（m）。

        posterior_min_tau_s: 后验线性系统中允许使用的最小 tau（秒）。
            - 用于避免 tau 过小导致的病态/数值不稳定。

        posterior_optimize_tb: 是否在后验拟合中联合估计反弹时刻 t_b。
            - 采用一维网格搜索（对每个候选单独搜索 tb），最小化 J_post。
            - 默认关闭以保持历史行为。
        posterior_tb_search_window_s: tb 搜索窗口半宽（秒），以 bounce.t_rel 为中心。
        posterior_tb_search_step_s: tb 搜索步长（秒）。
        posterior_tb_prior_sigma_s: tb 先验标准差（秒），用于抑制过度漂移。

        bounce_detector_v_down_mps: 分段检测器的“稳定下降”速度门限（m/s）。
        bounce_detector_v_up_mps: 分段检测器的“稳定上升”速度门限（m/s）。
        bounce_detector_eps_y_m: 分段检测器的近地高度容忍（m，球心高度）。
        bounce_detector_local_min_window: 分段检测器用于寻找局部最小的窗口长度（点数）。
        bounce_detector_down_debounce_s: 分段检测器对“稳定下降”的去抖时间常数（秒）。
        bounce_detector_up_debounce_s: 分段检测器对“稳定上升”的去抖时间常数（秒）。
        bounce_detector_min_points: 分段检测器最少需要的点数（过少时不触发）。

        bounce_detector_gap_freeze_enabled: 是否启用基于“时间缺口（gap）”的安全冻结。
        bounce_detector_gap_mult: 判定 gap 的倍数阈值（thr=gap_mult*median(dt)）。
        bounce_detector_gap_tb_margin_s: 判断 tb_pred 是否落在 gap 内的时间裕量（秒）。
        bounce_detector_gap_fit_points: 用于 tb_pred 拟合的 gap 左侧末端点数窗口。

        corridor_quantile_levels: corridor 分位数输出水平（0..1）。
            - 若为空，则不计算分位数（仅输出均值/协方差）。
            - 建议默认输出 5/95 或 10/90 以应对多峰走廊。

        corridor_components_k: corridor 混合分量数（仅对按平面输出生效）。
            - 1 表示不输出分量。
            - 2 表示输出两个简单分簇（沿 (x,z) 主轴投影做二分）。

        online_prior_enabled: 是否启用“在线参数沉淀”（docs/curve.md §7）。
            - 默认关闭，保证 v3 的行为与历史版本一致。
        online_prior_path: 在线权重池持久化路径（JSON）。
            - 为 None 时只在内存中沉淀，不做落盘。
        online_prior_ema_alpha: EMA 更新系数 α（0<α<=1）。
        online_prior_eps: 权重下限，避免出现 0 权重导致“学不回来”。
        online_prior_autosave: 是否在每次融合后自动保存权重池。
    """

    gravity: float = 9.8

    # 地面法向：默认水平地面（y-up）。
    ground_normal: tuple[float, float, float] = (0.0, 1.0, 0.0)

    e_bins: Sequence[float] = (0.55, 0.70, 0.85)
    kt_bins: Sequence[float] = (0.45, 0.65, 0.85)
    # e_bins: Sequence[float] = (0.55, 0.70)
    # kt_bins: Sequence[float] = (0.45, 0.65)

    # 参数裁剪：用于兜底防止配置/数据异常导致的数值发散。
    # 说明：e 允许略大于 1 以容纳等效误差，但不建议过大。
    e_range: tuple[float, float] = (0.05, 1.25)

    # 后验拟合模式：仅保留递推（rls）。
    # 说明：当 posterior_rls_lambda=1 时，等价于批量正规方程。
    fit_mode: Literal["rls"] = "rls"

    fit_params: Literal["v_only", "v+axz"] = "v+axz"
    # fit_params: Literal["v_only", "v+axz"] = "v_only"
    # 工程建议：第二阶段只使用少量 post 点（典型 N<=5）。
    # 默认不强制截断（设为较大值），避免离线回放/评估时被意外裁掉信息。
    max_post_points: int = 999

    weight_sigma_m: float = 0.15

    # 后验观测噪声（用于 MAP 求解与 J_post 评分）；None 表示沿用 weight_sigma_m。
    posterior_obs_sigma_m: float | None = None

    # 后验融合相关参数：
    # 这些默认值刻意设置得比较“温和”，用于提供一个弱锚点；
    # 当反弹后点足够有信息时，不会阻止模型做出较强校正。
    posterior_prior_strength: float = 1.0
    posterior_prior_sigma_v: float = 2.0
    posterior_prior_sigma_a: float = 8.0
    posterior_anchor_weight: float = 0.15

    # 后验细节：用于避免 tau 过小导致的病态（保持默认值与历史实现一致）。
    posterior_min_tau_s: float = 1e-6

    # 方案3：联合估计 tb（默认关闭；需要时由上层/实验显式开启）。
    posterior_optimize_tb: bool = False
    posterior_tb_search_window_s: float = 0.05
    posterior_tb_search_step_s: float = 0.002
    posterior_tb_prior_sigma_s: float = 0.03

    # RLS（信息形式递推）参数。
    posterior_rls_lambda: float = 1.0

    corridor_dt: float = 0.05
    corridor_horizon_s: float = 1.2

    net_height_1: float = _try_get_bot_motion_constant("NET_HEIGHT_1", 0.4)
    net_height_2: float = _try_get_bot_motion_constant("NET_HEIGHT_2", 1.1)

    legacy_receive_dt: float = 0.02
    legacy_too_close_to_land_s: float = 0.03

    z_speed_range: tuple[float, float] = (1.0, 27.0)

    # 反弹事件：观测 y 为球心高度时，触地/反弹时刻应满足 y(t)=r（球半径）。
    # 为避免默认值与 ball_radius_m 强耦合，这里用 None 表示“跟随 ball_radius_m”。
    bounce_contact_y_m: float | None = None

    # 候选权重退火：默认开启，前 2 个点用较小 beta 以降低锁错分支风险。
    candidate_beta_warmup_points: int = 2
    candidate_beta_min: float = 0.3

    # 分段检测器（prefit 冻结）：按 docs/curve.md §2.4.4 的默认建议值。
    # 说明：实现中使用“时间去抖”以适配不同 FPS。
    bounce_detector_v_down_mps: float = 0.6
    bounce_detector_v_up_mps: float = 0.4
    bounce_detector_eps_y_m: float = 0.04
    bounce_detector_down_debounce_s: float = 0.03
    bounce_detector_up_debounce_s: float = 0.03
    bounce_detector_local_min_window: int = 7
    bounce_detector_min_points: int = 6

    # 安全冻结：用于处理反弹附近不可见（例如 y<0.2m 不输出）导致 near_ground 永不成立的场景。
    # 触发逻辑在 utils/bounce_detector.py：最后一个时间 gap + 竖直模型预测 tb 落在 gap 内。
    bounce_detector_gap_freeze_enabled: bool = True
    bounce_detector_gap_mult: float = 3.0
    bounce_detector_gap_tb_margin_s: float = 0.033  # 30 fps 下约 1 帧，33ms
    bounce_detector_gap_fit_points: int = 12

    # 第一段（prefit）增强：水平面采用等效常加速度（二次）模型 + 1 次鲁棒重加权。
    # 说明：按 docs/curve.md 的工程规范，x/z 的二次模型为必选，不提供线性退化模式。
    # x/z 拟合短窗口点数：触地附近只用末端少量点来估计水平速度/加速度。
    prefit_xz_window_points: int = 12
    prefit_robust_iters: int = 1
    prefit_robust_delta_m: float = 0.12
    prefit_min_inlier_points: int = 5

    # 切向映射增强：标量 kt + 可选偏转角（等价于一个 2x2 线性映射：缩放 + 旋转）。
    # 默认保持“不启用偏转”，以避免在未调参场景下候选数膨胀并改变历史行为。
    # 若 φ（3 档），可设置：
    #     kt_angle_bins_rad = (-0.08, 0.0, 0.08)
    # 若 φ（1 档），可设置-直线运动：
    #     kt_angle_bins_rad: Sequence[float] = (0.0,)
    # 则候选数为 len(e_bins)*len(kt_bins)*len(kt_angle_bins_rad)。
    kt_angle_bins_rad: Sequence[float] = (0.0,)
    kt_range: tuple[float, float] = (-1.2, 1.2)

    # 球半径：用于定义触地时刻的球心高度（bounce_contact_y）。
    ball_radius_m: float = 0.033

    # 走廊（corridor）表达增强：按 docs/curve.md §4.3 推荐，默认提供分位数包络。
    corridor_quantile_levels: Sequence[float] = (0.05, 0.95)

    # 可选增强：多峰走廊的混合分量表示（K=1~2）。
    corridor_components_k: int = 2

    # 在线参数沉淀（docs/curve.md §7）：默认关闭。
    online_prior_enabled: bool = False
    online_prior_path: str | None = None
    online_prior_ema_alpha: float = 0.05
    online_prior_eps: float = 1e-8
    online_prior_autosave: bool = True

    # 低 SNR（低信噪比）策略：用于处理“噪声填充导数”的典型问题。
    # 说明：
    # - 默认开启，但只有当上层为 BallObservation 提供 conf（置信度）时，
    #   才会对权重与退化判别产生实质影响；conf 缺失时大多数路径等价于不启用。
    # - 该策略的目标是：在噪声很大时，避免导数/加速度被噪声“填满”导致发散。
    low_snr_enabled: bool = True

    # prefit 阶段 low SNR 判别窗口长度（点数）：仅取 prefit 段末尾 N 点做 analyze_window。
    # 说明：
    # - 该窗口用于决定是否触发 ignore/freeze/strong prior 等退化策略。
    # - 取尾窗而不是全量点，可避免早期噪声长期影响在线判别。
    low_snr_prefit_window_points: int = 7

    # conf 的下限（避免 1/sqrt(conf) 发散）。
    low_snr_conf_cmin: float = 0.05

    # 三轴基础噪声尺度 σ0（米），相当于 conf=1 时的观测标准差。
    low_snr_sigma_x0_m: float = 0.10
    low_snr_sigma_y0_m: float = 0.05
    low_snr_sigma_z0_m: float = 0.10

    # 退化判别阈值（见 docs/curve_low_snr_quick.md）。
    low_snr_delta_k_freeze_a: float = 4.0
    low_snr_delta_k_strong_v: float = 2.0
    low_snr_delta_k_ignore: float = 1.0
    low_snr_min_points_for_v: int = 3
    low_snr_disallow_ignore_y: bool = True

    # STRONG_PRIOR_V 的强度：对速度先验 σ_v 做缩放（<1 表示更强）。
    # 例如 0.1 表示比默认先验强约 10 倍（更“钉死”）。
    low_snr_strong_prior_v_scale: float = 0.1

    # prefit 阶段速度强先验的绝对尺度（m/s）。该值越小，越不允许被噪声带跑。
    low_snr_prefit_strong_sigma_v_mps: float = 0.5

    def bounce_contact_y(self) -> float:
        """返回反弹时刻的球心高度（m）。

        Returns:
            若配置 bounce_contact_y_m 为 None，则返回 ball_radius_m；否则返回 bounce_contact_y_m。
        """

        if self.bounce_contact_y_m is None:
            return float(self.ball_radius_m)
        return float(self.bounce_contact_y_m)

    def candidate_likelihood_beta(self, num_points: int) -> float:
        """计算候选似然权重的退火系数 beta。

        设计目标：在反弹后点数较少时，避免 exp(-0.5*J) 的权重过于尖锐导致
        过早锁定错误分支；随着点数增加，beta 逐渐回到 1.0。

        Args:
            num_points: 用于打分/重加权的反弹后点数。

        Returns:
            beta，范围 (0, 1]。
        """

        warmup = int(self.candidate_beta_warmup_points)
        if warmup <= 0:
            return 1.0

        beta_min = float(self.candidate_beta_min)
        beta_min = min(max(beta_min, 1e-3), 1.0)

        n = int(max(num_points, 0))
        # 线性预热：n=0 -> beta_min, n>=warmup -> 1.0
        alpha = min(float(n) / float(warmup), 1.0)
        return float(beta_min + (1.0 - beta_min) * alpha)
