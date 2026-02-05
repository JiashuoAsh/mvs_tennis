# create on 0513， 2024. 监听来自img_detect_topic 和user_command_topic。持续追踪网球的状态，有以下几种状态：
#   idle： 闲置状态，等待用户指令进行发球
#   shot_without_img: 接收发球指令，或者在对打中发出了球. 但是还没有检测到球，此时要求小车yaw角始终与发射是一致，等待检测
#   shot_with_img: 检测到了发射的球，仍在往前飞行过程中
#   return_ball: 检测到了用户回击的球，使用curve预测落点与击打

# ==============================================================================
# BallTracer - 网球轨迹追踪与状态管理
# ==============================================================================
#
# 实时追踪网球轨迹，管理状态转换，并为网球机器人系统预测击球点。
#
# 主要功能-函数:
#   1. 球位置追踪: 收集和维护检测到的球位置历史记录
#      - calc_target_loc_with_ballpos(): 主入口，接收相机检测数据并转换坐标
#      - calc_target_loc_with_abs_ballloc(): 将球位置添加到历史队列 ball_locs
#      - get_bot_loc_at_time(): 获取指定时刻小车位置，用于坐标转换
#
#   2. 状态机管理: 管理空闲、发球和回球状态之间的转换
#      - is_user_return_ball(): 检测用户是否开始回球（连续变近判定）
#      - is_fire_ball(): 检测机器人发球是否成功（连续变远判定）
#      - status_check(): 超时检查，防止状态长期停留
#      - reset(): 重置状态机到初始状态（IDLE或USER_SERVE）
#
#   3. 轨迹拟合: 使用五次多项式曲线拟合球的轨迹
#      - curve.add_frame(): 添加球位置点并拟合用户回球曲线
#      - fire_curve.add_frame(): 拟合机器人发球曲线
#
#   4. 击球点预测: 基于拟合曲线计算最优击球点
#      - curve.add_frame() 返回 shot_point_list: 预测击球点列表
#      - get_return_ball_in_camera(): 获取回球在相机坐标系下的预测位置
#      - get_curve_stage(): 判断当前处于落地前还是反弹后阶段
#
#   5. 坐标转换: 在相机坐标系和世界坐标系之间进行转换
#      - utils.rotate_2d_point(): 旋转坐标系（小车坐标系 ↔ 世界坐标系）
#      - calc_target_loc_with_ballpos(): 相机坐标 → 世界坐标
#      - get_return_ball_in_camera(): 世界坐标 → 相机坐标
#
# 数据流:
#   相机检测(真实) → img_info["ball_loc"] → calc_target_loc_with_ballpos()
#   → abs_ball_loc(真实) → ball_locs(真实历史) → curve.add_frame()
#   → shot_point_list(预测)
#
# 数据类型:
#   - 真实检测数据:
#     * ball_locs: deque队列，存储检测到的球位置历史 [x, y, z, time]，世界坐标系
#     * abs_ball_loc: 当前帧检测到的球位置（世界坐标系）
#     * img_info["ball_loc"]: 相机检测到的球，相机坐标系 [x, y, z]
#     * bot_loc_queue: 小车位置历史队列 [x, y, z, yaw, time]
#
#   - 预测数据:
#     * curve: 用户回球的拟合曲线对象 (Curve对象)
#     * fire_curve: 机器人发球的拟合曲线对象 (Curve对象)
#     * shot_point_list: 预测的击球点列表 [[x,y,z,t], ...]
#     * land_speed: 预测的落地速度 [vx, vy, vz]
#
# 状态机:
#   IDLE(空闲) → FIRE_BALL_WITH_GUESS(发球猜测，数据不足)
#   → FIRE_BALL_WITH_CURVE(发球曲线，已拟合) → RETURN_BALL(检测到回球) → IDLE
#        ↓
#   USER_SERVE(用户发球) → RETURN_BALL(检测到回球) → IDLE
#
# 使用示例:
#   tracer = BallTracer()
#   # 每帧调用，传入图像检测信息
#   data = tracer.calc_target_loc_with_ballpos(img_info)
#   if "shot_point_list" in data:
#       # 使用预测的击球点进行运动规划
#       target_point = data["shot_point_list"][0]
#   if "return_ball" in data:
#       # 检测到用户回球，记录时间
#       return_time = data["return_ball"]
# ==============================================================================


import time
import numpy as np

from collections import deque
import traceback

from curve.curve_v2.bot_motion_config import BotMotionConfig
from curve.curve_v2.curve2 import Curve
from curve.curve_v2.logging_utils import get_logger

from enum import Enum


def rotate_2d_point(x: float, z: float, yaw_rad: float) -> tuple[float, float]:
    """绕 y 轴旋转 (x,z) 平面点。

    legacy 代码里该函数来自外部 utils；这里提供最小实现，避免额外依赖。

    Args:
        x: 平面 x。
        z: 平面 z。
        yaw_rad: 旋转角（弧度），右手系绕 +y。

    Returns:
        (x_rot, z_rot): 旋转后的坐标。
    """

    c = float(np.cos(yaw_rad))
    s = float(np.sin(yaw_rad))
    return (c * x - s * z, s * x + c * z)


class Status(Enum):
    """
    球追踪系统的状态机枚举

    定义了网球机器人在不同阶段的状态，用于管理发球、回球和空闲等场景。
    状态转换通过 BallTracer 中的检测方法（is_user_return_ball、is_fire_ball）触发。
    """

    # ==================== 空闲状态 ====================
    IDLE = 1
    """闲置状态 - 系统等待开始新的回合

        - 没有正在追踪的球
        - 等待用户指令进行下一次发球
        - 所有轨迹数据已清空

    触发条件：
        - 初始化时的默认状态（当 IS_USER_SERVE=False）
        - 回合结束后调用 reset() 方法
        - 状态超时（status_check 检测到超时）
        - 曲线拟合错误时重置

    转换到：
        - FIRE_BALL_WITH_GUESS: 调用 reset(is_bot_serve=True) 时
        - USER_SERVE: 初始化时如果 IS_USER_SERVE=True
    """

    # SHOT_WITHOUT_IMG = 2 # 接收发球指令，或者在对打中发出了球. 但是还没有检测到球，此时要求小车yaw角始终与发射是一致，等待检测
    # SHOT_WITH_IMG = 3 # 检测到了发射的球，仍在往前飞行过程中
    # 对于机器人来说，是对方打回的球

    # ==================== 机器人发球阶段 ====================
    FIRE_BALL_WITH_GUESS = 2
    """机器人发球猜测阶段 - 已发球但数据不足

        - 机器人刚刚发出一个球
        - 开始检测球的位置，但样本数量还不够拟合轨迹曲线
        - 等待收集足够的帧数（通常需要 5 帧）

    触发条件：
        - 调用 reset(is_bot_serve=True) 时进入此状态
        - 表示机器人执行了发球动作

    转换到：
        - FIRE_BALL_WITH_CURVE: 当 is_fire_ball(5) 检测成功
          条件：连续5帧球持续变远且高度>0.4m
        - RETURN_BALL: 如果 is_user_return_ball() 检测到用户回球
        - IDLE: 超时（>3.5秒）时重置
    """

    FIRE_BALL_WITH_CURVE = 3
    """机器人发球曲线拟合阶段 - 可以拟合轨迹

        - 已收集足够的球位置数据
        - 可以拟合出机器人击出的球的飞行曲线
        - 持续追踪球的轨迹，更新 fire_curve

    触发条件：
        - 从 FIRE_BALL_WITH_GUESS 转换而来
        - is_fire_ball(5) 返回 True（连续5帧检测到球变远）

    转换到：
        - RETURN_BALL: 当 is_user_return_ball() 检测到用户回球
          条件：连续4帧球变近 + 距离/速度/时间条件满足
        - IDLE: 超时（>3.5秒）时重置
    """

    # ==================== 用户回球阶段 ====================
    RETURN_BALL = 4
    """用户回球阶段 - 检测到回球并预测击球点

        - 检测到用户击回的球正在向机器人飞来
        - 使用 curve 拟合球的飞行轨迹
        - 预测落点和最佳击球点（shot_point_list）
        - 机器人根据预测结果规划运动轨迹

    触发条件：
        - 从 FIRE_BALL_WITH_CURVE、FIRE_BALL_WITH_GUESS 或 USER_SERVE 转换而来
        - is_user_return_ball() 返回 True，需满足所有条件：
          1. 至少有 RETURN_BALL_LEN (4) 帧数据
          2. 连续帧中球持续变近（z坐标减小）
          3. 球高度始终 > 0.6m
          4. 球距离机器人 > RETURN_BALL_MINIMUS_DIS
          5. 最大飞行距离 > RETURN_BALL_BEGIN_DIS
          6. 检测时间窗口 < 0.5s
          7. 首帧距离差 > RETURN_BALL_FIRST_GAP

    转换到：
        - IDLE: 正常完成回球（超时 >3.5秒）
        - IDLE: 曲线拟合错误（curve.add_frame() 返回 -1）

    数据输出：
        - shot_point_list: 预测的击球点列表
        - land_speed: 球落地时的速度
        - return_ball: 首次检测到回球的时间戳
    """

    # ==================== 用户发球模式 ====================
    USER_SERVE = 5
    """用户发球状态 - 等待用户主动发球

        - 系统配置为用户发球模式（IS_USER_SERVE=True）
        - 机器人不主动发球，等待用户发球
        - 持续监测球的出现

    触发条件：
        - 初始化时 BotMotionConfig.IS_USER_SERVE=True
        - 调用 reset() 且 is_bot_serve=False 时

    转换到：
        - RETURN_BALL: 当 is_user_return_ball() 检测到用户发球
          检测条件与从其他状态转换相同
        - IDLE: 根据配置可能不会转到 IDLE，而是保持在 USER_SERVE
    """

    # ==================== 异常状态 ====================
    GENERAL_MISS = 101
    """通用失误状态 - 预留的异常处理状态
        - 表示发生了某种失误或异常情况
        - 当前代码中未被使用，预留给未来扩展

    触发条件：
        - ？
    """


class BallTracer:
    RETURN_BALL_LEN = 4  # 追踪回球的长度， 目前设置为3帧

    def __init__(self):
        self.logger = get_logger(
            "ball_tracer",
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

        self.logger.info("=" * 60)
        self.logger.info("初始化 BallTracer")
        self.logger.info("=" * 60)

        # 最新的小车位置
        self.last_frame_loc = [0, 0, 0, 0, time.perf_counter()]  # [x, y, z, yaw, ct]
        self.bot_loc_queue = deque()  # 小车状态信息， 目前包含： {loc:[x, y, z, yaw, camera_yaw,  time]}, 由于是字典可以后续添加其他信息
        self.ball_locs = deque()  # 记录所有检测到的球的位置信息，用于检测状态和预测落点

        # 相机配置（legacy 字段保留）。
        # 说明：本仓库的 curve_v2 用于离线对比，不强依赖双目参数；
        # 如需由像素坐标恢复 3D，请在上层完成后调用 `calc_target_loc_with_abs_ballloc`。
        self.cam1 = None
        self.cam2 = None

        if BotMotionConfig.IS_USER_SERVE:
            self.default_status = Status.USER_SERVE
            self.logger.info(f"配置: 用户发球模式 (IS_USER_SERVE=True)")
        else:
            self.default_status = Status.IDLE
            self.logger.info(f"配置: 机器人发球模式 (IS_USER_SERVE=False)")

        self.logger.info(f"回球检测参数:")
        self.logger.info(f"   - RETURN_BALL_LEN: {BallTracer.RETURN_BALL_LEN}")
        self.logger.info(
            f"   - RETURN_BALL_MINIMUS_DIS: {BotMotionConfig.RETURN_BALL_MINIMUS_DIS}m"
        )
        self.logger.info(
            f"   - RETURN_BALL_BEGIN_DIS: {BotMotionConfig.RETURN_BALL_BEGIN_DIS}m"
        )
        self.logger.info(
            f"   - RETURN_BALL_FIRST_GAP: {BotMotionConfig.RETURN_BALL_FIRST_GAP}m"
        )

        self.first_return_ball_time = None
        self.last_ball_start_time = 0

        # 人击出的球的曲线
        self.curve = Curve()
        # 机器人击出的球的曲线
        self.fire_curve = Curve()

        self.reset()
        # self.camera_processor.is_detect_full = Trueres
        pass

    # 重置tracer， 如果是机器人发球，需要设置is_bot_shot为True。 否则进入默认状态，可以是IDELE或者USER_SERVE（取决于配置）
    def reset(self, is_bot_serve=False):
        old_status = self.status.name if hasattr(self, "status") else "UNKNOWN"

        self.bot_loc_queue = deque()  # 小车状态信息， 目前包含： {loc:[x, y, z, yaw,  time]}, 由于是字典可以后续添加其他信息
        self.shot_loc, self.current_loc, self.target_loc = (
            None,
            None,
            None,
        )  # 分别记录发射球时的车位置、当前位置和目标移动位置

        # 记录上个球的开始时间
        if self.first_return_ball_time is not None:
            self.last_ball_start_time = self.first_return_ball_time
        # 首次回球时间
        self.first_return_ball_time = None

        # 人击出的球的曲线
        self.curve.reset()
        # 机器人击出的球的曲线
        self.fire_curve.reset()

        if is_bot_serve:
            self.status = Status.FIRE_BALL_WITH_GUESS
        else:
            self.status = self.default_status

        self.farthest_ball_loc = None  # 记录当前追踪的球最远球index
        self.farthest_ball_frame_cnt = 0  # 记录距离最远球隔了多少帧

        self.logger.info(
            f"重置球追踪器 | {old_status} -> {self.status.name} | is_bot_serve={is_bot_serve}"
        )

    # def is_ball_near(self):
    #     if not self.curve.first_ball_land_point:
    #         return False

    #     ct = time.perf_counter() - self.curve.time_base # 曲线内的时间标准

    #     real_time_ball_loc = np.array(self.curve.get_point_at_time(ct+0.05)) # 提前计算50ms后的朝向位置
    #     bot_loc = self.get_current_bot_loc()

    #     target_loc = real_time_ball_loc[:3] - bot_loc[:3]
    #     if target_loc[0] ** 2 + target_loc[2] ** 2 < 0.65 ** 2:
    #         return True

    #     return False

    # 用于获得用户回球曲线上，球在相机坐标系下的位置
    # update 20241209 by Yi
    # 只返回return ball时预测的球，其他情况使用search 和last ball追踪
    # 注意此处不进行是否有return ball 检查，因此必须在调用前保证有return ball
    def get_return_ball_in_camera(self, t=time.perf_counter()):
        """
        获取预测球的位置（同时包含世界坐标系和小车坐标系）

        Args:
            t: 预测时间戳

        Returns:
            dict: {
                'world': [x, y, z, t],  # 世界坐标系
                'rela2car': [x, y, z]   # 小车坐标系（相对于小车）
            }
            或 None（如果计算失败）
        """
        relative_t = t - self.curve.time_base
        ball_loc = self.curve.get_point_at_time(relative_t)
        if ball_loc is None:
            return None

        bot_loc = self.get_current_bot_loc()
        relative_loc = ball_loc[:3] - bot_loc[:3]

        x, z = rotate_2d_point(float(relative_loc[0]), float(relative_loc[2]), -float(bot_loc[3]))

        return {
            "world": ball_loc,  # 世界坐标系 [x, y, z, t]
            "rela2car": [x, relative_loc[1], z],  # 小车坐标系 [x, y, z]
        }  # 每次在bot agent loop中调用，用于更新tracer状态

    def status_check(self, last_fire_time):
        ct = time.perf_counter()

        if (
            self.status == Status.FIRE_BALL_WITH_CURVE
            or self.status == Status.FIRE_BALL_WITH_GUESS
        ) and ct > last_fire_time + 3.5:
            elapsed = ct - last_fire_time
            self.logger.warning(
                f"发球阶段超时 | 状态:{self.status.name} | 已持续:{elapsed:.2f}s > 3.5s | 重置追踪器"
            )
            self.reset()

        if (
            self.status == Status.RETURN_BALL and ct > self.first_return_ball_time + 3.5  # type: ignore
        ):  # return ball 最多持续时间，如果超过了重置状态
            # self.save_curve_speed()
            elapsed = ct - self.first_return_ball_time  # type: ignore
            self.logger.warning(
                f"回球阶段超时 | 已持续:{elapsed:.2f}s > 3.5s | 重置追踪器"
            )
            self.reset()

    def calc_target_loc_with_abs_ballloc(self, abs_ball_loc):
        data = {}
        self.ball_locs.append(abs_ball_loc)
        data["abs_loc"] = list(abs_ball_loc)

        # self.guess_x_loc = None
        while len(self.ball_locs) > 60:
            self.ball_locs.popleft()

        if self.status == Status.IDLE:
            return data

        # self.logger.info(f"self_statue:{self.status}, ball_loc:{abs_ball_loc}, first return ball time: {self.first_return_ball_time}")

        # ==================== 如果是机器发球后的监测阶段 ====================
        fire_ball_len = 5
        if self.status == Status.FIRE_BALL_WITH_GUESS and self.is_fire_ball(
            fire_ball_len
        ):
            self.status = Status.FIRE_BALL_WITH_CURVE
            self.logger.info(
                f"检测到机器人发球 | 状态切换: FIRE_BALL_WITH_GUESS -> FIRE_BALL_WITH_CURVE | 样本数:{fire_ball_len}"
            )
            for i in range(fire_ball_len):
                ball = self.ball_locs[-fire_ball_len + i]
                self.fire_curve.add_frame(ball, is_bot_fire=1)

        elif self.status == Status.FIRE_BALL_WITH_CURVE:
            self.fire_curve.add_frame(abs_ball_loc, is_bot_fire=1)

        # ==================== 检测是否已经进入回球阶段 ====================
        if (
            self.status == Status.FIRE_BALL_WITH_CURVE
            or self.status == Status.USER_SERVE
            or self.status == Status.FIRE_BALL_WITH_GUESS
        ):
            # 用户开始回球
            if self.is_user_return_ball():
                # TODO：保存发球轨迹落点 为json
                # self.logger.info(f"detect return ball, start to trace")
                self.first_return_ball_time = abs_ball_loc[-1]

                self.logger.info(
                    f"检测到用户回球 | 状态切换: {self.status.name} -> RETURN_BALL | 首次回球时间:{self.first_return_ball_time:.3f}s"
                )
                self.logger.info(
                    f"   回球位置: x={abs_ball_loc[0]:.3f}m, y={abs_ball_loc[1]:.3f}m, z={abs_ball_loc[2]:.3f}m"
                )
                self.logger.info(
                    f"   球队列长度:{len(self.ball_locs)} | 使用样本数:{BallTracer.RETURN_BALL_LEN}"
                )

                self.status = Status.RETURN_BALL
                data["return_ball"] = self.ball_locs[-1][-1]  # 返回球的时间

                for i in range(BallTracer.RETURN_BALL_LEN - 1):
                    ball = self.ball_locs[-(BallTracer.RETURN_BALL_LEN - 1) + i]
                    self.curve.add_frame(ball)

        # ==================== 回球阶段返回的数据 ====================
        if self.status == Status.RETURN_BALL:
            # update 0617: 拟合时间为1ms级别
            # update 0619: 加入拟合错误的判断（error_rate < 0). 如果曲线不对，则马上发送Target None， 同时设置为SHOT_WITHOUT_IMG状态
            shot_point_list = self.curve.add_frame(abs_ball_loc)
            if shot_point_list == -1:  # 曲线错误，马上停止小车
                data["samples_cnt"] = self.curve.curve_samples_cnt[0]
                data["reset"] = True
                self.reset()
                self.logger.error(
                    f"曲线拟合错误 | 样本数:{data['samples_cnt']} | 重置状态"
                )
            elif shot_point_list is not None:
                data["shot_point_list"] = shot_point_list

            # update 20241202: 增加记录球落地时速度
            land_speed = self.curve.land_speed.copy()
            data["land_speed"] = land_speed
        return data

    # 判断给定时间，是在第0曲线还是反弹曲线阶段。 如果是第2曲线，只取落地后的至多1.0秒
    def get_curve_stage(self, ft):
        if self.status != Status.RETURN_BALL:
            return 0  # 表示第0曲线，用mipi相机
        if self.curve.curve_samples_cnt[0] <= 8:
            return 0  # 表示第0曲线还不够，只考虑第0曲线

        land_time = (
            self.curve.land_point[0][-1] + self.curve.time_base
        )  # 计算落地时间   # type: ignore
        if ft < land_time - 0.03:
            return 0  # 距离落地还早，不考虑反弹曲线
        elif ft < land_time + 0.03:
            return -1  # 落地期间，一个都不考虑
        # emmmmm，下降段有可能超过一秒啊
        elif ft < land_time + 1.5:
            return 1  # 表示反弹曲线
        else:
            return 0  # 其他情况都用第0曲线

    # 后续的过滤球的代码可以移到这里，添加基于曲线过滤的逻辑。
    def calc_target_loc_with_ballpos(self, img_info):
        # 计算绝对位置, 并加入到队列中, 同时只保持队列至多60个球，以方便保存发球轨迹。

        # if "left_ball_pos" not in img_info or "right_ball_pos" not in img_info:
        #     return {}
        if "ball_loc" not in img_info:
            return {}
        ft = img_info["ft"]

        bot_loc = self.get_bot_loc_at_time(ft)
        # 得到相对于小车的位置
        # relative_ball_loc = utils.ball_estimation_by_cv2(img_info["left_ball_pos"], img_info["right_ball_pos"], self.cam1, self.cam2)
        relative_ball_loc = img_info["ball_loc"]

        # 得到球的绝对位置 格式为(x, y, z, ct)，注意ball_loc时间index为3， 并加入轨迹数组
        x, z = relative_ball_loc[0], relative_ball_loc[2]
        x, z = rotate_2d_point(float(x), float(z), float(bot_loc[3]))  # 把小车坐标系旋转回大地坐标系

        abs_ball_loc = np.array(
            [x + bot_loc[0], relative_ball_loc[1], z + bot_loc[2], ft]
        )

        data = self.calc_target_loc_with_abs_ballloc(abs_ball_loc)
        data["ball_loc"] = list(relative_ball_loc) + [ft]
        return data

    # 检测是否球往回了,
    # 条件是最近3个球都依次变近。 后续可能要更新该方案
    # def is_user_return_ball(self):
    #     n = len(self.ball_locs)
    #     if n < BallTracer.RETURN_BALL_LEN:
    #         return False
    #     for i in range(BallTracer.RETURN_BALL_LEN - 1):
    #         # 最近一帧球如果比上一帧还变远，说明不对！
    #         # update 由于此处过于敏感，先不加误差容忍
    #         if (
    #             self.ball_locs[n - BallTracer.RETURN_BALL_LEN + i][2] + 0
    #             < self.ball_locs[n - BallTracer.RETURN_BALL_LEN + i + 1][2]
    #             or self.ball_locs[n - BallTracer.RETURN_BALL_LEN + i][1] < 0.6
    #         ):
    #             return False
    #     # 如果球距离小车太近，也不对
    #     if (
    #         self.ball_locs[-1][2] - self.get_current_bot_loc()[2]
    #         < BotMotionConfig.RETURN_BALL_MINIMUS_DIS
    #     ):
    #         return False

    #     max_dis = 0
    #     start_index = max(len(self.ball_locs) - 12, 0)
    #     for i in range(start_index, len(self.ball_locs)):
    #         max_dis = max(max_dis, self.ball_locs[i][2] - self.ball_locs[-1][2])

    #     # return max_dis > 0.6
    #     # return max_dis>0.6 and self.ball_locs[-1][-1] - self.ball_locs[-BallTracer.RETURN_BALL_LEN][-1] < 0.5

    #     # return ball满足的条件：最远飞行距离超过了设定值，RETURN_BALL_LEN之间的总时间小于设定值，且第一个球和第二个球的距离超过了设定值, 且时间在0.5s内
    #     return (
    #         max_dis > BotMotionConfig.RETURN_BALL_BEGIN_DIS
    #         and self.ball_locs[-1][-1] - self.ball_locs[-BallTracer.RETURN_BALL_LEN][-1]
    #         < 0.5
    #         and self.ball_locs[-BallTracer.RETURN_BALL_LEN][2]
    #         - self.ball_locs[-BallTracer.RETURN_BALL_LEN + 1][2]
    #         > BotMotionConfig.RETURN_BALL_FIRST_GAP
    #     )

    def is_user_return_ball(self):
        """检测用户是否开始回球

        判断球是否正在向机器人方向移动（用户回球），
        而不是远离机器人（机器人发球）。

        检测条件（必须全部满足）：
            1. 样本数量：至少需要 RETURN_BALL_LEN 帧的球位置数据
            2. 连续接近：球在连续帧中沿 z 方向持续变近，且全程高度 > 0.6m
            3. 最小距离：球必须距离机器人当前位置至少 RETURN_BALL_MINIMUS_DIS
            4. 飞行距离：最大飞行距离（最近12帧内）必须超过 RETURN_BALL_BEGIN_DIS 阈值
            5. 时间窗口：检测帧的总时间跨度 < 0.5s（确保快速检测无延迟）
            6. 初始速度：首帧距离差必须超过 RETURN_BALL_FIRST_GAP（过滤慢速或静止球）

        Returns:
            bool: 检测到用户回球返回 True，否则返回 False

        数据结构:
            self.ball_locs: deque 队列，元素格式 [x, y, z, timestamp]
            - x, z: 世界坐标系下的水平位置
            - y: 垂直高度
            - timestamp: 检测时间戳

        Notes:
            - 用于触发从 FIRE_BALL_* 或 USER_SERVE 状态转换到 RETURN_BALL 状态
            - 所有距离比较主要使用 z 轴（深度）作为判断依据
        """
        n = len(self.ball_locs)

        # 条件1：检查样本数量是否足够
        if n < BallTracer.RETURN_BALL_LEN:
            return False

        # 条件2：验证球是否连续接近（z 减小）且保持有效高度
        for i in range(BallTracer.RETURN_BALL_LEN - 1):
            z_curr = self.ball_locs[n - BallTracer.RETURN_BALL_LEN + i][2]
            z_next = self.ball_locs[n - BallTracer.RETURN_BALL_LEN + i + 1][2]
            y_curr = self.ball_locs[n - BallTracer.RETURN_BALL_LEN + i][1]

            # 球应该在每个连续帧中变得更近（z 减小）
            if z_curr < z_next:
                return False
            # 球必须保持最小高度（避免地面球或噪声）
            if y_curr < 0.6:
                return False

        # 条件3：确保球距离机器人不会太近（安全性/有效性检查）
        bot_z = self.get_current_bot_loc()[2]
        ball_z = self.ball_locs[-1][2]
        ball_bot_dis = ball_z - bot_z
        if ball_bot_dis < BotMotionConfig.RETURN_BALL_MINIMUS_DIS:
            return False

        # 条件4：计算近期历史（最近12帧或更少）中的最大飞行距离
        # 这确保球在接近之前已经飞行了足够的距离
        max_dis = 0
        start_index = max(len(self.ball_locs) - 12, 0)
        for i in range(start_index, len(self.ball_locs)):
            max_dis = max(max_dis, self.ball_locs[i][2] - self.ball_locs[-1][2])

        if max_dis <= BotMotionConfig.RETURN_BALL_BEGIN_DIS:
            return False

        # 条件5：验证检测发生在合理的时间窗口内
        # 防止慢速移动物体或陈旧数据造成的误检
        time_gap = (
            self.ball_locs[-1][-1] - self.ball_locs[-BallTracer.RETURN_BALL_LEN][-1]
        )
        if time_gap >= 0.5:
            return False

        # 条件6：检查初始速度（首帧间隔）以确保球有足够的速度
        # 过滤掉球几乎不动或静止的情况
        first_gap = (
            self.ball_locs[-BallTracer.RETURN_BALL_LEN][2]
            - self.ball_locs[-BallTracer.RETURN_BALL_LEN + 1][2]
        )
        if first_gap <= BotMotionConfig.RETURN_BALL_FIRST_GAP:
            return False

        # 所有条件满足 - 检测到用户回球
        self.logger.info(
            f"检测到回球 | max_dis={max_dis:.3f}m | time_gap={time_gap:.3f}s | first_gap={first_gap:.3f}m | ball_bot_dis={ball_bot_dis:.3f}m"
        )
        return True

    # 判断是否时机器发球，逻辑与判断是否用户回球类似
    # todo 需要根据测试结果进一步优化
    def is_fire_ball(self, fire_ball_len=4):
        n = len(self.ball_locs)
        if n < fire_ball_len:
            return False

        for i in range(fire_ball_len - 1):
            z_curr = self.ball_locs[n - fire_ball_len + i][2]
            z_next = self.ball_locs[n - fire_ball_len + i + 1][2]
            y_curr = self.ball_locs[n - fire_ball_len + i][1]

            # 最近一帧球如果比上一帧还变近，说明不对！如果高度小于0.6，说明也不对
            if z_curr > z_next:
                return False
            if y_curr < 0.4:
                return False
            if z_curr > 8:
                return False

        self.logger.info(f"机器人发球检测成功")
        return True

    # 用于获取指定时间的小车位置
    def get_bot_loc_at_time(self, t):
        try:
            # last_frame_loc是最接近t的前一帧。next_frame则是最接近t的后一帧。
            while self.bot_loc_queue and t > self.bot_loc_queue[0][-1]:
                self.last_frame_loc = self.bot_loc_queue.popleft()

            if self.bot_loc_queue:
                next_frame = self.bot_loc_queue[0]
                tl = t - self.last_frame_loc[-1]
                tr = next_frame[-1] - t
                self.last_frame_loc = (
                    np.array(self.last_frame_loc) * tr + np.array(next_frame) * tl
                ) / (tl + tr)
                self.last_frame_loc[-1] = t

        except Exception as e:
            self.logger.error(f"获取小车位置出错: {e}")
            traceback.print_exc()

        return self.last_frame_loc

    def get_current_bot_loc(self):
        if self.bot_loc_queue:
            return np.array(self.bot_loc_queue[-1])
        else:
            return np.array(self.last_frame_loc)

    def log_ball_locs_summary(self):
        """记录球位置队列的摘要信息，用于调试"""
        if not self.ball_locs:
            return

        n = len(self.ball_locs)
        first_ball = self.ball_locs[0]
        last_ball = self.ball_locs[-1]

        # 计算z方向的趋势
        if n >= 2:
            z_trend = "接近" if last_ball[2] < self.ball_locs[-2][2] else "远离"
            z_change = abs(last_ball[2] - self.ball_locs[-2][2])
        else:
            z_trend = "未知"
            z_change = 0

        self.logger.debug(
            f"球队列状态 | 数量:{n} | 趋势:{z_trend}({z_change:.3f}m) | 最新:z={last_ball[2]:.3f}m,y={last_ball[1]:.3f}m | 时间跨度:{last_ball[3] - first_ball[3]:.3f}s"
        )
