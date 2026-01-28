# 1126 创建的用于单组双目（单个cap 只读取一张图片）

"""
============================================================================
Camera-BotAgent 时序同步与预测性截图系统
============================================================================
流程示意：
Camera检测球 → BotAgent预测轨迹 → Camera使用预测截图

============================================================================
主要数据流
============================================================================
1. Camera → BotAgent (球检测结果)
   Topic: img_info_topic
   发送函数: run_ball_detect_result_handle() [408行]
   数据格式: {"ball_loc": [x, y, z], "ft": timestamp, ...}

2. BotAgent → Camera (球位置预测)
   Topic: estimate_loc_topic
   接收函数: estimate_loc_callback() [333行]
   消费函数: deal_estimate_msg() [1253行]
   数据格式: {
       "pred_ball_loc_rela2car": [x, y, z],   # 预测球相对于小车的位置
       "pred_ball_loc_world": [x, y, z, t],   # 预测球在世界坐标系的位置
       "ts": timestamp,                       # 预测时间 (当前时间 + 33ms)
       "bot_loc": [x, y, z, yaw],             # 小车当前位置
       "fire_time": time,                     # 发球时间
       "source": "return_curve_prediction" | "mipi_as_source"
   }

============================================================================
时序循环详解 (以30fps为例，帧间隔33ms)
============================================================================

T=0ms   [Camera读取第1帧]
        └─ read_frame_cut() [1100行]
           ├─ expect_time = last_read_time + frame_interval  (预测下一帧时间)
           ├─ get_ball_detect_cut_param(expect_time) [905行]
           │  └─ 使用 self.last_ball_loc 计算截图框位置
           └─ cap.read_crop_multi() 读取并裁剪图片

T=5ms   [球检测推理]
        └─ model.predict() 异步推理

T=8ms   [检测结果处理]
        └─ run_ball_detect_result_handle() [408行]
           ├─ 获取检测框，计算球的3D位置
           └─ 发布到 img_info_topic
              → self.last_frame_detect_time = ft

T=10ms  [BotAgent接收并处理]
        └─ bot_agent.deal_with_img_info()
           ├─ ball_tracer.calc_target_loc_with_ballpos()
           │  └─ curve.add_frame() 更新球轨迹
           └─ 预测未来33ms的球位置

T=12ms  [BotAgent发布预测]
        └─ bot_agent.publish_ball_tracer() [bot_agent.py:683]
           ├─ predict_ball_time = time.perf_counter() + 0.033
           ├─ ball_data = ball_tracer.get_return_ball_in_camera(predict_ball_time)
           └─ 发布到 estimate_loc_topic
              data_to_send = {
                  "pred_ball_loc_rela2car": [...],
                  "ts": predict_ball_time,  ← 关键：预测的是"下一帧"的时间
                  ...
              }

T=15ms  [Camera接收预测]
        └─ estimate_loc_callback() [333行]
           └─ self.estimate_msg_queue.put(data)

T=30ms  [Camera准备读取第2帧]
        └─ read_frame_cut()
           ├─ expect_time = 33ms (预测下一帧时间)
           ├─ deal_estimate_msg(expect_time + 33ms) [1253行]
           │  ├─ 遍历 estimate_msg_queue
           │  ├─ if data["ts"] > ft: break  (只使用时间合理的预测)
           │  └─ self.last_ball_loc = data["pred_ball_loc_rela2car"]
           │     → self.last_estimate_ball_update_time = data["ts"]
           ├─ get_ball_detect_cut_param(33ms) [905行]
           │  └─ 使用更新后的 last_ball_loc 计算截图框
           └─ 读取第2帧并在预测位置附近截图

(循环继续...)

============================================================================
三级降级策略 (容错机制)
============================================================================

优先级1: 曲线预测模式 (最优)
    触发条件: 最近检测到球 && 收到最新预测
    截图位置: self.last_ball_loc (来自 pred_ball_loc_rela2car)
    来源标记: self.cut_box_from = "return_curve_prediction"

优先级2: 发球搜索模式 (发球后短期丢球)
    触发条件: (ft - last_frame_detect_time > 0.4)
              AND (ft - last_estimate_ball_update_time > 0.3)
              AND (ft - last_fire_time < 1.2s)
    截图位置: self.fire_search_ball_loc_list[index]  (固定位置列表)
    来源标记: self.cut_box_from = "fire search box"

优先级3: 回球搜索模式 (长期丢球)
    触发条件: 长时间未检测到球且非发球阶段
    截图位置: self.search_ball_loc_list[index]  (九宫格搜索)
    来源标记: self.cut_box_from = "return search box"

判断逻辑: get_ball_detect_cut_param() [905行]

============================================================================
相关状态变量
============================================================================

时间戳：
    self.last_frame_detect_time           - 最后一次检测到球的时间
    self.last_estimate_ball_update_time   - 最后一次收到预测的时间
    self.last_fire_time                   - 最后一次发球的时间

位置数据：
    self.last_ball_loc                    - 当前用于截图的球位置 [x, y, z]
    self.current_bot_loc                  - 小车当前位置 [x, y, z, yaw]

队列：
    self.estimate_msg_queue               - 存储来自bot_agent的预测 (Queue)
    self.frame_queue_ball                 - 存储待推理的帧信息 (Queue)
    self.left_camera_crop_queue           - 左相机裁剪后的图片队列
    self.right_camera_crop_queue          - 右相机裁剪后的图片队列

搜索配置：
    self.search_ball_loc_list             - 回球搜索位置列表 (九宫格)
    self.fire_search_ball_loc_list        - 发球搜索位置列表
    self.search_ball_loc_index            - 当前搜索位置索引

状态标记：
    self.cut_box_from                     - 截图来源标记 (用于debug和日志)

============================================================================
主要线程
============================================================================

1. read_frame_cut(camera_id) [1100行]
   从相机读取帧并进行多任务裁剪 (球、T点、人体姿态、视频录制)
   频率: 30fps
   关键操作:
   - 预测下一帧时间: expect_time = last_read_time + frame_interval
   - 获取截图参数: get_ball_detect_cut_param(expect_time)
   - 读取并裁剪: cap.read_crop_multi()

2. deal_two_frames_cut() [1324行]
   同步左右相机的裁剪图片并分发到各检测任务
   关键操作:
   - 左右图时间对齐 (误差<50ms)
   - 调用 deal_estimate_msg() 更新预测位置
   - 分发到球检测、T点检测、姿态检测

3. run_ball_detect_result_handle() [408行]
   处理球检测推理结果并发布
   关键操作:
   - 获取推理结果: model.get_result()
   - 计算3D位置: 双目视觉计算
   - 发布检测结果: img_info_pub.publish()
   - 更新时间戳: self.last_frame_detect_time = ft

============================================================================
性能参数 (对应代码位置和变量)
============================================================================

帧率: 30fps
    └─ 配置: BotMotionConfig.CAMERA_FPS
    └─ 使用位置: read_frame_cut() [约1279行]
       frame_fps = BotMotionConfig.CAMERA_FPS
       frame_interval = 1.0 / frame_fps

预测提前量: 33ms (一帧间隔)
    └─ 计算: 1.0 / 30fps = 0.033s
    └─ 使用位置:
       - bot_agent.publish_ball_tracer() [bot_agent.py:683行]
         predict_ball_time = time.perf_counter() + 0.033
       - deal_estimate_msg() [约1430行]
         self.deal_estimate_msg(ft + 1.0 / BotMotionConfig.CAMERA_FPS)

时间对齐容差: 50ms
    └─ 变量: 硬编码常量 0.050
    └─ 使用位置: deal_two_frames_cut() [约1347行]
       if abs(left_crop[-1] - right_crop[-1]) > 0.050:
           print(f"shift one img, {left_crop[-1]}, {right_crop[-1]}")

丢球判定阈值:
    └─ 检测间隔阈值: 400ms (0.4s)
       变量: 硬编码常量 0.4
       位置: get_ball_detect_cut_param() [约1082行]
       条件: ft - self.last_frame_detect_time > 0.4

    └─ 预测间隔阈值: 300ms (0.3s)
       变量: 硬编码常量 0.3
       位置: get_ball_detect_cut_param() [约1082行]
       条件: ft - self.last_estimate_ball_update_time > 0.3

    完整判定逻辑:
       need_search = (
           ft - self.last_frame_detect_time > 0.4
           and ft - self.last_estimate_ball_update_time > 0.3
       )

发球搜索持续时间: 1.2s
    └─ 变量: self.fire_ball_search_duration
    └─ 初始化: __init__() [约271行]
       self.fire_ball_search_duration = 1.2
    └─ 使用位置: get_ball_detect_cut_param() [约1087行]
       if ft - self.last_fire_time < self.fire_ball_search_duration:

搜索模式刷新策略: 每帧切换到下一个搜索位置
    └─ 回球搜索索引: self.search_ball_loc_index
       更新位置: deal_two_frames_cut() [约1412行]
       self.search_ball_loc_index = (self.search_ball_loc_index + 1) % len(
           self.search_ball_loc_list
       )

    └─ 发球搜索索引: self.fire_serach_ball_loc_index
       更新位置: deal_two_frames_cut() [约1409行]
       self.fire_serach_ball_loc_index = (
           self.fire_serach_ball_loc_index + 1
       ) % len(self.fire_search_ball_loc_list)

T点检测间隔: 900ms (0.9s)
    └─ 变量: 硬编码常量 0.9
    └─ 使用位置: deal_two_frames_cut() [约1384行]
       if ft - self.last_t_predict_time > 0.9:
           self.is_predict_t_flag = True

人体姿态检测间隔: 120ms (0.12s)
    └─ 变量: 硬编码常量 0.12
    └─ 使用位置: deal_two_frames_cut() [约1391行]
       if (self.enable_person_pos_detect
           and ft - self.last_add_person_pos_detect_time > 0.12):

============================================================================
"""

import cv2
import time
import numpy as np
from queue import Queue
import threading
import json
import traceback

import subprocess
import numpy as np
import sys
import os
import math
import logging


# from . import rknn_inference
# from .rknn_inference_muti import Inference
# yolov8 v10的后处理可以一样，所以其实换个模型就行
from .camera.rknn_inference_pose import Inference_Pose
from .camera.yolov10_inference_muti import Inference
from .camera import VideoCaptureV4L2c
from .camera import Inference_seg_i8  # type: ignore

from .cfgs import config
from .cfgs.bot_motion_config import BotMotionConfig
from .ball import filter_ball
from .utils import utils
from .camera.camera_writer import CameraWriter

from ament_index_python.packages import get_package_share_directory


import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# create 1130, 适用于4k旋转摄像头的camera processor。
# 1）整体图片是需要逆时针旋转后得到实际观察（由于4:3相机限制只能这么做）
# - 不对原图片旋转（因为太费时）， 转而在crop_frame方法中截取对应的部分，然后在get_ball_center_with_base中还原
# 2）截图时根据网球预测位置在相机上的投影来截，提高双目对齐率. done
# 3）设置预测位置后，同时更新待搜索的范围（目前使用九宫格模式），如果第一位置没有检测到就进入搜索模式 （未完成）
# 4）在截图时加入缩放机制， 以更好的平衡可见范围和图片搜索区域. done


def is_point_on_edge(point, img_width=640, img_height=640, edge_threshold=50):
    x, y = point

    return (
        x <= edge_threshold
        or img_width - x <= edge_threshold
        or y <= edge_threshold
        or img_height - y <= edge_threshold
    )


def is_edge(box, img_width=640, img_height=640, edge_threshold=5):
    x1, y1, x2, y2 = box

    # 检查框是否靠近图像的四个边界
    near_left_edge = x1 <= edge_threshold
    near_right_edge = img_width - x2 <= edge_threshold
    near_top_edge = y1 <= edge_threshold
    near_bottom_edge = img_height - y2 <= edge_threshold

    # 如果框靠近任何一个边界，返回 True
    return near_left_edge or near_right_edge or near_top_edge or near_bottom_edge


class CameraProcesser(Node):
    def __init__(self):
        super().__init__("CameraProcesser")
        self.detect_size = 640  # 图片检测的分辨率

        # ================================  网球和T点的截图信息 ===================================#
        self.is_predict_t_flag = (
            True  # 是否预测T点, 预测T点必然用近焦， 同时只要保持3hz的检测频率即可
        )
        self.last_t_predict_time = 0  # 上一次T点预测时间，用以控制预测频率
        self.current_bot_loc = [0, 0, 0, 0, math.pi / 2]
        self.current_bot_yaw_speed = 0

        self.vision_stand_dis = 15  # 不缩放图片进行推理的标准距离，如果小于此距离，进行等比例缩放 来覆盖相应的范围
        self.cut_box_from = ""

        # 每次设置下次检测球位时， 专门生成基于深度的9宫格待检索区域， 第0个值为last detect ball或bot_agent发来的信息。 当长时间没检测到球时就index+1搜索模式
        # self.search_ball_loc_list = [[0, 0, 8], [-2.0, 1.0, 12], [0, 1.0, 12], [2.0, 1.0, 12], [0, 1.0, 8], [3.0, 1.0, 8]]
        # 用户回球检测时的搜索设置
        self.search_ball_loc_list = [
            [0, 0, 8],
            [-2.0, 1.0, 12],
            [0, 1.0, 12],
            [2.0, 1.0, 12],
            [3.5, 1.0, 12],
            [0, 1.0, 8],
        ]
        # self.search_ball_loc_list = [[0, 2, 5], [0, 0.5, 5]]
        self.search_ball_loc_index = 0

        # 发射球时的搜索设置。 只在发射一段时间内进行发射搜索模式，发射搜索配置与回球搜索不同，同时last ball box 不进行位置上调。
        # self.fire_search_ball_loc_list = [[0, 1, 3], [0, 1, 5], [0, 2, 5], [-1, 1, 5], [-1, 2, 5], [1, 1, 5], [1, 2, 5]]
        self.fire_search_ball_loc_list = [[0, 2, 5], [0, 0.5, 5]]
        self.last_ball_loc = [0, 0, 10]

        self.fire_serach_ball_loc_index = 0
        self.fire_ball_search_duration = 1.2  # 发射搜索持续时间

        # IDLE状态下的搜索人姿势设置
        self.search_person_loc_list = [
            [-1.2, 0.8, 10],
            [-0.5, 0.8, 10],
            [0, 0.8, 10],
            [0.5, 0.8, 10],
            [1.2, 0.8, 10],
        ]
        self.search_person_loc_index = 0
        self.last_person_loc = [0, 0, 10]

        # T点的搜索位置
        self.search_t_list = [
            [0, 0, 5],
            # [0, 0, 5],
            # [0, 0, 7],
            # [0, 0, 10],
            # [0, 0, 15],
            # [0, 0, 20],
        ]
        self.search_t_index = 0
        self.need_search_t = False
        # 暂时未用到，上一次能找到T点的位置，目前只有一个固定找T点的位置
        # self.last_refind_loc_t = None

        # 记录检测或发送的预估球时间
        self.last_frame_detect_time = 0
        self.last_estimate_ball_update_time = 0
        self.last_fire_time = 0  # 上一次机器发球的时间

        # =========================================== 推理时暂存的信息， 用于和推理结果对齐 =================#
        self.frame_queue = Queue()  # 用来记录缓存收到摄像头后待检测帧的左右图片中心点 和 时间， 格式为（left_base, right_base, time)

        # update by gwj in 20241213，frame_queue分成两个，一个用于球，一个用于t点
        self.frame_queue_ball = Queue()
        self.frame_queue_t_point = Queue()
        self.frame_queue_person_pos = Queue()

        self.video_buf1 = Queue()  # 未加label的video buffer
        # update by gwj in 20241213，video_buf1分成两个，一个用于球，一个用于t点
        self.video_buf1_ball = Queue()
        self.video_buf1_t_point = Queue()
        self.video_buf1_person_pos = Queue()

        self.detected_frames_cnt = 0  # 累计已经检测的图片数量
        self.camera_sync_read_frames_cnt = 0  # 相机累计读取的图片数量
        self.camera_single_read_frames_cnt = 0  # 单个相机累计读取的图片数量

        # 读取图片延迟
        self.read_frame_latency = 0.0

        self.ball_inference_latency = 0.0
        self.ball_inference_cnt = 0

        self.world_base_time = 0

        # ======================================== 模型信息============================================#
        package_share_dir = get_package_share_directory("hit")
        config_path = os.path.join(package_share_dir, "models", "tennis_20241227.rknn")

        self.model = Inference(config_path)

        self.model.start()

        self.model_seg = Inference_seg_i8.Inference_seg(BotMotionConfig.GROUND_MODEL_ID)
        self.model_seg.start()

        pose_path = os.path.join(package_share_dir, "models", "yolov8n-pose_2.rknn")
        self.model_person_pos = Inference_Pose(pose_path)
        self.model_person_pos.start()

        self.filter = filter_ball.BallFilter()

        # ============================== ROS 发布关于球状态信息话题 ====================================#
        self.img_info_pub = self.create_publisher(String, "img_info_topic", 10)
        self.img_analysis_pub = self.create_publisher(String, "img_analysis_topic", 10)

        # update: 1205
        # 接收来自bot_agent，在给定时间下要瞄准的图片位置，也就是ball_pos，
        # 格式为{"ball_loc": [0,1,2],  "shot_time": 123.4] } 分别表示球的位置和发射时机
        self.estimate_msg_queue = Queue()
        self.estimate_loc_sub = self.create_subscription(
            String, "estimate_loc_topic", self.estimate_loc_callback, 10
        )

        self.person_pos_detect_sub = self.create_subscription(
            String, "person_pos_detect_topic", self.person_pos_detect_callback, 10
        )
        self.last_person_pos_detect_time = 0  # 上一次检测到人的时间
        self.enable_person_pos_detect = False

        self.last_add_person_pos_detect_time = 0  # 上一次添加检测人任务的时间

        self.bot_switch_sub = self.create_subscription(
            String, "bot_switch_topic", self.bot_switch_callback, 10
        )

        self.need_search_t_sub = self.create_subscription(
            String, "need_search_t_topic", self.need_search_t_callback, 10
        )

        self.ball_camera_writer = None
        self.t_camera_writer = None
        self.person_camera_writer = None
        self.video_camera_writer = None

        self.need_record_video = False
        self.last_record_video_time = 0
        self.record_video_frame_queue = Queue()

        if BotMotionConfig.SAVE_VIDEO_ENABLE:
            time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
            self.ball_camera_writer = CameraWriter(time_str + "_ball", "ball")
            self.t_camera_writer = CameraWriter(time_str + "_t", "t")
            # self.person_camera_writer = CameraWriter(time_str + "_person")
            self.video_camera_writer = CameraWriter(time_str + "_video", "video")

        #  =============================== 启动线程 =================================================#
        self.left_camera_crop_queue = (
            Queue()
        )  # 分别存储做存储左右摄像头的截图队列， 用于软同步
        self.right_camera_crop_queue = Queue()

        self.stop_event = threading.Event()

        # self.read_frame_thread_0 = threading.Thread(target=self.read_front_frames, args=(0,))
        # self.read_frame_thread_0.start()
        # self.read_frame_thread_1 = threading.Thread(target=self.read_front_frames, args=(1,))
        # self.read_frame_thread_1.start()
        # self.trigger_thread = threading.Thread(target=self.trigger_cmd, args=())
        # self.trigger_thread.start()

        # self.detect_thread = threading.Thread(target=self.run, args=())
        # self.detect_thread.start()

        self.ball_detect_result_handle_thread = threading.Thread(
            target=self.run_ball_detect_result_handle, args=()
        )
        self.ball_detect_result_handle_thread.start()

        self.t_detect_result_handle_thread = threading.Thread(
            target=self.run_t_detect_result_handle, args=()
        )
        self.t_detect_result_handle_thread.start()

        self.pose_detect_result_handle_thread = threading.Thread(
            target=self.run_person_pos_detect_result_handle, args=()
        )
        self.pose_detect_result_handle_thread.start()

        self.deal_video_record_thread = threading.Thread(
            target=self.deal_video_record, args=()
        )
        self.deal_video_record_thread.start()

        # self.deal_frame_thread = threading.Thread(target=self.deal_two_frames)
        # self.deal_frame_thread.start()

        self.left_camera_can_read = threading.Semaphore(1)
        self.right_camera_can_read = threading.Semaphore(1)

        self.read_frame_thread_0 = threading.Thread(
            target=self.read_frame_cut, args=(0,)
        )
        self.read_frame_thread_0.start()

        self.read_frame_thread_1 = threading.Thread(
            target=self.read_frame_cut, args=(1,)
        )
        self.read_frame_thread_1.start()

        self.deal_frame_thread = threading.Thread(target=self.deal_two_frames_cut)
        self.deal_frame_thread.start()

        # self.profile_memory_operations()
        self.model_missed_frame_cnt = 0

        pass

    def profile_memory_operations(self):
        """测试不同内存操作的性能"""
        print("开始内存操作性能分析...")

        # 测试MIPI帧的内存操作性能
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)  # 模拟25MB帧

        # 测试大块内存复制
        start = time.perf_counter()
        frame_copy = frame.copy()  # 完整复制
        copy_time = time.perf_counter() - start

        # 测试裁剪操作
        start = time.perf_counter()
        crop = frame[500:1500, 1000:2000, :]  # 视图创建
        crop_time = time.perf_counter() - start

        # 测试裁剪+复制
        start = time.perf_counter()
        crop_copy = frame[500:1500, 1000:2000, :].copy()  # 裁剪并复制
        crop_copy_time = time.perf_counter() - start

        # 测试实际裁剪+调整大小操作
        start = time.perf_counter()
        resized = cv2.resize(frame[500:1500, 1000:2000, :], (640, 640))
        resize_time = time.perf_counter() - start

        print(f"内存性能分析结果:")
        print(f"  完整帧复制: {copy_time * 1000:.2f}ms")
        print(f"  裁剪视图创建: {crop_time * 1000:.2f}ms")
        print(f"  裁剪+复制: {crop_copy_time * 1000:.2f}ms")
        print(f"  裁剪+调整大小: {resize_time * 1000:.2f}ms")

        # 返回结果以便进一步分析
        return {
            "copy_time": copy_time,
            "crop_time": crop_time,
            "crop_copy_time": crop_copy_time,
            "resize_time": resize_time,
        }

    def estimate_loc_callback(self, msg):
        # ball pos格式为 [left_pos, right_pos, ct]
        data = json.loads(msg.data)
        self.estimate_msg_queue.put(data)

    def person_pos_detect_callback(self, msg):
        data = json.loads(msg.data)
        enable_flag = data["detect"]
        if enable_flag == "enable":
            # print("turn on person pos detect")
            self.enable_person_pos_detect = True
        else:
            # print("turn off person pos detect")
            self.enable_person_pos_detect = False

    def bot_switch_callback(self, msg):
        print(f"receive bot_switch_callback msg: {msg.data}")
        data = json.loads(msg.data)

        mode = data["mode"]
        switch_type = data["type"]
        print("turn off all cameraWriter")
        if self.ball_camera_writer is not None:
            self.ball_camera_writer.close()
            self.ball_camera_writer = None
        if self.t_camera_writer is not None:
            self.t_camera_writer.close()
            self.t_camera_writer = None
        if self.video_camera_writer is not None:
            self.video_camera_writer.close()
            self.video_camera_writer = None

        if "run_start" == switch_type:
            # time_str = time.strftime("%Y%m%d%H%M%S", time.localtime(start_time))
            start_time = data["start_time"]
            time_str = str(start_time)
            print(f"receive bot start")
            if mode == "rally":
                if BotMotionConfig.SAVE_VIDEO_ENABLE:
                    print(f"start to record ball and t, time_str: {time_str}")
                    self.ball_camera_writer = CameraWriter(time_str + "_ball", "ball")
                    self.t_camera_writer = CameraWriter(time_str + "_t", "t")
                    self.video_camera_writer = CameraWriter(
                        time_str + "_video", "video"
                    )
                else:
                    print("not save video")
            if mode == "ball_machine":
                print(f"start to record video, time_str: {time_str}")
                self.video_camera_writer = CameraWriter(time_str + "_video", "video")
        pass

    def need_search_t_callback(self, msg):
        data = json.loads(msg.data)
        # print(data)
        need_search_t = data["need_search_t"]
        if need_search_t:
            # print(f"need search t point")
            self.need_search_t = True
        else:
            # print(f"no need search t point")
            self.need_search_t = False
        # FIXME 临时加的， 后续和bot_agent协商好再改，永远不搜 t 点
        self.need_search_t = False

    # 对于行扫描相机，逆时针旋转才会正，相当于变成从左往右扫描， 结合拍摄时的小车yaw speed，对于y坐标做补偿
    # yaw速度误差在0.1rad/s级别， 整体行扫描延迟在20ms级别， 因此会有0.002 rad级别的误差，2k宽度像素，大约是4个像素误差（可以接受）
    def scan_compensate_x(self, x, yaw_speed):
        time_gap = x * 10e-6  # 5us一列，相比于第一列的时间间隔
        f = config.CMAERA_LEFT_PARAMS["mtx"][0][0]
        angle_in_take_moment = math.atan2(
            x - config.CAMERA_WIDTH // 2, f
        )  # 假设光学中心位于中点，相对光学中心的角度
        angle_in_first_colunm = (
            angle_in_take_moment - yaw_speed * time_gap
        )  # 相对于第一列，多旋转了time_gap时间，因此增加了 yaw_speed * time_gap
        new_x = config.CAMERA_WIDTH // 2 + f * math.tan(angle_in_first_colunm)
        return new_x

    def run_ball_detect_result_handle(self):
        record_frame_cnt = 0
        while True:
            two_boxes = self.model.result_que.get()
            # print(f"two_boxes:{two_boxes}")
            if two_boxes is None:
                break
            origin_frame = self.video_buf1_ball.get()
            lcut, rcut, ft, cut_box_from, yaw_speed, loc_and_pos = (
                self.frame_queue_ball.get()
            )
            # print(f"gap:{time.perf_counter()- ft}")
            self.ball_inference_cnt += 1
            self.ball_inference_latency += time.perf_counter() - ft

            if two_boxes is None:
                # 方便后面的处理
                two_boxes = [[[], []], [[], []]]

            lboxes = two_boxes[0]
            rboxes = two_boxes[1]

            # 索引0是网球，20241220之前的0是网球袋，1是网球
            # tennis_box[0] 是位置， tennis_box[1] 是置信度
            # rknn_inference_muti用
            # left_tennis_boxes = lboxes[0]
            # right_tennis_boxes = rboxes[0]

            # yolov10_inference_muti用
            left_tennis_boxes = lboxes
            right_tennis_boxes = rboxes

            info_to_send_dict = {
                "lball_center_list": [],  # 使用[x,y]表示网球中心在图片的像素位置
                "lball_box_list": [],  # 使用[x, y, w, h] 表示网球box在世界图片中的位置
                "lball_ori_infer_box_list": [],  # 使用[x, y, w, h] 表示网球box在原始图片中的位置
                "rball_center_list": [],  # 同上
                "rball_box_list": [],
                "rball_ori_infer_box_list": [],
                "ft": ft,  # photo time
                "left_cut": lcut,  # 左截图框，用于label视频截图表示
                "right_cut": rcut,
                "box_from": cut_box_from,  # 截图框来源
                "car_loc": self.current_bot_loc,
                "unix_timestamp": int(time.time() * 1000),
            }
            if loc_and_pos is not None:
                info_to_send_dict["predict_ball_loc"] = [
                    float(loc_and_pos[0][0]),
                    float(loc_and_pos[0][1]),
                    float(loc_and_pos[0][2]),
                ]

            # 添加网球信息
            for i in range(len(left_tennis_boxes)):
                if left_tennis_boxes[i][0] is not None and (
                    not is_edge(left_tennis_boxes[i][0])
                ):
                    info_to_send_dict["lball_ori_infer_box_list"].append(
                        left_tennis_boxes[i][0].tolist()
                    )
                    l_center, l_box = self.get_ball_center_with_base(
                        left_tennis_boxes[i][0], lcut, yaw_speed
                    )
                    info_to_send_dict["lball_center_list"].append(l_center)
                    info_to_send_dict["lball_box_list"].append(l_box)

            for i in range(len(right_tennis_boxes)):
                if right_tennis_boxes[i][0] is not None and (
                    not is_edge(right_tennis_boxes[i][0])
                ):
                    info_to_send_dict["rball_ori_infer_box_list"].append(
                        right_tennis_boxes[i][0].tolist()
                    )
                    r_center, r_box = self.get_ball_center_with_base(
                        right_tennis_boxes[i][0], rcut, yaw_speed
                    )
                    info_to_send_dict["rball_center_list"].append(r_center)
                    info_to_send_dict["rball_box_list"].append(r_box)

            # with threadpool_limits(limits=1):
            # filter_balls 数据格式： [[left_center_x,left_center_y],[right_center_x,right_center_y],left_ball_list[i],right_ball_list[j], [error_l, error_r], loc_result])
            # filter_balls = self.filter.dual_filter_without_loc(info_to_send_dict['lball_box_list'], info_to_send_dict['rball_box_list'])

            # 加入基于球场边界的识别，注意此处bot_loc的yaw与拍摄时间可能有一定误差，且基于单目，因此整体准确率会偏低（具体根据球场测试调整）
            # filter_balls = self.filter.dual_filter_with_bot_loc(info_to_send_dict['lball_box_list'], info_to_send_dict['rball_box_list'], self.current_bot_loc)

            # filter_balls = self.filter.dual_filter_with_bot_loc(info_to_send_dict['lball_box_list'], info_to_send_dict['rball_box_list'], self.current_bot_loc)

            if loc_and_pos is None:
                filter_balls = self.filter.dual_filter_with_bot_loc(
                    info_to_send_dict["lball_box_list"],
                    info_to_send_dict["rball_box_list"],
                    self.current_bot_loc,
                )
            else:
                filter_balls = (
                    self.filter.dual_filter_with_bot_loc_and_predict_ball_loc(
                        info_to_send_dict["lball_box_list"],
                        info_to_send_dict["rball_box_list"],
                        self.current_bot_loc,
                        [
                            float(loc_and_pos[0][0]),
                            float(loc_and_pos[0][1]),
                            float(loc_and_pos[0][2]),
                        ],
                    )
                )
            if len(filter_balls) == 1:
                info_to_send_dict["left_ball_pos"] = filter_balls[0][0]
                info_to_send_dict["right_ball_pos"] = filter_balls[0][1]
                info_to_send_dict["project_error"] = filter_balls[0][4]
                info_to_send_dict["ball_loc"] = filter_balls[0][5].tolist()

                # ball_loc = utils.ball_estimation_by_cv2(left_ball_pos, right_ball_pos, config.CMAERA_LEFT_PARAMS_SHORT, config.CMAERA_RIGHT_PARAMS_SHORT)

                ball_loc = filter_balls[0][5]

                # 假设一帧行扫描延迟时间20ms， 一个像素是20ms / 2160
                info_to_send_dict["ft"] += (
                    filter_balls[0][6] / 2160 * BotMotionConfig.SCAN_DELAY
                )

                self.last_frame_detect_time = ft  # 图片检测时间全部换成拍照时刻
                ct = time.perf_counter()
                if (
                    self.last_estimate_ball_update_time + 0.3 < ct
                ):  # 预估网球时间已经超时, 预估网球的时间和当前时间比较（因为下一帧接近当前时间）。
                    # 如果处在搜索发球模式中
                    if ct - self.last_fire_time < self.fire_ball_search_duration:
                        self.last_ball_loc = [ball_loc[0], ball_loc[1], ball_loc[2]]
                        # self.fire_search_ball_loc_list[0] = [ball_loc[0], ball_loc[1], ball_loc[2]]
                        # self.fire_serach_ball_loc_index = 0

                        self.cut_box_from = "last fire ball"
                    else:
                        # self.search_ball_loc_list[0] = [ball_loc[0], ball_loc[1]+BotMotionConfig.LAST_BOX_Z_CENTER_MOVE, ball_loc[2]]
                        # self.search_ball_loc_index = 0
                        self.last_ball_loc = [
                            ball_loc[0],
                            ball_loc[1] + BotMotionConfig.LAST_BOX_Z_CENTER_MOVE,
                            ball_loc[2],
                        ]
                        self.cut_box_from = "last return ball"

                print(f"ball loc is {ball_loc}")

            if self.ball_camera_writer is not None:
                info_to_send_dict["video_time_str"] = self.ball_camera_writer.file_name
            try:
                self.img_info_pub.publish(String(data=json.dumps(info_to_send_dict)))
            except Exception as e:
                print(f"publish error: {e}")
                pass
            # 要写入的数据
            if self.ball_camera_writer is not None:
                if record_frame_cnt % 5 == 0:
                    self.ball_camera_writer.write(origin_frame, info_to_send_dict)
                record_frame_cnt += 1

        print("ball detect result handle thread exit")
        if self.ball_camera_writer is not None:
            self.ball_camera_writer.close()

    def run_t_detect_result_handle(self):
        while True:
            T_info_and_net_strap_info = self.model_seg.result_pos.get()
            if T_info_and_net_strap_info is None:
                break
            origin_frame = self.video_buf1_t_point.get()
            lcut, rcut, ft, cut_box_from, yaw_speed = self.frame_queue_t_point.get()

            T_info = T_info_and_net_strap_info["T_info"]
            net_strap_info = T_info_and_net_strap_info["net_strap_info"]

            info_to_send_dict = {
                "ft": ft,
                "left_cut": lcut,
                "right_cut": rcut,
                "box_from": cut_box_from,
                "car_loc": self.current_bot_loc,
                "unix_timestamp": int(time.time() * 1000),
            }

            if (
                net_strap_info is not None
                and len(net_strap_info[0]) == 2
                and len(net_strap_info[1]) == 2
                and (not is_point_on_edge(net_strap_info[0][0]))
                and (not is_point_on_edge(net_strap_info[1][0]))
            ):
                strap_l_up = self.get_t_center_with_base(net_strap_info[0][0], lcut)
                strap_l_down = self.get_t_center_with_base(net_strap_info[0][1], lcut)
                strap_r_up = self.get_t_center_with_base(net_strap_info[1][0], rcut)
                strap_r_down = self.get_t_center_with_base(net_strap_info[1][1], rcut)
                strap_points = [[strap_l_up, strap_l_down], [strap_r_up, strap_r_down]]
                strap_ori_points = [
                    [net_strap_info[0][0], net_strap_info[0][1]],
                    [net_strap_info[1][0], net_strap_info[1][1]],
                ]
                print(f"ft:{ft}, strap_points: {strap_points}")

                net_strap_top_relative_loc = utils.ball_estimation_by_cv2(
                    strap_points[0][0],
                    strap_points[1][0],
                    config.CMAERA_LEFT_PARAMS,
                    config.CMAERA_RIGHT_PARAMS,
                )
                info_to_send_dict["net_strap_top_relative_loc"] = (
                    net_strap_top_relative_loc.tolist()
                )

            else:
                strap_points = [[], []]
                strap_ori_points = [[], []]
                info_to_send_dict["net_strap_top_relative_loc"] = None

            info_to_send_dict["strap_left_points"] = strap_points[0]
            info_to_send_dict["strap_right_points"] = strap_points[1]
            info_to_send_dict["strap_left_ori_infer_points"] = strap_ori_points[0]
            info_to_send_dict["strap_right_ori_infer_points"] = strap_ori_points[1]

            # T_info 数据格式为[[[left_up_x, left_up_y], [left_down_x, left_down_y]], [[right_up_x], right_up_x], [right_down_x, right_down_y]] ]
            # 如果点数是对的，且上点不在边缘
            if (
                len(T_info[0]) == 2
                and len(T_info[1]) == 2
                and (not is_point_on_edge(T_info[0][0]))
                and (not is_point_on_edge(T_info[1][0]))
            ):
                l_up = self.get_t_center_with_base(T_info[0][0], lcut)
                l_down = self.get_t_center_with_base(T_info[0][1], lcut)
                r_up = self.get_t_center_with_base(T_info[1][0], rcut)
                r_down = self.get_t_center_with_base(T_info[1][1], rcut)
                t_points = [[l_up, l_down], [r_up, r_down]]

                # 传入down点是否在边缘，不在边缘的话，使用down点双目测距会更准
                is_down_on_edge_flag = is_point_on_edge(
                    T_info[0][1]
                ) or is_point_on_edge(T_info[1][1])

                loc_by_t = utils.calc_loc_by_T_with_option_test(
                    config.CMAERA_LEFT_PARAMS,
                    config.CMAERA_RIGHT_PARAMS,
                    t_points,
                    strap_points,
                    is_down_on_edge_flag,
                )
                info_to_send_dict["loc_by_t"] = loc_by_t
                info_to_send_dict["left_points"] = [l_up, l_down]
                info_to_send_dict["right_points"] = [r_up, r_down]
                info_to_send_dict["left_ori_infer_points"] = [
                    T_info[0][0],
                    T_info[0][1],
                ]
                info_to_send_dict["right_ori_infer_points"] = [
                    T_info[1][0],
                    T_info[1][1],
                ]

                print(f" find loc by t is {loc_by_t}")
            else:
                info_to_send_dict["loc_by_t"] = None

            if self.t_camera_writer is not None:
                info_to_send_dict["video_time_str"] = self.t_camera_writer.file_name
            try:
                self.img_info_pub.publish(String(data=json.dumps(info_to_send_dict)))
            except Exception as e:
                print(f"publish error: {e}")
                pass
            # 要写入的数据
            if self.t_camera_writer is not None:
                self.t_camera_writer.write(origin_frame, info_to_send_dict)

        print("t detect result handle thread exit")
        if self.t_camera_writer is not None:
            self.t_camera_writer.close()

    def run_person_pos_detect_result_handle(self):
        while True:
            predboxes = self.model_person_pos.result_queue.get()
            if predboxes is None:
                break
            origin_frame = self.video_buf1_person_pos.get()
            lcut, rcut, ft, cut_box_from, loc = self.frame_queue_person_pos.get()
            print(f"detected person cnt : {len(predboxes)}")

            info_to_send_dict = {
                "ft": ft,
                "left_cut": lcut,
                "right_cut": rcut,
                "box_from": cut_box_from,
                "car_loc": self.current_bot_loc,
                "unix_timestamp": int(time.time() * 1000),
            }

            has_one_raised_hand = False
            person_boxes = []
            for i in range(len(predboxes)):
                predbox = predboxes[i]
                l_up = self.get_t_center_with_base([predbox.xmin, predbox.ymin], lcut)
                l_down = self.get_t_center_with_base([predbox.xmax, predbox.ymax], lcut)

                self.last_person_pos_detect_time = ft
                self.last_person_loc = loc

                person_boxes.append([l_up, l_down])
                if self.is_right_hand_raised(predbox):
                    print(f"detected person raised hand !!!")
                    self.last_person_pos_detect_time = ft
                    self.last_person_loc = loc
                    # info_to_send_dict["raised_hand_person_index"] = i
                    has_one_raised_hand = True
                    points = predbox.keypoint.reshape(-1, 3)
                    person_key_points = []
                    for j in range(len(points)):
                        person_key_points.append(
                            self.get_t_center_with_base(
                                [points[j][0], points[j][1]], lcut
                            )
                        )
                    info_to_send_dict["raised_hand_person_key_points"] = (
                        person_key_points
                    )

                    break
            info_to_send_dict["person_boxes"] = person_boxes
            info_to_send_dict["person_raised_hand"] = has_one_raised_hand

            if self.person_camera_writer is not None:
                info_to_send_dict["video_time_str"] = (
                    self.person_camera_writer.file_name
                )
            try:
                self.img_info_pub.publish(String(data=json.dumps(info_to_send_dict)))
            except Exception as e:
                print(f"publish error: {e}")
                pass
            # 要写入的数据
            if self.person_camera_writer is not None:
                self.person_camera_writer.write(origin_frame, info_to_send_dict)

        print("person pos detect result handle thread exit")
        if self.person_camera_writer is not None:
            self.person_camera_writer.close()

    def is_right_hand_raised(self, detect_box: Inference_Pose.DetectBox):
        if detect_box is None:
            return False

        keypoints = detect_box.keypoint.reshape(-1, 3)
        # print(f"keypoints: {keypoints}")
        # 获取右肩、右肘、右腕的关键点
        right_shoulder = keypoints[6]  # 右肩
        right_elbow = keypoints[8]  # 右肘
        right_wrist = keypoints[10]  # 右腕

        # 检查左肩、左肘和左腕的置信度是否足够高
        if right_shoulder[2] < 0.5 or right_elbow[2] < 0.5 or right_wrist[2] < 0.5:
            # print(f"left_shoulder: {left_shoulder[2]}, left_elbow: {left_elbow[2]}, left_wrist: {left_wrist[2]}")
            return False  # 如果有任何一个置信度低于0.5，认为无法确定

        # print(f"left_shoulder: {left_shoulder[1]}, left_elbow: {left_elbow[1]}, left_wrist: {left_wrist[1]}")
        # 判断左腕是否高于左肩和左肘
        if right_wrist[1] < right_shoulder[1] and right_wrist[1] < right_elbow[1]:
            # 如果左腕的y坐标明显小于左肩和左肘的y坐标，则可能是举起了
            return True
        return False

    def deal_video_record(self):
        while True:
            frame = self.record_video_frame_queue.get()
            if self.video_camera_writer is not None:
                if frame is None:
                    self.video_camera_writer.close()
                    break
                self.video_camera_writer.write(frame, None)

    # # 分别传入原始照片，需要截图的真实投影位置， 截图图片的深度，默认真实图片需要进行逆时针旋转
    # def crop_frame(self, frame, pos, depth, rotate = 1):
    #     st = time.perf_counter()
    #     zoom_size = abs(int(self.detect_size * (self.vision_stand_dis / depth)))
    #     size = min(config.CAMERA_WIDTH // 2, zoom_size)  # 至多只按一半来进行截图
    #     sw = int(min(config.CAMERA_WIDTH - size, max(0, pos[0] - size // 2)))
    #     sh = int(min(config.CAMERA_HEIGHT - size, max(0, pos[1] - size // 2)))

    #     # 旋转后截取范围是 sw : sw+size, sh : sh+size
    #     # 旋转前, 左上角点为(H-sh-size, sw)
    #     # update 1225, 取消选择，直接使用sw sh
    #     # c_sw, c_sh = config.CAMERA_HEIGHT - sh - size, sw
    #     # img = frame[c_sh: c_sh+size, c_sw : c_sw+size]
    #     img = frame[sh:sh+size, sw:sw+size]

    #     if img is None or len(img) == 0 or img[0].size == 0:
    #         print(f"crop img error!! origin frame shape: {frame.shape},  size: {size}, c_sw: {sw}, c_sh: {sh}")
    #         return None, None, None, None
    #     # 只运行camera模块，简单统计下，单张图片旋转需要5ms， 缩放也需要5ms。因此坚决不旋转！
    #     img = cv2.resize(img, (self.detect_size, self.detect_size))

    #     # print(f"total time: {time.perf_counter() - st}")
    #     # img = cv2.resize(frame[sh:sh+size, sw:sw+size], (self.detect_size, self.detect_size))

    #     return img, sw, sh, size

    def get_crop_param(self, pos, depth):
        zoom_size = abs(int(self.detect_size * (self.vision_stand_dis / depth)))
        size = min(
            config.CAMERA_WIDTH // 2, max(zoom_size, self.detect_size)
        )  # 至多只按一半来进行截图
        sw = int(min(config.CAMERA_WIDTH - size, max(0, pos[0] - size // 2)))
        sh = int(min(config.CAMERA_HEIGHT - size, max(0, pos[1] - size // 2)))
        # 都变成偶数, 对nv12格式的图片，截取要求是偶数
        if size % 2 == 1:
            size += 1
        if sw % 2 == 1:
            sw -= 1
        if sh % 2 == 1:
            sh -= 1
        return sw, sh, size

    # 分别返回球的中心点（x, y)和球的box框格式为(x, y, w, h)
    # 传入的box格式为(x1, y1, x2, y2)， base表示旋转后截图的(sw, sh) 和世界图片上的截图尺寸
    # 注意推理的图片结果是旋转之前的，而需要返回的是旋转后的， 且补偿了相机扫描后的结果
    # update 1225, 不再需要旋转
    def get_ball_center_with_base(self, box, base, yaw_speed):
        ratio = base[2] / self.detect_size  # 当前识别框中缩放比例系数

        # 推理图片中， 旋转后的x、y、w、h
        # x, y = box[1], self.detect_size - box[2]
        # w, h = box[3] - box[1], box[2] - box[0]
        # 得到推理图片中的 x, y, w, h
        x, y = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]

        # 将推理图片中的坐标 转换成 世界图片坐标
        x = x * ratio + base[0]
        y = y * ratio + base[1]
        w *= ratio
        h *= ratio

        # update 1225, 改成了从上到下行扫描，当前版本误差40个像素，至多0.6ms时延，忽略之
        # x2 = x + w
        # x = self.scan_compensate_x(x, yaw_speed)
        # x2 = self.scan_compensate_x(x2, yaw_speed)
        # w = x2 - x

        return [int(x + 0.5 * w), int(y + 0.5 * h)], [int(x), int(y), int(w), int(h)]

    def get_t_center_with_base(self, point, base):
        ratio = base[2] / self.detect_size  # 当前识别框中缩放比例系数
        x = point[0] * ratio + base[0]
        y = point[1] * ratio + base[1]

        return [int(x), int(y)]

    # def add_ball_dectect_frames(self, left_frame, right_frame, ft):
    #     # 进行推理，如果等待处理过多直接扔掉避免延迟
    #     if self.frame_queue_ball.qsize() <= 3:
    #         self.detected_frames_cnt += 1
    #         if self.detected_frames_cnt % 20 == 1:
    #             print(f"add {self.detected_frames_cnt} frame to predict, frame_time: {ft:.3f} ")

    #         # 如果frame detect和predict，都间隔太长，就启动搜索
    #         need_search = ft - self.last_frame_detect_time > 0.4 and ft - self.last_estimate_ball_update_time > 0.3
    #         if need_search:
    #             if ft - self.last_fire_time < self.fire_ball_search_duration:
    #                 self.fire_serach_ball_loc_index = (self.fire_serach_ball_loc_index + 1) % len(self.fire_search_ball_loc_list)
    #                 self.cut_box_from = "fire search box"
    #                 loc = self.fire_search_ball_loc_list[self.fire_serach_ball_loc_index]
    #             else:
    #                 self.search_ball_loc_index = (self.search_ball_loc_index + 1) % len(self.search_ball_loc_list)
    #                 self.cut_box_from = "return search box"
    #                 loc = self.search_ball_loc_list[self.search_ball_loc_index]
    #         else:
    #             loc = self.last_ball_loc

    #         left_pos = utils.get_project_with_full_camera_1(loc, config.CMAERA_LEFT_PARAMS)
    #         right_pos = utils.get_project_with_full_camera_1(loc, config.CMAERA_RIGHT_PARAMS)

    #         left_img, lw, lh, lsize = self.crop_frame(left_frame, left_pos, loc[2])
    #         # print(f"lw lh lsize: {lw}, {lh}, {lsize}")
    #         right_img, rw, rh, rsize = self.crop_frame(right_frame, right_pos, loc[2])

    #         if left_img is not None and right_img is not None:
    #             # 加入识别队列
    #             if self.cut_box_from == "return_ball_predict":
    #                 self.frame_queue_ball.put([[lw, lh, lsize], [rw, rh, rsize], ft, str(self.cut_box_from), self.current_bot_yaw_speed, [loc.copy(), left_pos, right_pos]])
    #             else:
    #                 self.frame_queue_ball.put([[lw, lh, lsize], [rw, rh, rsize], ft, str(self.cut_box_from), self.current_bot_yaw_speed, None])

    #             self.model.predict(left_img, right_img)
    #             self.video_buf1_ball.put((left_frame, right_frame))
    #         return True
    #     else:
    #         # self.model_missed_frame_cnt += 1
    #         # if self.model_missed_frame_cnt % 1 == 0:
    #         #     print("dump frame due to model queue is full, result queue size: ", self.model.result_que.qsize())
    #         return False

    def get_ball_detect_cut_param(self, ft):
        need_search = (
            ft - self.last_frame_detect_time > 0.4
            and ft - self.last_estimate_ball_update_time > 0.3
        )
        # print(f"need_search: {need_search}, last_frame_detect_time: {self.last_frame_detect_time}, last_estimate_ball_update_time: {self.last_estimate_ball_update_time}")
        if need_search:
            if ft - self.last_fire_time < self.fire_ball_search_duration:
                # self.fire_serach_ball_loc_index = (self.fire_serach_ball_loc_index + 1) % len(self.fire_search_ball_loc_list)
                self.cut_box_from = "fire search box"
                loc = self.fire_search_ball_loc_list[self.fire_serach_ball_loc_index]
            else:
                self.cut_box_from = "return search box"
                loc = self.search_ball_loc_list[self.search_ball_loc_index]
        else:
            loc = self.last_ball_loc

        left_pos = utils.get_project_with_full_camera_1(loc, config.CMAERA_LEFT_PARAMS)
        right_pos = utils.get_project_with_full_camera_1(
            loc, config.CMAERA_RIGHT_PARAMS
        )

        lw, lh, lsize = self.get_crop_param(left_pos, loc[2])
        rw, rh, rsize = self.get_crop_param(right_pos, loc[2])
        dict = {}
        dict["left_cut"] = [lw, lh, lsize]
        dict["right_cut"] = [rw, rh, rsize]
        dict["ft"] = ft
        dict["cut_box_from"] = str(self.cut_box_from)
        dict["current_bot_yaw_speed"] = self.current_bot_yaw_speed
        # 兼容新旧source值：return_curve_prediction 或 return_ball_predict
        if (
            self.cut_box_from == "return_curve_prediction"
            or self.cut_box_from == "return_ball_predict"
        ):
            dict["loc_and_pos"] = [loc.copy(), left_pos, right_pos]
        else:
            dict["loc_and_pos"] = None
        return dict

    # # update 0930, 对于左右摄像头图片，裁剪后进行球的识别或者是T点识别
    # def add_t_detect_frames(self, left_frame, right_frame, ct):
    #     # self.current_camera_id = 1
    #     # 0910 add lock 只允许有一个线程执行
    #     if BotMotionConfig.T_DOWN_POINT["z"] - self.current_bot_loc[2] <= 1:
    #         print(f"ERROR: T point is too close. Current bot loc is {self.current_bot_loc}, T point is {BotMotionConfig.T_DOWN_POINT}")
    #         return

    #     #  先根据T点位置和小车位置，计算截图的pos
    #     middle_t_x = (BotMotionConfig.T_DOWN_POINT["x"] + BotMotionConfig.T_UP_POINT["x"]) * 0.5
    #     middle_t_z = (BotMotionConfig.T_DOWN_POINT["z"] + BotMotionConfig.T_UP_POINT["z"]) * 0.5
    #     middle_t_point = [middle_t_x, 0, middle_t_z]
    #     left_pos = utils.get_project_with_full_camera_1(middle_t_point,  config.CMAERA_LEFT_PARAMS, self.current_bot_loc)
    #     right_pos = utils.get_project_with_full_camera_1(middle_t_point, config.CMAERA_RIGHT_PARAMS, self.current_bot_loc)

    #     left_img, lw, lh, lsize = self.crop_frame(left_frame, left_pos, BotMotionConfig.T_DOWN_POINT["z"] - self.current_bot_loc[2])
    #     right_img, rw, rh, rsize = self.crop_frame(right_frame, right_pos, BotMotionConfig.T_DOWN_POINT["z"] - self.current_bot_loc[2])

    #     # 加入T点的识别队列
    #     if left_img is not None and right_img is not None:
    #         # 对left_img 和 right_img 进行旋转，seg推理中，只输出640图片旋转后的T点位置
    #         # 1225 同理不再需要旋转
    #         # left_img = cv2.rotate(left_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #         # right_img = cv2.rotate(right_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    #         # 加入识别队列
    #         self.frame_queue_t_point.put([[lw, lh, lsize], [rw, rh, rsize], ct, "T BOX", self.current_bot_yaw_speed])
    #         self.model_seg.add_img(left_img, right_img)
    #         self.video_buf1_t_point.put((left_frame, right_frame))

    #         return True

    def get_t_detect_cut_param(self, ct):
        # # FIXME 把 t search 删了
        if self.need_search_t:
            loc = self.search_t_list[self.search_t_index % len(self.search_t_list)]
            self.search_t_index += 1

            left_pos = utils.get_project_with_full_camera_1(
                loc, config.CMAERA_LEFT_PARAMS
            )
            right_pos = utils.get_project_with_full_camera_1(
                loc, config.CMAERA_RIGHT_PARAMS
            )
            lw, lh, lsize = self.get_crop_param(left_pos, loc[2])
            rw, rh, rsize = self.get_crop_param(right_pos, loc[2])
            print(
                f"search t detect cut param left:{lw}, {lh}, {lsize}, right: {rw}, {rh}, {rsize}"
            )

            dict = {}
            dict["left_cut"] = [lw, lh, lsize]
            dict["right_cut"] = [rw, rh, rsize]
            dict["ft"] = ct
            dict["cut_box_from"] = "SEARCH T BOX"
            dict["current_bot_yaw_speed"] = self.current_bot_yaw_speed
            return dict

        else:
            middle_t_x = (
                BotMotionConfig.T_DOWN_POINT["x"] + BotMotionConfig.T_UP_POINT["x"]
            ) * 0.5
            middle_t_z = (
                BotMotionConfig.T_DOWN_POINT["z"] + BotMotionConfig.T_UP_POINT["z"]
            ) * 0.5
            middle_t_point = [middle_t_x, 0, middle_t_z]
            left_pos = utils.get_project_with_full_camera_1(
                middle_t_point, config.CMAERA_LEFT_PARAMS, self.current_bot_loc
            )
            right_pos = utils.get_project_with_full_camera_1(
                middle_t_point, config.CMAERA_RIGHT_PARAMS, self.current_bot_loc
            )

            lw, lh, lsize = self.get_crop_param(
                left_pos, BotMotionConfig.T_DOWN_POINT["z"] - self.current_bot_loc[2]
            )
            rw, rh, rsize = self.get_crop_param(
                right_pos, BotMotionConfig.T_DOWN_POINT["z"] - self.current_bot_loc[2]
            )

            dict = {}
            dict["left_cut"] = [lw, lh, lsize]
            dict["right_cut"] = [rw, rh, rsize]
            dict["ft"] = ct
            dict["cut_box_from"] = "T BOX"
            dict["current_bot_yaw_speed"] = self.current_bot_yaw_speed
            return dict

    # # TODO: 添加注释
    # def add_person_detect_frames(self, left_frame, right_frame, ct):
    #     if self.frame_queue_person_pos.qsize() > 3:
    #         print("dump frame due to model queue is full, result queue size: ", self.model_person_pos.result_queue.qsize())
    #         return False

    #     need_search = ct - self.last_person_pos_detect_time > 0.3
    #     if need_search:
    #         self.search_person_loc_index = (self.search_person_loc_index + 1) % len(self.search_person_loc_list)
    #         loc = self.search_person_loc_list[self.search_person_loc_index]
    #     else:
    #         loc = self.last_person_loc

    #     left_pos = utils.get_project_with_full_camera_1(loc, config.CMAERA_LEFT_PARAMS)

    #     left_img, lw, lh, lsize = self.crop_frame(left_frame, left_pos, loc[2])

    #     if left_img is not None:
    #         self.frame_queue_person_pos.put([[lw, lh, lsize], [lw, lh, lsize], ct, "person pos",loc])
    #         self.model_person_pos.predict(left_img)
    #         self.video_buf1_person_pos.put((left_frame, right_frame))

    #         return True

    def get_person_detect_cut_param(self, ct):
        need_search = ct - self.last_person_pos_detect_time > 0.3
        if need_search:
            self.search_person_loc_index = (self.search_person_loc_index + 1) % len(
                self.search_person_loc_list
            )
            loc = self.search_person_loc_list[self.search_person_loc_index]
        else:
            loc = self.last_person_loc

        left_pos = utils.get_project_with_full_camera_1(loc, config.CMAERA_LEFT_PARAMS)
        lw, lh, lsize = self.get_crop_param(left_pos, loc[2])

        dict = {}
        dict["left_cut"] = [lw, lh, lsize]
        dict["right_cut"] = [lw, lh, lsize]
        dict["ft"] = ct
        dict["cut_box_from"] = "person pos"
        dict["loc"] = loc
        return dict

    # def read_front_frames(self, camera_id = 0):
    #     # 使用c获得驱动层时间，时间测试非常准确，误差在1ms级别。
    #     # 用0 表示左摄像头，用1 表示右摄像头
    #     if camera_id == 0: # 左摄像头
    #         cap = VideoCaptureV4L2c.VideoCapture(f"/dev/video{config.CAMERA_ORDER[0]}")
    #         camera_queue = self.left_camera_queue
    #     else:
    #         cap = VideoCaptureV4L2c.VideoCapture(f"/dev/video{config.CAMERA_ORDER[1]}")
    #         camera_queue = self.right_camera_queue

    #     if cap.open() and cap.start_capture():
    #         print(f"Open Camera {camera_id} successfully")

    #     # take camera frames and put in queue. sync dual frames in another thread.
    #     # while not self.stop_event.is_set():
    #     while camera_queue.qsize() < 20:
    #         frame, ct = cap.read()
    #         if (frame is None) or (ct is None):
    #             continue
    #         if frame.shape[1] != config.CAMERA_WIDTH or frame.shape[0] != config.CAMERA_HEIGHT:
    #             print(f"frame shpae is not right: {frame.shape}")
    #             continue

    #         # 1209手动摇车测得的对齐时间，x因为yaw的偏差，在0.1m级别
    #         ct *= 1e-6
    #         ct += 0.020
    #         camera_queue.put((frame, ct))

    def read_frame_cut(self, camera_id=0):
        if camera_id == 0:  # 左摄像头
            cap = VideoCaptureV4L2c.VideoCapture(f"/dev/video{config.CAMERA_ORDER[0]}")
            camera_queue = self.left_camera_crop_queue
        else:
            cap = VideoCaptureV4L2c.VideoCapture(f"/dev/video{config.CAMERA_ORDER[1]}")
            camera_queue = self.right_camera_crop_queue

        if cap.open() and cap.start_capture():
            print(f"Open Camera {camera_id} successfully")

        frame_fps = BotMotionConfig.CAMERA_FPS
        frame_interval = 1.0 / frame_fps
        last_read_time = time.perf_counter() - frame_interval

        # rk3588 一共有三个rga核心，0:rga3 2:rga3 4:rga2, rga2不支持处理4k, 左右摄像头各用一个rga3
        if camera_id == 0:
            rgaCoreNum = 0
        else:
            rgaCoreNum = 2

        # read_frame_cnt = 0

        while not self.stop_event.is_set():
            if camera_id == 0:
                self.left_camera_can_read.acquire()
            else:
                self.right_camera_can_read.acquire()

            expect_time = last_read_time + frame_interval
            # print(f"expect_time: {expect_time}")

            # ballCropRegionArray = None
            ball_cut_param = self.get_ball_detect_cut_param(expect_time)
            if camera_id == 0:
                cut_param = ball_cut_param["left_cut"]
            else:
                cut_param = ball_cut_param["right_cut"]
            ballCropRegionArray = [
                cut_param[0],
                cut_param[1],
                cut_param[2],
                cut_param[2],
            ]

            tCropRegionArray = None
            t_cut_param = None
            if self.is_predict_t_flag:
                t_cut_param = self.get_t_detect_cut_param(expect_time)
                if camera_id == 0:
                    cut_param = t_cut_param["left_cut"]
                else:
                    cut_param = t_cut_param["right_cut"]
                # print(f"t cut param: {cut_param}")
                tCropRegionArray = [
                    cut_param[0],
                    cut_param[1],
                    cut_param[2],
                    cut_param[2],
                ]

            poseCropRegionArray = None
            person_cut_param = None
            if self.enable_person_pos_detect:
                person_cut_param = self.get_person_detect_cut_param(expect_time)
                if camera_id == 0:
                    cut_param = person_cut_param["left_cut"]
                else:
                    cut_param = person_cut_param["right_cut"]
                poseCropRegionArray = [
                    cut_param[0],
                    cut_param[1],
                    cut_param[2],
                    cut_param[2],
                ]

            videoRecordCropRegionArray = None
            if self.video_camera_writer is not None and self.need_record_video:
                videoRecordCropRegionArray = [1080, 800, 1680, 1120]
                # videoRecordCropRegionArray = None

            ball_image, t_image, pose_image, video_record_image, ct = (
                cap.read_crop_multi(
                    rgaCoreNum=rgaCoreNum,
                    ballCropRegionArray=ballCropRegionArray,
                    tCropRegionArray=tCropRegionArray,
                    poseCropRegionArray=poseCropRegionArray,
                    videoRecordCropRegionArray=videoRecordCropRegionArray,
                )
            )
            if ct is None:
                continue
            if ball_image.shape[1] != 640 or ball_image.shape[0] != 640:
                print(f"ball frame shpae is not right: {ball_image.shape}")
                continue

            if t_image is not None:
                if t_image.shape[1] != 640 or t_image.shape[0] != 640:
                    print(f"t frame shpae is not right: {t_image.shape}")
                    continue

            if pose_image is not None:
                if pose_image.shape[1] != 640 or pose_image.shape[0] != 640:
                    print(f"pose frame shpae is not right: {pose_image.shape}")
                    continue

            # print(f"video_record_image shape: {video_record_image.shape}")

            # 1209手动摇车测得的对齐时间，x因为yaw的偏差，在0.1m级别
            ct *= 1e-6
            ct += (
                BotMotionConfig.CAMERA_IMU_LANTENCY
            )  # 0422 测试获得mipi相机相对于imu偏移的最新标准！

            # 以下是0422在59机器上的测试记录：
            # ct += 0.040 # 0422在59机器上的测试结果，该值下，机器向左旋转， 固定位置网球先会在x方向变大. x变大0.5
            # ct += 0.140 # 理论上该值，会使机器左旋时，x减小. 实际上则是出现了不规则抖动。why？
            # ct += 0.3 # 这在理论上一定是x减小了吧？？ 依然是震荡。。。？？？ 因为直接使用了最新的imu时间，这样不对
            # ct += -0.20  # 0422测试， 此状态下，机器向左旋转，固定网球位置在x方向一直变大，imu明确滞后，照片时间应该往后（ct增加）
            # ct += 0.3 # 再次尝试， 总算是对了，没有震荡，要在启动前先摆正机器！！ 也就是ct增加，x往负方向。
            # ct += 0.02  # x在正负之间有震荡，间隔0.1。 说明区间值是对的。但是照片对应的imu依然不够准确。imu值的颗粒度太粗。 =》 找到了原因，静止模式下睡眠过久！
            # ct += 0.02 # 重新测试： 左移x变大。但差值已经到了0.04
            # ct += 0.025 # 左移x稳健变负数 没有反弹，基本是正确了。
            # ct += 0.030 # 机器左旋，会使得x下降后再上升。

            # 读取到图片后，进行截图，然后进行推理
            # read_frame_cnt += 1
            # if read_frame_cnt % 30 == 0:
            #     # print(f"read frame at {ct}, camera_id: {camera_id}")
            #     pass

            camera_queue.put(
                (
                    ball_image,
                    ball_cut_param,
                    t_image,
                    t_cut_param,
                    pose_image,
                    person_cut_param,
                    video_record_image,
                    None,
                    ct,
                )
            )
            last_read_time = ct

        print(f"camera {camera_id} read frame thread exit")
        cap.release()
        camera_queue.put(None)

    # 更新预估网球和小车位置，用于截图位置
    def deal_estimate_msg(self, ft):
        while self.estimate_msg_queue.qsize() > 0:
            data = self.estimate_msg_queue.get()

            # 预估的位置应该在接近当前的拍摄时间，如果超前了就退出。因为queue中数据永远是递增的。

            # 首先更新车的位置和发射时间
            self.current_bot_loc = data["bot_loc"]
            self.last_fire_time = data["fire_time"]

            # 如果有预测球位置（相对于小车坐标系），则更新预测球
            # 优先使用新的键名 pred_ball_loc_rela2car，兼容旧的 ball_loc
            ball_loc_key = None
            if (
                "pred_ball_loc_rela2car" in data
                and data["pred_ball_loc_rela2car"] is not None
            ):
                ball_loc_key = "pred_ball_loc_rela2car"
            elif "ball_loc" in data and data["ball_loc"] is not None:
                ball_loc_key = "ball_loc"  # 兼容旧数据

            # FIXME 键值对向后兼容
            if ball_loc_key:
                self.last_ball_loc = data[ball_loc_key]
                self.last_estimate_ball_update_time = data.get(
                    "ts_pred", data.get("ts")
                )
                self.cut_box_from = data.get("source", "unknown")

            if data.get("ts_pred", data.get("ts")) > ft:
                break

    # # take camera frames from queue, sync them and send them to detect.
    # def deal_two_frames(self):
    #     left_img, right_img = None, None
    #     while not self.stop_event.is_set():
    #         if left_img is None:
    #             left_img = self.left_camera_queue.get()
    #             self.camera_single_read_frames_cnt += 1

    #         if right_img is None:
    #             right_img = self.right_camera_queue.get()
    #             self.camera_single_read_frames_cnt += 1

    #         if abs(left_img[1]-right_img[1]) > 0.010:
    #             print(f"shif one img, {left_img[1]},  {right_img[1]}")
    #             if left_img[1] < right_img[1]:
    #                 left_img = None
    #             else:
    #                 right_img = None
    #             continue
    #         # print(f"shift count: {shift_count},  sync count: {sync_count}")
    #         # now the two imgs are almost at same time, send them to detect
    #         ft = (left_img[1] + right_img[1]) * 0.5
    #         self.camera_sync_read_frames_cnt += 1
    #         if self.camera_sync_read_frames_cnt % 100 == 0:
    #             # 用于log 相机读取帧率和检测帧率，查看掉帧情况
    #             img_info = {"camera_read_frames_cnt": self.camera_sync_read_frames_cnt, "detect_frames_cnt": self.detected_frames_cnt, "single_frames_cnt": self.camera_single_read_frames_cnt, "ft": ft}
    #             self.img_info_pub.publish(String(data=json.dumps(img_info)))

    #             # print(f"read {self.camera_read_frames_cnt} frame from camera, frame_time: {ft:.3f} ")
    #         self.deal_estimate_msg(ft)
    #         if self.is_predict_t_flag and ft - self.last_t_predict_time > 0.9:
    #             self.last_t_predict_time = ft
    #             self.add_t_detect_frames(left_img[0], right_img[0], ft)

    #         if self.enable_person_pos_detect and ft - self.last_add_person_pos_detect_time > 0.12:
    #             self.last_add_person_pos_detect_time = ft
    #             self.add_person_detect_frames(left_img[0], right_img[0], ft)

    #         self.add_ball_dectect_frames(left_img[0], right_img[0], ft)

    #         # print(f"put img latency:{time.perf_counter() - ft}")
    #         left_img = None
    #         right_img = None

    def deal_two_frames_cut(self):
        from hit.utils.utils import set_thread_name

        try:
            set_thread_name("Cam_process")
        except:
            pass

        deal_frams_cnt = 0
        left_crop, right_crop = None, None

        while True:
            if left_crop is None:
                left_crop = self.left_camera_crop_queue.get()
                if left_crop is None:
                    print("left_crop is None")
                    break
                # print(f"get left crop at {left_crop[-1]}")
                self.camera_single_read_frames_cnt += 1

            if right_crop is None:
                right_crop = self.right_camera_crop_queue.get()
                if right_crop is None:
                    print("right_crop is None")
                    break
                # print(f"get right crop at {right_crop[-1]}")
                self.camera_single_read_frames_cnt += 1

            # if abs(left_crop[-1] - right_crop[-1]) > 0.010:
            if abs(left_crop[-1] - right_crop[-1]) > 0.050:
                print(f"shif one img, {left_crop[-1]},  {right_crop[-1]}")
                if left_crop[-1] < right_crop[-1]:
                    left_crop = None
                    self.left_camera_can_read.release()
                else:
                    right_crop = None
                    self.right_camera_can_read.release()
                continue

            ft = (left_crop[-1] + right_crop[-1]) * 0.5
            self.read_frame_latency += time.perf_counter() - ft
            self.camera_sync_read_frames_cnt += 1
            if self.camera_sync_read_frames_cnt % 100 == 0:
                # 用于log 相机读取帧率和检测帧率，查看掉帧情况
                img_info = {
                    "camera_read_frames_cnt": self.camera_sync_read_frames_cnt,
                    "detect_frames_cnt": self.detected_frames_cnt,
                    "single_frames_cnt": self.camera_single_read_frames_cnt,
                    "read_frame_latency": self.read_frame_latency / 100,
                    "ball_inference_latency": self.ball_inference_latency
                    / max(self.ball_inference_cnt, 1),
                    "ft": ft,
                    "unix_timestamp": int(time.time() * 1000),
                }
                try:
                    self.img_analysis_pub.publish(String(data=json.dumps(img_info)))
                except Exception as e:
                    print(f"img_info_pub publish error: {e}")
                    pass
                self.read_frame_latency = 0.0
                self.ball_inference_latency = 0.0
                self.ball_inference_cnt = 0

                # print(f"read {self.camera_read_frames_cnt} frame from camera, frame_time: {ft:.3f} ")
            self.deal_estimate_msg(ft + 1.0 / BotMotionConfig.CAMERA_FPS)

            if left_crop[2] is not None and right_crop[2] is not None:
                self.add_t_detect_crop(left_crop, right_crop, ft)
                self.last_t_predict_time = ft
                self.is_predict_t_flag = False

            if ft - self.last_t_predict_time > 0.9:
                # FIXME 禁用t点检测
                # self.is_predict_t_flag = True
                self.is_predict_t_flag = False
                # print(f"start to predict T point at {ft:.3f}")

            # if self.is_predict_t_flag and ft - self.last_t_predict_time > 0.9:
            #     self.last_t_predict_time = ft
            #     self.add_t_detect_crop(left_crop, right_crop, ft)

            if (
                self.enable_person_pos_detect
                and ft - self.last_add_person_pos_detect_time > 0.12
            ):
                self.last_add_person_pos_detect_time = ft
                self.add_person_detect_crop(left_crop, right_crop, ft)

            if left_crop[6] is not None and right_crop[6] is not None:
                self.record_video_frame_queue.put((left_crop[6], right_crop[6]))
                # print(f"record video at {ft:.3f}")
                self.last_record_video_time = ft
                self.need_record_video = False

            if deal_frams_cnt % 3 == 0:
                self.need_record_video = True

            self.add_ball_detect_crop(left_crop, right_crop, ft)

            self.fire_serach_ball_loc_index = (
                self.fire_serach_ball_loc_index + 1
            ) % len(self.fire_search_ball_loc_list)
            self.search_ball_loc_index = (self.search_ball_loc_index + 1) % len(
                self.search_ball_loc_list
            )

            left_crop = None
            right_crop = None
            self.left_camera_can_read.release()
            self.right_camera_can_read.release()
            deal_frams_cnt += 1

        print(f"deal_two_frames_cut exit")
        self.add_detect_result_queue_none()

    def add_t_detect_crop(self, left_crop, right_crop, ft):
        left_t_img = left_crop[2].copy()
        left_t_cut_param = left_crop[3]
        right_t_img = right_crop[2].copy()
        right_t_cut_param = right_crop[3]
        self.model_seg.add_img(left_t_img, right_t_img)
        self.frame_queue_t_point.put(
            [
                left_t_cut_param["left_cut"],
                right_t_cut_param["right_cut"],
                ft,
                "T BOX",
                (
                    left_t_cut_param["current_bot_yaw_speed"]
                    + right_t_cut_param["current_bot_yaw_speed"]
                )
                // 2,
            ]
        )
        self.video_buf1_t_point.put((left_t_img, right_t_img))

    def add_person_detect_crop(self, left_crop, right_crop, ft):
        left_pos_img = left_crop[4].copy()
        left_pos_cut_param = left_crop[5]
        right_pos_img = right_crop[4].copy()
        right_pos_cut_param = right_crop[5]
        self.frame_queue_person_pos.put(
            [
                left_pos_cut_param["left_cut"],
                left_pos_cut_param["left_cut"],
                ft,
                "person pos",
                left_pos_cut_param["loc"],
            ]
        )
        self.model_person_pos.predict(left_pos_img)
        self.video_buf1_person_pos.put((left_pos_img, right_pos_img))

    def add_ball_detect_crop(self, left_crop, right_crop, ft):
        if self.frame_queue_ball.qsize() < 2:
            self.detected_frames_cnt += 1
            if self.detected_frames_cnt % 3000 == 0:
                print(
                    f"add {self.detected_frames_cnt} frame to predict, frame_time: {ft:.3f} , mipi"
                )
            left_ball_img = left_crop[0].copy()
            left_ball_cut_param = left_crop[1]
            right_ball_img = right_crop[0].copy()
            right_ball_cut_param = right_crop[1]
            self.model.predict(left_ball_img, right_ball_img)
            # msg = [
            #         left_ball_cut_param["left_cut"],
            #         right_ball_cut_param["right_cut"],
            #         ft,
            #         left_ball_cut_param["cut_box_from"],
            #         (
            #             left_ball_cut_param["current_bot_yaw_speed"]
            #             + right_ball_cut_param["current_bot_yaw_speed"]
            #         )
            #         // 2,
            #         left_ball_cut_param["loc_and_pos"],
            #     ]
            # print(f"put ball frame to model queue: {msg}")
            self.frame_queue_ball.put(
                [
                    left_ball_cut_param["left_cut"],
                    right_ball_cut_param["right_cut"],
                    ft,
                    left_ball_cut_param["cut_box_from"],
                    (
                        left_ball_cut_param["current_bot_yaw_speed"]
                        + right_ball_cut_param["current_bot_yaw_speed"]
                    )
                    // 2,
                    left_ball_cut_param["loc_and_pos"],
                ]
            )
            self.video_buf1_ball.put((left_ball_img, right_ball_img))
        else:
            # print(
            #     f"dump frame due to model queue is full, frame_queue_ball is {self.frame_queue_ball.qsize()}"
            # )
            self.model_missed_frame_cnt += 1
            if self.model_missed_frame_cnt % 10000 == 0:
                self.model_missed_frame_cnt = 0
                print(
                    "dump frame due to model queue is full, result queue size: ",
                    self.model.result_que.qsize(),
                )
            return False

    def add_detect_result_queue_none(self):
        self.model.result_que.put(None)
        self.model_seg.result_pos.put(None)
        self.model_person_pos.result_queue.put(None)
        self.record_video_frame_queue.put(None)

    def stop_camera_processor(self):
        self.stop_event.set()
        # self.read_frame_thread_0.join()
        # self.read_frame_thread_1.join()
        # self.deal_frame_thread.join()
        # self.ball_detect_result_handle_thread.join()
        # self.t_detect_result_handle_thread.join()
        # self.pose_detect_result_handle_thread.join()


def main(args=None):
    np.set_printoptions(suppress=True, precision=3)

    rclpy.init(args=args)
    processer = CameraProcesser()
    try:
        rclpy.spin(processer)
    except (KeyboardInterrupt, Exception) as e:
        print(f"CameraProcesser error: {e}")
        print("start to quit")
        processer.stop_event.set()
        processer.stop_camera_processor()
        print("quit done")
    finally:
        processer.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
