import cv2
import threading
from collections import deque
from rknnlite.api import RKNNLite
from .rknn_executor import RKNN_model_container
import numpy as np
import time
import logging
import math

# from . import utils
from queue import Queue
import os
import subprocess
# from . import config

from ament_index_python.packages import get_package_share_directory


# 修正 logging 模块的 _checkLevel 函数，使其能够识别字符串形式的等级
original_checkLevel = logging._checkLevel


def patched_checkLevel(level):
    if isinstance(level, str):
        if level.upper() == "WARNING":
            return logging.WARNING
        # 这里可以添加更多字符串到常量的映射
    return original_checkLevel(level)


logging._checkLevel = patched_checkLevel

# 现在可以安全地导入其他模块
import torch


class Inference_seg:
    OBJ_THRESH = 0.25
    IMG_SIZE = (640, 640)
    OUT_seg = 0
    IN_seg = 1
    CAINIAO_seg = 2
    POINT = ()

    id = OUT_seg

    def __init__(self, id) -> None:
        self.id = id

        package_share_dir = get_package_share_directory("hit")
        model_path = os.path.join(package_share_dir, "models", "tennis_20241227.rknn")

        if id == self.OUT_seg:
            # model_path = r'/home/cat/Desktop/dualView/catkin_ws/model/middle_line_and_net_strap_20250314.rknn' # 室外场地
            model_path = os.path.join(
                package_share_dir, "models", "middle_line_and_net_strap_20250426.rknn"
            )

        if id == self.IN_seg:
            # model_path = r'/home/cat/Desktop/dualView/catkin_ws/model/t_indoor_20241217_2.rknn' # 室内场地
            model_path = os.path.join(
                package_share_dir, "models", "t_indoor_20241217_2.rknn"
            )
        if id == self.CAINIAO_seg:
            model_path = os.path.join(
                package_share_dir,
                "models",
                "middle_line_and_net_strap_cainiao_20250219_2.rknn",
            )
            # model_path = r'/home/cat/Desktop/dualView/catkin_ws/model/middle_line_and_net_strap_cainiao_20250219_2.rknn' # 菜鸟场地

        self.model0 = self.setup_model(model_path, core_mask=RKNNLite.NPU_CORE_2)
        self.model1 = self.setup_model(model_path, core_mask=RKNNLite.NPU_CORE_2)
        # core_2拿来做T点识别，换成用单线程会不会好点？

        self.pre_stop_event = threading.Event()
        self.inference_stop_event0 = threading.Event()
        self.inference_stop_event1 = threading.Event()
        self.post_stop_event = threading.Event()
        self.all_inputs0 = Queue()
        self.all_inputs1 = Queue()
        self.pre_inputs0 = Queue()
        self.pre_inputs1 = Queue()
        # self.left_points = Queue()
        # self.right_points = Queue()
        # self.ct_queue     = Queue()
        self.all_outputs0 = Queue()
        self.all_outputs1 = Queue()
        self.result_deque = Queue()
        self.result_pos = Queue()
        self.post_time = 0

        self.nms_time = 0
        self.wait_count = 0

        self.semaphore_pre = threading.Semaphore(0)  # 初始信号量为0
        self.semaphore_infer0 = threading.Semaphore(0)
        self.semaphore_infer1 = threading.Semaphore(0)
        self.semaphore_post0 = threading.Semaphore(0)
        self.semaphore_post1 = threading.Semaphore(0)
        self.semaphore_result = threading.Semaphore(0)

    def setup_model(self, model_path, core_mask):
        model = RKNN_model_container(model_path, core_mask)
        print("Model-{}, starting val".format(model_path))
        return model

    def custom_softmax(self, x, axis=None):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    def dfl_sigle(self, position, idx3=0, idx4=0):
        x = torch.tensor(position, device="cpu")
        n, c, h, w = x.shape
        p_num = 4
        mc = c // p_num
        y = x.reshape(n, p_num, mc, h, w)
        y_np = y.numpy()
        y_np = self.custom_softmax(y_np, axis=2)  # 实现softmax的效果
        y = torch.from_numpy(y_np).float()
        acc_metrix = torch.tensor(range(mc)).float().reshape(1, 1, mc, 1, 1)
        y = (y * acc_metrix).sum(2)

        return y.numpy()  # Return as NumPy array

    def box_process(self, position, h, w, idx3, idx4):
        grid_h, grid_w = h, w
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array(
            [Inference_seg.IMG_SIZE[1] // grid_h, Inference_seg.IMG_SIZE[0] // grid_w]
        ).reshape(1, 2, 1, 1)
        # print(position.shape)

        position = self.dfl_sigle(position, idx3, idx4)

        # print(position.shape)
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

        return xyxy

    """
    output0 get boxes 框的值即可
    resized_mask 计算得到的掩码区域,需上采样到图片原来的大小
    max_idx 可以不要
    裁剪box的原因是为了过滤掩码识别出来的错误的部分 并选择掩码均值最大的部分
    """

    def process_single_box_col(self, boxes, resized_mask):
        # 获取框坐标
        # x, y, w, h = map(int, [output0[0][0][max_idx], output0[0][1][max_idx], output0[0][2][max_idx], output0[0][3][max_idx]])
        x1 = max(0, int(boxes[0]))
        x2 = max(0, int(boxes[2]))
        y1 = max(0, int(boxes[1]))
        y2 = max(0, int(boxes[3]))

        # 裁剪掩码区域
        cropped_mask = resized_mask[y1:y2, x1:x2]
        # print(f"Cropping coordinates: y1={y1}, y2={y2}, x1={x1}, x2={x2}")
        # cv2.imwrite("mask.jpg",resized_mask)

        window_size = 20
        half_window = window_size // 2
        row0 = cropped_mask[0]
        max_mean_value0 = -1
        max_mean_index0 = -1
        mean_value0 = 0
        for i in range(half_window, len(row0) - half_window):
            mean_value0 = np.mean(
                row0[i - half_window : i + half_window + 1]
            )  # 计算5个点的均值
            if mean_value0 > max_mean_value0:
                max_mean_value0 = mean_value0
                max_mean_index0 = i

        row = cropped_mask[-1]
        half_window = window_size // 2
        max_mean_value = -1
        max_mean_index = -1
        for i in range(half_window, len(row) - half_window):
            mean_value = np.mean(
                row[i - half_window : i + half_window + 1]
            )  # 计算`window_size`个点的均值
            if mean_value > max_mean_value:
                max_mean_value = mean_value
                max_mean_index = i
        # 返回结果
        return [(x1 + max_mean_index0, y1), (x1 + max_mean_index, y2)]

    def process_single_box_col_polyfit(self, boxes, resized_mask):
        x1 = max(0, int(boxes[0]))
        y1 = max(0, int(boxes[1]))
        x2 = max(0, int(boxes[2]))
        y2 = max(0, int(boxes[3]))

        cropped_mask = resized_mask[y1:y2, x1:x2]
        y, x = np.where(cropped_mask > 0.5)
        if len(x) == 0 or len(y) == 0:
            return []
        # print(f"shape:{y.shape}")

        # 将坐标转换到整个图像的坐标系
        x = x + x1  # 恢复到原图像坐标
        y = y + y1  # 恢复到原图像坐标

        # 对这些点进行线性拟合
        coeffs = np.polyfit(
            y, x, 1
        )  # 拟合一次多项式（线性拟合）,将x和y调换，因为T点的线容易垂直
        # slope, intercept = coeffs  # 提取斜率和截距
        # print(f"拟合直线的方程是: x = {slope:.2f}y + {intercept:.2f}")

        point = [
            (round(np.polyval(coeffs, y1)), y1),
            (round(np.polyval(coeffs, y2)), y2),
        ]

        fit_values = np.polyval(coeffs, y)
        mse = np.mean((fit_values - x) ** 2)
        # print(f"point:{point}, mse:{mse}")
        if mse > 200:  # 拟合方差大于200的就不要了
            print(f"mse too large, pass, point:{point}, mse:{mse}")
            # point = []

        return point

    # def post_process_seg(self,input_data):  # 非量化的模型计算方法
    #     output0 = input_data[0]
    #     output1 = input_data[1]

    #     # 处理两个不同的最大值索引
    #     max_idx_4 = torch.max(torch.from_numpy(output0[0][4]), 0).indices.item()
    #     # max_idx_5 = torch.max(torch.from_numpy(output0[0][5]), 0).indices.item()

    #     mask_weights = output0[0, 6:38, max_idx_4]  # 从第6通道到第37通道
    #     final_mask = np.zeros((160, 160))
    #     for i in range(32):  # 因为有32个掩码
    #         final_mask += mask_weights[i] * output1[0, i, :, :]

    #     # 应用阈值以二值化掩码
    #     threshold_value = 0.5  # 阈值可以根据需要进行调整
    #     _, binary_mask = cv2.threshold(final_mask, threshold_value, 1, cv2.THRESH_BINARY)

    #     # 上采样到1280x1280
    #     resized_mask = cv2.resize(binary_mask, (640, 640), interpolation=cv2.INTER_LINEAR)

    #     # 处理两个框
    #     # points_box1 = self.process_single_box_row(output0, resized_mask, max_idx_4)
    #     x, y, w, h = map(int, [output0[0][0][max_idx_4], output0[0][1][max_idx_4], output0[0][2][max_idx_4], output0[0][3][max_idx_4]])
    #     boxes = [x-w/2,y-h/2,x+w/2,y+h/2]
    #     points_box1 = self.process_single_box_col(boxes, resized_mask)
    #     # points_box1 = [(0,0),(100,100)]
    #     # print("Box1 Points:", points_box1)
    #     return [points_box1]

    """
    len(input_data) = 13
    12:分三轮代表 8400个框的组合
    1*64*80*80 经过处理后可以得到 1*4*80*80的数据 代表框
    1*2*80*80 代表两个类别的置信度。80*80对应框的80*80 现在的模型训练的时候只有一类
    1*1*80*80 代表置信度的和。目前还没用上这个
    1*32*80*80 代表最后需要计算segment的权重
    80*80 + 40*40 + 20*20 = 8400
    13: 1*32*160*160 代表最后计算segment的掩码原型
    """

    def post_process_seg_i8(self, input_data):
        defualt_branch = 3
        pair_per_branch = 4
        maskPrototype = input_data[12]
        final_mask = np.zeros((160, 160))
        result = []
        box0 = []
        mask_weights = []
        max_0 = 0
        id = 0
        # Python 忽略 score_sum 输出

        for i in range(defualt_branch):
            n, c, h, w = input_data[pair_per_branch * i + 1].shape

            flat_index_channel_0 = np.argmax(
                input_data[pair_per_branch * i + 1][0, 0, :, :]
            )
            # result_indices0 = np.where(input_data[pair_per_branch*i+1][0, 0, :, :] > 0.6)
            # print(result_indices0)
            max_value_channel_0 = input_data[pair_per_branch * i + 1][0, 0, :, :].flat[
                flat_index_channel_0
            ]
            # print(flat_index_channel_0)
            index_channel_0 = np.unravel_index(flat_index_channel_0, (h, w))
            # print(index_channel_0)

            if max_value_channel_0 > max_0:
                last_boxes_input0 = input_data[pair_per_branch * i][
                    :, :, index_channel_0[0], index_channel_0[1]
                ][..., None, None]
                last_index00 = index_channel_0[0]
                last_index01 = index_channel_0[1]
                last_h0 = h
                last_w0 = w
                max_0 = max_value_channel_0
                last_weight_input0 = input_data[pair_per_branch * i + 3]

        if max_0 > Inference_seg.OBJ_THRESH:
            # for i in range(10):
            bag_output = self.box_process(last_boxes_input0, last_h0, last_w0, 0, 0)
            mask_weights = last_weight_input0[:, :, last_index00, last_index01]
            # print(last_input0.shape,bag_output.shape)
            mask_weights = np.squeeze(mask_weights)
            box0 = bag_output[:, :, last_index00, last_index01][0]

        maskPrototype = input_data[12]
        if len(mask_weights) == 0:
            result = []
        else:
            final_mask = np.zeros((160, 160))
            for i in range(32):  # 因为有32个掩码
                final_mask += mask_weights[i] * maskPrototype[0, i, :, :]

            # 应用阈值以二值化掩码
            threshold_value = 0.5  # 阈值可以根据需要进行调整
            _, binary_mask = cv2.threshold(
                final_mask, threshold_value, 1, cv2.THRESH_BINARY
            )

            # 上采样到1280x1280
            resized_mask = cv2.resize(
                binary_mask, (640, 640), interpolation=cv2.INTER_LINEAR
            )

            # print(f"resized_mask :{resized_mask.shape}")
            # array = resized_mask * 255
            # array = array.astype(np.uint8)
            # cv2.rectangle(array, (int(box0[0]), int(box0[1])), (int(box0[2]), int(box0[3])), 255, 1)
            # cv2.imwrite('output_image.png', array)

            # points_box1 = self.process_single_box_col(box0, resized_mask)
            polyfit_point = self.process_single_box_col_polyfit(box0, resized_mask)
            result = polyfit_point

        return result

    def post_process_seg_i8_net_strap(self, input_data):
        defualt_branch = 3
        pair_per_branch = 4
        maskPrototype = input_data[12]
        final_mask = np.zeros((160, 160))
        result = []
        box0 = []
        mask_weights = []
        max_0 = 0
        id = 0
        # Python 忽略 score_sum 输出

        for i in range(defualt_branch):
            n, c, h, w = input_data[pair_per_branch * i + 1].shape

            flat_index_channel_0 = np.argmax(
                input_data[pair_per_branch * i + 1][0, 1, :, :]
            )
            # result_indices0 = np.where(input_data[pair_per_branch*i+1][0, 1, :, :] > 0.6)
            # print(result_indices0)
            max_value_channel_0 = input_data[pair_per_branch * i + 1][0, 1, :, :].flat[
                flat_index_channel_0
            ]
            # print(flat_index_channel_0)
            index_channel_0 = np.unravel_index(flat_index_channel_0, (h, w))
            # print(index_channel_0)

            if max_value_channel_0 > max_0:
                last_boxes_input0 = input_data[pair_per_branch * i][
                    :, :, index_channel_0[0], index_channel_0[1]
                ][..., None, None]
                last_index00 = index_channel_0[0]
                last_index01 = index_channel_0[1]
                last_h0 = h
                last_w0 = w
                max_0 = max_value_channel_0
                last_weight_input0 = input_data[pair_per_branch * i + 3]

        if max_0 > Inference_seg.OBJ_THRESH:
            # for i in range(10):
            bag_output = self.box_process(last_boxes_input0, last_h0, last_w0, 0, 0)
            mask_weights = last_weight_input0[:, :, last_index00, last_index01]
            # print(last_input0.shape,bag_output.shape)
            mask_weights = np.squeeze(mask_weights)
            box0 = bag_output[:, :, last_index00, last_index01][0]

        maskPrototype = input_data[12]
        if len(mask_weights) == 0:
            result = []
        else:
            final_mask = np.zeros((160, 160))
            for i in range(32):  # 因为有32个掩码
                final_mask += mask_weights[i] * maskPrototype[0, i, :, :]

            # 应用阈值以二值化掩码
            threshold_value = 0.5  # 阈值可以根据需要进行调整
            _, binary_mask = cv2.threshold(
                final_mask, threshold_value, 1, cv2.THRESH_BINARY
            )

            # 上采样到1280x1280
            resized_mask = cv2.resize(
                binary_mask, (640, 640), interpolation=cv2.INTER_LINEAR
            )

            # points_box1 = self.process_single_box_col(box0, resized_mask)
            polyfit_point = self.process_single_box_col_polyfit(box0, resized_mask)

            result = polyfit_point

        return result

    def pre_thread(self):
        # 前处理
        from hit.utils.utils import set_thread_name

        try:
            set_thread_name("RKNN_pre")
        except:
            pass

        print("start pre")
        while True:
            self.semaphore_pre.acquire()  # 等待队列pre中有数据
            img0 = self.pre_inputs0.get()
            img1 = self.pre_inputs1.get()

            if img0 is None and img1 is None:
                self.all_inputs0.put(None)
                self.all_inputs1.put(None)
                self.semaphore_infer0.release()
                self.semaphore_infer1.release()
                print("pre deal finish")
                break

            # 增加一个判断，当flag == 1时，不做识别，直接往结果里添加内容
            if img0 is None and img1 is not None:
                while self.wait_count != 0:
                    time.sleep(0.001)
                    continue

                # 不做识别，把输入的起点拿出来，然后往最后的结果里添加None
                # left_point = self.left_points.get()
                # right_point = self.right_points.get()
                self.result_pos.put({})
            else:
                self.wait_count += 1

                # 取消旋转
                # for rotated camera
                # rotated_img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
                # rotated_img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
                # img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
                # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

                # img0 = cv2.merge([
                #     img0[:,:,1],  # B
                #     img0[:,:,0],  # G
                #     img0[:,:,2]   # R
                # ])

                # img1 = cv2.merge([
                #     img1[:,:,1],  # B
                #     img1[:,:,0],  # G
                #     img1[:,:,2]   # R
                # ])

                # filename = f"/home/cat/Desktop/frame_1.jpg"
                # cv2.imwrite(filename, img0)
                # print(f"[Python] Saved frame")

                img0_input = np.expand_dims(img0, 0)
                img1_input = np.expand_dims(img1, 0)
                self.all_inputs0.put(img0_input)
                self.all_inputs1.put(img1_input)

                # img0 = np.expand_dims(img0, 0)
                # img1 = np.expand_dims(img1, 0)
                # self.all_inputs0.put(img0)
                # self.all_inputs1.put(img1)
                self.semaphore_infer0.release()
                self.semaphore_infer1.release()
            # time.sleep(0.001)

    def inference_thread0(self):
        # 推理
        from hit.utils.utils import set_thread_name

        try:
            set_thread_name("RKNN_npu0")
        except:
            pass

        model = self.model0
        print("start inference0")
        while True:
            self.semaphore_infer0.acquire()
            img = self.all_inputs0.get()
            if img is None:
                self.all_outputs0.put(None)
                self.semaphore_post0.release()
                print("npu0 inference finish")
                break
            outputs = model.run([img])
            self.all_outputs0.put(outputs)
            self.semaphore_post0.release()

    def inference_thread1(self):
        # 推理
        from hit.utils.utils import set_thread_name

        try:
            set_thread_name("RKNN_npu1")
        except:
            pass

        model = self.model1
        print("start inference1")
        while True:
            self.semaphore_infer1.acquire()
            img = self.all_inputs1.get()
            if img is None:
                self.all_outputs1.put(None)
                self.semaphore_post1.release()
                print("npu1 inference finish")
                break
            outputs = model.run([img])
            self.all_outputs1.put(outputs)
            self.semaphore_post1.release()

    def post_thread(self):
        # 后处理
        from hit.utils.utils import set_thread_name

        try:
            set_thread_name("RKNN_post")
        except:
            pass

        print("start post")
        i = 0
        time_start = time.perf_counter()
        while True:
            self.semaphore_post0.acquire()
            self.semaphore_post1.acquire()
            output0 = self.all_outputs0.get()
            output1 = self.all_outputs1.get()
            i += 1
            if output0 is None and output1 is None:
                print("post deal finish")
                self.result_deque.put({"T_info": None, "net_strap_info": None})
                self.semaphore_result.release()
                break
            if output0 is not None:
                result0 = self.post_process_seg_i8(output0)
                if self.id == self.OUT_seg:
                    result0_net_strap = self.post_process_seg_i8_net_strap(output0)
            if output1 is not None:
                result1 = self.post_process_seg_i8(output1)
                if self.id == self.OUT_seg:
                    result1_net_strap = self.post_process_seg_i8_net_strap(output1)

            result_data = {"T_info": [result0, result1]}
            if self.id == self.OUT_seg:
                result_data["net_strap_info"] = [result0_net_strap, result1_net_strap]
            else:
                result_data["net_strap_info"] = None
            self.result_deque.put(result_data)
            self.semaphore_result.release()

        self.post_time = (time.perf_counter() - time_start) / i

    def start(self):
        self.thread_npu = threading.Thread(target=self.inference_thread0)
        self.thread_npu1 = threading.Thread(target=self.inference_thread1)
        self.thread_pre = threading.Thread(target=self.pre_thread)
        self.thread_post = threading.Thread(target=self.post_thread)
        self.thread_pos_calcalate = threading.Thread(
            target=self.process_images_and_calculate_position
        )

        self.thread_pre.start()
        self.thread_npu.start()
        self.thread_npu1.start()
        self.thread_post.start()
        self.thread_pos_calcalate.start()

    def add_img(self, img0, img1):
        self.pre_inputs0.put(img0)
        self.pre_inputs1.put(img1)
        # self.left_points.put(left_point)
        # self.right_points.put(right_point)
        # self.ct_queue.put(ct)
        self.semaphore_pre.release()

    def stop(self):
        #
        self.add_img(None, None)
        self.thread_pre.join()
        self.thread_npu.join()
        self.thread_npu1.join()
        self.thread_post.join()
        self.thread_pos_calcalate.join()

    def calculate_origin_position(self, xb, yb, xp, yp, theta_deg):
        """
        xb,yb 原坐标 xp,yp 旋转后的坐标 ,theta_deg 旋转角度
        """
        theta_rad = math.radians(theta_deg)
        cos_theta = math.cos(theta_rad)
        sin_theta = math.sin(theta_rad)

        rotated_x = cos_theta * xp - sin_theta * yp
        rotated_y = sin_theta * xp + cos_theta * yp

        xa = xb - rotated_x
        ya = yb - rotated_y

        return (xa, ya)

    def process_images_and_calculate_position(self):
        from hit.utils.utils import set_thread_name

        try:
            set_thread_name("RKNN_pos_calc")
        except:
            pass

        while True:
            self.semaphore_result.acquire()
            all_data = self.result_deque.get()
            # print(f"all_data:{all_data}")
            if all_data["T_info"] is None:
                print("pos calculate done")
                break
            # left_point = self.left_points.get()
            # right_point = self.right_points.get()
            # ct = self.ct_queue.get()
            # startx_left = left_point[0]
            # starty_left = left_point[1]
            # startx_right = right_point[0]
            # starty_right = right_point[1]
            # points_left = all_data[0]
            # points_right = all_data[1]

            self.result_pos.put(all_data)

            self.wait_count -= 1

            # if len(points_left)!= 0:
            #     left_upx = points_left[0][0] + startx_left
            #     left_upy = points_left[0][1] + starty_left
            #     left_downx = points_left[1][0] + startx_left
            #     left_downy = points_left[1][1] + starty_left
            #     points_left =[(left_upx, left_upy),(left_downx, left_downy)]
            # if len(points_right)!=0:
            #     right_upx = points_right[0][0] + startx_right
            #     right_upy = points_right[0][1] + starty_right
            #     right_downx = points_right[1][0] + startx_right
            #     right_downy = points_right[1][1] + starty_right
            #     points_right = [(right_upx, right_upy),(right_downx, right_downy)]
            # if len(points_left) == 0 or len(points_right) == 0:
            #     # print(points_left,points_right)
            #     self.result_pos.put([None,points_left,points_right,[(startx_left,starty_left),(startx_right,starty_right)],ct])
            #     # self.resulpoints_left,points_right,[(startx_left,starty_left),(startx_right,starty_right)],ct])
            # else:
            #     # loc = utils.calc_loc_by_T(config.FRONT_LEFT_CAMERA_PRAMS_1360, config.FRONT_RIGHT_CAMERA_PRAMS_1360, points_left, points_right)
            #     loc = utils.calc_loc_by_T(config.CMAERA_LEFT_PARAMS_SHORT, config.CMAERA_RIGHT_PARAMS_SHORT, points_left, points_right)
            #     self.result_pos.put([loc, points_left, points_right,[(startx_left,starty_left),(startx_right,starty_right)],ct])

            # self.wait_count -= 1

            # Calculate the relative location of the detected points
            # 0 代表label 目前只有竖线 第二个代表上面和下面的点 第三个代表x，y
            # relative_ball_loc_up   = utils.ball_estimation_by_cv2(
            #     [left_upx, left_upy],
            #     [right_upx, right_upy],
            #     config.FRONT_LEFT_CAMERA_PRAMS_1360,
            #     config.FRONT_RIGHT_CAMERA_PRAMS_1360

            # )
            # relative_ball_loc_down = utils.ball_estimation_by_cv2(
            #     [left_downx, left_downy],
            #     [right_downx, right_downy],
            #     config.FRONT_LEFT_CAMERA_PRAMS_1360,
            #     config.FRONT_RIGHT_CAMERA_PRAMS_1360
            # )

            # # Calculate the angle
            # def calculate_angle(x1, y1, x2, y2):
            #     return math.atan2(y2 - y1, x2 - x1)

            # angle1 = calculate_angle(relative_ball_loc_down[0], relative_ball_loc_down[2], relative_ball_loc_up[0], relative_ball_loc_up[2])
            # angle_degrees = math.degrees(math.pi/2 - angle1)
            # # print("roll_angle", angle_degrees)

            # x_down, y_down = self.calculate_origin_position(self.x, self.z, relative_ball_loc_down[0], relative_ball_loc_down[2], angle_degrees)

            # x_up, y_up = self.calculate_origin_position(self.x, self.z + 1.22, relative_ball_loc_up[0], relative_ball_loc_up[2], angle_degrees)
            # print("loc", x, y)
            # self.result_pos.put([[x, y, angle_degrees],[(left_upx, left_upy),(left_downx, left_downy)],[(right_upx, right_upy),(right_downx, right_downy)],ct])
            # temp
            # loc = utils.calc_loc_by_T(config.FRONT_LEFT_CAMERA_PRAMS_1360, config.FRONT_RIGHT_CAMERA_PRAMS_1360, points_left, points_right)
            # self.result_pos.put([[x_down, y_down, angle_degrees],[(left_upx, left_upy),(left_downx, left_downy)],[(right_upx, right_upy),(right_downx, right_downy)],[(startx_left,starty_left),(startx_right,starty_right)],ct])
            # self.result_pos.put([loc, points_left, points_right,[(startx_left,starty_left),(startx_right,starty_right)],ct])


def draw(boxes, img):
    for i in range(len(boxes)):
        x, y, w, h = map(int, boxes[i])  # 转换为整数
        x1 = int(x - w / 2)
        x2 = int(x + w / 2)
        y1 = int(y - h / 2)
        y2 = int(y + h / 2)
        cv2.rectangle(
            img, (x1, y1), (x2, y2), (0, 255, 0), 2
        )  # 绘制绿色边框，边框宽度为2

        # 绘制分数
        # score = boxes[i][1]
        score_text = f"{i:.2f}"
        cv2.putText(
            img,
            score_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    cv2.imwrite(f"output_image_test.jpg", img)


def draw_points_on_image(img, points, output_image_path):
    """
    在图像上绘制点。

    """
    # 加载图像
    # output_image_path = "/home/cat/Desktop/rk_basic_code/inference/images/seg_label/0730_1941_left.jpg"
    color = (0, 255, 0)
    radius = 1
    save_image = True
    # img = cv2.imread(image_path)
    if img is None:
        print("Error: 图像未找到，请检查路径。")
        return

    # 在图像上绘制点
    print(points)
    if len(points) != 0:
        for idx, point in enumerate(points):
            # for point in group:
            cv2.circle(img, point, radius, color, -1)  # -1 表示填充圆
            cv2.putText(
                img,
                f"{idx}",
                (point[0] + 10, point[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.1,
                color,
                1,
            )

    # 根据需要保存图像
    if save_image:
        cv2.imwrite(output_image_path, img)
        print(f"图像已保存到 {output_image_path}")


def calculate_angle(x1, y1, x2, y2):
    # 计算坐标差
    dx = x2 - x1
    dy = y2 - y1

    # print(dx,dy)

    # 使用atan2计算角度（弧度）
    angle_radians = math.atan2(dy, dx)

    # 转换为度数
    # angle_degrees = math.degrees(angle_radians)

    return angle_radians


def crop_center(img, cropx, cropy):
    y, x, _ = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2 + 200
    return img[starty : starty + cropy, startx : startx + cropx], startx, starty


def get_pos(model_inference):
    while True:
        if model_inference.result_pos.empty():
            time.sleep(0.001)
            continue
        pos = model_inference.result_pos.get()
        print("time_gap", time.perf_counter() - pos[-1])


"""
temp_deal:
pos:[[x,z,angle],[left_points],[right_points],[start_x,start_y],ct]
last_deal:

正确识别的情况下：
[[x,z,angle],[left_points],[right_points],ct]
"""


def deal_video(model_inference, video_path, width, height):
    filename_with_extension = os.path.basename(video_path)

    # 使用os.path.splitext分离文件名和扩展名
    filename, extension = os.path.splitext(filename_with_extension)
    output_file = f"/home/cat/Desktop/rk_basic_code/inference/video/label_test/{filename}_label_use.mp4"

    # 初始化VideoWriter对象
    command = [
        "ffmpeg",
        "-y",  # 覆盖输出文件
        "-f",
        "rawvideo",  # 告诉 FFmpeg 输入将是原始视频数据
        "-vcodec",
        "rawvideo",  # 输入视频编码格式为原始
        # '-s', f'{1440}x{1280}',  # 分辨率
        "-s",
        f"{width}x{height}",  # 分辨率
        "-pix_fmt",
        "bgr24",  # OpenCV 默认以 BGR 格式输出
        "-r",
        "30",  # 帧率
        "-i",
        "-",  # 从 stdin 读取输入
        "-c:v",
        "h264_rkmpp",  # 输出视频使用 x264 编码
        # '-c:v', 'h264_v4l2m2m', # 使用 V4L2 mem2mem H.264 编码器
        "-pix_fmt",
        "yuv420p",  # 输出视频的像素格式
        "-preset",
        "veryfast",  # 编码速度最快，牺牲压缩率
        "-tune",
        "zerolatency",  # 减少编码延迟
        "-loglevel",
        "quiet",
        "-movflags",
        "+faststart",
        f"{output_file}",  # 输出文件名
    ]
    process_video = subprocess.Popen(command, stdin=subprocess.PIPE)
    video = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not video.isOpened():
        print("Error: Could not open video.")
        exit()

    # 逐帧读取视频
    frame_id = 0
    color = (255, 255, 255)
    radius = 3
    gap = 0
    while True:
        # 读取一帧
        if model_inference.result_pos.empty():
            time.sleep(0.001)
            continue
        pos = model_inference.result_pos.get()
        # print("time gap ",time.perf_counter() - pos[-1])
        # pos:[[x,z,angle],[left_points],[right_points],[start_x,start_y],ct]
        ret, frame = video.read()
        frame_id += 1
        gap += time.perf_counter() - pos[-1]
        average_gap = gap / frame_id
        print("average time gap ", average_gap)
        frame = frame.copy()
        formatted_pos_down = []
        # formatted_pos_up =[]
        if len(pos[0]) != 0:
            formatted_pos_down = [f"{x:.2f}" for x in pos[0]]
            text = f"pos_down: {formatted_pos_down}"
        # if len(pos[1])!=0:
        #     formatted_pos_up = [f"{x:.2f}" for x in pos[1]]
        #     text_up = f"pos_up: {formatted_pos_up}"
        # deal frame 画裁剪的框 画识别的点，画位置
        if len(pos[0]) != 0:
            cv2.putText(
                frame, text, (1280, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2
            )
            # cv2.imwrite("test.jpg",frame)
        # if len(pos[1]) != 0:
        #     cv2.putText(frame, text_up, (1280, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2)
        # print(pos[4],pos[5])
        cv2.rectangle(frame, (480, 480), (1120, 1120), (0, 0, 255), 3)
        cv2.rectangle(frame, (480 + 1600, 480), (1120 + 1600, 1120), (0, 0, 255), 3)
        points_left = pos[1]
        points_right = pos[2]
        if len(points_left) != 0:
            for idx, point in enumerate(points_left):
                # for point in group:
                cv2.circle(frame, point, radius, color, -1)  # -1 表示填充圆
                cv2.putText(
                    frame,
                    f"{idx}",
                    (point[0] + 10, point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.1,
                    color,
                    2,
                )

        if len(points_right) != 0:
            for idx, point in enumerate(points_right):
                # for point in group:
                new_point = (point[0] + 1600, point[1])
                cv2.circle(frame, new_point, radius, color, -1)  # -1 表示填充圆
                cv2.putText(
                    frame,
                    f"{idx}",
                    (new_point[0] + 10, new_point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.1,
                    color,
                    2,
                )

        process_video.stdin.write(frame.tobytes())

        # 如果读取成功，ret为True
        if not ret:
            print("Reached end of video or failed to read the frame. ------ deal video")
            break

        # 打印帧编号
        print(f"Processing frame {frame_id}")

    video.release()


"""
0928
室内新布的线
down z = 1.48
up   z = 2.7
delay test
30帧输入的平均延时 200ms
"""
if __name__ == "__main__":
    model_inference = Inference_seg(Inference_seg.IN_seg)
    # model_inference.start()

    # img = cv2.imread(r"/home/cat/Desktop/dualView/pics/test1.jpg")
    # left_img = img[:, :640]
    # left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    # print(left_img.shape)
    # right_img = img[:, 640:]
    # right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
    # # img1 = cv2.imread(r"/home/cat/Desktop/dualView/pics/20250116-122624.jpg")
    # # img2 = cv2.imread(r"/home/cat/Desktop/dualView/pics/20250116-122624.jpg")

    #     # video_path = "/home/cat/Desktop/rk_basic_code/inference/video/row/63.mp4"
    #     # width = 3200
    #     # height = 1200
    #     # ----------------处理视频用于验证识别效果------------------
    #     # video = cv2.VideoCapture(video_path)
    #     # thread_deal_video = threading.Thread(target=deal_video,args=(model_inference,video_path,width,height))
    #     # thread_deal_video.start()
    #     # -----------------------end-------------------------

    #     # 当前裁切处理，左右图片中心
    #     # crop_width, crop_height = 640, 640
    #     # while True:
    #     #     ret, frame = video.read()
    #     #     # 如果读取成功，ret为True
    #     #     if not ret:
    #     #         print("Reached end of video or failed to read the frame. ------ read video")
    #     #         break
    #     #     img_left  = frame[:, :width//2]
    #     #     img_right = frame[:, width//2:]
    #     #     cropped_img_left,startx_left,starty_left    = crop_center(img_left, crop_width, crop_height)
    #     #     cropped_img_right,startx_right,starty_right = crop_center(img_right, crop_width, crop_height)
    #     #     # 输入样例
    #     #     # 左边640*640的图像，右边640*640的图像，左边截图的起点，右边截图的起点，时间
    #     #     model_inference.add_img(cropped_img_left,cropped_img_right,(startx_left,starty_left),(startx_right,starty_right),time.perf_counter())
    #     #     time.sleep(0.1)
    # model_inference.add_img(left_img, right_img)

    # time.sleep(0.5)

    # aaa = model_inference.result_pos.get()
    # print(f"main:T_info:{aaa['T_info']}")
    # print(f"main:net_strap_info:{aaa['net_strap_info']}")

    # # 结果的含义 计算位置的函数：process_images_and_calculate_position    在这个函数里实现了车位置的计算
    # # pos[0]: [x_down, y_down, angle_degrees]                         通过下面的点计算出来的车的位置
    # # pos[1]: [(left_upx, left_upy),(left_downx, left_downy)]         左边图像识别出来的上面的点和下面的点
    # # pos[2]: [(right_upx, right_upy),(right_downx, right_downy)]     右边图像识别出来的上面的点和下面的点
    # # pos[3]: [(startx_left,starty_left),(startx_right,starty_right)] 截图的起始位置
    # # pos[4]: ct                                                      输入图片的时间
    # # 拿到结果：
    # # pos = model_inference.result_pos.get()

    # model_inference.stop()
