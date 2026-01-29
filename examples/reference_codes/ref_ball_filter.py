"""
接收左右图片都识别到的球的信息然后做过滤，滤除不符合击球需要的球。

实验调试方法，基于json数据来做，
需要收集的信息：
record_json:{
    left_ball:
    right_ball:
}

可以用现成的video 和 json做测试，可以先基于球的位置裁，然后调用匹配过滤算法。看效果。
过滤之前需要做匹配（暂时先用暴力方法匹配）
可以过滤的球的类型:地上的球(计算loc_y),大小不对的球(通过z的计算实现)。

"""

"""
0906：目前存在的问题，双目过滤可能过滤调一些对的球，因为目前用的单目计算高度由于运动中球的抖动影响比较大。
导致最后双目计算出来的深度在比较的时候出错了。
"""

import json
from hit.cfgs import config
import math
import numpy as np
import time
from hit.cfgs.bot_motion_config import BotMotionConfig
from hit.utils import utils, get_logger


MAX_DEPTH = 15


class BallFilter:
    def __init__(self):
        # 定义了球的直径
        self.ball_size = 0.065
        self.logger = get_logger(
            "ball_filter",
            console_output=True,
            file_output=True,
            # console_level="INFO",
            console_level="INFO",
            file_level="INFO",
            time_interval=-0.1,
            use_buffering=True,
            buffer_capacity=100,
            flush_interval=1.0,
            propagate=False,
        )

    def get_project_with_distortion(self, obj_point, camera, r_matrix=None, k1=0, k2=0):
        if r_matrix is None:
            r_matrix = utils.get_r_matrix_with_radians(camera[3:6])
        vp = np.dot(r_matrix, obj_point - camera[:3])
        # print(f"relative point is: {obj_point - camera[:3]}, after rotation is : {vp}")

        if vp[2] <= 0:
            return np.array([-1e9, -1e9])

        x = vp[0] / vp[2]
        y = vp[1] / vp[2]
        r_square = x**2 + y**2
        radial = 1 + r_square * k1 + r_square**2 * k2
        # print(f"radial is {radial}, {[x*radial*camera[6], y*radial*camera[6]]}")
        return np.array(
            [x * radial * camera[6] + camera[7], -y * radial * camera[6] + camera[8]]
        )

    def calc_error_after_translation(self, rps, ips, camera):
        r_matrix = utils.get_r_matrix_with_radians(camera[3:6])
        pp = []
        n = len(rps)
        for i in range(n):
            # print(rps[i])
            project_point = self.get_project_with_distortion(
                rps[i], camera, r_matrix, camera[9], camera[10]
            )

            # print(f"real point: {rps[i]},  camera: {camera},  project point: {project_point}")

            pp.append(project_point)

        error = 0
        for i in range(n):
            error += math.sqrt(
                (pp[i][0] - ips[i][0]) ** 2 + (pp[i][1] - ips[i][1]) ** 2
            )
        error /= n
        return error

    # camera_id 0:short_left,1:short_right,2:long_left,3:long_right
    # 单目过滤地上的球 cam[6]： 焦距， cam[8]： y_center， cam[1]： height， cam[4]： pitch
    # def single_filter(self, ball_pixel_loc_list,camera_id):
    #     if camera_id == 0:
    #         cam = config.FRONT_LEFT_CAMERA_PRAMS_1360
    #     if camera_id == 1:
    #         cam = config.FRONT_RIGHT_CAMERA_PRAMS_1360
    #     if camera_id == 2:
    #         cam = config.FRONT_LEFT_CAMERA_PRAMS_2800
    #     if camera_id == 3:
    #         cam = config.FRONT_RIGHT_CAMERA_PRAMS_2800

    #     x,y,w,h = ball_pixel_loc_list
    #     pixel_width = w
    #     deep = cam[6] * self.ball_size / pixel_width
    #     y_gap = (cam[8] - y - h/2)
    #     height_ball =  cam[1] + y_gap * self.ball_size / pixel_width
    #     H_ball = deep * math.tan(cam[4]) + height_ball
    #     if H_ball < 0.18:
    #         return None
    #     return ball_pixel_loc_list

    # # 双目过滤 基于重投影误差计算。以及大小匹配计算，过滤球场的球,过滤之后可以实现返回，左右相机都有的并且匹配的球
    # # TODO 修改深度比较，改成基于三维深度算出这个深度下的球的pixel和实际识别出来的物体的pixel做比较
    # def dual_filter(self,left_ball_list,right_ball_list,camera_id):
    #     filter_ball = []
    #     filter_left  = []
    #     filter_right = []
    #     if camera_id == 0:
    #         cam_l = config.FRONT_LEFT_CAMERA_PRAMS_1360
    #         cam_r = config.FRONT_RIGHT_CAMERA_PRAMS_1360
    #     if camera_id == 1:
    #         cam_l = config.FRONT_LEFT_CAMERA_PRAMS_2800
    #         cam_r = config.FRONT_RIGHT_CAMERA_PRAMS_2800
    #     for i in range(len(left_ball_list)):
    #         for j in range(len(right_ball_list)):
    #             left_center_x = left_ball_list[i][0] + left_ball_list[i][2]/2
    #             left_center_y = left_ball_list[i][1] + left_ball_list[i][3]/2

    #             right_center_x = right_ball_list[j][0] + right_ball_list[j][2]/2
    #             right_center_y = right_ball_list[j][1] + right_ball_list[j][3]/2
    #             loc_result = utils.ball_estimation_by_cv2([left_center_x,left_center_y],[right_center_x,right_center_y],cam_l,cam_r)
    #             # 计算这个位置下的重投影误差，左右都差不多就认为匹配上了
    #             # print("loc_result is ",loc_result)
    #             error_l = self.calc_error_after_translation([loc_result],[[left_center_x,left_center_y]] ,cam_l)
    #             error_r = self.calc_error_after_translation([loc_result],[[right_center_x,right_center_y]],cam_r)
    #             # print("error_l",error_l,error_r)
    #             if (error_l + error_r) < 30: # 左右匹配上了
    #                 x,y,w,h = left_ball_list[i]
    #                 pixel_width = (w + h) / 2
    #                 pixel_width_world = cam_l[6] * self.ball_size / loc_result[2]
    #                 deep = cam_l[6] * self.ball_size / pixel_width
    #                 y_gap = (cam_l[8] - y - h/2)
    #                 height_ball =  cam_l[1] + y_gap * self.ball_size / pixel_width
    #                 H_ball = deep * math.tan(cam_l[4]) + height_ball
    #                 if abs(H_ball - loc_result[1]) < 0.3 and abs(pixel_width_world - pixel_width) < 5: # 大小也匹配
    #                     filter_ball.append([[left_center_x,left_center_y],[right_center_x,right_center_y],left_ball_list[i],right_ball_list[j]])
    #     return filter_ball

    def calc_error_after_translation_test(self, rps, ips, camera):
        project_point = utils.get_project_with_full_camera_1(rps, camera)

        # print(f"real point: {rps},   project point: {project_point}, ips is {ips}")
        error = math.sqrt(
            (project_point[0] - ips[0]) ** 2 + (project_point[1] - ips[1]) ** 2
        )

        return error

    # camera_id 0:short_left,1:short_right,2:long_left,3:long_right
    # 单目过滤地上的球 cam[6]： 焦距， cam[8]： y_center， cam[1]： height， cam[4]： pitch
    def single_filter(self, ball_pixel_loc_list, camera_id):
        if camera_id == 0:
            cam = config.CMAERA_LEFT_PARAMS
        if camera_id == 1:
            cam = config.CMAERA_RIGHT_PARAMS
        if camera_id == 2:
            cam = config.CMAERA_LEFT_PARAMS_LONG
        if camera_id == 3:
            cam = config.CMAERA_RIGHT_PARAMS_LONG

        H_ball, f, deep = self.calculate_ball_H_from_single(ball_pixel_loc_list, cam)

        if H_ball < 0.21:
            return None
        return ball_pixel_loc_list

    def calculate_ball_H_from_single(self, ball_info, cam):
        f = cam["mtx"][0][0]
        x, y, w, h = ball_info
        pixel_width = (w + h) / 2
        # 网球深度
        deep = f * self.ball_size / pixel_width

        # 构造像素坐标
        pixel_coords = np.array([x + w / 2, y + h / 2, 1], dtype=np.float32)
        # camera_coords = deep * np.dot(np.linalg.inv(cam["mtx"]), pixel_coords)
        camera_coords = deep * np.dot(cam["MTX_INV"], pixel_coords)
        world_coords = np.array(cam["R"]).T @ camera_coords - np.array(
            [cam["T"][0][0], cam["T"][1][0], cam["T"][2][0]]
        )

        return -world_coords[1] + BotMotionConfig.CAMERA_TO_CAR_CENTER["y"], f, deep

    # 返回相机坐标系，和世界坐标系的网球坐标
    def calculate_abs_ball_loc_from_single(self, ball_info, cam, bot_loc):
        f = cam["mtx"][0][0]
        x, y, w, h = ball_info
        pixel_width = (w + h) / 2
        # 网球深度
        deep = f * self.ball_size / pixel_width

        # 构造像素坐标
        pixel_coords = np.array([x + w / 2, y + h / 2, 1], dtype=np.float32)
        camera_coords = deep * np.dot(
            cam["MTX_INV"], pixel_coords
        )  # 相机视角下的空间坐标
        relative_ball_loc = np.array(cam["R"]).T @ camera_coords - np.array(
            [cam["T"][0][0], cam["T"][1][0], cam["T"][2][0]]
        )  # 以小车中心为原点的，3d坐标

        relative_ball_loc[1] *= -1
        return relative_ball_loc, utils.get_ball_in_world(relative_ball_loc, bot_loc)

    # 判断网球是否在球场界内, 以及是否高于地面
    def is_ball_in_court(self, ball_loc):
        if ball_loc[1] < BotMotionConfig.FILTER_BALL_HEIGHT:
            return False
        if (
            ball_loc[0] < BotMotionConfig.COURT_X_RANGE[0]
            or ball_loc[0] > BotMotionConfig.COURT_X_RANGE[1]
        ):
            return False
        if (
            ball_loc[2] < BotMotionConfig.COURT_Z_RANGE[0]
            or ball_loc[2] > BotMotionConfig.COURT_Z_RANGE[1]
        ):
            return False

        return True

    # 基于小车位置，过滤掉隔壁球场的球，也就是对识别网球区域建立虚拟边界
    # 但并没有加入基于曲线过滤
    def dual_filter_with_bot_loc(self, left_ball_list, right_ball_list, bot_loc):
        filter_ball = []
        cam_l = config.CMAERA_LEFT_PARAMS
        cam_r = config.CMAERA_RIGHT_PARAMS
        f = cam_l["mtx"][0][0]

        for i in range(len(left_ball_list)):
            # 判断单目下左球在球场空中
            ball_in_camera, ball_in_world = self.calculate_abs_ball_loc_from_single(
                left_ball_list[i], cam_l, bot_loc
            )
            if not self.is_ball_in_court(ball_in_world):
                continue

            for j in range(len(right_ball_list)):
                # 判断单目下右球在球场空中
                ball_in_camera, ball_in_world = self.calculate_abs_ball_loc_from_single(
                    right_ball_list[j], cam_r, bot_loc
                )
                if not self.is_ball_in_court(ball_in_world):
                    continue

                _, _, wl, hl = left_ball_list[i]
                _, _, wr, hr = right_ball_list[j]
                # 網球像素誤差大小，應該在一定範圍内, 避免球場上其他形狀的干擾
                if (
                    2 * abs(wl - wr) / (wl + wr + 10) > 0.2
                    or 2 * abs(hl - hr) / (hl + hr + 10) > 0.3
                ):
                    continue

                left_center_x = left_ball_list[i][0] + left_ball_list[i][2] / 2
                left_center_y = left_ball_list[i][1] + left_ball_list[i][3] / 2

                right_center_x = right_ball_list[j][0] + right_ball_list[j][2] / 2
                right_center_y = right_ball_list[j][1] + right_ball_list[j][3] / 2
                dual_ball_loc = utils.ball_estimation_by_cv2(
                    [left_center_x, left_center_y],
                    [right_center_x, right_center_y],
                    cam_l,
                    cam_r,
                )

                # 只判断z在一定范围的球
                if dual_ball_loc[2] < 0.5 or dual_ball_loc[2] > MAX_DEPTH:
                    continue

                # 计算这个位置下的重投影误差，左右都差不多就认为匹配上了
                error_l = self.calc_error_after_translation_test(
                    dual_ball_loc.copy(), [left_center_x, left_center_y], cam_l
                )
                error_r = self.calc_error_after_translation_test(
                    dual_ball_loc.copy(), [right_center_x, right_center_y], cam_r
                )

                # 近距离的球，有可能重投影误差被放大，因此乘上距离作为总的误差。
                # 1218在mark6 ar0822上测得，2m处约有100误差。
                project_error = (error_l + error_r) * math.sqrt(dual_ball_loc[2]) * 2
                # print(f"loc_result:loc_result, {dual_ball_loc}, error_l: {error_l}, error_r: {error_r}")
                if project_error < 200:  # 左右匹配上了
                    # H_ball,f,deep = self.calculate_ball_H_from_single(left_ball_list[i],cam_l)
                    x, y, w, h = left_ball_list[i]
                    pixel_width = (w + h) / 2
                    pixel_width_world = f * self.ball_size / dual_ball_loc[2]
                    # print(H_ball,'  ', loc_result[1], '   ', pixel_width_world,'   ', pixel_width)
                    pixel_gap = abs(pixel_width_world - pixel_width)
                    pixel_error_rate = pixel_gap / (
                        (pixel_width + pixel_width_world) * 0.5
                    )

                    # if abs(left_h_ball - dual_ball_loc[1]) < 0.3 and pixel_error_rate < 0.3: # 大小也匹配
                    if pixel_error_rate < 0.3:  # 大小也匹配
                        filter_ball.append(
                            [
                                [left_center_x, left_center_y],
                                [right_center_x, right_center_y],
                                left_ball_list[i],
                                right_ball_list[j],
                                [project_error, pixel_error_rate],
                                dual_ball_loc,
                                0.5 * (left_center_y + right_center_y),
                            ]
                        )

        return filter_ball

    def close_to_predict_ball_loc(self, ball, predict_ball_loc):
        timestamp = time.perf_counter()
        x_gap = ball[0] - predict_ball_loc[0]
        y_gap = ball[1] - predict_ball_loc[1]
        z_gap = ball[2] - predict_ball_loc[2]

        msg = (
            f"[t={timestamp:.3f}s] Ball position check - "
            f"Predicted: [{predict_ball_loc[0]:.2f}m, {predict_ball_loc[1]:.2f}m, {predict_ball_loc[2]:.2f}m], "
            f"Current: [{ball[0]:.2f}m, {ball[1]:.2f}m, {ball[2]:.2f}m], "
            f"Error: [x:{x_gap:.2f}m, y:{y_gap:.2f}m, z:{z_gap:.2f}m]"
        )
        self.logger.info(msg)

        if abs(x_gap) > 0.5:  # x误差
            return False
        if abs(y_gap) > 0.5:  # y误差
            return False
        if abs(z_gap) > 2.0:  # z误差
            return False

        return True

    # 基于小车位置，过滤掉隔壁球场的球，也就是对识别网球区域建立虚拟边界
    # 同时加入基于预测球的过滤
    def dual_filter_with_bot_loc_and_predict_ball_loc(
        self, left_ball_list, right_ball_list, bot_loc, predict_ball_loc
    ):
        filter_ball = []
        cam_l = config.CMAERA_LEFT_PARAMS
        cam_r = config.CMAERA_RIGHT_PARAMS
        f = cam_l["mtx"][0][0]

        best_loc_gap = 1e9
        best_result = None
        # 先遍历两个列表，筛选掉不合适的球，再对两个列表里剩下的球进行匹配，应该会好点
        for i in range(len(left_ball_list)):
            # 判断单目下左球在球场空中, 或者在地面距离预测球位置非常近
            ball_in_camera, ball_in_world = self.calculate_abs_ball_loc_from_single(
                left_ball_list[i], cam_l, bot_loc
            )
            if not (
                self.is_ball_in_court(ball_in_world)
                or self.close_to_predict_ball_loc(ball_in_camera, predict_ball_loc)
            ):
                continue

            for j in range(len(right_ball_list)):
                # 判断单目下右球在球场空中， 或者在地面距离预测球位置非常近
                ball_in_camera, ball_in_world = self.calculate_abs_ball_loc_from_single(
                    right_ball_list[j], cam_r, bot_loc
                )
                if not (
                    self.is_ball_in_court(ball_in_world)
                    or self.close_to_predict_ball_loc(ball_in_camera, predict_ball_loc)
                ):
                    continue

                # 开始进入常规的过滤筛选， 经过筛选后，只保留距离predict ball最近的球
                _, _, wl, hl = left_ball_list[i]
                _, _, wr, hr = right_ball_list[j]
                # 網球像素誤差大小，應該在一定範圍内, 避免球場上其他形狀的干擾
                if (
                    2 * abs(wl - wr) / (wl + wr + 10) > 0.2
                    or 2 * abs(hl - hr) / (hl + hr + 10) > 0.3
                ):
                    continue

                left_center_x = left_ball_list[i][0] + left_ball_list[i][2] / 2
                left_center_y = left_ball_list[i][1] + left_ball_list[i][3] / 2

                right_center_x = right_ball_list[j][0] + right_ball_list[j][2] / 2
                right_center_y = right_ball_list[j][1] + right_ball_list[j][3] / 2
                dual_ball_loc = utils.ball_estimation_by_cv2(
                    [left_center_x, left_center_y],
                    [right_center_x, right_center_y],
                    cam_l,
                    cam_r,
                )

                # 只判断z在一定范围的球
                if dual_ball_loc[2] < 0.5 or dual_ball_loc[2] > MAX_DEPTH:
                    continue

                # 计算这个位置下的重投影误差，左右都差不多就认为匹配上了
                error_l = self.calc_error_after_translation_test(
                    dual_ball_loc.copy(), [left_center_x, left_center_y], cam_l
                )
                error_r = self.calc_error_after_translation_test(
                    dual_ball_loc.copy(), [right_center_x, right_center_y], cam_r
                )

                # 近距离的球，有可能重投影误差被放大，因此乘上距离作为总的误差。
                # 1218在mark6 ar0822上测得，2m处约有100误差。
                project_error = (error_l + error_r) * math.sqrt(dual_ball_loc[2]) * 2
                # print(f"loc_result:loc_result, {project_error}, error_l: {error_l}, error_r: {error_r}")
                if project_error < 200:  # 左右匹配上了
                    x, y, w, h = left_ball_list[i]
                    pixel_width = (w + h) / 2
                    pixel_width_world = f * self.ball_size / dual_ball_loc[2]
                    # print(H_ball,'  ', loc_result[1], '   ', pixel_width_world,'   ', pixel_width)
                    pixel_gap = abs(pixel_width_world - pixel_width)
                    pixel_error_rate = pixel_gap / (
                        (pixel_width + pixel_width_world) * 0.5
                    )

                    loc_gap = np.sqrt(
                        (dual_ball_loc[0] - predict_ball_loc[0]) ** 2
                        + (dual_ball_loc[1] - predict_ball_loc[1]) ** 2
                        + (dual_ball_loc[2] - predict_ball_loc[2]) ** 2
                    )
                    # if abs(left_h_ball - dual_ball_loc[1]) < 0.3 and pixel_error_rate < 0.3: # 大小也匹配

                    if (
                        pixel_error_rate < 0.3 and loc_gap < best_loc_gap
                    ):  # 大小匹配，且距离最近
                        best_result = [
                            [left_center_x, left_center_y],
                            [right_center_x, right_center_y],
                            left_ball_list[i],
                            right_ball_list[j],
                            [project_error, pixel_error_rate],
                            dual_ball_loc,
                            0.5 * (left_center_y + right_center_y),
                        ]
                        best_loc_gap = loc_gap
        if best_result is not None:
            filter_ball.append(best_result)
        return filter_ball
