"""两级 ROI：相机侧 AOI（OffsetX/OffsetY）+ 主机侧软件裁剪（detector crop）。

目标：
- 相机侧 AOI：降低带宽/解码压力（输出图像更小）。
- 主机侧软件裁剪：降低 detector 的输入分辨率（进一步提速）。
- 坐标系一致：detector 输出 bbox 一律回写到“满幅像素坐标系”，以匹配满幅标定内参。

重要约束：
- 当相机 ROI 的 OffsetX/OffsetY 在运行中发生变化时，**不能**使用
  `apply_sensor_roi_to_calibration()` 做一次性主点平移；因为 offset 是逐组变化的。
  解决方案是：在 pipeline/core 里对每张图像返回 (total_offset_x, total_offset_y)，
  让检测结果逐帧回写到满幅坐标。

本模块只实现控制逻辑，不负责加载 MVS binding。
在线入口（tennis3d.apps.online_mvs_localize）会把相机句柄的 set_offset 方法注入进来。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

from mvs.runtime_roi import IntNodeInfo, clamp_and_align
from tennis3d.geometry.calibration import CalibrationSet
from tennis3d.geometry.triangulation import project_point
from tennis3d.preprocess import crop_bgr
from tennis3d.pipeline.roi import RoiController


class CameraOffsetApplier(Protocol):
    """把 (OffsetX, OffsetY) 写入相机的抽象接口。

    说明：
    - 之所以抽象，是为了让控制逻辑可单测（无需真相机）。
    - 在线模式下会注入一个实现：对 MVS 相机调用 SetIntValue。
    """

    def set_offsets(self, *, camera: str, offset_x: int, offset_y: int) -> bool:  # pragma: no cover
        ...


@dataclass
class SoftwareCropConfig:
    """主机侧软件裁剪配置。"""

    crop_width: int = 640
    crop_height: int = 640

    # 平滑：0 表示完全跟随新值，1 表示完全不动。
    smooth_alpha: float = 0.2

    # 每组最大移动像素（避免抖动/误检导致 ROI 瞬移）。0 表示不限制。
    max_step_px: int = 120

    # 连续多少组没有球则重置预测（回到“不过度裁剪”的状态）。
    reset_after_missed: int = 8

    # 允许使用的时间字段；按顺序尝试。
    time_keys: tuple[str, ...] = ("capture_t_abs", "group_index", "frame_id")


@dataclass
class CameraAoiRuntimeConfig:
    """相机侧 AOI 运行中平移配置。"""

    enabled: bool = False

    # 每隔多少组才尝试更新一次（降低写节点频率，减少抖动/开销）。
    update_every_groups: int = 2

    # 小于该像素的变化不更新（避免频繁写微小抖动）。
    min_move_px: int = 8

    # 平滑：0 立即跟随，1 完全不动。
    smooth_alpha: float = 0.3

    # 每次更新最大移动像素（限速）。0 表示不限制。
    max_step_px: int = 160

    # 连续多少组无球后，开始把 AOI 缓慢拉回初始位置（0 表示禁用 recenter）。
    recenter_after_missed: int = 30


@dataclass
class CameraAoiState:
    """单相机 AOI 状态（以满幅坐标系为基准）。"""

    aoi_width: int
    aoi_height: int

    offset_x: int
    offset_y: int

    # 写入对齐与 clamp 约束。
    offset_x_info: IntNodeInfo
    offset_y_info: IntNodeInfo

    # 用于 recenter 的初始值。
    initial_offset_x: int
    initial_offset_y: int


def _parse_time(meta: dict[str, Any], *, keys: tuple[str, ...]) -> float | None:
    for k in keys:
        if k not in meta:
            continue
        v = meta.get(k)
        if v is None:
            continue
        try:
            t = float(v)
        except Exception:
            continue
        if np.isfinite(t):
            return float(t)
    return None


def _clamp_origin(*, x0: int, y0: int, crop_w: int, crop_h: int, img_w: int, img_h: int) -> tuple[int, int]:
    crop_w = int(max(1, min(int(crop_w), int(img_w))))
    crop_h = int(max(1, min(int(crop_h), int(img_h))))

    x0 = int(max(0, min(int(x0), int(img_w) - crop_w)))
    y0 = int(max(0, min(int(y0), int(img_h) - crop_h)))
    return int(x0), int(y0)


def _smooth_and_limit(*, prev: int, target: int, alpha: float, max_step: int) -> int:
    """一维平滑+限速。"""

    alpha = float(max(0.0, min(1.0, float(alpha))))
    v = int(round(int(prev) + (1.0 - alpha) * float(int(target) - int(prev))))

    step = int(max_step)
    if step > 0:
        dv = int(v - int(prev))
        if dv > step:
            v = int(prev + step)
        elif dv < -step:
            v = int(prev - step)

    return int(v)


class TwoStageKinematicRoiController(RoiController):
    """两级 ROI 控制器。

    - preprocess_for_detection：在当前 AOI 图像上做软件裁剪；同时返回 total_offset（相机 offset + 裁剪 offset）。
    - update_after_output：用当前组的 3D 输出更新状态；可选更新相机 OffsetX/OffsetY（用于下一组）。

    关键策略：
    - 未锁定到第一颗球之前，不做软件裁剪（避免闭环自锁死）。
    - 相机 AOI 的更新频率可控（update_every_groups + min_move_px），避免过度写节点。
    """

    def __init__(
        self,
        *,
        crop_cfg: SoftwareCropConfig,
        camera_cfg: CameraAoiRuntimeConfig,
        aoi_state_by_camera: dict[str, CameraAoiState],
        applier: CameraOffsetApplier | None = None,
    ) -> None:
        self._crop_cfg = crop_cfg
        self._camera_cfg = camera_cfg
        self._aoi_state_by_camera = dict(aoi_state_by_camera)
        self._applier = applier

        # 3D 运动学状态。
        self._last_t: float | None = None
        self._last_X_w: np.ndarray | None = None
        self._last_v_w: np.ndarray | None = None
        self._missed = 0

        # 用于“下一组提前量”的 dt 估计（指数平滑）。
        self._dt_est: float | None = None

        # 回退：上一组每相机的 2D 观测（满幅坐标）。
        self._last_uv_full_by_camera: dict[str, tuple[float, float]] = {}

        # 软件裁剪窗口左上角（AOI 局部坐标）。
        self._crop_origin_local_by_camera: dict[str, tuple[int, int]] = {}

    def preprocess_for_detection(
        self,
        *,
        meta: dict[str, Any],
        camera: str,
        img_bgr: np.ndarray,
        calib: CalibrationSet,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        cam = str(camera)

        if img_bgr is None or img_bgr.size == 0:
            return img_bgr, (0, 0)

        st = self._aoi_state_by_camera.get(cam)
        cam_ox = int(st.offset_x) if st is not None else 0
        cam_oy = int(st.offset_y) if st is not None else 0

        cfg = self._crop_cfg
        if int(cfg.crop_width) <= 0 or int(cfg.crop_height) <= 0:
            # 即使不做软件裁剪，也必须把相机 AOI 的 offset 回写到满幅坐标。
            return img_bgr, (cam_ox, cam_oy)

        h, w = int(img_bgr.shape[0]), int(img_bgr.shape[1])
        cw = int(max(1, min(int(cfg.crop_width), w)))
        ch = int(max(1, min(int(cfg.crop_height), h)))

        t_now = _parse_time(meta, keys=cfg.time_keys)
        uv_full = self._predict_uv_full(camera=cam, t_now=t_now, calib=calib)
        if uv_full is None:
            # 未锁定前：不裁剪，避免永远找不到球。
            if self._last_X_w is None and cam not in self._last_uv_full_by_camera:
                return img_bgr, (cam_ox, cam_oy)

            # 回退：已有历史但本次预测失败时，退回到居中裁剪（在 AOI 局部坐标系）。
            u_local = 0.5 * float(w)
            v_local = 0.5 * float(h)
        else:
            u_local = float(uv_full[0]) - float(cam_ox)
            v_local = float(uv_full[1]) - float(cam_oy)

        target_x0 = int(round(u_local - 0.5 * float(cw)))
        target_y0 = int(round(v_local - 0.5 * float(ch)))

        target_x0, target_y0 = _clamp_origin(
            x0=target_x0,
            y0=target_y0,
            crop_w=cw,
            crop_h=ch,
            img_w=w,
            img_h=h,
        )

        prev = self._crop_origin_local_by_camera.get(cam)
        x0, y0 = target_x0, target_y0
        if prev is not None:
            px0, py0 = int(prev[0]), int(prev[1])
            x0 = _smooth_and_limit(prev=px0, target=x0, alpha=float(cfg.smooth_alpha), max_step=int(cfg.max_step_px))
            y0 = _smooth_and_limit(prev=py0, target=y0, alpha=float(cfg.smooth_alpha), max_step=int(cfg.max_step_px))
            x0, y0 = _clamp_origin(x0=x0, y0=y0, crop_w=cw, crop_h=ch, img_w=w, img_h=h)

        self._crop_origin_local_by_camera[cam] = (int(x0), int(y0))

        cropped, (ox_local, oy_local) = crop_bgr(
            img_bgr,
            crop_width=cw,
            crop_height=ch,
            offset_x=int(x0),
            offset_y=int(y0),
        )

        # total_offset：把“AOI 局部坐标系”回写到“满幅坐标系”。
        return cropped, (int(cam_ox + ox_local), int(cam_oy + oy_local))

    def update_after_output(self, *, out_rec: dict[str, Any], calib: CalibrationSet) -> None:
        crop_cfg = self._crop_cfg

        balls = out_rec.get("balls") or []
        best = balls[0] if isinstance(balls, list) and balls and isinstance(balls[0], dict) else None

        if best is None:
            self._missed += 1
            if int(crop_cfg.reset_after_missed) > 0 and self._missed >= int(crop_cfg.reset_after_missed):
                self._reset_prediction_state()
            self._maybe_update_camera_offsets(out_rec=out_rec, calib=calib, have_ball=False)
            return

        X = best.get("ball_3d_world")
        if not (isinstance(X, (list, tuple)) and len(X) == 3):
            self._missed += 1
            if int(crop_cfg.reset_after_missed) > 0 and self._missed >= int(crop_cfg.reset_after_missed):
                self._reset_prediction_state()
            self._maybe_update_camera_offsets(out_rec=out_rec, calib=calib, have_ball=False)
            return

        try:
            X_w = np.array([float(X[0]), float(X[1]), float(X[2])], dtype=np.float64)
        except Exception:
            self._missed += 1
            if int(crop_cfg.reset_after_missed) > 0 and self._missed >= int(crop_cfg.reset_after_missed):
                self._reset_prediction_state()
            self._maybe_update_camera_offsets(out_rec=out_rec, calib=calib, have_ball=False)
            return

        t_now = _parse_time(out_rec, keys=crop_cfg.time_keys)

        # 速度估计 + dt 估计（用于提前量）。
        if t_now is not None and self._last_t is not None and self._last_X_w is not None:
            dt = float(t_now) - float(self._last_t)
            if dt > 1e-6 and np.isfinite(dt):
                v = (X_w.reshape(3) - self._last_X_w.reshape(3)) / dt
                if np.all(np.isfinite(v)):
                    self._last_v_w = v.astype(np.float64)

                # dt_est：指数平滑，避免时间戳偶发抖动。
                if self._dt_est is None:
                    self._dt_est = float(dt)
                else:
                    self._dt_est = float(0.7 * float(self._dt_est) + 0.3 * float(dt))

        self._last_t = float(t_now) if t_now is not None else None
        self._last_X_w = X_w
        self._missed = 0

        # 2D 观测回退（满幅坐标）。
        obs = best.get("obs_2d_by_camera")
        if isinstance(obs, dict):
            for cam, o in obs.items():
                if not isinstance(o, dict):
                    continue
                uv = o.get("uv")
                if not (isinstance(uv, (list, tuple)) and len(uv) == 2):
                    continue
                try:
                    u, v = float(uv[0]), float(uv[1])
                except Exception:
                    continue
                if np.isfinite(u) and np.isfinite(v):
                    self._last_uv_full_by_camera[str(cam)] = (u, v)

        self._maybe_update_camera_offsets(out_rec=out_rec, calib=calib, have_ball=True)

    def _predict_uv_full(
        self,
        *,
        camera: str,
        t_now: float | None,
        calib: CalibrationSet,
    ) -> tuple[float, float] | None:
        # 优先：3D 外推 + 投影。
        if self._last_X_w is not None:
            X = self._last_X_w
            if t_now is not None and self._last_t is not None and self._last_v_w is not None:
                dt = float(t_now) - float(self._last_t)
                if not np.isfinite(dt) or dt < 0:
                    dt = 0.0
                X = (self._last_X_w.reshape(3) + self._last_v_w.reshape(3) * float(dt)).astype(np.float64)

            try:
                cam_calib = calib.require(str(camera))
            except KeyError:
                cam_calib = None

            if cam_calib is not None:
                uv = project_point(cam_calib.P, np.asarray(X, dtype=np.float64).reshape(3))
                u, v = float(uv[0]), float(uv[1])
                if np.isfinite(u) and np.isfinite(v):
                    return (u, v)

        # 回退：上一帧 2D 观测（满幅坐标）。
        return self._last_uv_full_by_camera.get(str(camera))

    def _maybe_update_camera_offsets(self, *, out_rec: dict[str, Any], calib: CalibrationSet, have_ball: bool) -> None:
        cam_cfg = self._camera_cfg
        if not bool(cam_cfg.enabled):
            return

        # 没有 applier 时，不做真实写入（但仍允许单测/逻辑验证）。
        if self._applier is None:
            return

        # 无 3D 状态时，无法驱动相机 AOI。
        if self._last_X_w is None:
            return

        # 降频：按 group_index 控制。
        group_index = out_rec.get("group_index")
        if group_index is not None:
            try:
                gi = int(group_index)
            except Exception:
                gi = None
            if gi is not None and int(cam_cfg.update_every_groups) > 1:
                if (gi % int(cam_cfg.update_every_groups)) != 0:
                    return

        # 目标时刻：略微提前到“下一组”，减少闭环滞后。
        t_now = _parse_time(out_rec, keys=self._crop_cfg.time_keys)
        t_target = t_now
        if t_target is not None and self._dt_est is not None and np.isfinite(self._dt_est):
            t_target = float(t_target) + float(max(0.0, float(self._dt_est)))

        # 无球时的 recenter：把 AOI 缓慢拉回初始 offset，增加重捕获机会。
        do_recenter = False
        if not have_ball and int(cam_cfg.recenter_after_missed) > 0 and self._missed >= int(cam_cfg.recenter_after_missed):
            do_recenter = True

        for cam, st in list(self._aoi_state_by_camera.items()):
            if int(st.aoi_width) <= 0 or int(st.aoi_height) <= 0:
                continue

            # 预测/回退得到满幅 uv。
            uv_full = self._predict_uv_full(camera=str(cam), t_now=t_target, calib=calib)
            if uv_full is None:
                continue

            if do_recenter:
                desired_ox = int(st.initial_offset_x)
                desired_oy = int(st.initial_offset_y)
            else:
                desired_ox = int(round(float(uv_full[0]) - 0.5 * float(st.aoi_width)))
                desired_oy = int(round(float(uv_full[1]) - 0.5 * float(st.aoi_height)))

            # 平滑 + 限速。
            next_ox = _smooth_and_limit(
                prev=int(st.offset_x),
                target=int(desired_ox),
                alpha=float(cam_cfg.smooth_alpha),
                max_step=int(cam_cfg.max_step_px),
            )
            next_oy = _smooth_and_limit(
                prev=int(st.offset_y),
                target=int(desired_oy),
                alpha=float(cam_cfg.smooth_alpha),
                max_step=int(cam_cfg.max_step_px),
            )

            # 按 Inc 对齐并 clamp。
            next_ox = clamp_and_align(next_ox, info=st.offset_x_info, mode="nearest")
            next_oy = clamp_and_align(next_oy, info=st.offset_y_info, mode="nearest")

            dx = int(next_ox - int(st.offset_x))
            dy = int(next_oy - int(st.offset_y))
            if max(abs(dx), abs(dy)) < int(cam_cfg.min_move_px):
                continue

            ok = False
            try:
                ok = bool(self._applier.set_offsets(camera=str(cam), offset_x=int(next_ox), offset_y=int(next_oy)))
            except Exception:
                ok = False

            if ok:
                # 写入成功才更新内部状态。
                self._aoi_state_by_camera[str(cam)] = CameraAoiState(
                    aoi_width=int(st.aoi_width),
                    aoi_height=int(st.aoi_height),
                    offset_x=int(next_ox),
                    offset_y=int(next_oy),
                    offset_x_info=st.offset_x_info,
                    offset_y_info=st.offset_y_info,
                    initial_offset_x=int(st.initial_offset_x),
                    initial_offset_y=int(st.initial_offset_y),
                )

    def _reset_prediction_state(self) -> None:
        # 清空 3D/2D 预测，让软件裁剪回到“不过度裁剪”的状态。
        self._last_t = None
        self._last_X_w = None
        self._last_v_w = None
        self._dt_est = None
        self._last_uv_full_by_camera.clear()
        self._missed = 0
