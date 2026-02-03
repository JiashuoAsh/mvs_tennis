"""软件裁剪（动态 ROI）控制。

背景：
- 相机侧 ROI（Width/Height/OffsetX/OffsetY）在 MVS 中通常建议在 StartGrabbing 前配置。
  运行中动态改 offset 可能受限（节点锁定/需要停流重启/时延抖动）。
- 为了在保持相机输出不变的前提下，降低 detector 的输入分辨率，同时不丢球，
  这里提供“软件裁剪 + 坐标回写”的一条链路：
  1) 在 detector 前把图像裁成较小窗口
  2) detector 输出 bbox 在裁剪坐标系
  3) 把 bbox 加回裁剪窗口的 offset，使其回到原图像素坐标系

本文件只提供：
- 一个可被 pipeline/core 调用的最小接口 RoiController
- 一个基于 3D 结果做简单运动学外推的实现 KinematicRoiController

说明：
- 该实现是“闭环”的：第 k 帧裁剪使用 k-1 帧的 3D 输出外推到当前时刻。
- 它不依赖 curve stage（曲线拟合），但可作为后续接入 curve 预测的基础。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from tennis3d.geometry.calibration import CalibrationSet
from tennis3d.geometry.triangulation import project_point
from tennis3d.preprocess import crop_bgr


class RoiController(Protocol):
    """最小 ROI 控制器接口。

    约定：
    - preprocess_for_detection 在调用 detector 前被执行。
    - update_after_output 在每组输出生成后被执行，用于更新下一组的预测状态。
    """

    def preprocess_for_detection(
        self,
        *,
        meta: dict[str, Any],
        camera: str,
        img_bgr: np.ndarray,
        calib: CalibrationSet,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """返回用于 detector 的图像，以及 (offset_x, offset_y)。"""

        ...

    def update_after_output(self, *, out_rec: dict[str, Any], calib: CalibrationSet) -> None:
        """用本组输出更新内部状态（供后续组使用）。"""

        ...


@dataclass
class KinematicRoiConfig:
    """动态裁剪配置。"""

    crop_width: int = 640
    crop_height: int = 640

    # 平滑：0 表示完全跟随新值，1 表示完全不动。
    # 这里使用“指数平滑”的直觉：alpha 越大越稳，但跟随越慢。
    smooth_alpha: float = 0.2

    # 每组最大移动像素（避免抖动/误检导致 ROI 瞬移）。0 表示不限制。
    max_step_px: int = 120

    # 连续多少组没有球则重置预测（回到居中裁剪）。
    reset_after_missed: int = 8

    # 允许使用的时间字段；按顺序尝试。
    time_keys: tuple[str, ...] = ("capture_t_abs", "group_index", "frame_id")


class KinematicRoiController:
    """基于 3D 运动学外推的动态裁剪控制器。

    设计：
    - 使用 out_rec['balls'][0]['ball_3d_world'] 作为单球主目标。
    - 用相邻两次 3D 观测估计速度 v（常速度模型）。
    - 在新组到来时，根据 meta 的时间戳把上一帧状态外推到当前时刻，
      并投影到各相机平面作为裁剪中心。

    注意：
    - 这是“极简可用”的实现：它不做多目标跟踪，也不做复杂滤波。
    - 若需要更强鲁棒性，可在此基础上加入 Kalman/粒子滤波或 curve stage 预测。
    """

    def __init__(self, *, cfg: KinematicRoiConfig) -> None:
        self._cfg = cfg

        self._last_t: float | None = None
        self._last_X_w: np.ndarray | None = None
        self._last_v_w: np.ndarray | None = None

        self._missed = 0

        # 记录上一组每个相机的 2D 观测（用于 3D 投影失败时回退）。
        self._last_uv_by_camera: dict[str, tuple[float, float]] = {}

        # 记录每相机当前 ROI 左上角，用于平滑与限幅。
        self._origin_xy_by_camera: dict[str, tuple[int, int]] = {}

    def preprocess_for_detection(
        self,
        *,
        meta: dict[str, Any],
        camera: str,
        img_bgr: np.ndarray,
        calib: CalibrationSet,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        cfg = self._cfg
        if cfg.crop_width <= 0 or cfg.crop_height <= 0:
            return img_bgr, (0, 0)

        if img_bgr is None or img_bgr.size == 0:
            return img_bgr, (0, 0)

        h, w = int(img_bgr.shape[0]), int(img_bgr.shape[1])
        cw = int(max(1, min(int(cfg.crop_width), w)))
        ch = int(max(1, min(int(cfg.crop_height), h)))

        t_now = _parse_time(meta, keys=cfg.time_keys)

        uv = self._predict_uv(camera=str(camera), t_now=t_now, calib=calib)
        if uv is None:
            # 关键策略：在“尚未锁定到第一颗球”之前，不做裁剪。
            # 原因：中心裁剪很可能直接把球裁掉，导致永远无法进入闭环。
            # 一旦上游定位到球，update_after_output 会提供 3D/2D 观测，本控制器才开始跟随裁剪。
            if self._last_X_w is None and str(camera) not in self._last_uv_by_camera:
                return img_bgr, (0, 0)

            # 回退：已有上一帧 2D/3D 但本次投影失败时，退回到居中裁剪。
            u = 0.5 * float(w)
            v = 0.5 * float(h)
        else:
            u, v = float(uv[0]), float(uv[1])

        target_x0 = int(round(u - 0.5 * float(cw)))
        target_y0 = int(round(v - 0.5 * float(ch)))

        target_x0, target_y0 = _clamp_origin(
            x0=target_x0,
            y0=target_y0,
            crop_w=cw,
            crop_h=ch,
            img_w=w,
            img_h=h,
        )

        prev = self._origin_xy_by_camera.get(str(camera))
        x0, y0 = target_x0, target_y0
        if prev is not None:
            px0, py0 = int(prev[0]), int(prev[1])

            # 平滑：越靠近 1 越稳。
            alpha = float(cfg.smooth_alpha)
            alpha = float(max(0.0, min(1.0, alpha)))
            x0 = int(round(px0 + (1.0 - alpha) * float(x0 - px0)))
            y0 = int(round(py0 + (1.0 - alpha) * float(y0 - py0)))

            # 限速：避免 ROI 瞬移。
            step = int(cfg.max_step_px)
            if step > 0:
                dx = int(max(-step, min(step, x0 - px0)))
                dy = int(max(-step, min(step, y0 - py0)))
                x0 = int(px0 + dx)
                y0 = int(py0 + dy)

            x0, y0 = _clamp_origin(
                x0=x0,
                y0=y0,
                crop_w=cw,
                crop_h=ch,
                img_w=w,
                img_h=h,
            )

        self._origin_xy_by_camera[str(camera)] = (int(x0), int(y0))

        cropped, (ox, oy) = crop_bgr(
            img_bgr,
            crop_width=cw,
            crop_height=ch,
            offset_x=int(x0),
            offset_y=int(y0),
        )
        return cropped, (int(ox), int(oy))

    def update_after_output(self, *, out_rec: dict[str, Any], calib: CalibrationSet) -> None:
        cfg = self._cfg

        # 取“最佳球”（按质量排序的第一个）。
        balls = out_rec.get("balls") or []
        best = balls[0] if isinstance(balls, list) and balls and isinstance(balls[0], dict) else None

        if best is None:
            self._missed += 1
            if int(cfg.reset_after_missed) > 0 and self._missed >= int(cfg.reset_after_missed):
                self._reset_prediction_state()
            return

        X = best.get("ball_3d_world")
        if not (isinstance(X, (list, tuple)) and len(X) == 3):
            self._missed += 1
            if int(cfg.reset_after_missed) > 0 and self._missed >= int(cfg.reset_after_missed):
                self._reset_prediction_state()
            return

        try:
            X_w = np.array([float(X[0]), float(X[1]), float(X[2])], dtype=np.float64)
        except Exception:
            self._missed += 1
            if int(cfg.reset_after_missed) > 0 and self._missed >= int(cfg.reset_after_missed):
                self._reset_prediction_state()
            return

        t_now = _parse_time(out_rec, keys=cfg.time_keys)

        # 更新速度估计。
        if t_now is not None and self._last_t is not None and self._last_X_w is not None:
            dt = float(t_now) - float(self._last_t)
            if dt > 1e-6 and np.isfinite(dt):
                v = (X_w.reshape(3) - self._last_X_w.reshape(3)) / dt
                if np.all(np.isfinite(v)):
                    self._last_v_w = v.astype(np.float64)

        self._last_t = float(t_now) if t_now is not None else None
        self._last_X_w = X_w
        self._missed = 0

        # 存一份 2D 观测作为回退（以免某些相机 P/投影异常）。
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
                    self._last_uv_by_camera[str(cam)] = (u, v)

    def _predict_uv(
        self,
        *,
        camera: str,
        t_now: float | None,
        calib: CalibrationSet,
    ) -> tuple[float, float] | None:
        # 优先：用 3D 外推 + 投影。
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

        # 回退：上一帧 2D 观测。
        return self._last_uv_by_camera.get(str(camera))

    def _reset_prediction_state(self) -> None:
        # 说明：清空 3D/2D 预测，让裁剪回到“居中 + 平滑”。
        self._last_t = None
        self._last_X_w = None
        self._last_v_w = None
        self._last_uv_by_camera.clear()
        self._missed = 0


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


def _clamp_origin(
    *,
    x0: int,
    y0: int,
    crop_w: int,
    crop_h: int,
    img_w: int,
    img_h: int,
) -> tuple[int, int]:
    crop_w = int(max(1, min(int(crop_w), int(img_w))))
    crop_h = int(max(1, min(int(crop_h), int(img_h))))

    x0 = int(max(0, min(int(x0), int(img_w) - crop_w)))
    y0 = int(max(0, min(int(y0), int(img_h) - crop_h)))
    return int(x0), int(y0)
