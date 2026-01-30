"""网球检测推理封装（高内聚：只管模型加载与输出解析）。

注意：
- .rknn 推理通常需要 Rockchip 设备运行时（rknnlite）或 Linux 工具链（rknn-toolkit2）。
- Windows 环境大概率无法直接跑 .rknn。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tennis3d.adapters.rknn_runtime import load_rknn_runtime
from tennis3d.models import Detection
from tennis3d.preprocess import letterbox


class TennisDetector:
    """使用 RKNN/RKNNLite 运行 .rknn 模型并解析出 bbox。

        当前支持两类常见输出：
        1) “常见 YOLO NMS 后输出”
             - 单输出，形状 (N, 6) 或 (1, N, 6)
                 每行: [x1, y1, x2, y2, score, cls]
        2) “YOLOv8-DFL 多输出”（常见于 seg/pose 等头，可能 len(outputs)==13）
             - 采用全图最大 score 的位置解码出单个 bbox（适合单球场景）

    如果你的模型输出不同，可使用 --dump-raw-outputs 输出 raw shape/min/max 后再适配解析。
    """

    def __init__(
        self,
        model_path: Path,
        input_size: int = 640,
        conf_thres: float = 0.25,
        rgb: bool = True,
    ):
        self.model_path = model_path
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.rgb = rgb

        # 运行时加载被拆到独立模块，避免 detector.py 混入过多 SDK/环境处理细节。
        self._rknn = load_rknn_runtime(self.model_path)

    def infer_raw(self, img_bgr: np.ndarray) -> list[np.ndarray]:
        img = img_bgr
        if self.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_lb, _, _ = letterbox(img, (self.input_size, self.input_size))
        inp = np.expand_dims(img_lb, axis=0)

        outputs = self._rknn.inference(inputs=[inp])
        return [np.array(o) for o in outputs]

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        raw_outputs = self.infer_raw(img_bgr)
        dets = self._parse_outputs(raw_outputs)
        return [d for d in dets if d.score >= self.conf_thres]

    def _parse_outputs(self, outputs: list[np.ndarray]) -> list[Detection]:
        """解析 RKNN 输出。

        解析策略：
        1) 先尝试“常见 NMS 后输出”(N,6)/(1,N,6)
        2) 若不匹配，再尝试 YOLOv8-DFL 多输出（常见于 seg/pose 等头）
        """

        dets = self._parse_common_nms_output(outputs)
        if dets:
            return dets
        return self._parse_yolov8_dfl_multi_output(outputs)

    @staticmethod
    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    def _parse_yolov8_dfl_multi_output(self, outputs: list[np.ndarray]) -> list[Detection]:
        """解析 YOLOv8 风格的 DFL 头输出。

        适配特征（参考 examples/reference_rknn/ref_Inference_seg.py 的说明）：
        - len(outputs) == 13
        - 3 个尺度分支，每个分支 4 个输出：
          * boxes: (1, 4*reg_max, H, W)
          * scores: (1, C, H, W)
          * score_sum: (1, 1, H, W)  # 可能存在但不一定使用
          * mask_weights: (1, 32, H, W)  # seg 任务常见
        - 最后一个输出通常是原型： (1, 32, 160, 160)

        离线网球检测一般每张图只有一个球：这里采取“全图取最大 score”的方式解码出单个 bbox。
        """

        if len(outputs) != 13:
            return []

        pair_per_branch = 4
        branches = 3

        best = None
        best_score = float("-inf")

        for i in range(branches):
            scores = outputs[pair_per_branch * i + 1]
            if scores.ndim != 4 or scores.shape[0] != 1:
                return []

            h = scores.shape[2]
            w = scores.shape[3]

            flat = scores.reshape(-1)
            flat_idx = int(np.argmax(flat))
            raw_score = float(flat[flat_idx])

            # 兼容 logits：如果超出 [0,1] 范围太多，尝试做一次 sigmoid
            score = raw_score
            if raw_score > 1.5 or raw_score < -0.5:
                score = self._sigmoid(raw_score)

            if score > best_score:
                per_cls = h * w
                cls = flat_idx // per_cls
                rem = flat_idx % per_cls
                idx_h = rem // w
                idx_w = rem % w
                best_score = score
                best = (i, int(cls), int(idx_h), int(idx_w), int(h), int(w))

        if best is None or best_score < self.conf_thres:
            return []

        branch_i, cls, idx_h, idx_w, h, w = best
        boxes = outputs[pair_per_branch * branch_i]
        if boxes.ndim != 4 or boxes.shape[0] != 1:
            return []

        reg_ch = int(boxes.shape[1])
        if reg_ch % 4 != 0:
            return []
        reg_max = reg_ch // 4
        if reg_max <= 1:
            return []

        vec = boxes[0, :, idx_h, idx_w].astype(np.float32)
        vec = vec.reshape(4, reg_max)

        # DFL：softmax 后求期望
        vec = vec - np.max(vec, axis=1, keepdims=True)
        exp = np.exp(vec)
        prob = exp / np.sum(exp, axis=1, keepdims=True)
        bins = np.arange(reg_max, dtype=np.float32)
        dists = (prob * bins[None, :]).sum(axis=1)

        left, top, right, bottom = map(float, dists.tolist())

        stride_x = float(self.input_size) / float(w)
        stride_y = float(self.input_size) / float(h)

        # 与 Inference_seg.box_process 一致：grid + 0.5，再乘 stride
        cx = (float(idx_w) + 0.5)
        cy = (float(idx_h) + 0.5)

        x1 = (cx - left) * stride_x
        y1 = (cy - top) * stride_y
        x2 = (cx + right) * stride_x
        y2 = (cy + bottom) * stride_y

        return [
            Detection(
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                score=float(best_score),
                cls=int(cls),
            )
        ]

    def _parse_common_nms_output(self, outputs: list[np.ndarray]) -> list[Detection]:
        if not outputs:
            return []

        out = outputs[0]
        if out.ndim == 3 and out.shape[0] == 1:
            out = out[0]

        if out.ndim != 2 or out.shape[1] < 6:
            return []

        rows: list[Detection] = []
        for row in out:
            x1, y1, x2, y2, score, cls = row[:6]
            if score < 0:
                continue

            # 如果是归一化坐标，放大到输入尺度
            if max(x1, y1, x2, y2) <= 1.5:
                x1 *= self.input_size
                y1 *= self.input_size
                x2 *= self.input_size
                y2 *= self.input_size

            rows.append(
                Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    score=float(score),
                    cls=int(cls),
                )
            )

        return rows
