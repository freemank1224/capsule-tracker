"""onnxruntime 检测后端 (v1.2 §2.5/§3.1)。

- Provider 优先级: CUDAExecutionProvider → CPUExecutionProvider(已删除 DML)。
- CUDAProvider 不可用时自动回退 CPU 并 warn。
- 需与 YOLO 导出格式匹配的 letterbox 预处理和 NMS 后处理。
注意: onnxruntime 与 onnxruntime-gpu 不可共存。
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from .base import Detector, nms_numpy

logger = logging.getLogger(__name__)

PROVIDER_PRIORITY = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def letterbox(img: np.ndarray, new_shape: int,
              color: tuple = (114, 114, 114)) -> tuple[np.ndarray, float, tuple[int, int]]:
    """等比缩放 + 填充到正方形;返回 (图像, 缩放比, 填充量(dw, dh))。"""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = round(h * r), round(w * r)
    if (nh, nw) != (h, w):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (new_shape - nw) / 2.0, (new_shape - nh) / 2.0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


class OnnxDetector(Detector):
    name = "onnx"

    def __init__(self, cfg, warn_cb=None):
        import onnxruntime as ort

        self.cfg = cfg
        self.warn_cb = warn_cb or (lambda msg: None)
        self.imgsz = cfg.imgsz

        avail = list(ort.get_available_providers())
        chosen = [p for p in PROVIDER_PRIORITY if p in avail]
        if "CUDAExecutionProvider" not in chosen and "CUDAExecutionProvider" in PROVIDER_PRIORITY:
            logger.warning("CUDAExecutionProvider 不可用(可用: %s),onnx_det 回退 CPU 并发",
                           avail)
        self.session = ort.InferenceSession(cfg.weights, providers=chosen or avail)
        self.input_name = self.session.get_inputs()[0].name
        in_shape = self.session.get_inputs()[0].shape
        # 记录模型期望的输入尺寸(若为动态则用配置 imgsz)
        self.input_hw = (in_shape[2], in_shape[3]) if isinstance(
            in_shape[2], int) and isinstance(in_shape[3], int) else (self.imgsz, self.imgsz)
        logger.info("onnx 后端: weights=%s providers=%s",
                    cfg.weights, self.session.get_providers())

    def warmup(self, shape: tuple[int, int]) -> None:
        dummy = np.zeros((self.input_hw[0], self.input_hw[1], 3), dtype=np.uint8)
        for _ in range(3):
            self._infer(dummy)

    def detect(self, img_bgr: np.ndarray) -> np.ndarray:
        raw = self._infer(img_bgr)
        return self._postprocess(raw, img_bgr.shape[1], img_bgr.shape[0])

    # ---- 内部 ----
    def _infer(self, img_bgr: np.ndarray) -> np.ndarray:
        lb, r, (dw, dh) = letterbox(img_bgr, self.input_hw[1])
        blob = cv2.dnn.blobFromImage(lb, 1.0 / 255.0, swapRB=True, crop=False)
        out = self.session.run(None, {self.input_name: blob})[0]
        self._meta = (r, dw, dh)
        return out

    def _postprocess(self, raw: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
        r, dw, dh = getattr(self, "_meta", (1.0, 0.0, 0.0))
        pred = raw[0] if raw.ndim == 3 else raw
        # YOLOv8 导出格式: (4+nc, M) 需转置;(M, 4+nc) 直接用
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T
        boxes_cxcywh = pred[:, :4]
        class_scores = pred[:, 4:]
        cls_ids = class_scores.argmax(axis=1)
        confs = class_scores.max(axis=1)
        mask = confs >= self.cfg.detector.conf
        if not mask.any():
            return np.zeros((0, 6), dtype=np.float32)
        b = boxes_cxcywh[mask]
        c = confs[mask]
        k = cls_ids[mask]
        # cxcywh(模型输入像素) -> x1y1x2y2 -> 去填充/缩放回原图
        x1 = (b[:, 0] - b[:, 2] / 2 - dw) / r
        y1 = (b[:, 1] - b[:, 3] / 2 - dh) / r
        x2 = (b[:, 0] + b[:, 2] / 2 - dw) / r
        y2 = (b[:, 1] + b[:, 3] / 2 - dh) / r
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h)
        keep = nms_numpy(boxes, c, self.cfg.detector.iou, self.cfg.detector.max_det)
        keep = keep[: self.cfg.detector.max_det]
        out = np.concatenate(
            [boxes[keep], c[keep][:, None], k[keep][:, None].astype(np.float32)], axis=1)
        return out.astype(np.float32)
