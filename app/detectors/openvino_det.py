"""OpenVINO 检测后端(可选): 仅在 openvino 已安装时启用 (v1.0 §6.1)。

接受 .xml / .onnx 模型路径;自动选择设备(CPU/GPU/AUTO)。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .base import Detector, nms_numpy
from .onnx_det import letterbox

logger = logging.getLogger(__name__)


class OpenvinoDetector(Detector):
    name = "openvino"

    def __init__(self, cfg, warn_cb=None):
        import openvino as ov  # 延迟导入

        self.cfg = cfg
        self.warn_cb = warn_cb or (lambda msg: None)
        self.imgsz = cfg.imgsz
        core = ov.Core()
        device = "AUTO"
        if "GPU" in core.available_devices:
            device = "GPU"
        weights = Path(cfg.weights)
        model = core.read_model(str(weights))
        self.compiled = core.compile_model(model, device)
        self.infer = self.compiled.create_infer_request()
        p = self.compiled.input(0)
        ph = int(p.shape[2]) if isinstance(p.shape[2], int) else self.imgsz
        pw = int(p.shape[3]) if isinstance(p.shape[3], int) else self.imgsz
        self.input_hw = (ph, pw)
        logger.info("openvino 后端: weights=%s device=%s", cfg.weights, device)

    def warmup(self, shape: tuple[int, int]) -> None:
        dummy = np.zeros((self.input_hw[0], self.input_hw[1], 3), dtype=np.uint8)
        for _ in range(3):
            self._infer(dummy)

    def detect(self, img_bgr: np.ndarray) -> np.ndarray:
        raw = self._infer(img_bgr)
        return self._postprocess(raw, img_bgr.shape[1], img_bgr.shape[0])

    def _infer(self, img_bgr: np.ndarray) -> np.ndarray:
        lb, r, (dw, dh) = letterbox(img_bgr, self.input_hw[1])
        import cv2
        blob = cv2.dnn.blobFromImage(lb, 1.0 / 255.0, swapRB=True, crop=False)
        out = self.infer.infer(blob)
        self._meta = (r, dw, dh)
        return np.asarray(list(out.values())[0])

    def _postprocess(self, raw: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
        # 与 onnx_det 相同的解析逻辑
        r, dw, dh = self._meta
        pred = raw[0] if raw.ndim == 3 else raw
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T
        cls_scores = pred[:, 4:]
        cls_ids = cls_scores.argmax(axis=1)
        confs = cls_scores.max(axis=1)
        mask = confs >= self.cfg.detector.conf
        if not mask.any():
            return np.zeros((0, 6), dtype=np.float32)
        b = pred[mask][:, :4]
        c = confs[mask]
        k = cls_ids[mask]
        x1 = (b[:, 0] - b[:, 2] / 2 - dw) / r
        y1 = (b[:, 1] - b[:, 3] / 2 - dh) / r
        x2 = (b[:, 0] + b[:, 2] / 2 - dw) / r
        y2 = (b[:, 1] + b[:, 3] / 2 - dh) / r
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h)
        keep = nms_numpy(boxes, c, self.cfg.detector.iou, self.cfg.detector.max_det)
        return np.concatenate(
            [boxes[keep], c[keep][:, None], k[keep][:, None].astype(np.float32)],
            axis=1).astype(np.float32)
