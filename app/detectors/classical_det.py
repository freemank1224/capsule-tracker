"""传统背景差分检测后端: 无需模型,用于背景干净时的超低算力模式与联调 (v1.0 §6.1)。

流程: 灰度 → 与背景模型差分(首帧/空托盘标定帧) → 阈值 → 形态学开闭
→ findContours → 面积过滤 → 外接框。
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import Detector


class ClassicalDetector(Detector):
    name = "classical"

    def __init__(self, cfg):
        self.cfg = cfg
        cc = cfg.detector.classical
        self.diff_thresh = cc.diff_thresh
        self.min_area = cc.min_area
        self.max_area = cc.max_area
        # 闭运算核需覆盖 JPEG 8x8 块效应尺度(合并碎片),开运算用小核去噪点
        self.kernel = np.ones((cc.morph_kernel, cc.morph_kernel), np.uint8)
        self.small_kernel = np.ones((max(3, cc.morph_kernel // 3),) * 2, np.uint8)
        self._bg: np.ndarray | None = None

    def warmup(self, shape: tuple[int, int]) -> None:
        pass

    def calibrate(self, img_bgr: np.ndarray) -> None:
        """用"空托盘"帧标定背景模型。"""
        self._bg = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def detect(self, img_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if self._bg is None:
            # 首帧作为背景,本帧不输出检测
            self.calibrate(img_bgr)
            return np.zeros((0, 6), dtype=np.float32)
        diff = np.abs(gray.astype(np.float32) - self._bg)
        binary = (diff > self.diff_thresh).astype(np.uint8) * 255
        # 先闭后开: 闭运算合并块效应碎片,再用小核开运算去掉孤立噪点
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.small_kernel, iterations=1)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: list[list[float]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            out.append([x, y, x + w, y + h, 1.0, 0.0])
        if not out:
            return np.zeros((0, 6), dtype=np.float32)
        return np.array(out, dtype=np.float32)
