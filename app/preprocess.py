"""预处理: 亮度归一化(CLAHE/gamma)、ROI 裁剪。

保持 BGR uint8 输入输出,并记录坐标偏移以便检测结果映射回原图 (v1.0 §6.3)。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PreInfo:
    """记录本轮预处理的几何变换,供检测框映射回原图。"""
    offset_x: int = 0
    offset_y: int = 0
    crop_size: Optional[Tuple[int, int]] = None  # (w, h) 裁剪后尺寸;None=无裁剪


def normalize_brightness(img: np.ndarray, clip_limit: float = 2.0,
                         tile_grid: int = 8) -> np.ndarray:
    """对 L 通道做 CLAHE,减轻 iPhone 自动曝光波动。BGR in/out。"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_Lab2BGR)


def roi_is_degenerate(roi_norm: List[List[float]]) -> bool:
    """零面积/坐标不合法的 ROI(如所有点 x 相同)会让整幅图被涂黑,必须拦截。"""
    if len(roi_norm) < 3:
        return True
    for p in roi_norm:
        if not (0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0):
            return True
    xs = [p[0] for p in roi_norm]
    ys = [p[1] for p in roi_norm]
    return max(xs) - min(xs) < 1e-3 or max(ys) - min(ys) < 1e-3


def apply_roi(img: np.ndarray, roi_norm: Optional[List[List[float]]]) -> Tuple[np.ndarray, PreInfo]:
    """按归一化多边形/矩形 ROI 处理。

    传入矩形或多边形:区域外像素置黑,不裁剪尺寸(坐标系不变,PreInfo 全零);
    多边形为空时仅返回原图。
    """
    info = PreInfo()
    if not roi_norm:
        return img, info
    if roi_is_degenerate(roi_norm):
        logger.warning("ROI 无效(零面积或坐标越界),已忽略: %s", roi_norm)
        return img, info
    h, w = img.shape[:2]
    pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in roi_norm], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    out = cv2.bitwise_and(img, img, mask=mask)
    return out, info


def map_boxes_to_original(boxes: np.ndarray, info: PreInfo) -> np.ndarray:
    """把检测结果从预处理坐标系映射回原图坐标系。

    当前 ROI 方案不裁剪,为恒等映射;若未来改为裁剪方案,
    在此统一加上 offset 并按 crop 尺寸校验。
    """
    if boxes.size == 0:
        return boxes
    out = boxes.copy()
    out[:, [0, 2]] += info.offset_x
    out[:, [1, 3]] += info.offset_y
    return out


class Preprocessor:
    """按配置执行: 亮度归一化(可选) -> ROI(可选)。"""

    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, img: np.ndarray) -> Tuple[np.ndarray, PreInfo]:
        if self.cfg.preprocess.normalize_brightness:
            img = normalize_brightness(img)
        return apply_roi(img, self.cfg.preprocess.roi)
