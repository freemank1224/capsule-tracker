"""Detector 抽象基类 + 小目标切片推理(tiling)包装 (v1.0 §6.1)。"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class Detector(ABC):
    """所有检测后端的统一接口。detect 返回 (N,6): x1,y1,x2,y2,score,cls(像素坐标)。"""

    name: str = "base"

    @abstractmethod
    def warmup(self, shape: tuple[int, int]) -> None:
        """使用最终 imgsz 预热,避免首帧高延迟 (v1.2 §5.2)。"""

    @abstractmethod
    def detect(self, img_bgr: np.ndarray) -> np.ndarray:
        ...

    def close(self) -> None:
        pass


class DetectOOM(RuntimeError):
    """CUDA 显存不足。"""


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float,
              max_det: int = 500) -> np.ndarray:
    """纯 NumPy NMS,返回保留索引(按分数降序)。"""
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = np.argsort(-scores)
    keep: list[int] = []
    while order.size > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        iw = np.clip(ix2 - ix1, 0, None)
        ih = np.clip(iy2 - iy1, 0, None)
        inter = iw * ih
        union = areas[i] + areas[rest] - inter
        iou = inter / np.maximum(union, 1e-9)
        order = rest[iou <= iou_thr]
    return np.array(keep, dtype=int)


class TilingWrapper(Detector):
    """切片推理包装: 图像长边 > trigger_size 时按 size/overlap 切块检测,
    合并后做跨块 NMS。默认关闭,由配置开启 (v1.0 §6.1 小目标支持)。"""

    def __init__(self, inner: Detector, tile_cfg):
        self.inner = inner
        self.name = f"{inner.name}+tiling"
        self.cfg = tile_cfg

    def warmup(self, shape: tuple[int, int]) -> None:
        self.inner.warmup(shape)

    def detect(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if not self.cfg.enabled or max(w, h) <= self.cfg.trigger_size:
            return self.inner.detect(img)
        step = max(1, int(self.cfg.size * (1.0 - self.cfg.overlap)))
        all_boxes: list[np.ndarray] = []
        ys = list(range(0, max(1, h - self.cfg.size) + 1, step))
        xs = list(range(0, max(1, w - self.cfg.size) + 1, step))
        if ys and ys[-1] + self.cfg.size < h:
            ys.append(h - self.cfg.size)
        if xs and xs[-1] + self.cfg.size < w:
            xs.append(w - self.cfg.size)
        for y in ys:
            for x in xs:
                tile = img[y:y + self.cfg.size, x:x + self.cfg.size]
                dets = self.inner.detect(tile)
                if len(dets) > 0:
                    dets = dets.copy()
                    dets[:, [0, 2]] += x
                    dets[:, [1, 3]] += y
                    all_boxes.append(dets)
        if not all_boxes:
            return np.zeros((0, 6), dtype=np.float32)
        dets = np.vstack(all_boxes)
        keep = nms_numpy(dets[:, :4], dets[:, 4], 0.5)  # 跨块 NMS
        return dets[keep]

    def close(self) -> None:
        self.inner.close()
