"""代价矩阵与匹配: 小目标专用扩张 IoU + 中心距离惩罚 + 匈牙利匹配 (v1.0 §6.2)。

小框小位移下普通 IoU 接近 0,扩张 IoU(默认宽高各 x1.3)配合
1 - IoU_exp + λ * d_center_norm 代价可显著提高关联成功率。
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def expand_box(box: np.ndarray, ratio: float) -> np.ndarray:
    """保持中心,宽高各乘 (1+ratio)。box: (...,4) x1y1x2y2。"""
    if ratio <= 0:
        return box
    x1, y1, x2, y2 = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1) * (1.0 + ratio), (y2 - y1) * (1.0 + ratio)
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """普通 IoU 矩阵,(Ma,4)x(Mb,4) -> (Ma,Mb)。"""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    a, b = boxes_a[:, None, :], boxes_b[None, :, :]
    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_a = np.clip(a[..., 2] - a[..., 0], 0, None) * np.clip(a[..., 3] - a[..., 1], 0, None)
    area_b = np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def diagonal_norm(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """中心点距离 / 框对角线(取框对中较大者的对角线),(Ma,Mb)。"""
    ca = (boxes_a[:, :2] + boxes_a[:, 2:]) / 2.0
    cb = (boxes_b[:, :2] + boxes_b[:, 2:]) / 2.0
    dist = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=-1)
    da = np.linalg.norm(boxes_a[:, 2:] - boxes_a[:, :2], axis=-1)
    db = np.linalg.norm(boxes_b[:, 2:] - boxes_b[:, :2], axis=-1)
    diag = np.maximum(da[:, None], db[None, :])
    return np.where(diag > 0, dist / diag, dist)


def cost_matrix(pred_boxes: np.ndarray, det_boxes: np.ndarray,
                expand_ratio: float = 0.3,
                distance_penalty: float = 0.3) -> np.ndarray:
    """小目标专用代价: cost = 1 - IoU_expanded + λ * d_center_norm。"""
    if len(pred_boxes) == 0 or len(det_boxes) == 0:
        return np.zeros((len(pred_boxes), len(det_boxes)), dtype=np.float64)
    iou = iou_matrix(expand_box(pred_boxes, expand_ratio),
                     expand_box(det_boxes, expand_ratio))
    return 1.0 - iou + distance_penalty * diagonal_norm(pred_boxes, det_boxes)


def linear_assignment(cost: np.ndarray,
                      thresh: float) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    """匈牙利匹配 + 门限: 代价超过 thresh 的配对不接受。

    返回 (匹配对列表, 未匹配行索引, 未匹配列索引)。
    """
    if cost.size == 0:
        return [], np.arange(cost.shape[0]), np.arange(cost.shape[1])
    rows, cols = linear_sum_assignment(cost)
    matched: list[tuple[int, int]] = []
    rset, cset = set(range(cost.shape[0])), set(range(cost.shape[1]))
    for r, c in zip(rows, cols):
        if cost[r, c] <= thresh:
            matched.append((int(r), int(c)))
            rset.discard(r)
            cset.discard(c)
    unmatched_rows = np.array(sorted(rset), dtype=int)
    unmatched_cols = np.array(sorted(cset), dtype=int)
    return matched, unmatched_rows, unmatched_cols
