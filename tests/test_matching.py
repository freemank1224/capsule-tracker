"""匹配模块单元测试 (v1.0 §13.1)。

核心用例: 8x8 像素小框、位移 5 像素 —— 普通 IoU≈0 导致失配,
扩张 IoU + 中心距离惩罚的代价在门限内成功匹配。
"""
import numpy as np
import pytest

from app.tracking.matching import (
    cost_matrix, expand_box, iou_matrix, linear_assignment,
)


def _box(x, y, w=8, h=8):
    return np.array([x, y, x + w, y + h], dtype=np.float64)


def test_plain_iou_fails_small_displacement():
    """8x8 小框位移 5px: 普通 IoU = 24/104 ≈ 0.23;无扩张时总代价 0.90 超门限 0.8。"""
    iou = iou_matrix(_box(0, 0)[None], _box(5, 0)[None])[0, 0]
    assert iou < 0.3
    plain = cost_matrix(_box(0, 0)[None], _box(5, 0)[None],
                        expand_ratio=0.0, distance_penalty=0.3)
    assert plain[0, 0] > 0.8  # 普通 IoU 代价在默认门限下失配


def test_expanded_iou_matches_small_displacement():
    """扩张 0.3 后(宽高 x1.3),同一用例的代价应低于默认门限 0.8。"""
    cost = cost_matrix(_box(0, 0)[None], _box(5, 0)[None],
                       expand_ratio=0.3, distance_penalty=0.3)
    assert cost[0, 0] < 0.8


def test_expand_box_keeps_center():
    b = _box(10, 10, w=10, h=4)
    e = expand_box(b[None], 0.3)[0]
    assert np.allclose(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2),
                       ((e[0] + e[2]) / 2, (e[1] + e[3]) / 2))
    assert np.allclose(e[2] - e[0], 10 * 1.3)
    assert np.allclose(e[3] - e[1], 4 * 1.3)


def test_distance_penalty_increases_with_displacement():
    c1 = cost_matrix(_box(0, 0)[None], _box(1, 0)[None])[0, 0]
    c2 = cost_matrix(_box(0, 0)[None], _box(10, 0)[None])[0, 0]
    assert c2 > c1


def test_hungarian_gating():
    """代价超门限的配对不被接受,返回未匹配集合。"""
    preds = np.stack([_box(0, 0), _box(100, 100)])
    dets = np.stack([_box(2, 0), _box(500, 500)])
    cost = cost_matrix(preds, dets, 0.3, 0.3)
    matched, u_r, u_c = linear_assignment(cost, thresh=0.8)
    assert (0, 0) in matched
    # (100,100) 与 (500,500) 距离过远,代价超门限
    assert (1, 1) not in matched
    assert 1 in u_r and 1 in u_c


def test_empty_inputs():
    cost = cost_matrix(np.zeros((0, 4)), np.zeros((3, 4)))
    matched, u_r, u_c = linear_assignment(cost, 0.8)
    assert matched == [] and list(u_c) == [0, 1, 2]


def test_multiple_small_objects_matching():
    """疏密场景: 多个小框各自匹配正确的对应检测,离群检测不被强配。"""
    preds = np.stack([_box(0, 0), _box(20, 0), _box(40, 0)])
    dets = np.stack([_box(2, 1), _box(22, 0), _box(42, 1), _box(200, 200)])
    cost = cost_matrix(preds, dets, 0.3, 0.3)
    matched, _, u_c = linear_assignment(cost, 0.8)
    assert dict(matched) == {0: 0, 1: 1, 2: 2}
    assert list(u_c) == [3]  # 离群检测保持未匹配
