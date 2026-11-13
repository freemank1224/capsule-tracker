"""跟踪器单元测试 (v1.0 §13.1)。全部使用合成序列,不依赖 GPU / 模型。"""
import numpy as np
import pytest

from app.tracking.bytetrack import ByteTracker, STrack, TrackConfig


def make_cfg(**kw):
    return TrackConfig(**kw)


def det(x, y, w, h, score=0.9, cls=0.0):
    return [x, y, x + w, y + h, score, cls]


def run_sequence(tracker, frames):
    """frames: list of (N,6) 检测数组;返回每帧活跃轨迹列表。"""
    outs = []
    for dets in frames:
        outs.append(tracker.update(np.array(dets, dtype=np.float64)))
    return outs


# ---------- 验收: 50 个小框匀速运动,无遮挡,ID 切换数 = 0 ----------

def test_50_small_boxes_constant_velocity_no_id_switch():
    tracker = ByteTracker(make_cfg(), fps=15)
    n = 50
    rng = np.random.RandomState(0)
    starts = rng.uniform(0, 300, size=(n, 2)) + 50  # 疏密混合
    ids_per_frame = []
    frames = []
    for f in range(50):
        dets = [det(s[0] + f * 2.0, s[1] + f * 1.0, 8, 12, score=0.9)
                for s in starts]
        frames.append(np.array(dets))
    outs = run_sequence(tracker, frames)
    for tracks in outs[10:]:  # 跳过前几帧的建轨迹期
        ids_per_frame.append(sorted(t.track_id for t in tracks))
    ref = ids_per_frame[0]
    assert len(ref) == n
    for ids in ids_per_frame:
        assert ids == ref, "出现 ID 切换"


# ---------- 验收: 漏检 1~2 帧后恢复,ID 保持不变 ----------

def test_missed_detections_keep_id():
    tracker = ByteTracker(make_cfg(), fps=15)
    seq = []
    for f in range(20):
        if f in (7, 8):  # 漏检两帧
            seq.append(np.zeros((0, 6)))
        else:
            seq.append(np.array([det(100 + f * 1.5, 200, 8, 12)]))
    outs = run_sequence(tracker, seq)
    ids_before = {t.track_id for t in outs[6]}
    ids_after = {t.track_id for t in outs[11]}
    assert ids_before and ids_before == ids_after


def test_short_occlusion_second_object():
    tracker = ByteTracker(make_cfg(), fps=15)
    seq = []
    for f in range(25):
        dets = [det(100 + f, 100, 8, 12), det(300, 100, 8, 12)]
        if f in (10, 11, 12):
            dets.pop(1)  # 第二个目标消失 3 帧
        seq.append(np.array(dets))
    outs = run_sequence(tracker, seq)
    ids = {t.track_id for t in outs[20]}
    assert len(ids) == 2


# ---------- 验收: 低分检测二阶段回收 ----------

def test_low_score_second_stage_rescue():
    cfg = make_cfg(track_high_thresh=0.5, track_low_thresh=0.1, match_thresh=0.8)
    tracker = ByteTracker(cfg, fps=15)
    # 先建轨迹(高分)
    outs = run_sequence(tracker, [np.array([det(50, 50, 8, 12, 0.9)]) for _ in range(5)])
    tid = outs[-1][0].track_id
    # 接着只给低分检测(0.2~0.4): 应走第二阶段关联,ID 不变
    seq = [np.array([det(51, 50, 8, 12, 0.3)]), np.array([det(52, 50, 8, 12, 0.2)])]
    outs2 = run_sequence(tracker, seq)
    assert outs2[-1][0].track_id == tid
    assert outs2[-1][0].score == pytest.approx(0.2)


def test_very_low_score_not_matched():
    """低于 track_low_thresh 的检测应被过滤,轨迹转入 Lost。"""
    cfg = make_cfg()
    tracker = ByteTracker(cfg, fps=15)
    outs = run_sequence(tracker, [np.array([det(50, 50, 8, 12, 0.9)]) for _ in range(3)])
    tid = outs[-1][0].track_id
    outs2 = run_sequence(tracker, [np.array([det(51, 50, 8, 12, 0.05)]) for _ in range(2)])
    assert len(outs2[-1]) == 0  # 轨迹丢失,不输出
    assert tracker.lost_stracks[0].track_id == tid


# ---------- 验收: reset() 后 ID 重新计数 ----------

def test_reset_restarts_ids():
    tracker = ByteTracker(make_cfg(), fps=15)
    outs = run_sequence(tracker, [np.array([det(50, 50, 8, 12)]) for _ in range(3)])
    assert outs[-1][0].track_id >= 1
    tracker.reset()
    outs2 = run_sequence(tracker, [np.array([det(80, 80, 8, 12)])])
    assert outs2[0][0].track_id == 1


# ---------- 帧率自适应 buffer ----------

def test_buffer_adapts_to_fps():
    tracker = ByteTracker(make_cfg(track_buffer_seconds=2.0), fps=30)
    assert tracker.buffer_size == 60
    tracker.set_fps(10)
    assert tracker.buffer_size == 20
    tracker.set_fps(1)
    assert tracker.buffer_size >= 5  # 下限 5


# ---------- predict_only(隔帧检测) ----------

def test_predict_only_moves_track():
    tracker = ByteTracker(make_cfg(), fps=15)
    # 先喂匀速运动序列让卡尔曼估计出速度
    outs = run_sequence(tracker, [np.array([det(100 + f * 2.0, 100, 8, 12, 0.9)])
                                  for f in range(5)])
    x_before = outs[-1][0].tlbr[0]
    tracks = tracker.predict_only()
    assert len(tracks) == 1
    assert tracks[0].tlbr[0] > x_before  # 卡尔曼预测按估计速度前进
