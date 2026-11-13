"""协议与坐标映射测试 (v1.0 §13.1 test_protocol)。"""
import numpy as np
import pytest
from pydantic import ValidationError

from app.counting import Counter
from app.schemas import (
    ErrorMsg, InfoMsg, ObjOut, ResultMsg, WarnMsg,
    fit_containing, scale_box, scale_point,
)
from app.pipeline import AdaptiveController, ChangeDetector


# ---------- 消息结构校验 ----------

def test_result_msg_schema():
    msg = ResultMsg(
        frame_id=123, w=1920, h=1080,
        latency_ms={"decode": 4, "detect": 85, "track": 1, "total": 92},
        count=57, objs=[{"id": 3, "box": [10, 20, 30, 40], "score": 0.91,
                         "state": "tracked"}],
        stable=True, tier="gpu")
    d = msg.model_dump()
    assert d["type"] == "result"
    assert d["objs"][0]["box"] == [10, 20, 30, 40]
    assert d["latency_ms"]["total"] == 92


def test_result_msg_rejects_bad_obj():
    with pytest.raises(ValidationError):
        ObjOut(id=1, box="bad", score=0.9, state="tracked")


def test_simple_msgs():
    assert WarnMsg(msg="降级").model_dump()["type"] == "warn"
    assert ErrorMsg(msg="x").model_dump()["type"] == "error"
    info = InfoMsg(tier="gpu", detector="ultralytics", imgsz=960,
                   detect_every=1, conf=0.25, version="1.2.0").model_dump()
    assert info["type"] == "info"


# ---------- 坐标映射函数 ----------

def test_scale_point_and_box():
    assert scale_point((100, 50), (1000, 500), (500, 250)) == (50.0, 25.0)
    box = scale_box([0, 0, 100, 200], (1000, 1000), (500, 500))
    assert box == [0.0, 0.0, 50.0, 100.0]


def test_fit_containing_landscape_video_in_square():
    """1920x1080 视频放进 500x500 容器: 上下留黑边。"""
    x, y, w, h = fit_containing((500, 500), (1920, 1080))
    assert w == pytest.approx(500) and h == pytest.approx(281.25)
    assert y == pytest.approx((500 - 281.25) / 2)


def test_fit_containing_portrait():
    x, y, w, h = fit_containing((400, 800), (1080, 1920))
    assert w == pytest.approx(400) and h == pytest.approx(400 * 1920 / 1080)
    assert x == pytest.approx(0)


# ---------- 计数与稳定性 ----------

def test_roi_degenerate_ignored():
    """退化 ROI(零面积/越界)不能把整幅图涂黑 —— 必须被忽略。"""
    import numpy as np
    from app.preprocess import apply_roi, roi_is_degenerate

    img = np.full((100, 200, 3), 255, dtype=np.uint8)

    bad_cases = [
        [[0.0, 0.1], [0.0, 0.1], [0.0, 0.5], [0.0, 0.5]],  # x 全相同(本次线上故障)
        [[0.1, 0.2], [0.9, 0.2], [0.9, 0.2005], [0.1, 0.2005]],  # 高度差 <1e-3
        [[0.1, 0.2], [0.9, 0.2], [0.9, 0.8]],  # 正常三角形,对照
        [[1.5, 0.2], [2.0, 0.2], [1.8, 0.8]],  # 越界
        [[0.5, 0.5]],  # 点数不足
    ]
    assert roi_is_degenerate(bad_cases[0])
    assert roi_is_degenerate(bad_cases[1])
    assert not roi_is_degenerate(bad_cases[2])
    assert roi_is_degenerate(bad_cases[3])
    assert roi_is_degenerate(bad_cases[4])

    out1, _ = apply_roi(img, bad_cases[0])   # 退化 -> 原图返回
    out2, _ = apply_roi(img, bad_cases[2])   # 正常 -> 区域外置黑
    assert (out1 == img).all()
    assert not (out2 == img).all()


def test_counter_stable_and_unique():
    c = Counter(stable_frames=3)
    assert c.update([1, 2, 3]).stable is False
    assert c.update([1, 2, 3]).stable is False
    r = c.update([1, 2, 3]).stable
    assert c.update([1, 2, 3]).stable is True
    assert c.update([1, 2, 3, 4]).stable is False       # 新轨迹打破稳定
    assert c.update([1, 2, 3]).count == 3
    assert c.update([]).count == 0
    assert c.update([]).total_unique == 4


def test_counter_reset():
    c = Counter(stable_frames=1)
    c.update([1])
    c.reset()
    r = c.update([5])
    assert r.total_unique == 1


# ---------- 变化检测与自适应降级 ----------

def test_change_detector_static_vs_motion():
    cd = ChangeDetector()
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)
    d1 = cd.diff(frame)          # 首帧 = inf
    d2 = cd.diff(frame.copy())   # 完全相同
    assert d2 == 0.0
    moved = frame.copy()
    moved[:, 320:] = 200
    d3 = cd.diff(moved)
    assert d3 > d2


def _profile():
    from app.hardware import TierProfile
    return TierProfile("cpu_balanced", imgsz=800, imgsz_max=800,
                       detect_every=2, change_detect=True, tile=False)


def test_adaptive_downgrade_and_upgrade():
    ac = AdaptiveController(_profile(), target_latency_ms=100.0)
    notes = []
    for _ in range(30):
        note = ac.record(200.0)
        if note:
            notes.append(note)
    assert notes, "持续超时未触发降级"
    p = ac.profile
    assert p.imgsz < 800 or p.detect_every > 2
    # 长期稳定后应升回
    for _ in range(AdaptiveController.UPGRADE_FRAMES + 30):
        ac.record(50.0)
    assert ac._downgraded < 1


def test_adaptive_min_floor():
    p = _profile()
    p.imgsz = 640
    p.detect_every = 4
    ac = AdaptiveController(p, target_latency_ms=1.0)
    for _ in range(30):
        ac.record(999.0)
    assert ac.profile.imgsz == 640 and ac.profile.detect_every == 4  # 已到底
