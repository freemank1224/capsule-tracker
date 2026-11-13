"""单连接处理流水线: 解码 → 预处理 → (变化检测/隔帧) → 检测 → 跟踪 → 计数 → 序列化。

- 背压由调用方保证(一次只处理最新一帧)。
- 变化检测无变化时跳过检测,沿用上一帧结果,返回 skipped=true (v1.0 §6.4)。
- 隔帧检测时中间帧轨迹状态标记 predicted (v1.0 §7.4)。
"""
from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from .counting import Counter
from .preprocess import Preprocessor, map_boxes_to_original
from .schemas import LatencyOut, ObjOut, ResultMsg
from .tracking import ByteTracker, TrackConfig
from .tracking.kalman import KalmanConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- 变化检测

class ChangeDetector:
    """缩至 160px 宽灰度图,与上一"已处理帧"的平均绝对差判定画面是否变化。"""

    def __init__(self, width: int = 160):
        self.width = width
        self._prev: np.ndarray | None = None

    def diff(self, img_bgr: np.ndarray) -> float:
        h, w = img_bgr.shape[:2]
        nw = self.width
        nh = max(1, round(h * nw / w))
        small = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self._prev is None or self._prev.shape != gray.shape:
            self._prev = gray
            return float("inf")  # 首帧视为变化
        mad = float(np.mean(np.abs(gray - self._prev)))
        self._prev = gray
        return mad

    def reset(self) -> None:
        self._prev = None


# ---------------------------------------------------------------- 自适应降级

class AdaptiveController:
    """运行时自适应降级 (v1.0 §7.3): 最近 30 帧平均耗时超过 target_latency_ms
    自动下调一级;持续稳定后才升回,避免抖动。"""

    WINDOW = 30
    UPGRADE_FRAMES = 150  # 需连续这么多帧稳定才升回一级

    def __init__(self, profile, target_latency_ms: float, warn_cb=None):
        self.profile = profile
        self.target = target_latency_ms
        self.warn_cb = warn_cb or (lambda msg: None)
        self._lat: list[float] = []
        self._stable_run = 0
        self._downgraded: int = 0  # 当前降级步数

    def record(self, total_ms: float) -> str | None:
        """记录一帧耗时;返回触发的降级/升级说明(否则 None)。"""
        self._lat.append(total_ms)
        if len(self._lat) > self.WINDOW:
            self._lat.pop(0)
        if len(self._lat) < self.WINDOW:
            return None
        avg = sum(self._lat) / len(self._lat)
        if avg > self.target:
            self._stable_run = 0
            return self._downgrade(f"平均延迟 {avg:.0f}ms 超过目标 {self.target:.0f}ms")
        if avg < self.target * 0.7:
            self._stable_run += 1
            if self._stable_run >= self.UPGRADE_FRAMES and self._downgraded > 0:
                self._downgraded -= 1
                self._stable_run = 0
                self._lat.clear()  # 清窗口,重新积累再评估,避免抖动
                return f"延迟恢复,升回一级: imgsz={self.profile.imgsz} detect_every={self.profile.detect_every}"
        return None

    def _downgrade(self, reason: str) -> str:
        # 先降 imgsz,到底后再降检测频率
        if self.profile.imgsz > 640:
            self.profile.imgsz = max(640, {1920: 1280, 1280: 960, 960: 800}.get(
                self.profile.imgsz, 640))
            msg = f"自适应降级({reason}): imgsz 降为 {self.profile.imgsz}"
        elif self.profile.detect_every < 4:
            self.profile.detect_every += 1
            msg = f"自适应降级({reason}): 检测频率降为每 {self.profile.detect_every} 帧"
        else:
            return ""  # 已到底
        self._downgraded += 1
        self._lat.clear()  # 清窗口,重新积累一个完整窗口再评估
        logger.warning(msg)
        self.warn_cb(msg)
        return msg

    def set_imgsz(self, imgsz: int) -> None:
        self.profile.imgsz = imgsz


# ---------------------------------------------------------------- 会话流水线

class SessionPipeline:
    """每连接一份(跟踪器/计数器/变化检测有状态);detector 与 cfg 由外部共享。"""

    def __init__(self, cfg, profile, detector):
        self.cfg = cfg
        self.profile = profile
        self.detector = detector
        self.tracker = ByteTracker(TrackConfig(
            track_high_thresh=cfg.tracker.track_high_thresh,
            track_low_thresh=cfg.tracker.track_low_thresh,
            new_track_thresh=cfg.tracker.new_track_thresh,
            match_thresh=cfg.tracker.match_thresh,
            track_buffer_seconds=cfg.tracker.track_buffer_seconds,
            iou_expand_ratio=cfg.tracker.iou_expand_ratio,
            distance_penalty=cfg.tracker.distance_penalty,
            kf=KalmanConfig(
                std_weight_position=cfg.tracker.std_weight_position,
                std_weight_velocity=cfg.tracker.std_weight_velocity,
            ),
        ), fps=15.0)
        self.counter = Counter(cfg.counting.stable_frames)
        self.change = ChangeDetector()
        self.prep = Preprocessor(cfg)
        self.frame_idx = 0
        self._last_objs: list[ObjOut] = []
        self._last_force_time = 0.0

    def reset(self) -> None:
        self.tracker.reset()
        self.counter.reset()
        self.change.reset()
        self.frame_idx = 0
        self._last_objs = []

    # ---- 主入口 ----
    def process(self, jpeg_bytes: bytes) -> dict:
        t0 = time.perf_counter()
        self.frame_idx += 1

        # 解码
        buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("JPEG 解码失败")
        t_decode = (time.perf_counter() - t0) * 1000

        # 预处理(亮度归一化 / ROI)
        t1 = time.perf_counter()
        img, pre_info = self.prep.run(img)
        t_pre = (time.perf_counter() - t1) * 1000

        skipped = False
        detect_ms = 0.0
        track_ms = 0.0

        # 变化检测(静态托盘省算力)
        cd = self.cfg.change_detect
        now = time.perf_counter()
        if cd.enabled and self.profile.change_detect and self._last_objs:
            changed = self.change.diff(img) >= cd.change_thresh
            force_due = (now - self._last_force_time) >= cd.force_detect_interval_s
            if not changed and not force_due:
                skipped = True
                objs = [o.model_copy(update={"state": "predicted"}) for o in self._last_objs]
                res = self.counter.update([o.id for o in objs])
                result = self._build_result(objs, res, img, skipped,
                                            LatencyOut(decode=t_decode,
                                                       detect=0.0, track=0.0,
                                                       total=(time.perf_counter() - t0) * 1000))
                return result

        # 检测帧 / 预测帧(第 1 帧必检测)
        if (self.frame_idx - 1) % self.profile.detect_every == 0:
            td = time.perf_counter()
            dets = self.detector.detect(img)
            detect_ms = (time.perf_counter() - td) * 1000
            self._last_force_time = now
            tt = time.perf_counter()
            tracks = self.tracker.update(dets)
            track_ms = (time.perf_counter() - tt) * 1000
            objs = [ObjOut(**t.to_dict()) for t in tracks]
        else:
            tt = time.perf_counter()
            tracks = self.tracker.predict_only()
            track_ms = (time.perf_counter() - tt) * 1000
            objs = [ObjOut(**t.to_dict(state="predicted")) for t in tracks]

        # 检测框映射回原图坐标(ROI 裁剪方案时需要)
        if objs:
            boxes = map_boxes_to_original(
                np.array([o.box for o in objs], dtype=np.float64), pre_info)
            for o, b in zip(objs, boxes):
                o.box = [float(v) for v in b]

        self._last_objs = objs
        res = self.counter.update([t.track_id for t in tracks])
        total = (time.perf_counter() - t0) * 1000
        result = self._build_result(objs, res, img, skipped,
                                    LatencyOut(decode=t_decode, detect=detect_ms,
                                               track=track_ms, total=total + t_pre))
        return result

    def _build_result(self, objs, res, img, skipped, latency) -> dict:
        msg = ResultMsg(
            frame_id=self.frame_idx,
            w=img.shape[1], h=img.shape[0],
            latency_ms=latency,
            count=res.count,
            total_unique=res.total_unique,
            objs=objs,
            stable=res.stable,
            tier=self.profile.tier,
            skipped=skipped,
        )
        return msg.model_dump()
