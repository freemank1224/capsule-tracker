"""自实现 ByteTrack 多目标跟踪 (v1.0 §6.2)。

纯 NumPy + SciPy,不依赖 CUDA / Ultralytics;不使用外观特征(ReID),
只用运动 + 几何关联。状态: New / Tracked / Lost / Removed。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kalman import KalmanFilterXYAH, KalmanConfig, tlbr_to_xyah, xyah_to_tlbr
from .matching import cost_matrix, linear_assignment

# 轨迹状态 (与原版 ByteTrack 一致)
ST_NEW = 1
ST_TRACKED = 0
ST_LOST = 2
ST_REMOVED = 3
STATE_STR = {ST_NEW: "new", ST_TRACKED: "tracked", ST_LOST: "lost", ST_REMOVED: "removed"}


@dataclass
class TrackConfig:
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    match_thresh: float = 0.8
    track_buffer_seconds: float = 2.0
    iou_expand_ratio: float = 0.3
    distance_penalty: float = 0.3
    kf: KalmanConfig = field(default_factory=KalmanConfig)


class STrack:
    """单条轨迹。"""

    _count = 0

    def __init__(self, tlbr: np.ndarray, score: float, kf: KalmanFilterXYAH):
        tlbr = np.asarray(tlbr, dtype=np.float64)
        self.mean, self.covariance = kf.initiate(tlbr_to_xyah(tlbr))
        self.score = float(score)
        self.state = ST_NEW
        self.is_activated = False
        STrack._count += 1
        self.track_id = STrack._count
        self.frame_id = 0
        self.start_frame = 0
        self._tlbr = tlbr
        self._kf = kf

    @property
    def tlbr(self) -> np.ndarray:
        if self.state == ST_NEW:
            return self._tlbr
        return xyah_to_tlbr(self.mean)

    def predict(self) -> None:
        self.mean, self.covariance = self._kf.predict(self.mean, self.covariance)

    def activate(self, frame_id: int) -> None:
        """新轨迹激活(官方语义: 仅视频首帧立即确认,其余帧为"未确认",
        需下一帧再次匹配上才输出 —— 防止单帧误检进入计数)。"""
        self.state = ST_TRACKED
        self.is_activated = frame_id == 1
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, tlbr: np.ndarray, score: float, frame_id: int) -> None:
        """Lost 轨迹重新关联上。"""
        self.mean, self.covariance = self._kf.update(
            self.mean, self.covariance, tlbr_to_xyah(np.asarray(tlbr, dtype=np.float64)))
        self.score = float(score)
        self.state = ST_TRACKED
        self.is_activated = True
        self.frame_id = frame_id

    def update(self, tlbr: np.ndarray, score: float, frame_id: int) -> None:
        """Tracked 轨迹匹配成功,校正状态。"""
        self._tlbr = np.asarray(tlbr, dtype=np.float64)
        self.mean, self.covariance = self._kf.update(
            self.mean, self.covariance, tlbr_to_xyah(self._tlbr))
        self.score = float(score)
        self.state = ST_TRACKED
        self.is_activated = True
        self.frame_id = frame_id

    def mark_lost(self) -> None:
        self.state = ST_LOST
        self.is_activated = False

    def mark_removed(self) -> None:
        self.state = ST_REMOVED
        self.is_activated = False

    @property
    def center_velocity_norm(self) -> float:
        """中心位移速度(相对框高归一),供静态场景判定。"""
        h = max(self.mean[3], 1e-6)
        return float(np.hypot(self.mean[4], self.mean[5]) / h)

    def to_dict(self, state: str | None = None) -> dict:
        return {
            "id": self.track_id,
            "box": [round(float(v), 1) for v in self.tlbr],
            "score": round(self.score, 3),
            "state": state or STATE_STR[self.state],
        }


class ByteTracker:
    """两阶段关联 ByteTrack。用法: tracks = tracker.update(detections (N,6))。"""

    def __init__(self, cfg: TrackConfig | None = None, fps: float = 15.0):
        self.cfg = cfg or TrackConfig()
        self.kf = KalmanFilterXYAH(self.cfg.kf)
        self.tracked_stracks: list[STrack] = []   # Tracked(含未确认)
        self.lost_stracks: list[STrack] = []
        self.frame_id = 0
        self.buffer_size = 30
        self.set_fps(fps)

    # ---- 帧率自适应 track_buffer (v1.0 §6.2.6): buffer = round(s * fps), 至少 5 ----
    def set_fps(self, fps: float) -> None:
        fps = max(fps, 1e-3)
        self.buffer_size = max(5, round(self.cfg.track_buffer_seconds * fps))

    def reset(self) -> None:
        """重置全部轨迹,ID 重新计数。"""
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.frame_id = 0
        STrack._count = 0

    @property
    def active_tracks(self) -> list[STrack]:
        return [t for t in self.tracked_stracks if t.is_activated]

    def all_states(self) -> dict:
        return {"tracked": len(self.tracked_stracks), "lost": len(self.lost_stracks),
                "ids": [t.track_id for t in self.tracked_stracks]}

    # ---- 主入口 ----
    def update(self, detections: np.ndarray) -> list[STrack]:
        """输入 (N,6): x1,y1,x2,y2,score,cls;返回当前帧活跃轨迹。"""
        self.frame_id += 1
        frame_id = self.frame_id
        activated: list[STrack] = []
        refind: list[STrack] = []
        lost_out: list[STrack] = []
        removed_out: list[STrack] = []

        detections = self._filter(detections)
        if len(detections) > 0:
            scores = detections[:, 4]
            order = np.argsort(-scores)
            detections = detections[order]
            high_mask = scores[order] >= self.cfg.track_high_thresh
            dets_high = detections[high_mask]
            dets_low = detections[~high_mask & (scores[order] >= self.cfg.track_low_thresh)]
        else:
            dets_high = np.zeros((0, 6), dtype=np.float64)
            dets_low = np.zeros((0, 6), dtype=np.float64)

        unconfirmed = [t for t in self.tracked_stracks if not t.is_activated]
        tracked = [t for t in self.tracked_stracks if t.is_activated]

        # 预测步(Tracked + Lost 全部)
        strack_pool = tracked + self.lost_stracks
        for t in strack_pool:
            t.predict()

        # ---- 第一阶段: 高分检测 vs Tracked + Lost ----
        matched1, u_pool, u_det1 = self._assign(strack_pool, dets_high)
        for i_tr, i_det in matched1:
            t = strack_pool[i_tr]
            if t.state == ST_TRACKED:
                t.update(dets_high[i_det, :4], dets_high[i_det, 4], frame_id)
                activated.append(t)
            else:
                t.re_activate(dets_high[i_det, :4], dets_high[i_det, 4], frame_id)
                refind.append(t)

        # ---- 第二阶段: 低分检测 vs 第一阶段未匹配的 Tracked ----
        r_tracked = [strack_pool[i] for i in u_pool if strack_pool[i].state == ST_TRACKED]
        matched2, u_r2, u_low = self._assign(r_tracked, dets_low)
        for i_tr, i_det in matched2:
            t = r_tracked[i_tr]
            t.update(dets_low[i_det, :4], dets_low[i_det, 4], frame_id)
            activated.append(t)
        for i in u_r2:
            t = r_tracked[i]
            t.mark_lost()
            lost_out.append(t)

        # ---- 未确认轨迹(尚未激活的新轨迹) vs 剩余高分检测 ----
        dets_rest = dets_high[u_det1] if len(u_det1) > 0 else np.zeros((0, 6))
        matched3, u_unc, u_det2 = self._assign(unconfirmed, dets_rest)
        for i_tr, i_det in matched3:
            t = unconfirmed[i_tr]
            t.update(dets_rest[i_det, :4], dets_rest[i_det, 4], frame_id)
            activated.append(t)
        for i in u_unc:
            unconfirmed[i].mark_removed()
            removed_out.append(unconfirmed[i])

        # ---- 剩余高分检测按 new_track_thresh 新建轨迹 ----
        for i in u_det2:
            det = dets_rest[i]
            if det[4] >= self.cfg.new_track_thresh:
                t = STrack(det[:4], det[4], self.kf)
                t.activate(frame_id)
                activated.append(t)

        # ---- 状态流转 ----
        for t in self.tracked_stracks:
            if t not in activated and t not in refind and t.state == ST_TRACKED:
                t.mark_lost()
                lost_out.append(t)
        for t in self.lost_stracks:
            if t not in lost_out and t.state == ST_LOST and frame_id - t.frame_id > self.buffer_size:
                t.mark_removed()

        # 去重式重建列表: 匹配成功的在 activated/refind 中,不再重复保留
        self.tracked_stracks = [t for t in self.tracked_stracks
                                if t.state == ST_TRACKED
                                and t not in activated and t not in refind]
        self.tracked_stracks.extend(activated)
        self.tracked_stracks.extend(refind)
        self.lost_stracks = [t for t in self.lost_stracks
                             if t.state == ST_LOST and t not in lost_out]
        self.lost_stracks.extend(lost_out)

        return [t for t in self.tracked_stracks if t.is_activated]

    # ---- 仅预测(隔帧检测/画面无变化时中间帧调用),输出标记 predicted ----
    def predict_only(self) -> list[STrack]:
        self.frame_id += 1
        for t in self.tracked_stracks:
            t.predict()
        return [t for t in self.tracked_stracks if t.is_activated]

    # ---- 内部 ----
    def _filter(self, detections: np.ndarray) -> np.ndarray:
        if detections is None or len(detections) == 0:
            return np.zeros((0, 6), dtype=np.float64)
        dets = np.asarray(detections, dtype=np.float64)
        return dets[dets[:, 4] >= self.cfg.track_low_thresh]

    def _assign(self, tracks: list[STrack],
                dets: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
        """代价 = 1 - 扩张IoU + λ·中心距离,门限 match_thresh。"""
        if not tracks or len(dets) == 0:
            return [], np.arange(len(tracks)), np.arange(len(dets))
        pred = np.stack([t.tlbr for t in tracks])
        cost = cost_matrix(pred, dets[:, :4],
                           self.cfg.iou_expand_ratio, self.cfg.distance_penalty)
        return linear_assignment(cost, self.cfg.match_thresh)
