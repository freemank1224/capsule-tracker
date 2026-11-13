"""WebSocket 消息数据结构 (v1.0 §5.2)。"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ObjOut(BaseModel):
    id: int
    box: List[float]          # [x1,y1,x2,y2] 原图像素坐标
    score: float
    state: str                # tracked | predicted | lost


class LatencyOut(BaseModel):
    decode: float = 0.0
    detect: float = 0.0
    track: float = 0.0
    total: float = 0.0


class ResultMsg(BaseModel):
    type: str = "result"
    frame_id: int
    w: int
    h: int
    latency_ms: LatencyOut
    count: int
    total_unique: int = 0
    objs: List[ObjOut] = Field(default_factory=list)
    stable: bool = False
    tier: str
    skipped: bool = False     # 变化检测判定无变化,跳过了本次检测


class InfoMsg(BaseModel):
    type: str = "info"
    tier: str
    detector: str
    imgsz: int
    detect_every: int
    conf: float
    version: str


class WarnMsg(BaseModel):
    type: str = "warn"
    msg: str


class ErrorMsg(BaseModel):
    type: str = "error"
    msg: str


# ---------------------------------------------------- 客户端坐标映射辅助

def scale_point(pt: tuple[float, float],
                src_size: tuple[int, int],
                dst_size: tuple[int, int]) -> tuple[float, float]:
    """把服务端图像坐标映射到客户端显示坐标。"""
    sx = dst_size[0] / src_size[0]
    sy = dst_size[1] / src_size[1]
    return pt[0] * sx, pt[1] * sy


def scale_box(box: List[float],
              src_size: tuple[int, int],
              dst_size: tuple[int, int]) -> List[float]:
    """整框缩放 (x1,y1,x2,y2)。"""
    sx = dst_size[0] / src_size[0]
    sy = dst_size[1] / src_size[1]
    return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]


def fit_containing(display_size: tuple[int, int],
                   video_size: tuple[int, int]) -> tuple[float, float, float, float]:
    """object-fit: contain 模式下计算视频在容器中的实际渲染区域。

    返回 (offset_x, offset_y, draw_w, draw_h),客户端叠加层据此换算,避免框错位。
    """
    dw, dh = display_size
    vw, vh = video_size
    scale = min(dw / vw, dh / vh)
    w, h = vw * scale, vh * scale
    return (dw - w) / 2, (dh - h) / 2, w, h
