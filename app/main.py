"""FastAPI 应用: HTTP 路由 + WebSocket (/ws)。

- 推理/解码放入线程池,不阻塞事件循环;推理互斥锁避免多连接争用同一 GPU (v1.2 §5.3)。
- 服务端只保留最新一帧(背压, v1.0 §3.1)。
- 所有异常在 WebSocket 内捕获并发送 error 消息 (v1.0 §16)。
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import cv2
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import AppConfig, dump_config, hot_update
from .detectors.base import DetectOOM
from .hardware import TierProfile
from .pipeline import AdaptiveController, SessionPipeline
from .preprocess import map_boxes_to_original, roi_is_degenerate
from .schemas import ErrorMsg, InfoMsg, LatencyOut, ObjOut, ResultMsg, WarnMsg

logger = logging.getLogger(__name__)
STATS_INTERVAL_S = 5.0


class LatestFrameSlot:
    """只保留最新一帧的异步槽(服务端背压)。"""

    def __init__(self):
        self._data = None
        self._event = asyncio.Event()

    def set(self, data) -> None:
        self._data = data
        self._event.set()

    async def get(self):
        await self._event.wait()
        self._event.clear()
        data, self._data = self._data, None
        return data


class AppContext:
    """进程级共享状态: 配置 / 检测器 / 档位 / 自适应降级。"""

    def __init__(self, cfg: AppConfig, detector, profile: TierProfile,
                 config_path: str | None = None, warn_cb=None):
        self.cfg = cfg
        self.detector = detector
        self.profile = profile
        self.config_path = config_path
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="infer")
        self.gpu_lock = threading.Lock()  # v1.2 §5.3: 推理互斥
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None  # 启动时由 build_app 设置
        self._warn_tasks: set[asyncio.Task] = set()
        self._warn_cb = warn_cb
        self.adaptive = AdaptiveController(profile, cfg.target_latency_ms,
                                           warn_cb=self._broadcast_warn)
        self._frames = 0
        self._stat_start = time.perf_counter()
        self._stat_lat: list[float] = []

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def broadcast_warn(self, msg: str) -> None:
        self._broadcast_warn(msg)

    def _broadcast_warn(self, msg: str) -> None:
        """线程安全地向所有客户端广播 warn(检测器线程也会调用)。"""
        logger.warning("warn -> 客户端: %s", msg)
        payload = WarnMsg(msg=msg).model_dump()

        def _send() -> None:
            for ws in list(self.clients):
                t = asyncio.ensure_future(ws.send_json(payload))
                self._warn_tasks.add(t)
                t.add_done_callback(self._warn_tasks.discard)

        if self.loop is not None and self.loop.is_running():
            self.loop.call_soon_threadsafe(_send)

    def detect_locked(self, img: np.ndarray) -> np.ndarray:
        """带互斥锁的检测(GPU 串行化;CPU 档也顺带保护线程池)。"""
        with self.gpu_lock:
            try:
                return self.detector.detect(img)
            except DetectOOM:
                # v1.2 §5.4: OOM 到底 -> 回退 CPU 档(仍在锁内重建)
                self._rebuild_cpu_locked()
                return self.detector.detect(img)

    def _rebuild_cpu_locked(self) -> None:
        if getattr(self.detector, "device", "cpu") == "cpu":
            raise RuntimeError("检测器已在 CPU 仍失败")
        logger.error("GPU 推理彻底失败,回退 CPU 档重建检测器")
        self._broadcast_warn("GPU 显存不足且无法继续降级,已回退 CPU 档")
        from .detectors import create_detector
        try:
            self.detector.close()
        except Exception:
            pass
        self.detector = create_detector(self.cfg, device="cpu",
                                        warn_cb=self._broadcast_warn)
        self.profile.detect_every = max(self.profile.detect_every, 2)
        self.detector.warmup((self.cfg.detector.imgsz, self.cfg.detector.imgsz))

    def record_stats(self, total_ms: float) -> None:
        self._frames += 1
        self._stat_lat.append(total_ms)
        if len(self._stat_lat) > 100:
            self._stat_lat.pop(0)
        elapsed = time.perf_counter() - self._stat_start
        if elapsed >= STATS_INTERVAL_S:
            avg = sum(self._stat_lat) / max(len(self._stat_lat), 1)
            logger.info("吞吐统计: %.1f FPS | 平均延迟 %.0f ms | 近 %.0fs %d 帧",
                        self._frames / elapsed, avg, elapsed, self._frames)
            self._frames = 0
            self._stat_start = time.perf_counter()

    def apply_adaptive(self, note: str | None) -> None:
        """自适应降级/升级后,同步检测器的 imgsz。"""
        if note and hasattr(self.detector, "imgsz"):
            try:
                self.detector.imgsz = self.profile.imgsz
            except Exception:
                pass

    def save_roi(self, points) -> str | None:
        """设置/清除 ROI。返回错误提示(None=成功),供网页显示。"""
        if not points:
            self.cfg.preprocess.roi = None
            note = "ROI 已清除"
        elif roi_is_degenerate(points):
            logger.warning("客户端提交的 ROI 无效,已忽略: %s", points)
            self._broadcast_warn("ROI 无效(区域为零),已忽略")
            return "ROI 无效(区域为零)"
        else:
            self.cfg.preprocess.roi = points
            note = None
        if self.config_path:
            try:
                dump_config(self.cfg, self.config_path)
            except Exception as e:
                logger.warning("ROI 写回配置失败: %s", e)
        return note


def build_app(ctx: AppContext) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app):
        ctx.set_loop(asyncio.get_running_loop())
        yield

    app = FastAPI(title="capsule-tracker", version=__version__, lifespan=lifespan)
    web_dir = Path(__file__).resolve().parent.parent / "web"

    @app.get("/")
    async def index():
        return FileResponse(web_dir / "index.html")

    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

    @app.get("/api/info")
    async def api_info():
        return {
            "version": __version__,
            "tier": ctx.profile.tier,
            "detector": getattr(ctx.detector, "name", "unknown"),
            "imgsz": ctx.profile.imgsz,
            "detect_every": ctx.profile.detect_every,
            "conf": ctx.cfg.detector.conf,
            "change_detect": ctx.cfg.change_detect.enabled and ctx.profile.change_detect,
            "target_latency_ms": ctx.cfg.target_latency_ms,
        }

    @app.get("/api/config")
    async def api_config_get():
        return {
            "conf": ctx.cfg.detector.conf,
            "iou": ctx.cfg.detector.iou,
            "imgsz": ctx.profile.imgsz,
            "detect_every": ctx.profile.detect_every,
            "target_latency_ms": ctx.cfg.target_latency_ms,
            "roi": ctx.cfg.preprocess.roi,
        }

    @app.post("/api/config")
    async def api_config_post(updates: dict):
        applied = hot_update(ctx.cfg, updates)
        return {"applied": applied}

    @app.get("/cert/rootCA.pem")
    async def cert_download():
        p = Path(ctx.cfg.server.cert).parent / "rootCA.pem"
        if not p.exists():
            return JSONResponse({"error": f"根证书不存在: {p}"}, status_code=404)
        return FileResponse(p, media_type="application/x-pem-file",
                            filename="rootCA.pem")

    @app.post("/api/capture")
    async def capture_frame(request: Request):
        """保存当前帧到 captures/YYYYMMDD/,用于采集训练数据 (v1.0 §11)。

        请求体为原始 JPEG 字节(前端 fetch 直接传 blob,无需 multipart)。
        """
        data = await request.body()
        if not data:
            return JSONResponse({"error": "空文件"}, status_code=400)
        save_dir = Path(ctx.cfg.capture.save_dir) / time.strftime("%Y%m%d")
        save_dir.mkdir(parents=True, exist_ok=True)
        # datetime 才支持 %f 微秒(time.strftime 在 Windows 上会抛 ValueError)
        ts = datetime.now().strftime("%H%M%S_%f")[:-3]
        path = save_dir / f"{ts}.jpg"
        path.write_bytes(data)
        return {"saved": str(path)}

    # ---------------- WebSocket ----------------
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        ctx.clients.add(ws)
        pipeline = SessionPipeline(ctx.cfg, ctx.profile, ctx.detector)
        slot = LatestFrameSlot()
        info = InfoMsg(
            tier=ctx.profile.tier, detector=getattr(ctx.detector, "name", "unknown"),
            imgsz=ctx.profile.imgsz, detect_every=ctx.profile.detect_every,
            conf=ctx.cfg.detector.conf, version=__version__,
        ).model_dump()
        await ws.send_json(info)

        async def reader():
            try:
                while True:
                    msg = await ws.receive()
                    if msg["type"] == "websocket.disconnect":
                        slot.set(None)
                        return
                    if "bytes" in msg and msg["bytes"] is not None:
                        slot.set(msg["bytes"])
                    elif "text" in msg and msg["text"]:
                        await _handle_control(ws, ctx, pipeline, msg["text"])
            except WebSocketDisconnect:
                slot.set(None)

        async def worker():
            while True:
                jpeg = await slot.get()
                if jpeg is None:
                    return
                try:
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(
                        ctx.executor, _process_frame, ctx, pipeline, jpeg)
                    await ws.send_json(result)
                    ctx.record_stats(result["latency_ms"]["total"])
                    note = ctx.adaptive.record(result["latency_ms"]["total"])
                    ctx.apply_adaptive(note)
                    if note:
                        await ws.send_json(WarnMsg(msg=note).model_dump())
                except Exception as e:
                    logger.exception("处理帧失败")
                    try:
                        await ws.send_json(ErrorMsg(msg=f"{type(e).__name__}: {e}").model_dump())
                    except Exception:
                        return

        r_task = asyncio.create_task(reader())
        try:
            await worker()
        finally:
            r_task.cancel()
            ctx.clients.discard(ws)

    return app


def _process_frame(ctx: AppContext, pipeline: SessionPipeline, jpeg: bytes) -> dict:
    """在线程池中执行: 解码→预处理→检测(带锁)→跟踪→计数。"""
    buf = np.frombuffer(jpeg, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("JPEG 解码失败")
    t0 = time.perf_counter()

    img2, pre_info = pipeline.prep.run(img)
    t_pre = (time.perf_counter() - t0) * 1000

    skipped = False
    detect_ms = track_ms = 0.0
    pipeline.frame_idx += 1
    cd = ctx.cfg.change_detect
    now = time.perf_counter()

    # 变化检测: 无变化跳过检测,沿用上一帧结果 (v1.0 §6.4)
    if (cd.enabled and ctx.profile.change_detect and pipeline._last_objs
            and (now - pipeline._last_force_time) < cd.force_detect_interval_s):
        if pipeline.change.diff(img2) < cd.change_thresh:
            skipped = True
            objs = [o.model_copy(update={"state": "predicted"}) for o in pipeline._last_objs]
            res = pipeline.counter.update([o.id for o in objs])
            lat = LatencyOut(decode=0.0, detect=0.0, track=0.0,
                             total=(time.perf_counter() - t0) * 1000)
            return _result(ctx, pipeline, objs, res, img2, True, lat)

    if (pipeline.frame_idx - 1) % ctx.profile.detect_every == 0:
        td = time.perf_counter()
        dets = ctx.detect_locked(img2)
        detect_ms = (time.perf_counter() - td) * 1000
        pipeline._last_force_time = now
        tt = time.perf_counter()
        tracks = pipeline.tracker.update(dets)
        track_ms = (time.perf_counter() - tt) * 1000
        objs = [ObjOut(**t.to_dict()) for t in tracks]
    else:
        tt = time.perf_counter()
        tracks = pipeline.tracker.predict_only()
        track_ms = (time.perf_counter() - tt) * 1000
        objs = [ObjOut(**t.to_dict(state="predicted")) for t in tracks]

    if objs:
        boxes = map_boxes_to_original(np.array([o.box for o in objs]), pre_info)
        for o, b in zip(objs, boxes):
            o.box = [float(v) for v in b]

    pipeline._last_objs = objs
    res = pipeline.counter.update([t.track_id for t in tracks])
    lat = LatencyOut(decode=0.0, detect=detect_ms, track=track_ms,
                     total=(time.perf_counter() - t0) * 1000)
    return _result(ctx, pipeline, objs, res, img2, skipped, lat)


def _result(ctx, pipeline, objs, res, img, skipped, lat) -> dict:
    return ResultMsg(
        frame_id=pipeline.frame_idx, w=img.shape[1], h=img.shape[0],
        latency_ms=lat, count=res.count, total_unique=res.total_unique,
        objs=objs, stable=res.stable, tier=ctx.profile.tier, skipped=skipped,
    ).model_dump()


async def _handle_control(ws: WebSocket, ctx: AppContext,
                          pipeline: SessionPipeline, text: str) -> None:
    """文本控制消息: hello / reset / set / set_roi (v1.0 §5.2)。"""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        await ws.send_json(ErrorMsg(msg="控制消息不是合法 JSON").model_dump())
        return
    mtype = data.get("type")
    if mtype == "hello":
        fps = data.get("fps") or 15.0
        pipeline.tracker.set_fps(float(fps))
        # hello 携带的分辨率仅记录,不改变服务端行为
        logger.info("客户端 hello: ua=%s %sx%s torch=%s", data.get("ua", "?"),
                    data.get("w"), data.get("h"), data.get("torch"))
    elif mtype == "reset":
        pipeline.reset()
        await ws.send_json({"type": "info", "msg": "跟踪器与计数已重置"})
    elif mtype == "set":
        key, value = data.get("key"), data.get("value")
        applied = hot_update(ctx.cfg, {f"detector.{key}": value} if key in
                             ("conf", "iou", "imgsz", "max_det") else {key: value})
        if not applied:
            await ws.send_json(WarnMsg(msg=f"不支持的参数: {key}").model_dump())
        else:
            await ws.send_json({"type": "info", "msg": f"已更新 {key}={value}"})
    elif mtype == "set_classes":
        classes = [str(c).strip() for c in data.get("classes", []) if str(c).strip()]
        if not classes:
            await ws.send_json(WarnMsg(msg="classes 不能为空").model_dump())
            return
        try:
            with ctx.gpu_lock:
                ctx.detector.set_classes(classes)
            await ws.send_json({"type": "info",
                                "msg": f"Zero-shot 类别已切换: {', '.join(classes)}"})
        except Exception as e:
            await ws.send_json(ErrorMsg(msg=f"切换类别失败: {e}").model_dump())
    elif mtype == "set_roi":
        err = ctx.save_roi(data.get("points"))
        msg = err or "ROI 已保存"
        await ws.send_json({"type": "error" if err else "info", "msg": msg})
    else:
        await ws.send_json(WarnMsg(msg=f"未知消息类型: {mtype}").model_dump())
