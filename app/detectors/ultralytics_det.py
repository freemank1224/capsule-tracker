"""ultralytics (PyTorch) 检测后端。

v1.2 补丁要点:
- 仅 NVIDIA CUDA;half=auto 时用 warmup 同一输入实测 FP32/FP16 各 N 次(默认10)取更快者。
- 启动 warmup 至少 3 次(最终 imgsz),避免首帧高延迟。
- 只用 predict,不用 model.track();跟踪由自实现 ByteTrack 完成。
- CUDA OOM: empty_cache + imgsz 自动降一级(1920→1280→960→800→640)并通知客户端;
  降到最低仍失败则抛 DetectOOM,由上层回退 CPU 档。
"""
from __future__ import annotations

import logging
import time

import numpy as np

from .base import DetectOOM, Detector

logger = logging.getLogger(__name__)

IMSZ_LADDER = [1920, 1280, 960, 800, 640]  # v1.2 §5.4


class _HalfDeprecationFilter:
    """静音 ultralytics 对 half 参数的弃用提示(新版 quantize 是 INT8,并非 FP16 替代)。"""

    def filter(self, record: logging.LogRecord) -> bool:
        return "'half' is deprecated" not in record.getMessage()


def _silence_half_deprecation() -> None:
    from ultralytics.utils import LOGGER

    if not any(isinstance(f, _HalfDeprecationFilter) for f in LOGGER.filters):
        LOGGER.addFilter(_HalfDeprecationFilter())


class UltralyticsDetector(Detector):
    name = "ultralytics"

    def __init__(self, cfg, device: str = "auto", warn_cb=None):
        from ultralytics import YOLO  # 延迟导入,未安装时报清晰错误

        _silence_half_deprecation()
        self.cfg = cfg
        self.warn_cb = warn_cb or (lambda msg: None)
        self._torch = _import_torch()

        if device == "auto":
            device = "cuda:0" if self._torch.cuda.is_available() else "cpu"
        self.device = device
        self.imgsz = cfg.detector.imgsz
        self.half: bool | None = cfg.detector.half_bool  # None = auto
        self._half_decided = False
        self.model = YOLO(cfg.detector.weights)
        self.classes: list[str] = list(cfg.detector.classes)
        if self.classes and hasattr(self.model, "set_classes"):
            # Zero-shot 开放词汇检测(YOLO-World / YOLOE): 文本提示即类别,无需训练
            self.model.set_classes(self.classes)
            logger.info("Zero-shot 类别提示: %s", self.classes)
        elif self.classes:
            logger.info("模型 %s 不支持文本提示,忽略 classes 配置(自训练模型)",
                        cfg.detector.weights)
        logger.info("ultralytics 后端: weights=%s device=%s imgsz=%d half=%s",
                    cfg.detector.weights, self.device, self.imgsz, self.half)

    def set_classes(self, classes: list[str]) -> None:
        """运行时切换 Zero-shot 类别(线程安全: 由调用方的 GPU 锁保证)。"""
        if not hasattr(self.model, "set_classes"):
            raise ValueError(f"模型 {self.cfg.detector.weights} 不支持文本提示类别")
        self.classes = list(classes)
        self.model.set_classes(self.classes)
        self.cfg.detector.classes = self.classes
        logger.info("Zero-shot 类别已切换: %s", self.classes)

    # ---- warmup + FP16 自动选择 (v1.2 §5.1/§5.2) ----
    def warmup(self, shape: tuple[int, int]) -> None:
        if self.device.startswith("cuda") and self.half is None:
            self._auto_half(shape)
        dummy = np.zeros((shape[1], shape[0], 3), dtype=np.uint8)
        for i in range(3):  # 至少 3 次,使用最终 imgsz
            self._run(dummy)
        logger.info("ultralytics warmup 完成 (imgsz=%d, half=%s)", self.imgsz, self.half)

    def _auto_half(self, shape: tuple[int, int], n: int = 10) -> None:
        """同一输入各测 FP32 / FP16 平均耗时(各 n 次),自动选更快者。"""
        dummy = np.zeros((shape[1], shape[0], 3), dtype=np.uint8)
        times: dict[str, float] = {}
        for half in (False, True):
            self._run(dummy, half=half)  # 预跑一次不计入
            t0 = time.perf_counter()
            for _ in range(n):
                self._run(dummy, half=half)
            times["FP32" if not half else "FP16"] = (time.perf_counter() - t0) / n * 1000.0
        self.half = times["FP16"] < times["FP32"]
        self._half_decided = True
        logger.info("FP16 自动选择: FP32 平均 %.1f ms | FP16 平均 %.1f ms -> "
                    "选择 %s", times["FP32"], times["FP16"],
                    "FP16" if self.half else "FP32")

    # ---- 检测 ----
    def detect(self, img_bgr: np.ndarray) -> np.ndarray:
        try:
            return self._run(img_bgr)
        except DetectOOM:
            raise
        except self._torch.cuda.OutOfMemoryError:
            if self._handle_oom():
                return self._run(img_bgr)  # 降级后重试一次
            raise DetectOOM("CUDA OOM: 已降到最低 imgsz 仍失败,需回退 CPU 档")

    def _run(self, img_bgr: np.ndarray, half: bool | None = None) -> np.ndarray:
        try:
            results = self.model.predict(
                source=img_bgr, imgsz=self.imgsz, conf=self.cfg.detector.conf,
                iou=self.cfg.detector.iou, max_det=self.cfg.detector.max_det,
                device=self.device, half=half if half is not None else self.half,
                verbose=False,
            )
        except (RuntimeError, self._torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower() or isinstance(
                    e, self._torch.cuda.OutOfMemoryError):
                self._torch.cuda.empty_cache()
                raise self._torch.cuda.OutOfMemoryError(str(e)) from e
            raise
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32)
        boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = r.boxes.conf.cpu().numpy().astype(np.float32)
        classes = r.boxes.cls.cpu().numpy().astype(np.float32)
        return np.concatenate([boxes, scores[:, None], classes[:, None]],
                              axis=1).astype(np.float32)

    # ---- OOM 保护 (v1.2 §5.4) ----
    def _handle_oom(self) -> bool:
        """imgsz 自动降一级;返回是否成功降级。"""
        self._torch.cuda.empty_cache()
        for s in IMSZ_LADDER:
            if s < self.imgsz:
                self.imgsz = s
                msg = f"CUDA 显存不足,imgsz 自动降为 {s}"
                logger.warning(msg)
                self.warn_cb(msg)
                return True
        return False

    def close(self) -> None:
        if self.device.startswith("cuda"):
            self._torch.cuda.empty_cache()


def _import_torch():
    import torch
    return torch
