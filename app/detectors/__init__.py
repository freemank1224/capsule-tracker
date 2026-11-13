"""检测后端工厂: 按配置创建后端,失败时自动回退(深度后端不可用 → classical)。"""
from __future__ import annotations

import logging
from pathlib import Path

from .base import DetectOOM, Detector, TilingWrapper, nms_numpy

logger = logging.getLogger(__name__)

__all__ = ["DetectOOM", "Detector", "TilingWrapper", "nms_numpy", "create_detector"]


def _make_backend(cfg, device: str, warn_cb) -> Detector:
    backend = cfg.detector.backend
    if backend == "ultralytics":
        from .ultralytics_det import UltralyticsDetector
        return UltralyticsDetector(cfg, device=device, warn_cb=warn_cb)
    if backend == "onnx":
        weights = Path(cfg.detector.weights)
        if not weights.exists():
            weights = weights.with_suffix(".onnx")
        if not weights.exists():
            raise FileNotFoundError(f"ONNX 模型不存在: {cfg.detector.weights}(可运行 scripts/export_onnx.py 导出)")
        cfg.detector.weights = str(weights)
        from .onnx_det import OnnxDetector
        return OnnxDetector(cfg, warn_cb=warn_cb)
    if backend == "openvino":
        from .openvino_det import OpenvinoDetector
        return OpenvinoDetector(cfg, warn_cb=warn_cb)
    if backend == "classical":
        from .classical_det import ClassicalDetector
        return ClassicalDetector(cfg)
    raise ValueError(f"未知检测后端: {backend}")


def create_detector(cfg, device: str = "auto", warn_cb=None) -> Detector:
    """创建配置指定的后端;权重缺失/依赖未安装时警告并回退 classical 跑通链路。"""
    try:
        det = _make_backend(cfg, device, warn_cb)
    except Exception as e:
        if cfg.detector.backend == "classical":
            raise
        logger.warning("后端 %s 不可用(%s: %s),回退 classical 后端",
                       cfg.detector.backend, type(e).__name__, e)
        if warn_cb:
            warn_cb(f"检测后端 {cfg.detector.backend} 不可用,已回退 classical(无需模型)")
        cfg.detector.backend = "classical"
        from .classical_det import ClassicalDetector
        det = ClassicalDetector(cfg)
    if cfg.detector.tile.enabled:
        det = TilingWrapper(det, cfg.detector.tile)
    return det
