"""配置加载与校验 (pydantic v2)。非法值启动时报清晰错误。"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)

TierName = Literal["auto", "gpu_high", "gpu", "gpu_light", "cpu_balanced", "cpu_lite"]
DetectorBackend = Literal["ultralytics", "onnx", "openvino", "classical"]


class ServerCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str = "0.0.0.0"
    port: int = Field(8443, ge=1, le=65535)
    cert: str = "certs/cert.pem"
    key: str = "certs/key.pem"
    cert_download_port: int = Field(8080, ge=0, le=65535)  # 0 = 关闭


class HardwareCfg(BaseModel):
    """档位映射显存阈值 (v1.2 §3: 阈值可在配置中改)"""
    model_config = ConfigDict(extra="forbid")
    vram_high_mb: int = Field(20480, gt=0)
    vram_mb: int = Field(12288, gt=0)
    vram_light_mb: int = Field(7168, gt=0)
    force_cpu: bool = False  # 调试用: 强制忽略 CUDA


class TileCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    trigger_size: int = Field(1600, gt=0)
    size: int = Field(640, gt=0)
    overlap: float = Field(0.2, ge=0.0, lt=0.9)


class ClassicalCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    diff_thresh: int = Field(25, ge=1, le=255)
    min_area: int = Field(60, ge=1)
    max_area: int = Field(20000, ge=1)
    morph_kernel: int = Field(7, ge=1)


class DetectorCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: DetectorBackend = "ultralytics"
    weights: str = "yolov8s-worldv2.pt"  # Zero-shot 默认;也可用自训练 .pt
    classes: list[str] = []              # Zero-shot 文本类别提示(YOLO-World/YOLOE)
    # None = 由档位决定;显式设定则覆盖档位默认(小目标场景常需手动调高)
    imgsz: Optional[int] = Field(None, ge=320, le=1920)
    conf: float = Field(0.25, ge=0.0, le=1.0)
    iou: float = Field(0.5, ge=0.0, le=1.0)
    max_det: int = Field(500, ge=1)
    half: str = "auto"  # auto|true|false (v1.2 §5)

    tile: TileCfg = TileCfg()
    classical: ClassicalCfg = ClassicalCfg()

    @field_validator("half", mode="before")
    @classmethod
    def _norm_half(cls, v):
        if isinstance(v, bool):
            v = "true" if v else "false"
        v = str(v).lower()
        if v not in ("auto", "true", "false"):
            raise ValueError("detector.half 必须是 auto|true|false")
        return v

    @property
    def half_bool(self) -> Optional[bool]:
        return {"auto": None, "true": True, "false": False}[self.half]


class PreprocessCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    normalize_brightness: bool = False
    roi: Optional[list] = None  # [[x,y],...] 归一化坐标


class TrackerCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["bytetrack"] = "bytetrack"
    track_high_thresh: float = Field(0.5, ge=0.0, le=1.0)
    track_low_thresh: float = Field(0.1, ge=0.0, le=1.0)
    new_track_thresh: float = Field(0.6, ge=0.0, le=1.0)
    match_thresh: float = Field(0.8, ge=0.0, le=1.0)
    track_buffer_seconds: float = Field(2.0, gt=0.0)
    iou_expand_ratio: float = Field(0.3, ge=0.0, le=2.0)
    distance_penalty: float = Field(0.3, ge=0.0, le=2.0)
    std_weight_position: float = Field(0.05, gt=0.0)
    std_weight_velocity: float = Field(0.00625, gt=0.0)


class ChangeDetectCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    change_thresh: float = Field(2.0, gt=0.0)
    force_detect_interval_s: float = Field(3.0, gt=0.0)


class CountingCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stable_frames: int = Field(5, ge=1)


class CaptureCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    save_dir: str = "captures"


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    server: ServerCfg = ServerCfg()
    tier: TierName = "auto"
    target_latency_ms: float = Field(700.0, gt=0.0)
    hardware: HardwareCfg = HardwareCfg()
    detector: DetectorCfg = DetectorCfg()
    preprocess: PreprocessCfg = PreprocessCfg()
    tracker: TrackerCfg = TrackerCfg()
    change_detect: ChangeDetectCfg = ChangeDetectCfg()
    counting: CountingCfg = CountingCfg()
    capture: CaptureCfg = CaptureCfg()


def load_config(path: str | Path | None = None) -> AppConfig:
    """加载 YAML 并用 pydantic 校验;文件缺失/非法时给出清晰错误。"""
    if path is None:
        return AppConfig()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {p.resolve()}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"配置文件 YAML 语法错误 ({p}): {e}") from e
    if not isinstance(raw, dict):
        raise ValueError(f"配置文件顶层必须是键值映射 ({p})")
    try:
        return AppConfig(**raw)
    except ValidationError as e:
        lines = [f"  {err['loc'][0]}: {err['msg']}" for err in e.errors()]
        raise ValueError("配置校验失败:\n" + "\n".join(lines)) from e


def hot_update(cfg: AppConfig, updates: dict) -> list[str]:
    """热更新部分运行参数(仅允许白名单键),返回实际生效的键。"""
    allowed = {
        "detector.conf": "detector",
        "detector.iou": "detector",
        "detector.imgsz": "detector",
        "detector.max_det": "detector",
        "target_latency_ms": None,
        "change_detect.enabled": "change_detect",
        "preprocess.normalize_brightness": "preprocess",
    }
    applied = []
    for key, value in updates.items():
        if key not in allowed:
            continue
        sub = cfg if allowed[key] is None else getattr(cfg, allowed[key])
        field = key.split(".")[-1]
        try:
            sub.__pydantic_validator__.validate_assignment(sub, field, value)
            applied.append(key)
        except ValidationError:
            logger.warning("热更新参数非法,已忽略: %s=%r", key, value)
    return applied


def dump_config(cfg: AppConfig, path: str | Path) -> None:
    """把当前配置写回 YAML(供 ROI 等网页设置持久化)。"""
    data = copy.deepcopy(cfg.model_dump())
    Path(path).write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
