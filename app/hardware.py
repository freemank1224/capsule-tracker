"""硬件探测与性能档位选择。

v1.2 补丁要点:
- GPU 路线仅支持 NVIDIA CUDA(无 DirectML / ROCm)。
- select_tier(vram_mb, capability, has_cuda) 为纯函数,便于单元测试。
- 仅按显存与计算能力自动选择,显卡型号不写死;配置可强制覆盖。
"""
from __future__ import annotations

import logging
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# 显存档位默认阈值(MB),可被 config.hardware 覆盖
DEFAULT_VRAM_HIGH_MB = 20480   # >= -> gpu_high
DEFAULT_VRAM_MB = 12288        # >= -> gpu
DEFAULT_VRAM_LIGHT_MB = 7168   # <  -> gpu_light

RTX50_CAPABILITY = (12, 0)     # Blackwell sm_120
MIN_CUDA_FOR_RTX50 = (12, 8)


# ---------------------------------------------------------------- 数据结构

@dataclass
class GpuInfo:
    name: str
    vram_mb: int
    driver_version: str = ""
    capability: Optional[tuple] = None  # torch.cuda.get_device_capability()


@dataclass
class TorchInfo:
    version: str = ""
    cuda_version: str = ""       # torch.version.cuda,空串表示 CPU 版构建
    is_available: bool = False
    is_cuda_build: bool = False
    capability: Optional[tuple] = None  # torch.cuda.get_device_capability()


@dataclass
class HardwareReport:
    gpu: Optional[GpuInfo] = None             # nvidia-smi 探测结果
    torch: Optional[TorchInfo] = None         # torch 未安装为 None
    ort_package: str = ""                     # onnxruntime / onnxruntime-gpu / ""
    ort_providers: list[str] = field(default_factory=list)
    cpu_cores: int = 0
    mem_gb: float = 0.0
    os: str = ""


# ---------------------------------------------------------------- 探测

def probe_nvidia_smi() -> Optional[GpuInfo]:
    """nvidia-smi 查询显卡信息;命令不存在或失败视为无 NVIDIA 显卡 (v1.2 §2.3/§3.2)。"""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
    if len(parts) < 3:
        return None
    try:
        vram_mb = int(re.sub(r"[^0-9]", "", parts[1]) or 0)
    except ValueError:
        vram_mb = 0
    return GpuInfo(name=parts[0], vram_mb=vram_mb, driver_version=parts[2])


def _import_torch():
    import torch  # 延迟导入;未安装时抛 ImportError
    return torch


def probe_torch() -> Optional[TorchInfo]:
    try:
        torch = _import_torch()
    except ImportError:
        return None
    info = TorchInfo(version=torch.__version__, cuda_version=torch.version.cuda or "")
    info.is_cuda_build = bool(info.cuda_version)
    try:
        info.is_available = info.is_cuda_build and torch.cuda.is_available()
        if info.is_available:
            cap = torch.cuda.get_device_capability(0)
            info.capability = (int(cap[0]), int(cap[1]))
    except Exception as e:  # 驱动缺失等运行时错误
        logger.warning("torch.cuda 查询失败: %s", e)
        info.is_available = False
    return info


def probe_onnxruntime() -> tuple[str, list[str]]:
    """探测 onnxruntime 包名与可用 Provider;两者都不装返回 ("", [])。"""
    try:
        import onnxruntime as ort
    except ImportError:
        return "", []
    pkg = "onnxruntime"
    try:
        from importlib.metadata import metadata
        meta = metadata("onnxruntime-gpu")
        if meta is not None:
            pkg = "onnxruntime-gpu"
    except Exception:
        pass
    return pkg, list(ort.get_available_providers())


def collect_report(cfg_vram_thresholds: Optional[dict] = None) -> HardwareReport:
    """汇总硬件信息(不抛异常,任何探测失败都视为对应能力缺失)。"""
    rep = HardwareReport(
        gpu=probe_nvidia_smi(),
        torch=probe_torch(),
        cpu_cores=_cpu_cores(),
        mem_gb=_mem_gb(),
        os=f"{platform.system()} {platform.release()}",
    )
    rep.ort_package, rep.ort_providers = probe_onnxruntime()
    if rep.gpu is not None and rep.torch is not None and rep.torch.is_available:
        # 用 torch 读取的计算能力补充到 nvidia-smi 结果上
        rep.gpu.capability = rep.torch.capability
    return rep


def _cpu_cores() -> int:
    try:
        import os
        return os.cpu_count() or 0
    except Exception:
        return 0


def _mem_gb() -> float:
    try:
        import psutil  # 可选依赖
        return round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        try:
            import ctypes
            ctypes.windll.kernel32.GlobalMemoryStatusEx  # noqa: B018
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMORYSTATUSEX(dwLength=ctypes.sizeof(MEMORYSTATUSEX))
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return round(st.ullTotalPhys / 1024**3, 1)
        except Exception:
            return 0.0


# ---------------------------------------------------------------- 档位映射

def select_tier(vram_mb: int,
                capability: Optional[tuple],
                has_cuda: bool,
                cpu_cores: int = 0,
                thresholds: Optional[dict] = None) -> tuple[str, str]:
    """纯函数: 依据显存/计算能力/CUDA 可用性选择档位 (v1.2 §3 表格)。

    返回 (tier, reason)。显卡型号不参与判断。
    """
    th = {**{"high": DEFAULT_VRAM_HIGH_MB, "mid": DEFAULT_VRAM_MB, "light": DEFAULT_VRAM_LIGHT_MB},
          **(thresholds or {})}
    if not has_cuda:
        if cpu_cores >= 6:
            return "cpu_balanced", f"无可用 CUDA,CPU 物理核数 {cpu_cores} >= 6"
        return "cpu_lite", f"无可用 CUDA,CPU 物理核数 {cpu_cores} < 6"
    if vram_mb >= th["high"]:
        return "gpu_high", f"CUDA 可用,显存 {vram_mb} MB >= {th['high']} MB"
    if vram_mb >= th["mid"]:
        return "gpu", f"CUDA 可用,显存 {vram_mb} MB ∈ [{th['mid']}, {th['high']}) MB"
    if vram_mb >= th["light"]:
        return "gpu", (f"CUDA 可用,显存 {vram_mb} MB ∈ [{th['light']}, {th['mid']}) MB,"
                       f"imgsz 取 960 起步")
    return "gpu_light", f"CUDA 可用,显存 {vram_mb} MB < {th['light']} MB(建议 nano/small 模型)"


def version_tuple(v: str) -> tuple:
    """'12.8.1' -> (12, 8, 1);解析失败返回 ()。"""
    try:
        return tuple(int(x) for x in re.findall(r"\d+", v)[:3])
    except ValueError:
        return ()


@dataclass
class Issue:
    level: str   # "ERROR" | "WARNING" | "INFO"
    msg: str


def check_environment(rep: HardwareReport,
                      thresholds: Optional[dict] = None) -> list[Issue]:
    """启动自检判定 (v1.2 §4 表格),纯函数。"""
    issues: list[Issue] = []
    torch = rep.torch
    has_nvidia_gpu = rep.gpu is not None

    if not has_nvidia_gpu:
        issues.append(Issue("INFO", "未检测到 NVIDIA 显卡,使用 CPU 档位"))

    if has_nvidia_gpu and (torch is None or not torch.is_cuda_build):
        issues.append(Issue(
            "WARNING",
            "检测到 NVIDIA 显卡但安装的是 CPU 版 PyTorch,"
            "请按 README 重装 CUDA 版(>= 12.8);本次回退 CPU 档"))

    if torch is not None and torch.is_cuda_build and not torch.is_available:
        issues.append(Issue(
            "WARNING",
            "PyTorch 为 CUDA 版但 torch.cuda.is_available()=False,"
            "请更新 NVIDIA 驱动或检查兼容性;本次回退 CPU 档"))

    if has_nvidia_gpu and torch is not None and torch.is_available:
        cap = torch.capability or rep.gpu.capability
        cuda_v = version_tuple(torch.cuda_version)
        if cap is not None and cap >= RTX50_CAPABILITY and cuda_v and cuda_v < MIN_CUDA_FOR_RTX50:
            issues.append(Issue(
                "ERROR",
                f"RTX 50 系(计算能力 {cap[0]}.{cap[1]})需要 CUDA >= 12.8 的 PyTorch,"
                f"当前构建 CUDA 版本为 {torch.cuda_version},请重装;本次回退 CPU 档"))

    if rep.ort_package and _both_ort_installed():
        other = "onnxruntime-gpu" if rep.ort_package == "onnxruntime" else "onnxruntime"
        issues.append(Issue(
            "WARNING",
            f"同时安装了 onnxruntime 与 onnxruntime-gpu,二者不可共存,请卸载其一:"
            f" pip uninstall -y {rep.ort_package} {other}"))

    return issues


def _both_ort_installed() -> bool:
    try:
        from importlib.metadata import distributions
        names = {d.metadata["Name"].lower() for d in distributions() if d.metadata["Name"]}
        return "onnxruntime" in names and "onnxruntime-gpu" in names
    except Exception:
        return False


# ---------------------------------------------------------------- 档位参数

@dataclass
class TierProfile:
    tier: str
    imgsz: int
    imgsz_max: int          # 允许的上限(OOM 自动降级的起点回退)
    detect_every: int       # 每 N 帧检测一次,中间帧卡尔曼预测
    change_detect: bool     # 是否启用变化检测省算力
    tile: bool              # 是否建议开切片推理


def tier_profile(tier: str, vram_mb: int = 0) -> TierProfile:
    """把档位映射为具体运行参数 (v1.0 §7 + v1.2 §3)。"""
    if tier == "gpu_high":
        return TierProfile("gpu_high", imgsz=1280, imgsz_max=1920,
                           detect_every=1, change_detect=False, tile=False)
    if tier == "gpu":
        # 8GB 卡(5060)取 960 起步;>=12GB 可用 1280
        imgsz = 1280 if vram_mb >= DEFAULT_VRAM_MB else 960
        return TierProfile("gpu", imgsz=imgsz, imgsz_max=1920,
                           detect_every=1, change_detect=False, tile=False)
    if tier == "gpu_light":
        return TierProfile("gpu_light", imgsz=960, imgsz_max=960,
                           detect_every=1, change_detect=True, tile=False)
    if tier == "cpu_balanced":
        return TierProfile("cpu_balanced", imgsz=800, imgsz_max=800,
                           detect_every=2, change_detect=True, tile=False)
    return TierProfile("cpu_lite", imgsz=640, imgsz_max=640,
                       detect_every=3, change_detect=True, tile=True)


TIER_ORDER = ["cpu_lite", "cpu_balanced", "gpu_light", "gpu", "gpu_high"]


def choose_tier(rep: HardwareReport, cfg_tier: str,
                thresholds: Optional[dict] = None) -> tuple[str, list[Issue]]:
    """综合自检结果选择最终档位;配置可强制覆盖。返回 (tier, issues)。"""
    issues = check_environment(rep, thresholds)
    blocking = any(i.level == "ERROR" for i in issues)
    has_cuda = bool(
        rep.torch and rep.torch.is_cuda_build and rep.torch.is_available
    ) and rep.gpu is not None and not blocking
    vram = rep.gpu.vram_mb if has_cuda else 0

    if cfg_tier != "auto":
        if not cfg_tier.startswith("gpu"):
            return cfg_tier, issues  # 强制 CPU 档
        if has_cuda:
            return cfg_tier, issues  # 强制 GPU 档且可用
        issues.append(Issue("WARNING",
                            f"配置强制 {cfg_tier} 但 CUDA 不可用,回退自动选择"))
    tier, reason = select_tier(vram, rep.torch.capability if rep.torch else None,
                               has_cuda, rep.cpu_cores, thresholds)
    issues.append(Issue("INFO", f"档位选择: {tier} ({reason})"))
    return tier, issues
