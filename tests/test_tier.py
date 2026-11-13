"""档位映射与启动自检单元测试 (v1.2 验收项 11/12/13)。不依赖 GPU。"""
import pytest

from app.hardware import (
    GpuInfo, HardwareReport, Issue, TorchInfo,
    check_environment, select_tier, tier_profile,
)


# ---------- 验收项 13: 档位映射 ----------

@pytest.mark.parametrize("vram_mb,expected", [
    (6144, "gpu_light"),   # 6 GB  (3060)
    (8192, "gpu"),         # 8 GB  (5060)
    (16384, "gpu"),        # 16 GB (4060 Ti)
    (24576, "gpu_high"),   # 24 GB (3090)
])
def test_tier_mapping_by_vram(vram_mb, expected):
    tier, _ = select_tier(vram_mb, capability=(8, 6), has_cuda=True, cpu_cores=8)
    assert tier == expected


def test_tier_no_cuda():
    assert select_tier(0, None, has_cuda=False, cpu_cores=8)[0] == "cpu_balanced"
    assert select_tier(0, None, has_cuda=False, cpu_cores=4)[0] == "cpu_lite"


def test_tier_thresholds_overridable():
    tier, _ = select_tier(9000, (8, 6), True, 8,
                          thresholds={"high": 20000, "mid": 8000, "light": 4000})
    assert tier == "gpu"   # 9000 >= mid(8000),但 < high(20000)


def test_tier_profile_values():
    assert tier_profile("gpu_high").imgsz == 1280
    assert tier_profile("gpu", vram_mb=8192).imgsz == 960   # 5060 取 960 起步
    assert tier_profile("gpu", vram_mb=16384).imgsz == 1280
    assert tier_profile("gpu_light").imgsz == 960
    assert tier_profile("cpu_balanced").detect_every == 2
    assert tier_profile("cpu_lite").detect_every == 3


# ---------- 验收项 11/12: 启动自检判定 ----------

def _report(torch=None, gpu=None, ort="") -> HardwareReport:
    rep = HardwareReport(torch=torch, gpu=gpu, ort_package=ort,
                         ort_providers=["CPUExecutionProvider"] if ort else [],
                         cpu_cores=8, mem_gb=16.0, os="Windows 10")
    return rep


def test_no_nvidia_gpu_info():
    issues = check_environment(_report(torch=TorchInfo(version="2.4.0")))
    assert any(i.level == "INFO" and "CPU 档位" in i.msg for i in issues)


def test_cpu_torch_with_nvidia_gpu_warning():
    gpu = GpuInfo(name="RTX 4060 Ti", vram_mb=16384, driver_version="566.36")
    torch = TorchInfo(version="2.4.0", cuda_version="", is_cuda_build=False)
    issues = check_environment(_report(torch=torch, gpu=gpu))
    assert any(i.level == "WARNING" and "CPU 版 PyTorch" in i.msg for i in issues)


def test_blackwell_needs_cuda_128_error():
    """验收项 12: sm_120 + CUDA < 12.8 -> ERROR(模拟 mock 场景)。"""
    gpu = GpuInfo(name="RTX 5060", vram_mb=8192, driver_version="572.00")
    torch = TorchInfo(version="2.4.0", cuda_version="12.6", is_cuda_build=True,
                      is_available=True, capability=(12, 0))
    issues = check_environment(_report(torch=torch, gpu=gpu))
    errors = [i for i in issues if i.level == "ERROR"]
    assert errors and "12.8" in errors[0].msg


def test_blackwell_cuda_128_ok():
    gpu = GpuInfo(name="RTX 5060", vram_mb=8192, driver_version="572.00")
    torch = TorchInfo(version="2.8.0", cuda_version="12.8", is_cuda_build=True,
                      is_available=True, capability=(12, 0))
    issues = check_environment(_report(torch=torch, gpu=gpu))
    assert not any(i.level == "ERROR" for i in issues)


def test_cuda_build_unavailable_warning():
    gpu = GpuInfo(name="RTX 3090", vram_mb=24576)
    torch = TorchInfo(version="2.4.0", cuda_version="12.8", is_cuda_build=True,
                      is_available=False)
    issues = check_environment(_report(torch=torch, gpu=gpu))
    assert any(i.level == "WARNING" and "驱动" in i.msg for i in issues)


def test_both_onnxruntime_packages_warning(monkeypatch):
    import app.hardware as hw
    monkeypatch.setattr(hw, "_both_ort_installed", lambda: True)
    issues = check_environment(_report(ort="onnxruntime"))
    assert any(i.level == "WARNING" and "不可共存" in i.msg for i in issues)


def test_old_driver_ampere_ok():
    gpu = GpuInfo(name="RTX 3060", vram_mb=6144, driver_version="511.23")
    torch = TorchInfo(version="2.8.0", cuda_version="12.8", is_cuda_build=True,
                      is_available=True, capability=(8, 6))
    issues = check_environment(_report(torch=torch, gpu=gpu))
    assert not any(i.level == "ERROR" for i in issues)
    assert not any(i.level == "WARNING" for i in issues)
