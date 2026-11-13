"""启动自检 (v1.2 §4): 打印显卡/驱动/计算能力、torch 与 CUDA 版本、
onnxruntime 包名与 Providers、最终档位及原因。"""
from __future__ import annotations

import logging

from .hardware import (
    Issue, HardwareReport, TierProfile, collect_report, choose_tier, tier_profile,
)

logger = logging.getLogger("selfcheck")


def log_report(rep: HardwareReport) -> None:
    if rep.gpu is not None:
        cap = rep.gpu.capability
        cap_str = f"{cap[0]}.{cap[1]}" if cap else "未知"
        logger.info("显卡: %s | 显存: %d MB | 驱动: %s | 计算能力: %s",
                    rep.gpu.name, rep.gpu.vram_mb,
                    rep.gpu.driver_version or "未知", cap_str)
    else:
        logger.info("显卡: 未检测到 NVIDIA 显卡")

    if rep.torch is not None:
        logger.info("PyTorch: %s | 构建CUDA: %s | torch.cuda.is_available: %s",
                    rep.torch.version, rep.torch.cuda_version or "(CPU 版构建)",
                    rep.torch.is_available)
    else:
        logger.info("PyTorch: 未安装")

    if rep.ort_package:
        logger.info("onnxruntime 包: %s | Providers: %s",
                    rep.ort_package, rep.ort_providers)
    else:
        logger.info("onnxruntime: 未安装 (onnx_det 后端不可用)")

    logger.info("CPU 核数: %d | 内存: %.1f GB | 系统: %s",
                rep.cpu_cores, rep.mem_gb, rep.os)


def log_issues(issues: list[Issue]) -> None:
    for it in issues:
        if it.level == "ERROR":
            logger.error("%s", it.msg)
        elif it.level == "WARNING":
            logger.warning("%s", it.msg)
        else:
            logger.info("%s", it.msg)


def run_selfcheck(cfg_tier: str = "auto",
                  thresholds: dict | None = None,
                  rep: HardwareReport | None = None) -> tuple[HardwareReport, str, TierProfile]:
    """执行探测 -> 自检日志 -> 档位选择;返回 (报告, 档位, 档位参数)。"""
    if rep is None:
        rep = collect_report()
    logger.info("=" * 60)
    logger.info("启动自检 (v1.2 §4)")
    log_report(rep)
    tier, issues = choose_tier(rep, cfg_tier, thresholds)
    log_issues(issues)
    prof = tier_profile(tier, rep.gpu.vram_mb if rep.gpu else 0)
    logger.info("最终档位: %s (%s) | imgsz=%d detect_every=%d",
                tier, "配置强制" if cfg_tier != "auto" else "自动选择",
                prof.imgsz, prof.detect_every)
    logger.info("=" * 60)
    return rep, tier, prof
