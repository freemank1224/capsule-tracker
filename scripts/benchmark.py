"""离线基准测试: 输入视频文件或图像序列,输出各档位 FPS、平均延迟、
ID 切换数、计数误差 (v1.0 §12)。

v1.2 §7 新增: --device cuda|cpu 与 --half auto|true|false 参数,
输出 FP16 / FP32 对比与各档位 FPS。

用法:
  python scripts/benchmark.py --source test.mp4
  python scripts/benchmark.py --source test.mp4 --tier gpu,gpu_light
  python scripts/benchmark.py --source test.mp4 --device cuda --half auto
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.config import load_config  # noqa: E402
from app.detectors import create_detector  # noqa: E402
from app.hardware import collect_report, tier_profile  # noqa: E402
from app.pipeline import SessionPipeline  # noqa: E402

logger = logging.getLogger("benchmark")

ALL_TIERS = ["gpu_high", "gpu", "gpu_light", "cpu_balanced", "cpu_lite"]


def load_frames(source: str, max_frames: int, width: int) -> list[np.ndarray]:
    frames = []
    p = Path(source)
    if p.is_dir():  # 图像序列
        files = sorted(p.glob("*.jpg")) + sorted(p.glob("*.png"))
        for f in files[:max_frames]:
            img = cv2.imread(str(f))
            if img is not None:
                frames.append(img)
    else:  # 视频文件
        cap = cv2.VideoCapture(source)
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
    if width:
        frames = [cv2.resize(f, (width, round(f.shape[0] * width / f.shape[1])))
                  for f in frames]
    return frames


def run_tier(cfg, frames: list, tier: str, device: str, warn_cb) -> dict:
    profile = tier_profile(tier, cfg.hardware.vram_mb)
    cfg.detector.imgsz = profile.imgsz
    cfg.detector.half = "false" if device == "cpu" else cfg.detector.half

    detector = create_detector(cfg, device=device, warn_cb=warn_cb)
    detector.warmup((profile.imgsz, profile.imgsz))
    pipeline = SessionPipeline(cfg, profile, detector)

    lat_detect, lat_total = [], []
    id_sets, init_ids = [], set()
    new_ids_after_init = 0
    t_start = time.perf_counter()
    for i, frame in enumerate(frames):
        result = pipeline.process(cv2.imencode(".jpg", frame)[1].tobytes())
        lat_detect.append(result["latency_ms"]["detect"])
        lat_total.append(result["latency_ms"]["total"])
        ids = {o["id"] for o in result["objs"]}
        if i >= 5:
            new_ids_after_init += len(ids - init_ids)
            init_ids |= ids
        id_sets.append(ids)
    wall = time.perf_counter() - t_start

    # ID 切换数: 稳定期后新建轨迹数(近似)
    detector.close()
    return {
        "tier": tier,
        "fps": len(frames) / max(wall, 1e-9),
        "avg_detect_ms": float(np.mean(lat_detect)) if lat_detect else 0.0,
        "avg_total_ms": float(np.mean(lat_total)) if lat_total else 0.0,
        "final_count": len(pipeline.counter.seen_ids),
        "new_ids_after_init": new_ids_after_init,
        "imgsz": profile.imgsz,
        "detect_every": profile.detect_every,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="胶囊跟踪 基准测试")
    ap.add_argument("--source", required=True, help="视频文件或图像序列目录")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--tier", default="auto",
                    help="逗号分隔的档位列表,auto=按硬件选一个;all=全部档位")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--half", default=None, choices=["auto", "true", "false"],
                    help="覆盖 detector.half;cuda 下用 auto 可输出 FP16/FP32 对比")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--width", type=int, default=1920, help="测试帧统一缩放宽度")
    ap.add_argument("--gt-count", type=int, default=None,
                    help="人工点数的真实计数,用于输出计数误差")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    logging.getLogger("selfcheck").setLevel(logging.WARNING)

    frames = load_frames(args.source, args.max_frames, args.width)
    if not frames:
        print(f"未读取到帧: {args.source}")
        return 2
    print(f"测试序列: {len(frames)} 帧, {frames[0].shape[1]}x{frames[0].shape[0]}")

    cfg = load_config(args.config)
    if args.half:
        cfg.detector.half = args.half

    # 设备与档位
    rep = collect_report()
    if args.device == "cuda" and not (rep.torch and rep.torch.is_available):
        print("CUDA 不可用,请检查 PyTorch 安装(CUDA >= 12.8)与驱动")
        return 2
    if args.device == "auto":
        args.device = "cuda" if (rep.torch and rep.torch.is_available
                                 and rep.gpu) else "cpu"
    print(f"设备: {args.device} | torch={rep.torch.version if rep.torch else '未安装'} "
          f"cuda={rep.torch.cuda_version if rep.torch else '-'}")

    if args.tier == "all":
        tiers = [t for t in ALL_TIERS if not t.startswith("gpu")
                 or (args.device == "cuda" and rep.gpu)]
        if args.device == "cuda":
            tiers = [t for t in tiers if t.startswith("gpu")] or ["gpu"]
    elif args.tier == "auto":
        from app.selfcheck import run_selfcheck
        _, tier, _ = run_selfcheck(cfg.tier)
        tiers = [tier]
    else:
        tiers = args.tier.split(",")

    warn_cb = lambda m: print(f"  [warn] {m}")  # noqa: E731
    results = []
    for tier in tiers:
        print(f"\n--- 档位 {tier} (imgsz={tier_profile(tier, cfg.hardware.vram_mb).imgsz}) ---")
        try:
            r = run_tier(cfg.model_copy(deep=True), frames, tier, args.device, warn_cb)
        except Exception as e:
            print(f"  档位 {tier} 失败: {type(e).__name__}: {e}")
            continue
        r["id_switches"] = r.pop("new_ids_after_init")
        r["count_error"] = (None if args.gt_count is None
                            else abs(r["final_count"] - args.gt_count)
                            / max(args.gt_count, 1) * 100.0)
        results.append(r)
        print(f"  FPS={r['fps']:.1f} | 平均检测 {r['avg_detect_ms']:.0f} ms | "
              f"平均总延迟 {r['avg_total_ms']:.0f} ms | 计数 {r['final_count']} | "
              f"ID 切换(近似) {r['id_switches']}"
              + (f" | 计数误差 {r['count_error']:.1f}%" if r["count_error"] is not None else ""))

    # FP16 / FP32 对比 (v1.2 §7 / 验收项 16)
    if args.device == "cuda" and cfg.detector.backend == "ultralytics":
        print("\n--- FP16 / FP32 对比 (FP32 与 FP16 各测一段) ---")
        for half in ("false", "true"):
            cfg2 = load_config(args.config)
            cfg2.detector.half = half
            try:
                r = run_tier(cfg2, frames[: min(60, len(frames))], "gpu",
                             "cuda", warn_cb)
                print(f"  half={half:>5}: 平均检测 {r['avg_detect_ms']:.1f} ms | "
                      f"FPS {r['fps']:.1f}")
            except Exception as e:
                print(f"  half={half}: 失败 {type(e).__name__}: {e}")

    if len(results) > 1:
        print("\n=== 各档位 FPS 汇总 ===")
        for r in results:
            print(f"  {r['tier']:<13} FPS {r['fps']:>6.1f} | 延迟 {r['avg_total_ms']:>6.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
