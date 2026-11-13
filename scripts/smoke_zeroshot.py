"""Zero-shot YOLO-World 冒烟测试(完整管线:检测→ByteTrack→计数)。

加载 yolov8s-worldv2,set_classes(文本类别)后对输入图片/视频帧检测,
再过自实现 ByteTrack + Counter,验证开放词汇检测在 GPU 档位可用。

用法:
  python scripts/smoke_zeroshot.py --image path/to.jpg [--classes capsule,pill]
  python scripts/smoke_zeroshot.py --video path/to.mp4  (抽 30 帧连续跟踪)
不传 --image/--video 时用 ultralytics 自带的 bus.jpg 真实照片。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--half", default="auto")
    ap.add_argument("--weights", default="yolov8s-worldv2.pt")
    ap.add_argument("--image", default=None, help="测试图片路径(默认 bus.jpg)")
    ap.add_argument("--video", default=None, help="测试视频路径(抽前 30 帧连续跟踪)")
    ap.add_argument("--classes", default="bus,person")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--tier-imgsz", type=int, default=960)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app.config import load_config
    from app.counting import Counter
    from app.detectors.ultralytics_det import UltralyticsDetector
    from app.tracking.bytetrack import ByteTracker

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    cfg = load_config()
    cfg.detector.weights = args.weights
    cfg.detector.conf = args.conf
    cfg.detector.iou = 0.5
    cfg.detector.imgsz = args.tier_imgsz
    cfg.detector.half = args.half
    cfg.detector.classes = classes

    det = UltralyticsDetector(cfg, device=args.device)
    print(f"类别(文本提示): {classes}")

    tracker = ByteTracker()
    counter = Counter()

    if args.video:
        cap = cv2.VideoCapture(args.video)
        n_frames = 30
        frames = []
        for _ in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            print("视频读取失败", file=sys.stderr)
            return 2
    else:
        img_path = args.image or str(
            Path(__import__("ultralytics").__file__).parent / "assets" / "bus.jpg")
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"读图失败: {img_path}", file=sys.stderr)
            return 2
        frames = [frame] * 12  # 同一帧重复喂给 tracker,验证 ID 稳定

    t0 = time.perf_counter()
    det.warmup((args.tier_imgsz, args.tier_imgsz))
    print(f"warmup(含 FP16 auto 基准): {time.perf_counter() - t0:.1f}s")

    last = []
    t0 = time.perf_counter()
    for i, f in enumerate(frames):
        dets = det.detect(f)
        tracks = tracker.update(dets)
        res = counter.update([t.track_id for t in tracks])
        last = (tracks, res)
    dt = (time.perf_counter() - t0) / len(frames)

    print(f"共 {len(frames)} 帧, 平均 {dt * 1000:.0f} ms/帧 ({1 / dt:.1f} FPS)")
    tracks, res = last
    print(f"末帧目标: {[(t.track_id, round(float(t.score), 2)) for t in tracks]}")
    print(f"计数: count={res.count} 稳定={res.stable} 唯一累计={res.total_unique}")
    ok = len(tracks) > 0 and res.total_unique > 0
    print("冒烟" + ("通过 ✓" if ok else "未通过 ✗"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
