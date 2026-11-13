"""模型导出 ONNX: 把 ultralytics 训练好的 .pt 导出为 onnx_det 后端可用的模型。

用法: python scripts/export_onnx.py --weights models/capsule.pt --imgsz 960
说明: onnxruntime 与 onnxruntime-gpu 不可共存;CPU 推理装 onnxruntime 即可。
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/capsule.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    if not Path(args.weights).exists():
        print(f"权重不存在: {args.weights}")
        return 2

    from ultralytics import YOLO
    model = YOLO(args.weights)
    out = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset,
                       simplify=True, dynamic=False)
    print(f"已导出: {out}")
    print("在 config.yaml 中设置: detector.backend: onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
