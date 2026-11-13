"""合成测试序列生成器: 托盘背景 + 若干"胶囊"小目标匀速平移,输出 MP4。

用法: python tests/data/make_synth_video.py --out tests/data/synth.mp4 --n 50
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/data/synth.mp4")
    ap.add_argument("--n", type=int, default=60, help="帧数")
    ap.add_argument("--objs", type=int, default=30, help="目标数")
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    args = ap.parse_args()

    rng = np.random.RandomState(42)
    # 托盘背景: 带纹理的灰色平面
    bg = np.full((args.h, args.w, 3), 90, dtype=np.uint8)
    bg += rng.randint(-8, 8, bg.shape).astype(np.uint8)

    # 目标: 随机位置的彩色小胶囊(椭圆),匀速移动
    pos = rng.uniform([50, 50], [args.w - 100, args.h - 100], size=(args.objs, 2))
    vel = rng.uniform(-2.0, 2.0, size=(args.objs, 2))
    size = rng.uniform([6, 12], [10, 20], size=(args.objs, 2))  # (rx, ry)
    colors = rng.randint(60, 255, size=(args.objs, 3))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, 15, (args.w, args.h))
    for _ in range(args.n):
        frame = bg.copy()
        for i in range(args.objs):
            c = (int(pos[i][0]), int(pos[i][1]))
            cv2.ellipse(frame, c, (int(size[i][0]), int(size[i][1])),
                        rng.randint(0, 180), 0, 360, colors[i].tolist(), -1)
            pos[i] += vel[i]
            # 出界反弹
            for k in range(2):
                if pos[i][k] < 30 or pos[i][k] > (args.w, args.h)[k] - 30:
                    vel[i][k] *= -1
                    pos[i][k] = np.clip(pos[i][k], 30, (args.w, args.h)[k] - 30)
        writer.write(frame)
    writer.release()
    print(f"已生成 {args.out}: {args.n} 帧, {args.objs} 个目标")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
