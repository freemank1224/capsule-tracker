"""下载 YOLO-World zero-shot 所需的 CLIP 文本编码器权重。

YOLO-World 的 set_classes() 首次运行会自动下载 CLIP ViT-B/32 (~350MB),
国内网络直连 openaipublic 常因超时截断导致 SHA256 校验失败。
本脚本从同一 CDN 全量下载并校验,放到 ultralytics 期望的位置。

用法: python scripts/download_clip.py
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

# CLIP ViT-B/32 官方权重 URL(文件名含期望的 SHA256 前缀)
CLIP_URL = ("https://openaipublic.azureedge.net/clip/models/"
            "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt")
EXPECTED_SHA = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"


def weights_dir() -> Path:
    """ultralytics 的 WEIGHTS_DIR(默认 settings weights_dir,回退 ./weights)。"""
    try:
        from ultralytics.utils import WEIGHTS_DIR
        return Path(WEIGHTS_DIR)
    except Exception:
        return Path("weights")


def main() -> int:
    dest = weights_dir() / "clip" / "ViT-B-32.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        h = hashlib.sha256(dest.read_bytes()).hexdigest()
        if h.startswith(EXPECTED_SHA[:32]):
            print(f"已存在且校验通过: {dest}")
            return 0
        print(f"{dest} 校验失败,重新下载")

    print(f"下载 CLIP ViT-B/32 (~350MB) -> {dest}")
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(CLIP_URL, tmp)

    h = hashlib.sha256(tmp.read_bytes()).hexdigest()
    if not h.startswith(EXPECTED_SHA[:32]):
        print(f"SHA256 不匹配: {h[:16]}...,下载不完整,请重试")
        return 2
    tmp.rename(dest)
    print(f"完成: {dest} (SHA256 OK)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
