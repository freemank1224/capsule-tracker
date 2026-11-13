"""入口: python server.py [--config config.yaml] [--tier auto]

启动自检 (v1.2 §4) → 档位选择 → 检测器创建与 warmup → HTTPS 服务 (默认 0.0.0.0:8443)。
另起 HTTP 端口(默认 8080)仅供手机下载 mkcert 根证书。
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# 嵌入式 Python(python_embeded 的 ._pth)不会把脚本目录加进 sys.path,手动补上
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.config import AppConfig, load_config
from app.detectors import create_detector
from app.hardware import collect_report, tier_profile
from app.main import AppContext, build_app
from app.selfcheck import run_selfcheck

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("server")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="胶囊多目标跟踪系统 服务端")
    ap.add_argument("--config", default="config.yaml", help="配置文件路径")
    ap.add_argument("--tier", default=None,
                    help="强制档位: auto|gpu_high|gpu|gpu_light|cpu_balanced|cpu_lite")
    return ap.parse_args()


def start_cert_download_server(port: int, cert_path: str) -> None:
    """仅局域网的证书下载 HTTP 服务(独立线程, stdlib 实现)。"""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    root_ca = Path(cert_path).parent / "rootCA.pem"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") == "/cert/rootCA.pem" and root_ca.exists():
                data = root_ca.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/x-pem-file")
                self.send_header("Content-Disposition",
                                 'attachment; filename="rootCA.pem"')
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_error(404)

        def log_message(self, *args):  # 静默
            pass

    def run():
        try:
            HTTPServer(("0.0.0.0", port), Handler).serve_forever()
        except OSError as e:
            logger.warning("证书下载端口 %d 启动失败: %s", port, e)

    import threading
    threading.Thread(target=run, daemon=True, name="cert-download").start()
    logger.info("证书下载服务: http://<主机IP>:%d/cert/rootCA.pem", port)


def main() -> int:
    args = parse_args()
    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", e)
        return 2
    if args.tier:
        cfg.tier = args.tier

    # ---- 启动自检 (v1.2 §4) ----
    thresholds = {
        "high": cfg.hardware.vram_high_mb,
        "mid": cfg.hardware.vram_mb,
        "light": cfg.hardware.vram_light_mb,
    }
    rep, tier, profile = run_selfcheck(cfg.tier, thresholds)
    # 档位给默认 imgsz;用户在配置中显式设定的值优先(小目标场景常需调高)
    cfg.detector.imgsz = cfg.detector.imgsz or profile.imgsz
    if cfg.hardware.force_cpu and tier.startswith("gpu"):
        logger.info("hardware.force_cpu=true,已忽略 GPU 档位")
        tier = "cpu_balanced" if rep.cpu_cores >= 6 else "cpu_lite"
        profile = tier_profile(tier)

    # ---- 证书检查 ----
    cert, key = Path(cfg.server.cert), Path(cfg.server.key)
    if not cert.exists() or not key.exists():
        logger.error(
            "HTTPS 证书缺失(%s / %s)。\n"
            "请先运行 scripts\\setup_windows.ps1 或 scripts\\gen_cert.ps1 生成证书;"
            "iOS 摄像头访问与 wss:// 都要求 HTTPS。", cert, key)
        return 2

    # ---- 检测器创建 + warmup ----
    warn_cb = lambda msg: logger.warning("warn: %s", msg)  # noqa: E731
    detector = create_detector(cfg, device="auto", warn_cb=warn_cb)
    try:
        detector.warmup((cfg.detector.imgsz, cfg.detector.imgsz))
    except Exception as e:
        logger.error("检测器 warmup 失败: %s: %s", type(e).__name__, e)
        return 2

    ctx = AppContext(cfg, detector, profile, config_path=args.config)
    app = build_app(ctx)

    if cfg.server.cert_download_port > 0:
        start_cert_download_server(cfg.server.cert_download_port, cfg.server.cert)

    import uvicorn
    host, port = cfg.server.host, cfg.server.port
    logger.info("服务启动: https://<主机IP>:%d (本机访问 https://localhost:%d)",
                port, port)
    uvicorn.run(app, host=host, port=port, ssl_certfile=str(cert),
                ssl_keyfile=str(key), log_level="warning", access_log=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
