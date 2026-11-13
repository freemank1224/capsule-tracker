# 胶囊多目标跟踪计数系统 (capsule-tracker)

iPhone Safari 网页取景 → WSS 上传 JPEG → Windows 主机完成 **检测 + 多目标跟踪 + 计数** → 结果回显叠加到手机页面。

```
 iPhone Safari (index.html)                      Windows 主机 (Python)
┌──────────────────────────┐   WSS (二进制 JPEG)   ┌───────────────────────────────┐
│ getUserMedia(后置相机)    │ ───────────────────▶ │ FastAPI + uvicorn (HTTPS)       │
│ 定时抓帧 → canvas → JPEG │                       │  ├─ 解码 (cv2.imdecode)         │
│ torch 尝试开启(可选)     │ ◀─────────────────── │  ├─ Preprocess (可选归一化)      │
│ 叠加框 / ID / 计数        │   WSS (JSON 结果)     │  ├─ Detector (可插拔后端)        │
└──────────────────────────┘                       │  ├─ Tracker (自实现 ByteTrack)   │
                                                   │  └─ Counter / 状态机             │
                                                   └───────────────────────────────┘
```

## 亮点: Zero-shot 无需训练

**不需要针对每个物体训练模型。** 默认使用 **YOLO-World v2** 开放词汇检测器:
在网页"目标类别"框里输入任何文本(如 `capsule, pill, screw`),服务端即时切换检测类别,零样本运行。

- 模型: `yolov8s-worldv2.pt`(首次运行自动下载,约 20 MB),6GB 显卡可稳定运行
- 少样本(Few-shot)进阶: 可换 `YOLOE` 模型(`yoloe-11s-seg.pt`),支持"给一张目标图做视觉提示"
- 物体种类变化时只需改文字提示,无需重新训练
- 仍想自训练专属模型? 用网页"保存当前帧"采集 → Label Studio 标注 → Ultralytics 训练 → 替换 `detector.weights` 即可(可选,非必需)

## 环境要求

- Windows 10/11,Python 3.10+(GPU 路线: PyTorch **CUDA 版本必须 ≥ 12.8**,RTX 50 系/Blackwell sm_120 只在 cu128+ 构建上有内核)
- 有 NVIDIA 显卡: `nvidia-smi` 能输出即可;RTX 50 系请更新到最新 Game Ready / Studio 驱动
- pip 版 PyTorch 自带 CUDA 运行库,**无需单独安装 CUDA Toolkit**
- 无 NVIDIA 显卡时自动走 CPU 档位(目标 ≥ 2 FPS)
- **不使用 TensorRT**(需按架构分别编译引擎,本场景帧率需求低,PyTorch 已足够)

### PyTorch 安装(二选一)

1. **复用已有安装(推荐)**: 若机器上已有含 torch 的环境(如 ComfyUI 便携版 `python_embeded`,通常已是 cu12x 构建),直接用它运行本项目即可,无需任何下载:
   ```
   <ComfyUI路径>\python_embeded\python.exe -m pip install -r requirements.txt
   <ComfyUI路径>\python_embeded\python.exe server.py
   ```
2. **全新安装**: 到 [PyTorch 官网 Get Started](https://pytorch.org/get-started/locally/) 选 Windows / Pip / **CUDA ≥ 12.8**,复制安装命令(形如 `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`)执行。

> `scripts/setup_windows.ps1` 提供 `-TorchIndexUrl <url>` 参数传入上述 `--index-url` 地址;脚本不硬编码版本号。

## 一键部署

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
# 有 NVIDIA 显卡且想装 GPU 版 torch 时:
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -TorchIndexUrl https://download.pytorch.org/whl/cu128
```

脚本幂等,依次完成: 检查 Python ≥ 3.10 → venv + 依赖 → `nvidia-smi` 检查 → mkcert → 证书生成(gen_cert.ps1,自动取局域网 IP)→ 防火墙规则(需管理员,无权限时打印命令)→ 打印手机访问地址 + ASCII 二维码。

## 启动

```
python server.py [--config config.yaml] [--tier auto]
```

- 默认监听 `0.0.0.0:8443` (HTTPS),证书在 `certs/`;缺失时给出明确报错并提示运行 `gen_cert.ps1`
- 另起 `8080` 端口仅供手机下载根证书 `http://<IP>:8080/cert/rootCA.pem`(可在配置中关为 0)
- 启动日志输出(v1.2 §4 自检): 显卡型号/显存/驱动/计算能力、torch 与 CUDA 版本、onnxruntime 包名与 Providers、最终档位及原因
- 离线基准: `python scripts/benchmark.py --source test.mp4 [--device cuda|cpu] [--half auto|true|false] [--tier all]`

## 手机端(iPhone Safari)

1. 连同一 Wi-Fi,Safari 打开 `http://<IP>:8080/cert/rootCA.pem` 下载描述文件并安装
2. `设置 > 通用 > 关于本机 > 证书信任设置` → 对该根证书**打开完全信任**(图文见 `docs/phone_setup.md`)
3. Safari 打开 `https://<IP>:8443`,允许相机权限(每次询问属正常现象)
4. 页面自动显示检测框/ID/计数;底部抽屉可调上传帧率、JPEG 质量、置信度、**目标类别文本**;可重置计数、设置 ROI、保存当前帧(存入 `captures/`)

## 性能档位(v1.2 §3)

同一套代码、同一个模型文件,按**显存与计算能力**自动选择(显卡型号不写死,`config.yaml` 阈值可改,`tier:` 可强制覆盖):

| 档位 | 触发条件 | imgsz | 切片推理 | 说明 |
|---|---|---|---|---|
| `gpu_high` | CUDA,显存 ≥ 20GB | 1280(上限1920) | 按图像尺寸 | 24GB 卡(3090) |
| `gpu` | CUDA,显存 ≥ 12GB;<12GB 时 imgsz 960 起步 | 960~1280 | 按需 | 8GB 卡(5060)取 960 起步 |
| `gpu_light` | CUDA,显存 < 7GB | 960 | 默认关 | 6GB 卡(3060),优先 nano/small / YOLO-World s |
| `cpu_balanced` | 无 CUDA,核 ≥ 6 | 800 | — | 隔帧检测 + 变化检测 |
| `cpu_lite` | 无 CUDA,核 < 6 | 640 | 可开 | 隔帧检测 + 变化检测 + ROI |

- 运行时**自适应降级**: 最近 30 帧平均延迟超 `target_latency_ms`(默认700)自动下调一级并通过 `warn` 通知网页;持续稳定后才升回,避免抖动
- **OOM 保护**: 捕获 CUDA OOM → `empty_cache()` → imgsz 降一级(1920→1280→960→800→640)并 `warn`;到底仍失败自动回退 CPU 档
- **FP16 自动选择**: `detector.half: auto`(默认)启动时同输入实测 FP32/FP16 各 10 次取更快者并写日志;数值异常可手动设 `false`

## 配置

全部参数见 `config.yaml`(pydantic 校验,非法值启动时报清晰错误)。常用:

- `detector.weights`: 默认 `yolov8s-worldv2.pt`(zero-shot);换自训练 `.pt` 直接改路径
- `detector.classes`: zero-shot 文本类别,网页运行时可改,无需重启
- `detector.half`: `auto|true|false`
- `detector.tile`: 小目标切片推理(长边超 `trigger_size` 时自动切块+跨块NMS)
- `preprocess.roi`: 托盘区域(网页拖拽设置后自动写回配置)
- `change_detect`: 静态托盘省算力(画面无变化跳过检测,`skipped=true`)
- `tier`: `auto` 或强制档位

## 换用自己的模型(可选)

1. 网页"保存当前帧"采集数据(存 `captures/YYYYMMDD/`)
2. 用 Label Studio(免费开源)标注
3. Ultralytics 训练 nano/small 模型(单类 `capsule` 或按形状多类);训练分辨率不低于推理分辨率,样本覆盖疏密与多种光照
4. `python scripts/export_onnx.py --weights models/capsule.pt` 导出 ONNX(可选)
5. 改 `config.yaml` 的 `detector.weights`(+`backend: onnx` 如用 ONNX)

## 检测后端

| 后端 | 依赖 | 用途 |
|---|---|---|
| `ultralytics`(默认) | ultralytics + PyTorch | Zero-shot / 自训练模型,GPU/CPU |
| `onnx` | onnxruntime | 无 ultralytics 依赖;Provider: CUDA → CPU(CUDA 不可用自动回退并 warn) |
| `openvino` | openvino(可选) | Intel 平台 |
| `classical` | 仅 OpenCV | 背景差分,无模型联调/超低算力 |

> ⚠️ **不要同时安装 `onnxruntime` 与 `onnxruntime-gpu`**,二选一(启动自检会警告)。
> ⚠️ **许可提示**: Ultralytics 为 AGPL-3.0。若闭源分发,请使用 `onnx` 后端搭配自行训练/授权的模型(YOLO-World 权重许可另见其仓库)。

## 常见问题

| 现象 | 处理 |
|---|---|
| 手机打不开页面 | 防火墙未放行 8443/8080;主机 IP 变了(重新运行 gen_cert.ps1);路由器**客户端隔离**(AP 隔离)需关闭 |
| 证书警告 | 确认已在"证书信任设置"中**完全信任**根证书(仅安装描述文件不够) |
| 相机黑屏 | 必须是 `https://`;Safari 权限重新允许;切后台回来会自动重连 |
| 帧率低 | 查看启动日志档位;降低上传宽度/帧率;CPU 档开 ROI + classical;GPU 档检查 FP16 是否被禁用 |
| 启动报 "sm_120 is not compatible" | PyTorch CUDA 构建低于 12.8,按上文重装 cu128+ |
| 启动 WARNING "CPU 版 PyTorch" | 装 torch 时没选 CUDA 构建,按上文重装 |
| 网页开不了闪光灯 | iOS 网页大概率不支持 torch 控制,属正常;用外置 LED 补光或"屏幕补光"功能(另一台设备) |
| 检测不到目标 | 调整"目标类别"文本(英文提示词通常更准);降低置信度;光线不足时补光 |
| 首次 zero-shot 卡住 / "SHA256 checksum does not match" | YOLO-World 首次切换类别需下载 CLIP 文本编码器(~350MB),国内网络易截断。运行 `python scripts/download_clip.py` 从 CDN 全量下载并校验 |

## 目录结构

见 `doc/capsule_tracker_dev_spec.md` §4。测试: `.venv\Scripts\python -m pytest tests/ -q`(不依赖 GPU、不下载模型)。
