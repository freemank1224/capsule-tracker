# 开发文档补丁 v1.2:NVIDIA CUDA 多显卡兼容(合并并取代 v1.1)

与《胶囊多目标跟踪系统 开发文档 v1.0》配合使用。**本补丁取代 v1.1;若与 v1.0 冲突,以本补丁为准。**

## 1. 目标显卡与范围
- GPU 路线**仅支持 NVIDIA CUDA**。删除 v1.0 中所有 DirectML / ROCm / 非 NVIDIA 显卡内容。
- 必须用**同一套代码、同一个模型文件、同一个 PyTorch CUDA 环境**兼容下列显卡:

| 显卡 | 架构 | 显存 | 档位 |
|---|---|---|---|
| 笔记本 RTX 3060 | Ampere | 6 GB | `gpu_light` |
| 笔记本 RTX 5060 | Blackwell (sm_120) | 8 GB | `gpu` |
| 台式 RTX 4060 Ti | Ada | 16 GB | `gpu` |
| 台式 RTX 3090 | Ampere | 24 GB | `gpu_high` |

- 无 NVIDIA 显卡或 CUDA 不可用时,走 CPU 档位(`cpu_balanced` / `cpu_lite`),行为同 v1.0。
- 不使用 TensorRT(需按显卡架构分别编译引擎,增加兼容成本;本场景帧率需求低,PyTorch 已足够)。

## 2. 环境要求(关键)
1. **PyTorch 必须使用 CUDA ≥ 12.8 的构建**。原因:RTX 5060 为 Blackwell 架构(sm_120),旧版 CUDA 构建(如 cu121、cu126)不含其内核,运行时会报 "CUDA capability sm_120 is not compatible"。老卡(3060 / 4060 Ti / 3090)在 cu128+ 上同样可用,因此**所有机器统一安装 cu128 及以上版本**。
2. 安装方式:到 PyTorch 官网 "Get Started" 选 Windows / Pip / CUDA(≥12.8),复制 `pip install torch torchvision --index-url ...`。`setup_windows.ps1` 提供 `-TorchIndexUrl <url>` 参数;**脚本中不硬编码具体版本号**,但在 README 中说明 "CUDA 版本须 ≥ 12.8"。
3. 驱动:先运行 `nvidia-smi`,能输出即驱动可用。RTX 50 系需要较新的驱动,README 提示"请从 NVIDIA 官网更新到最新 Game Ready / Studio 驱动"。
4. pip 版 PyTorch 自带 CUDA 运行库,不需要单独安装 CUDA Toolkit。
5. `onnxruntime` 与 `onnxruntime-gpu` 二选一,不可共存。**ONNX GPU 路线不作为默认**(CUDA / cuDNN 版本匹配容易出错),仅在 CPU 降级和高级用户场景使用;`CUDAExecutionProvider` 不可用时自动回退 CPU 并发 `warn`。
6. 默认路线:PyTorch CUDA + `ultralytics_det`。

## 3. 对 v1.0 的具体修改
1. §6.1 `onnx_det`:Provider 优先级改为 `CUDAExecutionProvider → CPUExecutionProvider`,删除 `DmlExecutionProvider`。
2. §7 硬件探测:增加 `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader` 以及 `torch.cuda.get_device_capability()`,读取型号、显存、驱动、计算能力。命令不存在视为无 NVIDIA 显卡。
3. §7 档位表替换为下表(显存阈值可在配置中改):

| 档位 | 触发条件 | 检测输入 `imgsz` | 切片推理 | 备注 |
|---|---|---|---|---|
| `gpu_high` | CUDA 可用,显存 ≥ 20 GB | 1280(上限 1920) | 按图像尺寸触发 | 可选更大模型 |
| `gpu` | CUDA 可用,显存 ≥ 12 GB 且 < 20 GB;或 7 GB ≤ 显存 < 12 GB 时 `imgsz` 取 960~1280 | 960~1280 | 按需 | 8 GB 卡(5060)取 960 起步 |
| `gpu_light` | CUDA 可用,显存 < 7 GB | 960 | 默认关,按需开 | 6 GB 卡(3060)首选 nano / small 模型 |
| `cpu_balanced` / `cpu_lite` | 无可用 CUDA | 见 v1.0 §7 | — | — |

- 显卡型号不写死;**仅按显存与计算能力自动选择**,配置文件可强制覆盖。
- 笔记本显卡可能因散热降频:沿用 v1.0 的"自适应降级"(延迟超过 `target_latency_ms` 自动下调一级,稳定后才升回)。

## 4. 启动自检(server.py 必须实现)
启动日志输出:显卡型号、显存(MB)、驱动版本、计算能力;`torch.__version__`、`torch.version.cuda`、`torch.cuda.is_available()`;onnxruntime 包名与 Providers;最终选定的档位及原因。

| 情况 | 行为 |
|---|---|
| 检测到 NVIDIA 显卡,但 torch 为 CPU 版 | WARNING:"检测到 NVIDIA 显卡但安装的是 CPU 版 PyTorch,请按 README 重装 CUDA 版(≥12.8)";回退 CPU 档 |
| 计算能力为 12.0(sm_120)但 `torch.version.cuda` < 12.8 | **ERROR 级别提示**:"RTX 50 系需要 CUDA ≥ 12.8 的 PyTorch,请重装";回退 CPU 档 |
| torch 为 CUDA 版但 `is_available()` 为假 | WARNING:提示更新驱动或检查兼容性;回退 CPU 档 |
| 同时安装 onnxruntime 与 onnxruntime-gpu | WARNING:提示卸载其一 |
| 无 NVIDIA 显卡 | INFO:使用 CPU 档位 |

## 5. GPU 性能与稳定性
1. **FP16 不再默认强制开启**:启动时用 `warmup` 的同一输入各测 FP32 与 FP16 平均耗时(各 N 次,默认 10),自动选更快者并写入日志;配置项 `detector.half: auto|true|false`,默认 `auto`。出现数值异常可手动设为 `false`。
2. 启动 warmup 至少 3 次(使用最终 imgsz),避免首帧高延迟。
3. 解码、推理放入线程池 / `run_in_executor`,不阻塞事件循环;推理使用互斥锁,避免多连接争用同一 GPU。
4. 显存保护:捕获 CUDA OOM,调用 `torch.cuda.empty_cache()`,自动把 `imgsz` 降一级(如 1920→1280→960→800),向客户端发 `warn`;降到最低仍失败则回退 CPU 档。
5. 同一模型文件(单个 `.pt`)必须在四类显卡上都能直接运行;禁止为不同显卡生成不同模型或引擎文件。
6. 模型规格建议:默认 nano / small 级别检测器,确保 6 GB 显存也能稳定运行;大模型仅在 `gpu_high` 档作为可选项。

## 6. 新增 / 修改的验收项
| # | 验收项 | 通过标准 |
|---|---|---|
| 11 | CUDA 自检 | 启动日志按 §4 打印完整信息;故意装 CPU 版 torch 时出现对应警告并回退 CPU |
| 12 | Blackwell 检查 | 用低于 12.8 的 CUDA 构建模拟 sm_120 场景(可通过单元测试 mock `get_device_capability` 与 `torch.version.cuda`)时,出现 §4 的 ERROR 提示 |
| 13 | 档位映射 | 单元测试 mock 显存 6 / 8 / 16 / 24 GB,分别得到 `gpu_light` / `gpu` / `gpu` / `gpu_high` |
| 14 | GPU 加速生效 | `gpu*` 档单帧检测耗时明显低于 `cpu_balanced`(benchmark 输出对比) |
| 15 | OOM 保护 | 人为设置极大 `imgsz` 触发 OOM,服务不崩溃,自动降级并通知客户端 |
| 16 | FP16 自动选择 | `half: auto` 时日志给出 FP32 / FP16 耗时对比与最终选择;`half: false` 仍可运行 |
| 17 | 单模型兼容 | 同一 `.pt` 文件在 6 GB / 8 GB / 16 GB / 24 GB 显卡上均可启动运行(实机可用时验证) |

## 7. 任务补充(并入 v1.0 §12 阶段 3)
- 实现 §3 档位映射(纯函数 `select_tier(vram_mb, capability, has_cuda)`,便于单元测试)。
- 实现 §4 启动自检与 §5 的 FP16 自动选择、OOM 降级。
- `benchmark.py` 新增 `--device cuda|cpu` 与 `--half auto|true|false` 参数,输出 FP16 / FP32 对比与各档位 FPS。
- `setup_windows.ps1` 增加 `-TorchIndexUrl` 参数、`nvidia-smi` 检查,并在 README 中写明"CUDA ≥ 12.8"要求。
