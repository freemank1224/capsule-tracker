# 胶囊多目标跟踪系统 开发文档(交付 Claude Code 实现)

版本:v1.0  目标平台:Windows 10 / Windows 11  形态:纯网页前端 + Python 主机服务

---

## 1. 项目目标

用 iPhone 的后置摄像头拍摄托盘上的"各种形状的小胶囊",视频通过网页(Safari)传到 Windows 主机,主机完成检测 + 多目标跟踪 + 计数,结果回显到手机网页。

### 1.1 硬性约束
1. 全部使用免费 / 开源组件,不使用任何收费软件或服务。
2. 手机端**不开发 iOS App**,只是一个网页(HTTPS)。手机只负责:取景、(可能的)补光、图像上传、结果叠加显示。
3. 所有模型推理在 Windows 主机上运行。
4. 主机有 GPU 时用 GPU;**无 GPU 时 CPU 也必须"勉强可用"**(目标:CPU 下 ≥ 2 FPS 且结果正确)。算法对硬件要求越低越好。
5. 部署方式:`python server.py` 一条命令启动。不依赖 Docker、不依赖 npm 构建(前端是静态 HTML/JS)。
6. 局域网使用,手机与主机在同一 Wi-Fi。

### 1.2 场景特征(影响算法设计)
- 目标:小胶囊,形状多样,外观相似,尺寸小。
- 托盘上平铺,**基本不重叠**,但疏密差异大(稀疏区和密集区并存)。
- 背景固定(托盘),相机俯拍,基本固定。
- 因外观相似且小,**不使用 ReID / 外观特征**,只用运动 + 几何关联。

### 1.3 非目标(本期不做)
- iOS 原生 App、WebRTC、TURN/STUN、云部署、多用户并发、账号体系。
- 模型训练流程的完整实现(只提供接口与简要脚本,见 §11)。

---

## 2. 关键技术事实与限制(实现时必须遵守)

1. **iOS 只允许 HTTPS 页面访问摄像头**。必须提供受信任的本地 HTTPS 证书(用 mkcert 生成),否则 `getUserMedia` 不可用,且 `wss://` 会失败。
2. **iPhone Safari 网页大概率无法控制闪光灯(torch)**。网页只能"尝试"通过 `track.getCapabilities().torch` 检测并 `applyConstraints({advanced:[{torch:true}]})` 开启;检测不到就必须优雅降级,并在页面上提示用户改用外置补光(LED 面板灯 / 环形灯 / 另一台设备的白屏)。**不得假设 torch 可用。**
3. iOS Safari 上网页**无法可靠锁定曝光 / 对焦 / 白平衡**。因此服务端要有图像亮度归一化的容错(见 §6.3),并建议使用外置稳定光源。
4. iOS 相机权限在 Safari 中可能每次打开页面都重新询问,属正常现象。
5. 后端要处理"手机息屏 / 切后台导致视频停止"的情况:前端使用 Screen Wake Lock API(`navigator.wakeLock.request('screen')`,不支持则忽略)并在 `visibilitychange` 时重连。

---

## 3. 总体架构

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

### 3.1 设计原则
- 检测器、跟踪器、预处理全部是**可插拔接口**,通过配置文件切换。
- 跟踪器为**纯 NumPy + SciPy 实现**,不依赖 CUDA、不依赖 Ultralytics,保证 CPU 友好。
- 启动时**自动探测硬件**并选择性能档位(见 §7)。
- 背压:服务端处理慢时丢帧,不堆积延迟(客户端一次只允许一个在途帧,服务端也只保留最新一帧)。

---

## 4. 目录结构

```
capsule-tracker/
├─ README.md
├─ requirements.txt
├─ config.yaml                 # 全部可调参数
├─ server.py                   # 入口:启动 HTTPS 服务
├─ app/
│  ├─ __init__.py
│  ├─ main.py                  # FastAPI app, 路由, WebSocket
│  ├─ config.py                # 配置加载 + 校验 (pydantic)
│  ├─ hardware.py              # 硬件探测与档位选择
│  ├─ pipeline.py              # 单连接处理流水线
│  ├─ preprocess.py            # 亮度归一化、ROI 裁剪
│  ├─ detectors/
│  │  ├─ base.py               # Detector 抽象基类
│  │  ├─ ultralytics_det.py    # 后端1: ultralytics (PyTorch)
│  │  ├─ onnx_det.py           # 后端2: onnxruntime (CPU/DirectML/CUDA)
│  │  ├─ openvino_det.py       # 后端3 (可选): OpenVINO
│  │  └─ classical_det.py      # 后端4: 传统背景差分 + 轮廓 (无需模型)
│  ├─ tracking/
│  │  ├─ kalman.py             # 卡尔曼滤波 (xyah 或 xysr)
│  │  ├─ matching.py           # IoU/距离代价矩阵 + 匈牙利匹配
│  │  └─ bytetrack.py          # ByteTrack 实现
│  ├─ counting.py              # 计数 / 稳定性判定
│  └─ schemas.py               # WebSocket 消息数据结构
├─ web/
│  ├─ index.html
│  ├─ app.js
│  └─ style.css
├─ scripts/
│  ├─ setup_windows.ps1        # 一键环境准备 (venv, mkcert, 防火墙)
│  ├─ gen_cert.ps1             # 生成/更新证书 (自动取本机 IP)
│  ├─ export_onnx.py           # 模型导出 ONNX
│  └─ benchmark.py             # 离线视频基准测试
├─ models/                     # 放置模型权重 (不入库)
├─ tests/
│  ├─ test_tracker.py
│  ├─ test_matching.py
│  ├─ test_protocol.py
│  └─ data/                    # 合成测试序列
└─ docs/
   └─ phone_setup.md           # 手机端证书信任图文步骤
```

---

## 5. 通信协议

### 5.1 HTTP 路由
| 路径 | 方法 | 说明 |
|---|---|---|
| `/` | GET | 返回 `web/index.html` |
| `/static/*` | GET | 静态资源 (app.js / style.css) |
| `/api/info` | GET | 返回硬件档位、检测器名称、跟踪器参数、版本 |
| `/api/config` | GET / POST | 读取 / 热更新部分运行参数(conf 阈值、FPS 上限等) |
| `/ws` | WebSocket | 视频帧上行 + 结果下行 |
| `/cert/rootCA.pem` | GET | 下载 mkcert 根证书,方便手机安装 (仅局域网) |

### 5.2 WebSocket 消息
**上行(客户端 → 服务端)**
- 二进制消息:一帧 JPEG。
- 文本消息(JSON)用于控制:
  - `{"type":"hello","ua":"...","w":1920,"h":1080,"torch":true|false}`
  - `{"type":"reset"}` 重置跟踪器 / 计数
  - `{"type":"set","key":"conf","value":0.3}`

**下行(服务端 → 客户端),JSON**
```json
{
  "type": "result",
  "frame_id": 123,
  "w": 1920, "h": 1080,
  "latency_ms": {"decode": 4, "detect": 85, "track": 1, "total": 92},
  "count": 57,
  "objs": [
    {"id": 3, "box": [x1,y1,x2,y2], "score": 0.91, "state": "tracked"}
  ],
  "stable": true,
  "tier": "gpu|cpu_balanced|cpu_lite"
}
```
- 坐标基于服务端解码后图像的像素坐标,客户端按 `w,h` 与显示尺寸缩放。
- 其他消息 `type`:`info`(连接成功后下发配置摘要)、`warn`(如检测器降级)、`error`。

---

## 6. 核心模块规格

### 6.1 Detector 抽象
```python
class Detector(Protocol):
    name: str
    def warmup(self, shape: tuple[int,int]) -> None: ...
    def detect(self, img_bgr: np.ndarray) -> np.ndarray:
        """返回 (N,6): x1,y1,x2,y2,score,cls  (像素坐标, float32)"""
```
要求:
- `ultralytics_det`:加载 `.pt`;`device` 自动选择;`imgsz`、`conf`、`iou`、`max_det` 来自配置。不要使用 `model.track()`,只用 `predict`,跟踪由自实现模块完成。
- `onnx_det`:使用 onnxruntime,Provider 优先级 `CUDAExecutionProvider → DmlExecutionProvider → CPUExecutionProvider`(可用者自动选)。需实现与 YOLO 导出格式匹配的预处理(letterbox)和后处理(NMS)。
- `openvino_det`(可选):仅在 `openvino` 已安装时启用。
- `classical_det`:无模型备选。流程:灰度 → 与背景模型做差分(首帧或"空托盘"标定帧)→ 自适应阈值 → 形态学开闭 → `findContours` → 按面积过滤 → 输出外接框。用于背景干净时的超低算力模式。
- **小目标支持**:所有深度检测后端需支持**切片推理(tiling)**:当图像长边 > `tile.trigger_size` 时,按 `tile.size`、`tile.overlap` 切块检测,再合并并做跨块 NMS。默认关闭,由配置开启。
- **许可提示**:Ultralytics 为 AGPL-3.0。README 中注明:若要闭源分发,应使用 `onnx_det` 后端搭配自行训练/授权的模型。

### 6.2 Tracker:自实现 ByteTrack
输入:`(N,6)` 检测结果;输出:当前帧所有活跃轨迹。

要求:
1. 状态:`New / Tracked / Lost / Removed`。
2. 卡尔曼滤波:状态 `[cx, cy, a, h, vcx, vcy, va, vh]`,与原版 ByteTrack 一致;**参数可配置**(`std_weight_position`, `std_weight_velocity`)。
3. 两阶段关联:
   - 第一阶段:高分检测(`score ≥ track_high_thresh`)与已有轨迹(Tracked + Lost)匹配;
   - 第二阶段:低分检测(`track_low_thresh ≤ score < track_high_thresh`)与第一阶段未匹配的 Tracked 轨迹匹配;
   - 未匹配的高分检测按 `new_track_thresh` 新建轨迹。
4. **小目标专用相似度**(关键,默认开启,可配置):
   - 计算 IoU 前将检测框与预测框各向外扩张 `iou_expand_ratio`(默认 0.3,即宽高各 ×1.3)。
   - 代价 = `1 - IoU_expanded + λ · d_center_norm`,其中 `d_center_norm` 为中心点距离除以框对角线,`λ` 默认 0.3(`distance_penalty`)。
   - 门限:`match_thresh` 可配置,代价超过门限的配对不接受。
5. 匹配使用 `scipy.optimize.linear_sum_assignment`。
6. `track_buffer`(丢失保留帧数)按**帧率自适应**:`buffer = round(track_buffer_seconds * measured_fps)`,至少 5。
7. **静态场景优化**:当连续 `static_frames` 帧所有轨迹位移均 < `static_px` 时,进入 `static` 模式——跟踪器仍运行,但检测器降频(见 §7),直到画面发生变化(见 §6.4 变化检测)。
8. 提供 `reset()`。
9. 纯函数式、无全局状态,便于单元测试。

### 6.3 预处理 `preprocess.py`
- `normalize_brightness`:可选。基于 CLAHE(仅对 L 通道)或简单 gamma 校正,减轻 iPhone 自动曝光波动。默认关闭,配置开启。
- `roi`:可配置托盘区域多边形/矩形(归一化坐标),区域外像素置黑或直接裁剪,避免托盘外干扰。提供网页上拖拽设置 ROI 的简单交互(保存到 `config.yaml` 或本地 JSON)。
- 所有预处理必须保持 BGR uint8 输入输出,且记录坐标偏移,检测结果需映射回原图坐标。

### 6.4 变化检测(省算力)
- 对缩小到 160px 宽的灰度图计算与上一"已处理帧"的平均绝对差;低于 `change_thresh` 则判定"画面无变化"。
- 无变化时:跳过检测,沿用上一帧结果(或仅用卡尔曼预测),返回时 `"skipped": true`。
- 每 `force_detect_interval_s` 秒至少强制检测一次,防止漂移。

### 6.5 计数与稳定性 `counting.py`
- `count` = 当前 `Tracked` 轨迹数。
- `stable`:连续 `stable_frames` 帧(默认 5)计数不变,且无新轨迹产生,则为 true。网页据此显示"计数已稳定"。
- 同时提供"累计唯一 ID 数"(`total_unique`),用于运动场景。

---

## 7. 硬件探测与性能档位

`hardware.py` 启动时执行:
1. 检测 CUDA(`torch.cuda.is_available()`,torch 未安装则跳过)、onnxruntime 可用 Provider、CPU 物理核数、内存。
2. 选择档位(可在 `config.yaml` 中强制覆盖 `tier: auto|gpu|cpu_balanced|cpu_lite`):

| 档位 | 触发条件 | 检测输入 | 检测频率 | 其他 |
|---|---|---|---|---|
| `gpu` | CUDA 或 DirectML 可用 | imgsz 960~1280,可开 tiling | 每帧 | 关闭变化检测或低阈值 |
| `cpu_balanced` | 无 GPU,核数 ≥ 6 | imgsz 640~800 | 每 2 帧(中间帧卡尔曼预测) | 开启变化检测 |
| `cpu_lite` | 无 GPU,核数 < 6 或基准测试 < 2 FPS | imgsz 512~640 或 classical 后端 | 每 3~4 帧 | 开启变化检测 + ROI 裁剪 |

3. **自适应降级**:运行时统计最近 30 帧的平均处理耗时,若 > `target_latency_ms`(默认 700),自动下调一级(降 imgsz / 降检测频率),并通过 `warn` 消息通知客户端;恢复条件需持续一段时间稳定后才升回,避免来回抖动。
4. 隔帧检测时,中间帧输出的轨迹状态应标记 `"predicted"`。

---

## 8. 前端(`web/`)规格

纯静态,原生 JS,**不引入构建工具**。

### 8.1 功能
1. 相机:`getUserMedia({video:{facingMode:{ideal:'environment'}, width:{ideal:1920}, height:{ideal:1080}, frameRate:{ideal:15}}})`。
2. 页面元素:全屏预览 `<video playsinline muted>`;叠加 `<canvas>` 绘制框、ID;顶部状态栏(连接状态、FPS、延迟、档位、计数、稳定标识、torch 状态)。
3. 控制面板(底部抽屉):
   - 上传帧率 `FPS`(1~15 滑条)、JPEG 质量(0.5~0.95)、上传宽度(960/1280/1920)。
   - 检测置信度 `conf` 滑条(调用 `/api/config`)。
   - 按钮:重置跟踪 / 设置 ROI / 保存当前帧(用于采集训练数据,保存到主机 `captures/`)/ 开关 torch。
4. Torch:
   - 取得 track 后检查 `getCapabilities().torch`;可用则显示开关,不可用则显示提示"当前设备/浏览器不支持网页控制闪光灯,请使用外置补光"。
   - 提供"屏幕补光"备选:点击后弹出全白页面(可调亮度/色温)用于把另一设备当灯,**不在同一部手机上使用**(会遮挡摄像头画面)。
5. 防息屏:使用 Wake Lock;失败则忽略。
6. 重连:WebSocket 断开自动指数退避重连(1s → 2s → 4s,上限 10s)。
7. 背压:`busy` 标志,收到结果前不发下一帧;另加 `maxInflightMs` 超时(默认 3000ms)防卡死。
8. 抓帧用 `ImageBitmap` 或 `canvas.drawImage(video)`,编码用 `canvas.toBlob(cb,'image/jpeg',q)`。
9. 叠加坐标按 `object-fit` 实际渲染区域换算,避免框错位(需正确处理视频与容器宽高比不同的情形)。
10. 界面语言:中文。

### 8.2 兼容性
- 目标:iOS Safari 15+。必须在无 `ImageCapture`、`getCapabilities` 可能缺失 `torch` 的前提下正常工作。所有可选 API 要做特性检测。

---

## 9. Windows 部署

### 9.1 `scripts/setup_windows.ps1`(一键脚本,幂等)
1. 检查 Python ≥ 3.10(`py -3 --version`),缺失则提示安装方式,不自动下载。
2. 创建 venv 并 `pip install -r requirements.txt`。
3. 检查 `mkcert`,缺失则尝试 `winget install FiloSottile.mkcert`;失败则打印手动指引。
4. 执行 `mkcert -install`。
5. 调用 `gen_cert.ps1`:
   - 自动获取本机局域网 IPv4(过滤虚拟网卡 / 169.254.x.x);
   - `mkcert -cert-file certs\cert.pem -key-file certs\key.pem <IP> <主机名> localhost 127.0.0.1`;
   - 把 `mkcert -CAROOT` 下的 `rootCA.pem` 复制到 `certs\rootCA.pem`,供 `/cert/rootCA.pem` 下载。
6. 添加防火墙规则(需管理员,脚本检测权限,无权限则输出命令):
   `netsh advfirewall firewall add rule name="capsule-tracker-8443" dir=in action=allow protocol=TCP localport=8443`
7. 打印手机端访问地址和二维码(用 `qrcode` 库在终端 ASCII 输出)。

### 9.2 `requirements.txt`(建议分层)
```
fastapi
uvicorn[standard]
numpy
scipy
opencv-python-headless
pydantic>=2
pyyaml
qrcode
# 可选后端
ultralytics        # ultralytics_det
onnxruntime        # onnx_det (CPU); GPU 用户装 onnxruntime-gpu 或 onnxruntime-directml
openvino           # openvino_det (可选)
```
- README 说明:**不要同时安装 `onnxruntime` 与 `onnxruntime-gpu`/`-directml`**。
- Windows + NVIDIA 用户需自行安装带 CUDA 的 PyTorch;说明文档给出官方安装命令的查找方式,不要硬编码版本。

### 9.3 手机证书信任(写入 `docs/phone_setup.md`)
1. 手机连同一 Wi-Fi,Safari 访问 `http://<主机IP>:8080/cert/rootCA.pem`(提供一个仅用于下载证书的 HTTP 端口,默认 8080,仅局域网;可在配置中关闭),或用 AirDrop / 数据线传输。
2. 在"设置"中点击"已下载描述文件"→ 安装。
3. "设置 > 通用 > 关于本机 > 证书信任设置"→ 对该根证书打开完全信任。
4. Safari 访问 `https://<主机IP>:8443`,确认无证书警告。
- 注明:以上为通用 iOS 流程,不同 iOS 版本菜单文字可能略有差异。

### 9.4 启动
`python server.py [--config config.yaml] [--tier auto]`
- 默认监听 `0.0.0.0:8443`,使用 `certs/cert.pem` 与 `certs/key.pem`。
- 若证书缺失,给出明确报错与运行 `gen_cert.ps1` 的提示。
- 可选:`scripts/install_service.ps1` 通过"任务计划程序"开机自启(非必须)。

---

## 10. 配置文件 `config.yaml`(示例,全部带默认值)

```yaml
server:
  host: 0.0.0.0
  port: 8443
  cert: certs/cert.pem
  key: certs/key.pem
  cert_download_port: 8080       # 0 = 关闭
tier: auto                       # auto|gpu|cpu_balanced|cpu_lite
target_latency_ms: 700

detector:
  backend: ultralytics           # ultralytics|onnx|openvino|classical
  weights: models/capsule.pt
  imgsz: 960
  conf: 0.25
  iou: 0.5
  max_det: 500
  tile: {enabled: false, trigger_size: 1600, size: 640, overlap: 0.2}

preprocess:
  normalize_brightness: false
  roi: null                      # [[x,y],...] 归一化坐标

tracker:
  type: bytetrack
  track_high_thresh: 0.5
  track_low_thresh: 0.1
  new_track_thresh: 0.6
  match_thresh: 0.8
  track_buffer_seconds: 2.0
  iou_expand_ratio: 0.3
  distance_penalty: 0.3

change_detect:
  enabled: true
  change_thresh: 2.0
  force_detect_interval_s: 3.0

counting:
  stable_frames: 5

capture:
  save_dir: captures
```
- 使用 pydantic 校验,非法值启动时报清晰错误。

---

## 11. 模型训练接口(简要,非本期重点)

1. 网页的"保存当前帧"功能把原始 JPEG 存入 `captures/YYYYMMDD/`,用于采集训练数据。
2. 文档说明推荐流程:Label Studio(免费)或 Roboflow 之外的开源标注工具标注 → 用 Ultralytics 训练 nano/small 模型(单类 `capsule` 或按形状多类)→ `scripts/export_onnx.py` 导出 ONNX。
3. 小目标训练建议(写入 README):训练分辨率不低于推理分辨率;必要时使用切片训练;数据中要包含疏密不同的样本和多种光照。
4. 在没有自训练模型前,系统应能用 `classical` 后端跑通整条链路(用于联调)。

---

## 12. 实施步骤(给 Claude Code 的任务顺序)

**阶段 1:骨架**
1. 建立目录与 `requirements.txt`、`config.py`(pydantic)。
2. 实现 FastAPI、静态页面、HTTPS 启动、`/api/info`。
3. 实现前端取流 + WebSocket 回传,服务端先只返回帧尺寸(回环验证)。

**阶段 2:检测与跟踪**
4. 实现 `Detector` 基类与 `classical_det`、`ultralytics_det`。
5. 实现 `kalman.py`、`matching.py`、`bytetrack.py`,并写单元测试(§13)。
6. 在 `pipeline.py` 串联:解码 → 预处理 → 检测 → 跟踪 → 计数 → 序列化。

**阶段 3:性能与降级**
7. `hardware.py` 档位选择;隔帧检测;变化检测;自适应降级。
8. `onnx_det.py`、tiling。
9. `scripts/benchmark.py`:输入视频文件或图像序列,输出各档位 FPS、平均延迟、ID 切换数、计数误差。

**阶段 4:Windows 脚本与文档**
10. `setup_windows.ps1`、`gen_cert.ps1`、`phone_setup.md`、README。

每个阶段完成后运行对应测试并提交。

---

## 13. 测试与验收标准

### 13.1 单元测试(pytest)
- `test_matching.py`:扩张 IoU 在小框小位移下的匹配成功率高于普通 IoU(构造 8×8 像素框、位移 5 像素的用例)。
- `test_tracker.py`:
  - 合成序列:50 个小框匀速运动、无遮挡,ID 切换数 = 0。
  - 漏检 1~2 帧后恢复,ID 保持不变。
  - 低分检测二阶段回收用例。
  - `reset()` 后 ID 重新计数。
- `test_protocol.py`:消息结构校验,坐标映射函数正确。
- 所有测试不依赖 GPU、不下载模型即可运行。

### 13.2 集成验收(逐条可验证)
| # | 验收项 | 通过标准 |
|---|---|---|
| 1 | 手机访问 | 完成证书信任后,iPhone Safari 打开 `https://<IP>:8443` 无警告,可调用后置相机 |
| 2 | 数据链路 | 手机画面上叠加主机返回的框和 ID,框与胶囊对齐无明显偏移(含横屏/竖屏) |
| 3 | GPU 模式 | 有 NVIDIA GPU 时自动选 `gpu` 档,端到端延迟 ≤ 300ms(1920 宽上传,目标 ≥ 5 FPS) |
| 4 | CPU 模式 | 禁用 GPU(`tier: cpu_lite`)时,端到端 ≥ 2 FPS,无崩溃、无延迟累积 |
| 5 | 背压 | 人为让检测变慢(sleep),延迟不持续增长,客户端不积压 |
| 6 | 静态托盘 | 托盘静止时检测频率明显下降(日志可见 `skipped`),计数稳定且 `stable=true` |
| 7 | 疏密差异 | 同一托盘内稀疏区与密集区的胶囊均被检出,计数与人工点数误差 ≤ 2% (用自训练模型时验收) |
| 8 | 断线重连 | 关闭/重开 Wi-Fi 或切后台后,页面自动重连并恢复 |
| 9 | Torch 降级 | 在不支持 torch 的设备上,页面给出提示,其余功能不受影响 |
| 10 | 一键部署 | 全新 Win10/Win11 机器按 README 执行脚本后,能一次启动成功 |

---

## 14. 已知风险与处理

| 风险 | 处理 |
|---|---|
| iPhone 网页无法开闪光灯 | 特性检测 + 明确提示 + 外置光源建议,不作为阻塞项 |
| iOS 自动曝光抖动 | 可选亮度归一化;建议外置稳定光源;训练数据覆盖多种曝光 |
| 证书信任步骤繁琐 | 提供 `/cert/rootCA.pem` 下载、二维码、图文文档 |
| 小目标 IoU 失配 | 扩张 IoU + 中心距离惩罚(§6.2) |
| CPU 太慢 | 档位降级、隔帧检测、变化检测、ROI、classical 后端 |
| 通用 COCO 模型检不出胶囊 | 必须自训练模型;联调阶段用 classical 后端 |
| Ultralytics AGPL 许可 | README 说明,提供 ONNX 后端 |
| 局域网路由器客户端隔离 | README 的故障排查章节列出此项 |

---

## 15. README 必含内容
1. 项目简介与架构图。
2. Windows 环境准备与一键脚本用法。
3. 手机证书信任步骤(含截图占位)。
4. 常见问题:手机打不开页面(防火墙 / IP 变了 / 客户端隔离)、相机黑屏、证书警告、帧率低、torch 不可用。
5. 性能档位说明与调参指南。
6. 如何替换为自己的模型。

---

## 16. 代码规范
- Python 3.10+,类型标注,`ruff` 格式化。
- 日志使用 `logging`,默认 INFO;每 5 秒输出一次吞吐 / 延迟统计。
- 不得在代码中硬编码 IP、路径;全部来自配置或自动探测。
- 所有异常在 WebSocket 内捕获,发送 `error` 消息而不是让连接崩溃。
