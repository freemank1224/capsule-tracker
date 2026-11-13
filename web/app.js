/* 胶囊计数 前端逻辑(原生 JS,无构建工具)。
 * 功能: 相机取流 → JPEG 上行 → 叠加结果;背压 busy 标志;
 * 指数退避重连;Wake Lock;torch 特性检测 + 屏幕补光;ROI 拖拽设置。 */
'use strict';

const $ = (id) => document.getElementById(id);
const video = $('video'), overlay = $('overlay'), ctx = overlay.getContext('2d');

const state = {
  ws: null, busy: false, lastSendMs: 0, reconnectDelay: 1000,
  fps: 5, quality: 0.75, uploadWidth: 1280,
  roiMode: false, roiDrag: null, roiPoints: null,
  frameId: 0, wakeLock: null, videoSize: null,
  fpsCount: 0, fpsTime: performance.now(),
};

/* ---------------- WebSocket ---------------- */

function connect() {
  const proto = location.protocol === 'http:' ? 'ws' : 'wss';
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  state.ws = ws;

  ws.onopen = () => {
    setConn('ok', '已连接');
    state.reconnectDelay = 1000;
    const torch = (() => { try { return !!getTorchCap(); } catch (e) { return false; } })();
    ws.send(JSON.stringify({ type: 'hello', ua: navigator.userAgent,
      w: video.videoWidth || 0, h: video.videoHeight || 0, torch }));
  };
  ws.onmessage = (ev) => {
    if (ev.data instanceof ArrayBuffer) return;
    const msg = JSON.parse(ev.data);
    if (msg.type === 'result') onResult(msg);
    else if (msg.type === 'info' || msg.type === 'warn' || msg.type === 'error') {
      if (msg.msg) showMsg(msg.msg);
      if (msg.tier) $('st-tier').textContent = msg.tier;
    }
  };
  ws.onclose = () => {
    setConn('warn', '已断开');
    state.busy = false;
    setTimeout(connect, state.reconnectDelay);
    state.reconnectDelay = Math.min(state.reconnectDelay * 2, 10000); // 1s→2s→4s,上限10s
  };
  ws.onerror = () => ws.close();
}

function setConn(cls, text) {
  const el = $('st-conn');
  el.className = `chip ${cls}`;
  el.textContent = text;
}

function sendFrame(jpegBlob) {
  if (!state.ws || state.ws.readyState !== 1) return;
  state.busy = true;
  state.lastSendMs = performance.now();
  state.ws.send(jpegBlob);
}

/* ---------------- 相机 ---------------- */

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' },
               width: { ideal: 1920 }, height: { ideal: 1080 },
               frameRate: { ideal: 15 } },
      audio: false });
    video.srcObject = stream;
    await video.play();
    initTorch(stream);
    requestWakeLock();
  } catch (e) {
    showMsg('相机启动失败: ' + e.message);
  }
}

video.addEventListener('loadedmetadata', () => {
  state.videoSize = { w: video.videoWidth, h: video.videoHeight };
  resizeOverlay();
});
window.addEventListener('resize', resizeOverlay);

function resizeOverlay() {
  overlay.width = overlay.clientWidth * devicePixelRatio;
  overlay.height = overlay.clientHeight * devicePixelRatio;
}

/* object-fit: contain 实际渲染区域换算,避免框错位 */
function videoRect() {
  const d = { w: overlay.clientWidth, h: overlay.clientHeight };
  const v = state.videoSize || { w: 1, h: 1 };
  const s = Math.min(d.w / v.w, d.h / v.h);
  const w = v.w * s, h = v.h * s;
  return { x: (d.w - w) / 2, y: (d.h - h) / 2, w, h };
}

/* ---------------- 抓帧上传 ---------------- */

function captureLoop() {
  setInterval(() => {
    if (state.busy || video.readyState < 2 || !state.videoSize) return;
    const scale = state.uploadWidth / state.videoSize.w;
    const w = state.uploadWidth;
    const h = Math.round(state.videoSize.h * scale);
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    c.getContext('2d').drawImage(video, 0, 0, w, h);
    c.toBlob((blob) => {
      if (blob && !state.busy) sendFrame(blob);
    }, 'image/jpeg', state.quality);
  }, 1000 / state.fps);
}

/* maxInflightMs 超时防卡死 */
setInterval(() => {
  if (state.busy && performance.now() - state.lastSendMs > 3000) state.busy = false;
}, 500);

/* ---------------- 结果叠加 ---------------- */

function onResult(msg) {
  state.busy = false;
  state.frameId = msg.frame_id;
  state.lastResult = { w: msg.w, h: msg.h };  // ROI 坐标换算用
  const rect = videoRect();
  const sx = rect.w / msg.w, sy = rect.h / msg.h;   // 服务端像素 → 渲染区域
  const dx = rect.x, dy = rect.y;

  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.save();
  ctx.scale(devicePixelRatio, devicePixelRatio);
  for (const o of msg.objs) {
    const [x1, y1, x2, y2] = o.box;
    ctx.strokeStyle = o.state === 'tracked' ? '#31d158' : '#ffcc00';
    ctx.lineWidth = 2;
    ctx.strokeRect(dx + x1 * sx, dy + y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.font = '12px sans-serif';
    ctx.fillText(String(o.id), dx + x1 * sx, dy + y1 * sy - 3);
  }
  // ROI:已保存的画黄虚线框;拖动中的画白色实线预览
  const drawImgPoly = (pts, color, dash) => {
    if (!pts || pts.length < 3) return;
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash(dash || []);
    ctx.beginPath();
    pts.forEach(([x, y], i) => {
      const px = dx + x * sx, py = dy + y * sy;
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    });
    ctx.closePath(); ctx.stroke(); ctx.setLineDash([]);
  };
  drawImgPoly(state.roiPoints, '#ffcc00', [6, 4]);
  if (state.roiDrag) {
    const r = overlay.getBoundingClientRect();
    const a = state.roiDrag.start, b = state.roiDrag.cur;
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 2;
    ctx.strokeRect(Math.min(a.x, b.x) * r.width, Math.min(a.y, b.y) * r.height,
                   Math.abs(b.x - a.x) * r.width, Math.abs(b.y - a.y) * r.height);
  }
  ctx.restore();

  // 状态栏
  state.fpsCount++;
  const now = performance.now();
  if (now - state.fpsTime >= 1000) {
    $('st-fps').textContent = `${state.fpsCount} FPS`;
    state.fpsCount = 0; state.fpsTime = now;
  }
  $('st-lat').textContent = `${Math.round(msg.latency_ms.total)} ms`;
  $('st-tier').textContent = msg.tier + (msg.skipped ? '·skip' : '');
  $('st-count').textContent = `计数 ${msg.count}`;
  $('st-stable').classList.toggle('hidden', !msg.stable);
}

/* ---------------- torch(尽力而为,不假设可用) ---------------- */

function getTorchCap() {
  const track = video.srcObject && video.srcObject.getVideoTracks()[0];
  if (!track || !track.getCapabilities) return null;
  return track.getCapabilities().torch ? track : null;
}

function initTorch(stream) {
  const cap = (() => { try { return getTorchCap(); } catch (e) { return null; } })();
  if (cap) {
    $('st-torch').classList.remove('hidden');
    $('btn-torch').classList.remove('hidden');
  } else {
    // 优雅降级:提示改用外置补光
    showMsg('当前设备/浏览器不支持网页控制闪光灯,请使用外置补光(LED 面板灯/环形灯/另一台设备白屏)');
  }
}

async function toggleTorch() {
  const track = getTorchCap();
  if (!track) return;
  const on = !track.getSettings().torch;
  try {
    await track.applyConstraints({ advanced: [{ torch: on }] });
    $('btn-torch').textContent = on ? '关灯' : '开灯';
  } catch (e) { showMsg('闪光灯控制失败: ' + e.message); }
}

/* ---------------- Wake Lock + 重连 ---------------- */

async function requestWakeLock() {
  try {
    if ('wakeLock' in navigator) state.wakeLock = await navigator.wakeLock.request('screen');
  } catch (e) { /* 不支持则忽略 */ }
}

document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') {
    requestWakeLock();
    if (!state.ws || state.ws.readyState > 1) connect();  // 切后台回来重连
  }
});

/* ---------------- 控制面板 ---------------- */

$('panel-toggle').onclick = () => $('panel-body').classList.toggle('hidden');
$('ctl-fps').oninput = (e) => { state.fps = +e.target.value; $('v-fps').textContent = e.target.value; };
$('ctl-q').oninput = (e) => { state.quality = +e.target.value; $('v-q').textContent = e.target.value; };
$('ctl-width').onchange = (e) => { state.uploadWidth = +e.target.value; };
$('ctl-conf').oninput = (e) => {
  $('v-conf').textContent = e.target.value;
  clearTimeout($('ctl-conf')._t);
  $('ctl-conf')._t = setTimeout(() => {
    sendJson({ type: 'set', key: 'conf', value: +e.target.value });
  }, 300);
};
$('ctl-classes').addEventListener('change', (e) => {
  const classes = e.target.value.split(/[,，;；]/).map(s => s.trim()).filter(Boolean);
  if (!classes.length) return;
  sendJson({ type: 'set_classes', classes });
});
$('btn-reset').onclick = () => sendJson({ type: 'reset' });
$('btn-torch').onclick = toggleTorch;
$('btn-light').onclick = () => $('light-mask').classList.remove('hidden');
$('light-mask').onclick = () => $('light-mask').classList.add('hidden');
$('btn-capture').onclick = async () => {
  try {
    const scale = state.uploadWidth / state.videoSize.w;
    const c = document.createElement('canvas');
    c.width = state.uploadWidth;
    c.height = Math.round(state.videoSize.h * scale);
    c.getContext('2d').drawImage(video, 0, 0, c.width, c.height);
    const blob = await new Promise(r => c.toBlob(r, 'image/jpeg', 0.95));
    const resp = await fetch('/api/capture', { method: 'POST', body: blob });
    const j = await resp.json();
    showMsg(j.saved ? '已保存: ' + j.saved : (j.error || '保存失败'));
  } catch (e) { showMsg('保存失败: ' + e.message); }
};

function sendJson(obj) {
  if (state.ws && state.ws.readyState === 1) state.ws.send(JSON.stringify(obj));
}

let msgTimer = null;
function showMsg(text) {
  $('panel-msg').textContent = text;
  clearTimeout(msgTimer);
  msgTimer = setTimeout(() => { $('panel-msg').textContent = ''; }, 8000);
}

/* ---------------- ROI 拖拽(归一化矩形,移动端友好) ---------------- */

function evNorm(e) {
  const r = overlay.getBoundingClientRect();
  return { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height };
}

function endRoiDrag() {
  state.roiDrag = null;
  overlay.classList.remove('roi-active');
}

overlay.addEventListener('pointerdown', (e) => {
  if (!state.roiMode) return;
  e.preventDefault();
  overlay.setPointerCapture(e.pointerId);   // 手指滑出 overlay 也持续跟踪
  overlay.classList.add('roi-active');      // touch-action:none 防止 iOS 接管滚动
  state.roiDrag = { start: evNorm(e), cur: evNorm(e) };
});
overlay.addEventListener('pointermove', (e) => {
  if (!state.roiDrag) return;
  e.preventDefault();
  state.roiDrag.cur = evNorm(e);            // 实时预览(在 onResult 里绘制)
});
overlay.addEventListener('pointerup', (e) => {
  if (!state.roiDrag) return;
  const a = state.roiDrag.start, b = evNorm(e);
  endRoiDrag();
  // 拖动距离太小 = 误触轻点,不确认
  const r = overlay.getBoundingClientRect();
  if (Math.abs(b.x - a.x) * r.width < 20 || Math.abs(b.y - a.y) * r.height < 20) {
    showMsg('拖动距离太小(误触),请重新拖出一个矩形');
    return;
  }
  if (!state.lastResult) {
    showMsg('尚未收到画面数据,请等画面出现后再设置 ROI');
    return;
  }
  // 屏幕归一化坐标 → 服务端图像归一化坐标(考虑 contain 偏移)
  const rect = videoRect();
  const vs = state.lastResult;
  const sx = rect.w / vs.w, sy = rect.h / vs.h;
  const toImg = (p) => [
    Math.min(Math.max((p.x * r.width - rect.x) / sx / vs.w, 0), 1),
    Math.min(Math.max((p.y * r.height - rect.y) / sy / vs.h, 0), 1),
  ];
  const p1 = toImg(a), p2 = toImg(b);
  const x1 = Math.min(p1[0], p2[0]), x2 = Math.max(p1[0], p2[0]);
  const y1 = Math.min(p1[1], p2[1]), y2 = Math.max(p1[1], p2[1]);
  state.roiPoints = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]];
  sendJson({ type: 'set_roi', points: state.roiPoints });
  showMsg('ROI 已保存(黄虚线框),只统计框内');
  state.roiMode = false;
  $('roi-hint').classList.add('hidden');
});
overlay.addEventListener('pointercancel', () => {
  if (state.roiDrag) {
    endRoiDrag();
    showMsg('拖动被中断,请重新拖动');
  }
});

$('btn-roi').onclick = () => {
  state.roiMode = true;
  $('roi-hint').classList.remove('hidden');
  showMsg('在画面上按住并拖动,拖出矩形后松手确认');
};

$('btn-roi-clear').onclick = () => {
  state.roiPoints = null;
  endRoiDrag();
  sendJson({ type: 'set_roi', points: [] });
  showMsg('ROI 已清除,统计全部画面');
};

/* ---------------- 启动 ---------------- */

fetch('/api/info').then(r => r.json()).then(j => {
  $('st-tier').textContent = j.tier;
  $('ctl-conf').value = j.conf; $('v-conf').textContent = j.conf;
  if (j.roi && j.roi.length >= 3) state.roiPoints = j.roi;  // 回显已保存的 ROI
}).catch(() => {});
startCamera();
connect();
captureLoop();
