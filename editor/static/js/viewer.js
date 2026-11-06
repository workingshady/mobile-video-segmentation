// Center canvas: frame + masks/effects + prompts, and the click / box / brush / magic-brush tools.
import {
  S, on, emit, toast, frameUrl, overlayUrl, renderUrl, promptAt,
  sendPrompt, sendMaskPrompt, sendPromptLive, ensureObject,
} from './app.js';

// ---------- image cache (LRU) ----------
const cache = new Map();
export function img(url) {
  let im = cache.get(url);
  if (im) {
    cache.delete(url);
    cache.set(url, im);
    return im;
  }
  im = new Image();
  im.onload = () => { requestDraw(); emit('imgload'); };
  im.src = url;
  cache.set(url, im);
  if (cache.size > 900) cache.delete(cache.keys().next().value);
  return im;
}
const ready = (im) => im.complete && im.naturalWidth > 0;

function layerUrl(i) {
  if (S.view === 'preview') return renderUrl(i);
  return S.viewModel && S.tl.revs[i] > 0 ? overlayUrl(i) : null;
}

export function isReady(i) {
  if (!ready(img(frameUrl(i)))) return false;
  const u = layerUrl(i);
  return !u || ready(img(u));
}

export function preload(from, count = 10) {
  if (!S.project) return;
  for (let i = from; i < Math.min(S.project.num_frames, from + count); i++) {
    img(frameUrl(i));
    const u = layerUrl(i);
    if (u) img(u);
  }
}

// ---------- canvas + view transform ----------
const canvas = document.getElementById('viewer');
const ctx = canvas.getContext('2d');
const cursorInfo = document.getElementById('cursorInfo');
let scale = 1, ox = 0, oy = 0, fitted = false;
let userView = false; // true after the user zooms/pans: then resizes keep their view
let mouse = null; // last mouse position in image coords
let drag = null;  // current interaction
let spaceDown = false;

function dpr() { return window.devicePixelRatio || 1; }

function resize() {
  const r = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(r.width * dpr()));
  canvas.height = Math.max(1, Math.round(r.height * dpr()));
  if (!fitted || !userView) fit();
  requestDraw();
}
new ResizeObserver(resize).observe(canvas);

export function fit() {
  if (!S.project) return;
  const { width: w, height: h } = S.project;
  const pad = 20 * dpr();
  scale = Math.min((canvas.width - 2 * pad) / w, (canvas.height - 2 * pad) / h);
  ox = (canvas.width - w * scale) / 2;
  oy = (canvas.height - h * scale) / 2;
  fitted = true;
  userView = false;
  requestDraw();
}

function toImg(ev) {
  const r = canvas.getBoundingClientRect();
  const sx = (ev.clientX - r.left) * dpr();
  const sy = (ev.clientY - r.top) * dpr();
  return { x: (sx - ox) / scale, y: (sy - oy) / scale, sx, sy };
}

function inside(p) {
  return p.x >= 0 && p.y >= 0 && p.x < S.project.width && p.y < S.project.height;
}

// ---------- drawing ----------
let drawQueued = false;
export function requestDraw() {
  if (drawQueued) return;
  drawQueued = true;
  requestAnimationFrame(() => { drawQueued = false; draw(); });
}

function draw() {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#0b0c10';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!S.project) return;
  const { width: w, height: h } = S.project;
  const i = S.frame;
  ctx.setTransform(scale, 0, 0, scale, ox, oy);
  ctx.imageSmoothingEnabled = scale < 2;

  if (S.view === 'preview') {
    const r = img(renderUrl(i));
    if (ready(r)) ctx.drawImage(r, 0, 0, w, h);
    else drawFrameOnly(i, w, h);
  } else {
    drawFrameOnly(i, w, h);
    const u = layerUrl(i);
    if (u) {
      const ov = img(u);
      if (ready(ov)) ctx.drawImage(ov, 0, 0, w, h);
    }
  }

  if (drag?.kind === 'magic' && drag.pts.length) {
    const o = S.project.objects.find((x) => x.id === drag.oid);
    ctx.globalAlpha = 0.45;
    ctx.strokeStyle = drag.label ? (o?.color || '#fff') : '#ff5c6c';
    ctx.lineWidth = S.brushSize;
    ctx.lineCap = ctx.lineJoin = 'round';
    ctx.beginPath();
    drag.pts.forEach((q, k) => (k ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y + 0.01)));
    ctx.lineTo(drag.pts[drag.pts.length - 1].x + 0.01, drag.pts[drag.pts.length - 1].y);
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  if (drag?.kind === 'brush' && drag.canvas) {
    ctx.globalAlpha = 0.55;
    ctx.globalCompositeOperation = 'screen';
    ctx.drawImage(drag.canvas, 0, 0, w, h);
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;
  }

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  if (S.view === 'edit') drawPrompts();
  drawCursor();
}

function drawFrameOnly(i, w, h) {
  const f = img(frameUrl(i));
  if (ready(f)) ctx.drawImage(f, 0, 0, w, h);
}

const sx = (x) => x * scale + ox;
const sy = (y) => y * scale + oy;

function drawPrompts() {
  const d = dpr();
  for (const o of S.project.objects) {
    if (!o.visible) continue;
    const rec = promptAt(o.id, S.frame);
    if (!rec) continue;
    const isSel = o.id === S.sel;
    ctx.globalAlpha = isSel ? 1 : 0.55;
    if (rec.box) {
      ctx.setLineDash([6 * d, 4 * d]);
      ctx.lineWidth = 2 * d;
      ctx.strokeStyle = o.color;
      const [x1, y1, x2, y2] = rec.box;
      ctx.strokeRect(sx(x1), sy(y1), (x2 - x1) * scale, (y2 - y1) * scale);
      ctx.setLineDash([]);
    }
    rec.points.forEach(([x, y], k) => {
      const pos = rec.labels[k] === 1;
      const r = (rec.points.length > 6 ? (isSel ? 4 : 3) : (isSel ? 7 : 5)) * d;
      ctx.beginPath();
      ctx.arc(sx(x), sy(y), r, 0, Math.PI * 2);
      ctx.fillStyle = pos ? '#3ddc84' : '#ff5c6c';
      ctx.fill();
      ctx.lineWidth = 2 * d;
      ctx.strokeStyle = '#fff';
      ctx.stroke();
      ctx.fillStyle = '#0b0c10';
      ctx.font = `bold ${10 * d}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(pos ? '+' : '−', sx(x), sy(y) + 0.5);
    });
    if (rec.mask && isSel) {
      ctx.font = `${12 * d}px sans-serif`;
      ctx.textAlign = 'left';
      ctx.fillStyle = o.color;
      ctx.fillText('🖌 brush prompt on this frame', ox + 8 * d, oy + 18 * d);
    }
  }
  ctx.globalAlpha = 1;
  if (drag?.kind === 'box' && drag.cur) {
    const o = S.project.objects.find((x) => x.id === S.sel);
    ctx.setLineDash([6 * d, 4 * d]);
    ctx.lineWidth = 2 * d;
    ctx.strokeStyle = o?.color || '#fff';
    const x1 = Math.min(drag.start.x, drag.cur.x), y1 = Math.min(drag.start.y, drag.cur.y);
    ctx.strokeRect(sx(x1), sy(y1), Math.abs(drag.cur.x - drag.start.x) * scale, Math.abs(drag.cur.y - drag.start.y) * scale);
    ctx.setLineDash([]);
  }
}

function drawCursor() {
  if (!mouse || !(S.tool === 'brush' || S.tool === 'erase' || S.tool === 'magic')) return;
  ctx.beginPath();
  ctx.arc(sx(mouse.x), sy(mouse.y), (S.brushSize / 2) * scale, 0, Math.PI * 2);
  ctx.lineWidth = 1.5 * dpr();
  ctx.strokeStyle = S.tool === 'erase' ? '#ff5c6c' : '#fff';
  ctx.stroke();
}

// ---------- tools ----------
function hitPoint(rec, p) {
  if (!rec) return -1;
  const tol = 10 * dpr() / scale;
  return rec.points.findIndex(([x, y]) => Math.hypot(x - p.x, y - p.y) < tol);
}

function editModel() {
  return S.loaded || S.viewModel || 'manual';
}

async function startBrush(p) {
  const oid = await ensureObject();
  const { width: w, height: h } = S.project;
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const g = c.getContext('2d');
  g.fillStyle = '#000';
  g.fillRect(0, 0, w, h);
  drag = { kind: 'brush', canvas: c, g, oid, frame: S.frame, pts: [p], loaded: false, erase: S.tool === 'erase' };
  const base = new Image();
  base.onload = () => {
    if (drag?.canvas !== c) return;
    g.drawImage(base, 0, 0, w, h);
    drag.loaded = true;
    strokeAll();
  };
  base.onerror = () => { drag.loaded = true; strokeAll(); };
  base.src = `/api/objmask/${encodeURIComponent(editModel())}/${S.frame}/${oid}.png?p=${S.project.id}&t=${Date.now()}`;
}

function strokeSeg(a, b) {
  const g = drag.g;
  g.strokeStyle = g.fillStyle = drag.erase ? '#000' : '#fff';
  g.lineWidth = S.brushSize;
  g.lineCap = 'round';
  g.beginPath();
  g.moveTo(a.x, a.y);
  g.lineTo(b.x + 0.01, b.y);
  g.stroke();
}

function strokeAll() {
  for (let k = 0; k < drag.pts.length; k++) strokeSeg(drag.pts[Math.max(0, k - 1)], drag.pts[k]);
  requestDraw();
}

// ---------- magic brush ----------
// Paint over (part of) the object: the stroke is resampled into click points and the model segments
// the WHOLE object, live while you paint. More strokes keep adding to the same object.
// Alt / right-drag paints "not this" points to remove wrong areas. One stroke = one undo step.
const MAGIC_MAX_POINTS = 40;

function samplePath(pts, spacing, maxN) {
  let len = 0;
  for (let k = 1; k < pts.length; k++) len += Math.hypot(pts[k].x - pts[k - 1].x, pts[k].y - pts[k - 1].y);
  const n = Math.max(1, Math.min(maxN, Math.round(len / spacing) + 1));
  if (n === 1) return [pts[Math.floor(pts.length / 2)]];
  const step = len / (n - 1);
  const out = [pts[0]];
  let acc = 0;
  let next = step;
  for (let k = 1; k < pts.length && out.length < n - 1; k++) {
    const seg = Math.hypot(pts[k].x - pts[k - 1].x, pts[k].y - pts[k - 1].y);
    while (seg > 0 && acc + seg >= next && out.length < n - 1) {
      const t = (next - acc) / seg;
      out.push({ x: pts[k - 1].x + t * (pts[k].x - pts[k - 1].x), y: pts[k - 1].y + t * (pts[k].y - pts[k - 1].y) });
      next += step;
    }
    acc += seg;
  }
  out.push(pts[pts.length - 1]);
  return out;
}

function magicPrompt(d) {
  const samples = samplePath(d.pts, Math.max(24, S.brushSize), 12).map((q) => [Math.round(q.x), Math.round(q.y)]);
  let points = [...d.base.points, ...samples];
  let labels = [...d.base.labels, ...samples.map(() => d.label)];
  if (points.length > MAGIC_MAX_POINTS) {
    points = points.slice(-MAGIC_MAX_POINTS);
    labels = labels.slice(-MAGIC_MAX_POINTS);
  }
  return { points, labels, box: d.base.box };
}

function sendMagic(d, final) {
  d.lastSend = performance.now();
  d.sentLen = d.pts.length;
  const recordUndo = !d.firstSent;
  d.firstSent = true;
  d.inflight = sendPromptLive(d.oid, d.frame, magicPrompt(d), recordUndo, final)
    .catch((e) => toast(e.message, 'error', 6000))
    .finally(() => {
      d.inflight = null;
      if (!d.done) pumpMagic(d);
    });
  return d.inflight;
}

function pumpMagic(d) {
  // one request in flight at a time, live updates at most every ~180 ms
  if (d.inflight || d.done || d.sentLen === d.pts.length) return;
  const wait = 180 - (performance.now() - d.lastSend);
  if (wait > 0) {
    if (!d.timer) d.timer = setTimeout(() => { d.timer = null; pumpMagic(d); }, wait);
    return;
  }
  sendMagic(d, false);
}

async function startMagic(p, subtract) {
  const oid = await ensureObject();
  const old = promptAt(oid, S.frame);
  const base = old && !old.mask
    ? { points: old.points.map((q) => [...q]), labels: [...old.labels], box: old.box }
    : { points: [], labels: [], box: null };
  if (old?.mask) toast('Magic brush replaces the brush-mask prompt on this frame');
  drag = { kind: 'magic', oid, frame: S.frame, base, pts: [{ x: p.x, y: p.y }], label: subtract ? 0 : 1,
           lastSend: 0, sentLen: 0, firstSent: false, inflight: null, timer: null, done: false };
  pumpMagic(drag);
}

async function finishMagic(d) {
  d.done = true;
  clearTimeout(d.timer);
  if (d.inflight) await d.inflight;
  await sendMagic(d, true);
}

canvas.addEventListener('pointerdown', async (ev) => {
  if (!S.project) return;
  canvas.setPointerCapture(ev.pointerId);
  const p = toImg(ev);
  if (ev.button === 1 || S.tool === 'select' || spaceDown) {
    drag = { kind: 'pan', sx: p.sx, sy: p.sy, ox, oy };
    userView = true;
    return;
  }
  if (!(ev.button === 0 || (ev.button === 2 && S.tool === 'magic')) || !inside(p)) return;
  if (S.playing) { S.playing = false; emit('play'); }

  if (S.tool === 'pos' || S.tool === 'neg') {
    const oid = await ensureObject();
    const old = promptAt(oid, S.frame);
    let rec = old && !old.mask
      ? { points: old.points.map((q) => [...q]), labels: [...old.labels], box: old.box }
      : { points: [], labels: [], box: null };
    if (old?.mask) toast('Points replace the brush prompt on this frame');
    const hit = hitPoint(rec, p);
    if (hit >= 0) {
      rec.points.splice(hit, 1);
      rec.labels.splice(hit, 1);
    } else {
      rec.points.push([Math.round(p.x), Math.round(p.y)]);
      rec.labels.push(S.tool === 'pos' ? 1 : 0);
    }
    sendPrompt(oid, S.frame, rec);
  } else if (S.tool === 'box') {
    drag = { kind: 'box', start: p, cur: p };
  } else if (S.tool === 'magic') {
    await startMagic(p, ev.altKey || ev.button === 2);
  } else if (S.tool === 'brush' || S.tool === 'erase') {
    await startBrush(p);
  }
});

canvas.addEventListener('pointermove', (ev) => {
  if (!S.project) return;
  const p = toImg(ev);
  mouse = p;
  cursorInfo.textContent = inside(p) ? `x ${Math.round(p.x)}  y ${Math.round(p.y)}` : '';
  if (drag?.kind === 'pan') {
    ox = drag.ox + (p.sx - drag.sx);
    oy = drag.oy + (p.sy - drag.sy);
  } else if (drag?.kind === 'box') {
    drag.cur = p;
  } else if (drag?.kind === 'magic') {
    const last = drag.pts[drag.pts.length - 1];
    if (Math.hypot(p.x - last.x, p.y - last.y) > 2) drag.pts.push({ x: p.x, y: p.y });
    pumpMagic(drag);
  } else if (drag?.kind === 'brush') {
    const last = drag.pts[drag.pts.length - 1];
    drag.pts.push(p);
    if (drag.loaded) strokeSeg(last, p);
  }
  requestDraw();
});

canvas.addEventListener('pointerup', async () => {
  const d = drag;
  drag = null;
  if (!d) return;
  if (d.kind === 'box') {
    const x1 = Math.max(0, Math.min(d.start.x, d.cur.x)), y1 = Math.max(0, Math.min(d.start.y, d.cur.y));
    const x2 = Math.min(S.project.width, Math.max(d.start.x, d.cur.x)), y2 = Math.min(S.project.height, Math.max(d.start.y, d.cur.y));
    if (x2 - x1 < 4 || y2 - y1 < 4) return requestDraw();
    const oid = await ensureObject();
    const old = promptAt(oid, S.frame);
    const rec = old && !old.mask
      ? { points: old.points, labels: old.labels, box: null }
      : { points: [], labels: [], box: null };
    rec.box = [x1, y1, x2, y2].map(Math.round);
    sendPrompt(oid, S.frame, rec);
  } else if (d.kind === 'magic') {
    finishMagic(d);
  } else if (d.kind === 'brush') {
    if (!d.loaded) { // finish once the starting mask arrives
      await new Promise((r) => { const t = setInterval(() => { if (d.loaded) { clearInterval(t); r(); } }, 20); });
    }
    sendMaskPrompt(d.oid, d.frame, d.canvas.toDataURL('image/png'));
  }
  requestDraw();
});

canvas.addEventListener('pointerleave', () => { mouse = null; cursorInfo.textContent = ''; requestDraw(); });

canvas.addEventListener('wheel', (ev) => {
  if (!S.project) return;
  ev.preventDefault();
  const p = toImg(ev);
  const k = Math.exp(-ev.deltaY * 0.0015);
  const ns = Math.min(40, Math.max(0.05, scale * k));
  ox = p.sx - p.x * ns;
  oy = p.sy - p.y * ns;
  scale = ns;
  userView = true;
  requestDraw();
}, { passive: false });

canvas.addEventListener('contextmenu', (e) => e.preventDefault());

window.addEventListener('keydown', (e) => { if (e.code === 'Space' && e.target === document.body) spaceDown = true; });
window.addEventListener('keyup', (e) => { if (e.code === 'Space') spaceDown = false; });

export function setCursor() {
  canvas.style.cursor = { select: 'grab', pos: 'crosshair', neg: 'crosshair', box: 'crosshair', brush: 'none', erase: 'none', magic: 'none' }[S.tool];
}

on('frame', requestDraw);
on('timeline', requestDraw);
on('project', () => {
  if (!S.project) { fitted = false; }
  else if (!fitted) fit();
  requestDraw();
});
on('newproject', () => { fitted = false; fit(); });
