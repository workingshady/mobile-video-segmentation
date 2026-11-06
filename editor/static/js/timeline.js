// Bottom timeline: ruler, thumbnails, one track per object (mask coverage + keyframes),
// optional model-comparison strip, in/out range and the playhead.
import { S, on, emit, setFrame, thumbUrl, patchProject } from './app.js';
import { img } from './viewer.js';

const GUT = 150, RULER = 24, THUMB = 52, ROW = 26, CMP = 22;
const canvas = document.getElementById('timeline');
const ctx = canvas.getContext('2d');
const scroller = document.getElementById('timelineScroll');
const zoomEl = document.getElementById('tlZoom');
let px = 1;       // pixels per frame
let scroll = 0;   // horizontal scroll in px
let W = 0, H = 0;
let scrubbing = false;
let queued = false;

const dpr = () => window.devicePixelRatio || 1;
const n = () => (S.project ? S.project.num_frames : 1);
const fitPx = () => Math.max(0.01, (W - GUT - 12) / n());
const xOf = (f) => GUT + f * px - scroll;
const frameAt = (x) => Math.floor((x - GUT + scroll) / px);

function rows() {
  return S.project ? S.project.objects : [];
}

function layout() {
  W = scroller.clientWidth;
  const content = RULER + THUMB + Math.max(1, rows().length) * ROW + (S.compare ? CMP : 0) + 6;
  H = Math.max(content, scroller.clientHeight); // always fill the panel, scroll only when there are many objects
  canvas.style.height = H + 'px';
  canvas.width = Math.round(W * dpr());
  canvas.height = Math.round(H * dpr());
  applyZoom();
}

function applyZoom() {
  const f = fitPx();
  const maxPx = Math.max(f, 40);
  px = f * Math.pow(maxPx / f, zoomEl.value / 100);
  clampScroll();
}

function clampScroll() {
  const content = n() * px;
  scroll = Math.max(0, Math.min(scroll, content - (W - GUT - 12)));
  if (content <= W - GUT) scroll = 0;
}

export function requestDraw() {
  if (queued) return;
  queued = true;
  requestAnimationFrame(() => { queued = false; draw(); });
}

function fmtTime(sec) {
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, '0')}`;
}

function draw() {
  ctx.setTransform(dpr(), 0, 0, dpr(), 0, 0);
  ctx.fillStyle = '#12151b';
  ctx.fillRect(0, 0, W, H);
  if (!S.project) {
    ctx.fillStyle = '#5d6576';
    ctx.font = '13px "Segoe UI", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('The timeline appears after you import a video', W / 2, H / 2);
    ctx.textAlign = 'left';
    return;
  }
  const N = n();
  const fps = S.project.fps || 30;
  const objs = rows();
  const yObj = RULER + THUMB;
  const yCmp = yObj + objs.length * ROW;

  // --- clip region for the track area
  ctx.save();
  ctx.beginPath();
  ctx.rect(GUT, 0, W - GUT, H);
  ctx.clip();

  // ruler ticks
  const minGap = 120;
  const steps = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1500, 3000, 6000];
  const step = steps.find((s) => s * px >= minGap) || steps[steps.length - 1];
  ctx.fillStyle = '#8b93a5';
  ctx.strokeStyle = '#2a2f3a';
  ctx.font = '10px Consolas, monospace';
  ctx.textBaseline = 'top';
  const f0 = Math.max(0, Math.floor(frameAt(GUT) / step) * step);
  for (let f = f0; f < N; f += step) {
    const x = xOf(f);
    if (x > W) break;
    ctx.beginPath();
    ctx.moveTo(x + 0.5, 12);
    ctx.lineTo(x + 0.5, RULER);
    ctx.stroke();
    ctx.fillText(`${fmtTime(f / fps)}  #${f}`, x + 3, 3);
  }
  // minor ticks
  if (px >= 4) {
    ctx.strokeStyle = '#20242d';
    for (let f = Math.max(0, frameAt(GUT)); f < N; f++) {
      const x = xOf(f);
      if (x > W) break;
      ctx.beginPath();
      ctx.moveTo(x + 0.5, 19);
      ctx.lineTo(x + 0.5, RULER);
      ctx.stroke();
    }
  }

  // thumbnails
  const tw = Math.max(20, Math.round(THUMB * (S.project.width / S.project.height)));
  for (let x = GUT - ((scroll % tw) || 0); x < W; x += tw) {
    const f = Math.max(0, Math.min(N - 1, frameAt(Math.max(GUT, x) + 1)));
    const im = img(thumbUrl(f));
    if (im.complete && im.naturalWidth) ctx.drawImage(im, x, RULER, tw - 1, THUMB - 2);
    else { ctx.fillStyle = '#1c2029'; ctx.fillRect(x, RULER, tw - 1, THUMB - 2); }
  }

  // object tracks
  objs.forEach((o, k) => {
    const y = yObj + k * ROW;
    ctx.fillStyle = o.id === S.sel ? '#1b2130' : (k % 2 ? '#14171e' : '#161a22');
    ctx.fillRect(GUT, y, W - GUT, ROW);
    ctx.globalAlpha = o.visible ? 0.85 : 0.3;
    ctx.fillStyle = o.color;
    for (const [a, b] of S.tl.coverage?.[o.id] || []) {
      const x1 = xOf(a), x2 = xOf(b + 1);
      if (x2 < GUT || x1 > W) continue;
      ctx.fillRect(x1, y + 7, Math.max(1, x2 - x1), ROW - 14);
    }
    ctx.globalAlpha = 1;
    for (const f of S.tl.keyframes?.[o.id] || []) {
      const x = xOf(f + 0.5);
      if (x < GUT - 8 || x > W + 8) continue;
      const cy = y + ROW / 2;
      ctx.beginPath();
      ctx.moveTo(x, cy - 7); ctx.lineTo(x + 6, cy); ctx.lineTo(x, cy + 7); ctx.lineTo(x - 6, cy);
      ctx.closePath();
      ctx.fillStyle = '#fff';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = o.color;
      ctx.stroke();
    }
  });

  // compare strip
  if (S.compare && S.compareObj != null) {
    const vals = S.compare.per_frame[S.compareObj] || [];
    for (let f = Math.max(0, frameAt(GUT)); f < N; f++) {
      const x = xOf(f);
      if (x > W) break;
      const v = vals[f];
      ctx.fillStyle = v == null ? '#1c2029' : `hsl(${Math.round(Math.max(0, (v - 0.5) * 2) * 120)} 70% 45%)`;
      ctx.fillRect(x, yCmp + 4, Math.max(1, px), CMP - 8);
    }
  }

  // in/out shading
  const [inF, outF] = S.project.in_out || [0, N - 1];
  ctx.fillStyle = 'rgba(0,0,0,.5)';
  if (inF > 0) ctx.fillRect(GUT, 0, xOf(inF) - GUT, H);
  if (outF < N - 1) ctx.fillRect(xOf(outF + 1), 0, W, H);
  ctx.fillStyle = '#ffb547';
  ctx.fillRect(xOf(inF) - 1, 0, 2, H);
  ctx.fillRect(xOf(outF + 1) - 1, 0, 2, H);

  // playhead
  const ph = xOf(S.frame);
  if (px > 3 && px < 80) { // highlight the current frame cell (not when one frame fills the timeline, e.g. images)
    ctx.fillStyle = 'rgba(255,77,94,.18)';
    ctx.fillRect(ph, RULER, px, H - RULER);
  }
  const cx = ph + Math.max(px, 1) / 2;
  ctx.fillStyle = '#ff4d5e';
  ctx.fillRect(cx - 1, 0, 2, H);
  ctx.beginPath();
  ctx.moveTo(cx - 6, 0); ctx.lineTo(cx + 6, 0); ctx.lineTo(cx, 9);
  ctx.closePath();
  ctx.fill();
  ctx.restore();

  // gutter labels
  ctx.fillStyle = '#161920';
  ctx.fillRect(0, 0, GUT, H);
  ctx.strokeStyle = '#2a2f3a';
  ctx.beginPath(); ctx.moveTo(GUT + 0.5, 0); ctx.lineTo(GUT + 0.5, H); ctx.stroke();
  ctx.font = '12px "Segoe UI", sans-serif';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = '#8b93a5';
  ctx.fillText(S.tl.model ? `masks: ${S.tl.model}` : 'no masks yet', 10, RULER / 2);
  ctx.fillText('Video', 10, RULER + THUMB / 2);
  objs.forEach((o, k) => {
    const y = yObj + k * ROW;
    if (o.id === S.sel) { ctx.fillStyle = '#1f2533'; ctx.fillRect(0, y, GUT, ROW); }
    ctx.fillStyle = o.color;
    ctx.fillRect(10, y + 8, 10, 10);
    ctx.fillStyle = o.visible ? '#e6e8ee' : '#6b7385';
    ctx.fillText(o.name.length > 16 ? o.name.slice(0, 15) + '…' : o.name, 28, y + ROW / 2);
  });
  if (!objs.length) {
    ctx.fillStyle = '#6b7385';
    ctx.fillText('No objects yet (N)', 10, yObj + ROW / 2);
  }
  if (S.compare) {
    ctx.fillStyle = '#8b93a5';
    ctx.fillText('IoU A vs B', 10, yCmp + CMP / 2);
  }
}

// ---------- interaction ----------
function rowAt(y) {
  const k = Math.floor((y - RULER - THUMB) / ROW);
  return y >= RULER + THUMB && k >= 0 && k < rows().length ? rows()[k] : null;
}

canvas.addEventListener('pointerdown', (ev) => {
  if (!S.project) return;
  const r = canvas.getBoundingClientRect();
  const x = ev.clientX - r.left, y = ev.clientY - r.top;
  const o = rowAt(y);
  if (x < GUT) {
    if (o) { S.sel = o.id; emit('project'); }
    return;
  }
  if (o) {
    const kf = (S.tl.keyframes?.[o.id] || []).find((f) => Math.abs(xOf(f + 0.5) - x) < 7);
    if (kf != null) {
      S.sel = o.id;
      emit('project');
      setFrame(kf);
      return;
    }
  }
  if (S.playing) { S.playing = false; emit('play'); }
  scrubbing = true;
  canvas.setPointerCapture(ev.pointerId);
  setFrame(frameAt(x));
});
canvas.addEventListener('pointermove', (ev) => {
  if (!scrubbing) return;
  const r = canvas.getBoundingClientRect();
  setFrame(frameAt(ev.clientX - r.left));
});
canvas.addEventListener('pointerup', () => { scrubbing = false; });

canvas.addEventListener('wheel', (ev) => {
  if (!S.project) return;
  const r = canvas.getBoundingClientRect();
  const x = ev.clientX - r.left;
  if (ev.ctrlKey) {
    ev.preventDefault();
    const f = (x - GUT + scroll) / px;
    zoomEl.value = Math.max(0, Math.min(100, +zoomEl.value - Math.sign(ev.deltaY) * 6));
    applyZoom();
    scroll = f * px - (x - GUT);
    clampScroll();
    requestDraw();
  } else if (ev.shiftKey || Math.abs(ev.deltaX) > Math.abs(ev.deltaY) || px > fitPx() * 1.01) {
    ev.preventDefault();
    scroll += (ev.deltaX || ev.deltaY);
    clampScroll();
    requestDraw();
  }
}, { passive: false });

zoomEl.addEventListener('input', () => {
  const f = S.frame;
  const keepX = xOf(f);
  applyZoom();
  scroll = f * px - (keepX - GUT);
  clampScroll();
  requestDraw();
});

export function setIn() {
  if (!S.project) return;
  const [, b] = S.project.in_out;
  patchProject({ in_out: [S.frame, Math.max(S.frame, b)] });
}
export function setOut() {
  if (!S.project) return;
  const [a] = S.project.in_out;
  patchProject({ in_out: [Math.min(a, S.frame), S.frame] });
}

function follow() {
  if (!S.project) return;
  const x = xOf(S.frame);
  if (x < GUT || x > W - 20) {
    scroll = S.frame * px - (W - GUT) * 0.3;
    clampScroll();
  }
}

new ResizeObserver(() => { layout(); requestDraw(); }).observe(scroller);
on('frame', () => { follow(); requestDraw(); });
on('timeline', requestDraw);
on('imgload', requestDraw);
on('project', () => { layout(); requestDraw(); });
on('compare', () => { layout(); requestDraw(); });
