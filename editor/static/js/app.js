// Shared state, API client and actions used by every UI module.

export const S = {
  project: null,          // project.json from the server (+ id, models_with_masks, can_undo)
  models: [],             // catalog with status
  loaded: null,           // id of the model in GPU memory
  device: null,
  frame: 0,
  playing: false,
  tool: 'pos',
  brushSize: 24,
  sel: null,              // selected object id
  viewModel: null,        // whose masks are shown / exported
  view: 'edit',           // edit | preview
  tl: { coverage: {}, keyframes: {}, revs: [] },
  job: null,
  compare: null,
  compareObj: null,
};

const handlers = {};
export function on(ev, fn) { (handlers[ev] ||= []).push(fn); }
export function emit(ev, data) { (handlers[ev] || []).forEach((f) => f(data)); }

// ---------------- api ----------------
export async function api(path, opts = {}) {
  const o = { ...opts };
  if (o.json !== undefined) {
    o.method = o.method || 'POST';
    o.headers = { 'Content-Type': 'application/json' };
    o.body = JSON.stringify(o.json);
    delete o.json;
  }
  const r = await fetch('/api' + path, o);
  if (!r.ok) {
    let msg = r.statusText;
    try {
      const j = await r.json();
      msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail);
    } catch { /* not json */ }
    throw new Error(msg);
  }
  return (r.headers.get('content-type') || '').includes('json') ? r.json() : r;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export async function runJob(job) {
  S.job = job;
  emit('job');
  let lastTl = 0;
  while (job.status === 'running') {
    await sleep(300);
    job = await api('/jobs/' + job.id);
    S.job = job;
    emit('job');
    if (job.kind === 'propagate' && Date.now() - lastTl > 600) {
      lastTl = Date.now();
      refreshTimeline();
    }
  }
  if (job.status === 'error') toast(job.error, 'error', 10000);
  return job;
}

// ---------------- toasts ----------------
export function toast(msg, kind = 'info', ms = 4000) {
  const el = document.createElement('div');
  el.className = 'toast ' + kind;
  if (msg instanceof Node) el.appendChild(msg); else el.textContent = msg;
  document.getElementById('toasts').appendChild(el);
  setTimeout(() => el.remove(), ms);
}

export function guard(fn) {
  return async (...args) => {
    try { return await fn(...args); } catch (e) { toast(e.message, 'error', 8000); }
  };
}

// ---------------- refresh ----------------
export function viewModels() {
  const set = new Set(S.project?.models_with_masks || []);
  if (S.loaded) set.add(S.loaded);
  return [...set];
}

export async function refreshModels() {
  const r = await api('/models');
  S.models = r.models;
  S.tiers = r.tiers || {};
  S.tierOrder = r.tier_order || [];
  S.loaded = r.loaded;
  S.device = r.device;
  emit('models');
}

export async function refreshProject() {
  try { S.project = await api('/project'); } catch { S.project = null; }
  if (S.project) {
    if (!S.project.objects.some((o) => o.id === S.sel)) S.sel = S.project.objects[0]?.id ?? null;
    if (S.frame >= S.project.num_frames) S.frame = 0;
    const vms = viewModels();
    if (!S.viewModel || !vms.includes(S.viewModel)) {
      S.viewModel = S.loaded || S.project.active_model || vms[0] || null;
    }
  }
  emit('project');
}

export async function refreshTimeline() {
  if (!S.project) return;
  S.tl = await api('/timeline?model=' + encodeURIComponent(S.viewModel || ''));
  emit('timeline');
}

export async function refreshAll() {
  await refreshModels();
  await refreshProject();
  await refreshTimeline();
}

// ---------------- frames ----------------
export function setFrame(f) {
  if (!S.project) return;
  f = Math.max(0, Math.min(S.project.num_frames - 1, Math.round(f)));
  if (f === S.frame) return;
  S.frame = f;
  emit('frame');
}

export function lastFrame() {
  return S.project ? S.project.num_frames - 1 : 0;
}

// ---------------- urls (project id in every url so the browser cache never mixes projects) ----------------
function hashStr(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
  return (h >>> 0).toString(36);
}
export const frameUrl = (i) => `/api/frame/${i}.jpg?p=${S.project.id}`;
export const thumbUrl = (i) => `/api/thumb/${i}.jpg?p=${S.project.id}`;
export function overlayUrl(i) {
  const style = hashStr(JSON.stringify(S.project.objects.map((o) => [o.id, o.color, o.visible])));
  return `/api/overlay/${encodeURIComponent(S.viewModel)}/${i}.png?p=${S.project.id}&r=${S.tl.revs[i] || 0}&sel=${S.sel || 0}&s=${style}`;
}
// effectsRev only changes after the server has saved an effect change, so a preview
// image is never cached under settings the server hasn't applied yet.
let effectsRev = 0;
export function bumpEffects() { effectsRev++; emit('frame'); }
export function renderUrl(i) {
  const rev = S.viewModel ? (S.tl.revs[i] || 0) : 0;
  return `/api/render/${i}.jpg?p=${S.project.id}&model=${encodeURIComponent(S.viewModel || '')}&r=${rev}&k=${effectsRev}`;
}

// ---------------- prompts ----------------
export function promptAt(oid, f) {
  return S.project?.prompts?.[String(oid)]?.[String(f)] || null;
}

async function afterEdit(res) {
  if (res && res.ran === false && res.reason) toast(res.reason);
  if (res && res.ran && S.loaded && S.viewModel !== S.loaded) S.viewModel = S.loaded;
  await refreshProject();
  await refreshTimeline();
}

export async function ensureObject() {
  if (S.sel != null) return S.sel;
  const o = await api('/objects', { json: {} });
  S.sel = o.id;
  await refreshProject();
  return o.id;
}

export const sendPrompt = guard(async (oid, f, rec) => {
  // optimistic update so the point appears instantly
  const p = (S.project.prompts[String(oid)] ||= {});
  p[String(f)] = { ...rec, mask: null };
  emit('frame');
  const res = await api('/prompts', { json: { obj: oid, frame: f, points: rec.points, labels: rec.labels, box: rec.box } });
  await afterEdit(res);
});

// Magic brush: live updates while painting. Only refreshes the masks (cheap); the caller does a
// full refresh when the stroke ends. recordUndo is true only for the first update of a stroke.
export async function sendPromptLive(oid, f, rec, recordUndo, final) {
  const p = (S.project.prompts[String(oid)] ||= {});
  p[String(f)] = { ...rec, mask: null };
  const res = await api('/prompts', {
    json: { obj: oid, frame: f, points: rec.points, labels: rec.labels, box: rec.box, record_undo: recordUndo },
  });
  if (res.ran && S.loaded && S.viewModel !== S.loaded) S.viewModel = S.loaded;
  if (final) await afterEdit(res);
  else await refreshTimeline();
  return res;
}

export const sendMaskPrompt = guard(async (oid, f, dataUrl) => {
  const res = await api('/prompts/mask', { json: { obj: oid, frame: f, png: dataUrl } });
  await afterEdit(res);
});

export const clearPrompt = guard(async (oid, f) => {
  const res = await api(`/prompts/${oid}/${f}`, { method: 'DELETE' });
  await afterEdit(res);
});

export const undo = guard(async () => {
  const r = await api('/undo', { method: 'POST' });
  if (!r.ok) toast('Nothing to undo');
  await afterEdit();
});

// ---------------- objects ----------------
export const addObject = guard(async () => {
  if (!S.project) return toast('Import a video first');
  const o = await api('/objects', { json: {} });
  S.sel = o.id;
  await refreshProject();
  await refreshTimeline();
});

// Merge one level deep so debounced patches never drop an earlier change (e.g. effect.color then effect.opacity).
function mergePatch(into, patch) {
  for (const [k, v] of Object.entries(patch)) {
    into[k] = v && typeof v === 'object' && !Array.isArray(v) ? { ...(into[k] || {}), ...v } : v;
  }
  return into;
}

const patchTimers = {};
const pendingObj = {};
export function patchObject(oid, patch, immediate = false) {
  const o = S.project.objects.find((x) => x.id === oid);
  if (patch.effect) Object.assign(o.effect, patch.effect);
  Object.assign(o, { ...patch, effect: o.effect });
  emit('frame');
  mergePatch((pendingObj[oid] ||= {}), patch);
  clearTimeout(patchTimers[oid]);
  const send = guard(async () => {
    const body = pendingObj[oid];
    delete pendingObj[oid];
    await api('/objects/' + oid, { method: 'PATCH', json: body });
    bumpEffects();
    await refreshProject();
  });
  if (immediate) send(); else patchTimers[oid] = setTimeout(send, 200);
}

let projTimer;
let pendingProj = {};
export function patchProject(patch) {
  if (patch.background) Object.assign(S.project.background, patch.background);
  if (patch.settings) Object.assign(S.project.settings, patch.settings);
  if (patch.in_out) S.project.in_out = patch.in_out;
  emit('frame');
  emit('timeline');
  mergePatch(pendingProj, patch);
  clearTimeout(projTimer);
  projTimer = setTimeout(guard(async () => {
    const body = pendingProj;
    pendingProj = {};
    await api('/project', { method: 'PATCH', json: body });
    bumpEffects();
    await refreshProject();
  }), 200);
}

export const deleteObject = guard(async (oid) => {
  await api('/objects/' + oid, { method: 'DELETE' });
  await refreshProject();
  await refreshTimeline();
});

// ---------------- models / tracking ----------------
export const loadModel = guard(async (id) => {
  if (S.job?.status === 'running') return toast('Wait for the current job to finish');
  const job = await runJob(await api('/models/load', { json: { id } }));
  await refreshModels();
  if (job.status === 'done') {
    S.viewModel = S.loaded;
    toast(`${S.models.find((m) => m.id === S.loaded)?.name} loaded`, 'ok');
  }
  await refreshProject();
  await refreshTimeline();
});

export const track = guard(async (direction) => {
  if (!S.project) return toast('Import a video first');
  if (!S.loaded) return toast('Load a model first (top right)');
  if (S.job?.status === 'running') return toast('A job is already running');
  S.viewModel = S.loaded;
  const useRange = document.getElementById('useRange').checked;
  const job = await runJob(await api('/propagate', { json: { direction, use_range: useRange } }));
  await refreshProject();
  await refreshTimeline();
  if (job.status === 'done' && job.result) {
    const r = job.result;
    toast(`${r.model_name}: ${r.frames} frames in ${r.seconds}s → ${r.fps} fps (${r.ms_per_frame} ms/frame, ${r.peak_vram_mb} MB VRAM)`, 'ok', 7000);
  }
  emit('runs');
});

export const cancelJob = guard(async () => {
  if (S.job?.status === 'running') await api(`/jobs/${S.job.id}/cancel`, { method: 'POST' });
});
