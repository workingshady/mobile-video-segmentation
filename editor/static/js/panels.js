// Left objects list + right tabs (Effects, Models, Compare, Benchmarks) + model dropdowns.
import {
  S, on, emit, api, guard, toast, setFrame, promptAt, viewModels, refreshTimeline,
  addObject, patchObject, patchProject, deleteObject, loadModel, refreshModels, refreshProject, bumpEffects,
} from './app.js';

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const modelName = (id) => S.models.find((m) => m.id === id)?.name || id;

// =============== objects list ===============
function renderObjects() {
  const list = $('objectList');
  if (!S.project) { list.innerHTML = '<p class="hint">Import a video to start.</p>'; return; }
  if (!S.project.objects.length) {
    list.innerHTML = '<p class="hint">No objects. Press <b>＋ New</b> (or just click on the video with the ＋ tool).</p>';
  } else {
    list.innerHTML = '';
  }
  for (const o of S.project.objects) {
    const kf = (S.tl.keyframes?.[o.id] || []).length;
    const cov = (S.tl.coverage?.[o.id] || []).reduce((s, [a, b]) => s + b - a + 1, 0);
    const el = document.createElement('div');
    el.className = 'obj' + (o.id === S.sel ? ' sel' : '');
    el.innerHTML = `
      <div class="swatch" style="background:${o.color}" title="Color"><input type="color" value="${o.color}"></div>
      <div class="name" title="Double-click to rename">${esc(o.name)}</div>
      <div class="acts">
        <button class="vis ${o.visible ? '' : 'off'}" title="Show / hide">👁</button>
        <button class="fg ${o.foreground ? '' : 'off'}" title="Foreground (kept when background is replaced)">★</button>
        <button class="del" title="Delete object">✕</button>
      </div>
      <div class="sub">${kf} prompt frame${kf === 1 ? '' : 's'} · ${cov}/${S.project.num_frames} frames masked</div>`;
    el.addEventListener('click', () => { S.sel = o.id; emit('project'); });
    el.querySelector('input[type=color]').addEventListener('input', (e) => {
      el.querySelector('.swatch').style.background = e.target.value;
      patchObject(o.id, { color: e.target.value, effect: { color: e.target.value } });
    });
    el.querySelector('input[type=color]').addEventListener('click', (e) => e.stopPropagation());
    el.querySelector('.vis').addEventListener('click', (e) => { e.stopPropagation(); patchObject(o.id, { visible: !o.visible }, true); });
    el.querySelector('.fg').addEventListener('click', (e) => { e.stopPropagation(); patchObject(o.id, { foreground: !o.foreground }, true); });
    el.querySelector('.del').addEventListener('click', (e) => {
      e.stopPropagation();
      if (confirm(`Delete "${o.name}" and all its prompts and masks?`)) deleteObject(o.id);
    });
    el.querySelector('.name').addEventListener('dblclick', (e) => {
      const nameEl = e.currentTarget;
      nameEl.innerHTML = `<input value="${esc(o.name)}">`;
      const inp = nameEl.querySelector('input');
      inp.focus();
      inp.select();
      const done = () => patchObject(o.id, { name: inp.value.trim() || o.name }, true);
      inp.addEventListener('blur', done);
      inp.addEventListener('keydown', (k) => { if (k.key === 'Enter') inp.blur(); k.stopPropagation(); });
    });
    list.appendChild(el);
  }
  const rec = S.sel != null ? promptAt(S.sel, S.frame) : null;
  $('promptHint').textContent = !S.project.objects.length ? '' : rec
    ? (rec.mask ? 'This frame: brush prompt.' : `This frame: ${rec.points.length} point(s)${rec.box ? ' + box' : ''}.`)
    : 'No prompt on this frame for the selected object.';
  $('btnClearFrame').disabled = !rec;
}

// =============== effects tab ===============
const EFFECTS = [
  ['none', 'None'], ['highlight', 'Highlight'], ['outline', 'Outline'], ['blur', 'Blur (privacy)'],
  ['pixelate', 'Pixelate'], ['recolor', 'Recolor'], ['erase', 'Erase (inpaint)'],
];
const BACKGROUNDS = [
  ['keep', 'Keep'], ['remove', 'Remove (transparent)'], ['color', 'Solid color'], ['blur', 'Blur'],
  ['grayscale', 'Grayscale (color pop)'], ['image', 'Image…'],
];

function field(label, html, val = '') {
  return `<label class="field"><span>${label}</span>${html}<span class="val">${val}</span></label>`;
}
function range(key, v, min, max, step = 1) {
  return `<input type="range" data-k="${key}" min="${min}" max="${max}" step="${step}" value="${v}">`;
}
function options(list, cur) {
  return list.map(([v, l]) => `<option value="${v}" ${v === cur ? 'selected' : ''}>${l}</option>`).join('');
}

function renderEffects() {
  const el = $('tab-effects');
  if (!S.project) { el.innerHTML = '<p class="hint">Import a video first.</p>'; return; }
  const o = S.project.objects.find((x) => x.id === S.sel);
  const bg = S.project.background;
  const e = o?.effect || {};
  el.innerHTML = `
    <h4>Selected object</h4>
    ${o ? `
      ${field('Effect', `<select data-k="type">${options(EFFECTS, e.type)}</select>`)}
      ${field('Color', `<input type="color" data-k="color" value="${e.color || o.color}">`)}
      ${field('Opacity', range('opacity', e.opacity ?? 0.5, 0, 1, 0.05), (e.opacity ?? 0.5).toFixed(2))}
      ${field('Strength', range('strength', e.strength ?? 15, 1, 80), e.strength ?? 15)}
      ${field('Outline px', range('outline', e.outline ?? 2, 1, 12), e.outline ?? 2)}
      ${field('Grow/shrink', range('grow', e.grow ?? 0, -20, 20), e.grow ?? 0)}
      ${field('Feather', range('feather', e.feather ?? 2, 0, 20, 0.5), e.feather ?? 2)}
    ` : '<p class="hint">Select or create an object.</p>'}
    <h4>Background</h4>
    ${field('Mode', `<select data-bg="mode">${options(BACKGROUNDS, bg.mode)}</select>`)}
    ${field('Color', `<input type="color" data-bg="color" value="${bg.color}">`)}
    ${field('Blur', `<input type="range" data-bg="strength" min="1" max="80" value="${bg.strength}">`, bg.strength)}
    ${field('Image', `<button class="small" id="bgImageBtn">Choose…</button><input type="file" id="bgImageFile" accept="image/*" hidden>`, bg.image ? '✓' : '')}
    <p class="hint">Background effects apply around objects marked ★ (foreground).</p>
    <h4>Quality</h4>
    <label class="check"><input type="checkbox" id="tSmooth" ${S.project.settings.temporal_smooth ? 'checked' : ''}> Temporal smoothing (averages masks over ±1 frame, less flicker)</label>
    <p class="hint">Switch the viewer to <b>Preview</b> (P) to see effects. Export uses exactly the same renderer.</p>`;

  el.querySelectorAll('[data-k]').forEach((inp) => {
    inp.addEventListener('input', () => {
      const k = inp.dataset.k;
      const v = inp.type === 'range' ? +inp.value : inp.value;
      const valEl = inp.parentElement.querySelector('.val');
      if (valEl && inp.type === 'range') valEl.textContent = k === 'opacity' ? (+v).toFixed(2) : v;
      patchObject(o.id, { effect: { [k]: v } });
    });
  });
  el.querySelectorAll('[data-bg]').forEach((inp) => {
    inp.addEventListener('input', () => {
      const k = inp.dataset.bg;
      const v = inp.type === 'range' ? +inp.value : inp.value;
      const valEl = inp.parentElement.querySelector('.val');
      if (valEl && inp.type === 'range') valEl.textContent = v;
      patchProject({ background: { [k]: v } });
    });
  });
  $('tSmooth').addEventListener('change', (ev) => patchProject({ settings: { temporal_smooth: ev.target.checked } }));
  $('bgImageBtn').addEventListener('click', () => $('bgImageFile').click());
  $('bgImageFile').addEventListener('change', guard(async (ev) => {
    const f = ev.target.files[0];
    if (!f) return;
    const fd = new FormData();
    fd.append('file', f);
    await api('/project/background-image', { method: 'POST', body: fd });
    bumpEffects();
    await refreshProject();
    toast('Background image set', 'ok');
  }));
}

// =============== models tab ===============
const fmtSize = (mb) => (mb >= 1000 ? (mb / 1000).toFixed(1) + ' GB' : mb + ' MB');

export function tierBadge(tierId, small = false) {
  const t = S.tiers?.[tierId];
  if (!t) return '';
  return `<span class="tier tier-${tierId}${small ? ' sm' : ''}" title="${esc(t.desc)}">${t.icon} ${esc(t.label)}</span>`;
}

let showDesktop = false;

function renderModels() {
  const el = $('tab-models');
  let html = `<div class="model-tools">
      <button class="small" id="btnRefreshModels">↻ Re-scan</button>
      <button class="small" id="btnUnload" ${S.loaded ? '' : 'disabled'}>Unload</button>
      <label class="check"><input type="checkbox" id="showDesktop" ${showDesktop ? 'checked' : ''}>Show desktop models</label>
    </div>
    <p class="hint" style="margin-bottom:12px">Phone speeds are published device numbers. The Speed tab shows this PC, not the phone.</p>`;
  for (const tier of S.tierOrder || []) {
    if (tier === 'desktop' && !showDesktop) continue;
    const ms = S.models.filter((m) => m.tier === tier && m.group !== 'huge');
    if (!ms.length) continue;
    const t = S.tiers[tier];
    html += `<div class="tier-head">${tierBadge(tier)}<span class="hint">${esc(t.desc)}</span></div>`;
    for (const m of ms) {
      const loaded = m.id === S.loaded;
      const st = loaded ? 'loaded' : m.status;
      const inEditor = m.group === 'editor';
      html += `<div class="card model-card tier-edge-${tier} ${loaded ? 'loaded' : ''}">
        <div class="top"><span class="title">${esc(m.name)}</span>
          <span class="status ${st}">${inEditor ? st : (m.status === 'downloaded' ? 'file ✓' : m.status)}</span></div>
        <div class="phone-speed">📱 ${esc(m.phone_speed || '')}</div>
        <div class="notes">${esc(m.notes)}${m.size_mb ? ` · <b>${fmtSize(m.size_mb)}</b>` : ''}</div>
        ${m.reason && !m.runnable && inEditor ? `<div class="reason">${esc(m.reason)}</div>` : ''}
        ${m.runnable && !loaded ? `<button class="small primary" data-load="${m.id}" style="margin-top:8px">Load</button>` : ''}
      </div>`;
    }
  }
  if (!showDesktop) {
    const n = S.models.filter((m) => m.tier === 'desktop').length;
    html += `<p class="hint">${n} desktop-only models hidden (quality references). Tick “Show desktop models” to see them.</p>`;
  }
  el.innerHTML = html;
  $('showDesktop').addEventListener('change', (e) => { showDesktop = e.target.checked; renderModels(); });
  el.querySelectorAll('[data-load]').forEach((b) => b.addEventListener('click', () => loadModel(b.dataset.load)));
  $('btnRefreshModels').addEventListener('click', guard(async () => {
    await api('/models/refresh', { method: 'POST' });
    await refreshModels();
    toast('Model list refreshed');
  }));
  $('btnUnload').addEventListener('click', guard(async () => {
    await api('/models/unload', { method: 'POST' });
    await refreshModels();
    await refreshProject();
  }));
}

function renderModelSelects() {
  const sel = $('modelSelect');
  const runnable = S.models.filter((m) => m.group === 'editor');
  // grouped by tier: phone real-time first, desktop references last
  sel.innerHTML = (S.tierOrder || []).map((tier) => {
    const ms = runnable.filter((m) => m.tier === tier);
    if (!ms.length) return '';
    const t = S.tiers[tier];
    return `<optgroup label="${t.icon} ${esc(t.label)}">${ms.map((m) => {
      const tag = m.id === S.loaded ? '  ● loaded' : m.runnable ? `  · ${fmtSize(m.size_mb)}` : `  (${m.status})`;
      return `<option value="${m.id}" ${m.runnable ? '' : 'disabled'}>${esc(m.name)}${tag}</option>`;
    }).join('')}</optgroup>`;
  }).join('');
  if (S.loaded) sel.value = S.loaded;
  else {
    const first = runnable.find((m) => m.runnable && m.tier === 'phone-realtime') || runnable.find((m) => m.runnable);
    if (first) sel.value = first.id;
  }
  const cur = S.models.find((m) => m.id === sel.value);
  $('modelTier').innerHTML = cur ? tierBadge(cur.tier, true) : '';
  const chip = $('modelChip');
  const lm = S.models.find((m) => m.id === S.loaded);
  const where = lm?.backend === 'onnx-edgetam' ? `CPU · ${lm.threads || 4} threads` : (S.device || '').toUpperCase();
  chip.textContent = S.loaded ? `${modelName(S.loaded)} loaded · ${where}` : 'No model loaded';
  chip.classList.toggle('on', !!S.loaded);
  $('modelDot').classList.toggle('on', !!S.loaded);

  const vms = viewModels();
  const vsel = $('viewModelSelect');
  vsel.innerHTML = vms.length
    ? vms.map((id) => `<option value="${id}">${esc(modelName(id))}${id === S.loaded ? ' (loaded)' : ''}</option>`).join('')
    : '<option value="">— none yet —</option>';
  if (S.viewModel) vsel.value = S.viewModel;
}

$('modelSelect').addEventListener('change', (e) => {
  const m = S.models.find((x) => x.id === e.target.value);
  $('modelTier').innerHTML = m ? tierBadge(m.tier, true) : '';
});

$('viewModelSelect').addEventListener('change', async (e) => {
  S.viewModel = e.target.value || null;
  await refreshTimeline();
  emit('project');
});

// =============== compare tab ===============
function renderCompare() {
  const el = $('tab-compare');
  if (!S.project) { el.innerHTML = '<p class="hint">Import a video first.</p>'; return; }
  const vms = S.project.models_with_masks || [];
  if (vms.length < 2) {
    el.innerHTML = `<p class="hint">Track the same prompts with <b>two or more models</b>, then compare them here.<br><br>
      Tip: use <b>SAM 2.1 Large</b> as the reference (B) and a phone model (EdgeTAM, EfficientTAM) as A.
      Prompts are shared between models, so the test is fair.</p>`;
    return;
  }
  const opts = (cur) => vms.map((id) => `<option value="${id}" ${id === cur ? 'selected' : ''}>${esc(modelName(id))}</option>`).join('');
  const a = S.compare?.a || vms.find((x) => x !== 'sam2.1-large') || vms[0];
  const b = S.compare?.b || (vms.includes('sam2.1-large') ? 'sam2.1-large' : vms.find((x) => x !== a));
  let html = `
    ${field('A (test)', `<select id="cmpA">${opts(a)}</select>`)}
    ${field('B (reference)', `<select id="cmpB">${opts(b)}</select>`)}
    <button class="primary" id="cmpGo" style="width:100%">Compare masks</button>`;
  if (S.compare) {
    html += '<h4>Agreement (IoU, 1.0 = identical)</h4>';
    for (const o of S.project.objects) {
      const s = S.compare.summary[o.id];
      if (!s) continue;
      html += `<div class="card">
        <div class="top" style="display:flex;gap:8px;align-items:center">
          <span style="width:10px;height:10px;border-radius:3px;background:${o.color}"></span>
          <b style="flex:1">${esc(o.name)}</b>
          <button class="small" data-strip="${o.id}">${S.compareObj === o.id ? 'on timeline ✓' : 'show on timeline'}</button>
        </div>
        <div class="notes">mean <b>${s.mean_iou ?? '—'}</b> · min ${s.min_iou ?? '—'} · ${s.frames} frames</div>
        <canvas class="spark" data-spark="${o.id}"></canvas>
        <div class="worst">${s.worst.map((w) => `<button data-jump="${w.frame}" title="IoU ${w.iou}">#${w.frame} · ${w.iou.toFixed(2)}</button>`).join('')}</div>
      </div>`;
    }
  }
  el.innerHTML = html;
  $('cmpGo').addEventListener('click', guard(async () => {
    S.compare = await api('/compare', { json: { a: $('cmpA').value, b: $('cmpB').value } });
    S.compareObj = S.sel ?? S.project.objects[0]?.id ?? null;
    renderCompare();
    emit('compare');
  }));
  el.querySelectorAll('[data-jump]').forEach((btn) => btn.addEventListener('click', () => setFrame(+btn.dataset.jump)));
  el.querySelectorAll('[data-strip]').forEach((btn) => btn.addEventListener('click', () => {
    S.compareObj = +btn.dataset.strip;
    renderCompare();
    emit('compare');
  }));
  el.querySelectorAll('[data-spark]').forEach((c) => drawSpark(c, S.compare.per_frame[c.dataset.spark] || []));
}

function drawSpark(c, vals) {
  const d = window.devicePixelRatio || 1;
  c.width = c.clientWidth * d;
  c.height = c.clientHeight * d;
  const g = c.getContext('2d');
  g.fillStyle = '#0d0f14';
  g.fillRect(0, 0, c.width, c.height);
  g.strokeStyle = '#5b8cff';
  g.lineWidth = 1.5 * d;
  g.beginPath();
  let started = false;
  vals.forEach((v, i) => {
    if (v == null) { started = false; return; }
    const x = (i / Math.max(1, vals.length - 1)) * c.width;
    const y = c.height - v * (c.height - 4 * d) - 2 * d;
    if (started) g.lineTo(x, y); else { g.moveTo(x, y); started = true; }
  });
  g.stroke();
}

// =============== benchmarks tab ===============
async function renderRuns() {
  const el = $('tab-runs');
  if (!S.project) { el.innerHTML = '<p class="hint">Import a video first.</p>'; return; }
  const { runs } = await api('/runs');
  if (!runs.length) {
    el.innerHTML = '<p class="hint">Every Track run is timed here (fps, ms/frame, peak GPU memory). Track with several models to compare speed.</p>';
    return;
  }
  const latest = {};
  for (const r of runs) if (!r.cancelled && r.frames > 0) latest[r.model] = r;
  const best = Object.values(latest).reduce((m, r) => Math.max(m, r.fps), 0);
  let html = '<h4>Speed (latest full run per model, this PC)</h4><div class="bars">';
  for (const r of Object.values(latest).sort((x, y) => y.fps - x.fps)) {
    const tier = S.models.find((m) => m.id === r.model)?.tier;
    html += `<div class="bar"><span title="${esc(S.tiers?.[tier]?.label || '')}">${S.tiers?.[tier]?.icon || ''} ${esc(r.model_name)}</span><div class="track"><div class="fill fill-${tier}" style="width:${(100 * r.fps) / best}%"></div></div><span class="mono">${r.fps} fps</span></div>`;
  }
  html += `</div><p class="hint">PyTorch models run on this PC's GPU; 📲 phone-pipeline models run on CPU (4 threads) like a phone without NPU.
    Real phone speed with the NPU/GPU: see each model's 📱 line in the Models tab.</p>`;
  const staged = [...runs].reverse().find((r) => r.stage_ms && !r.cancelled);
  if (staged) {
    const total = Object.values(staged.stage_ms).reduce((a, b) => a + b, 0) || 1;
    html += `<h4>Where the phone pipeline spends time</h4><p class="hint">${esc(staged.model_name)} · ms per call</p><div class="bars">`;
    for (const [k, v] of Object.entries(staged.stage_ms)) {
      html += `<div class="bar"><span>${esc(k.replace(/_/g, ' '))}</span><div class="track"><div class="fill fill-phone-onnx" style="width:${(100 * v) / total}%"></div></div><span class="mono">${v} ms</span></div>`;
    }
    html += `</div><p class="hint">Memory attention and the image encoder dominate. On a phone these run on the NPU/GPU (CoreML on iPhone, QNN/NNAPI on Android).</p>`;
  }
  html += `
    <h4>All runs</h4><table class="grid"><tr><th>Model</th><th>Frames</th><th>fps</th><th>ms/f</th><th>VRAM MB</th></tr>`;
  for (const r of [...runs].reverse()) {
    html += `<tr class="${r.fps === best ? 'best' : ''}" title="${r.time} · ${r.direction} · ${r.resolution} · ${r.objects} obj">
      <td>${esc(r.model_name)}${r.cancelled ? ' (stopped)' : ''}</td><td>${r.frames}</td><td>${r.fps}</td><td>${r.ms_per_frame}</td><td>${r.peak_vram_mb}</td></tr>`;
  }
  el.innerHTML = html + '</table>';
}

// =============== tabs ===============
let tab = 'effects';
document.querySelectorAll('#rightTabs button').forEach((b) => b.addEventListener('click', () => {
  tab = b.dataset.tab;
  document.querySelectorAll('#rightTabs button').forEach((x) => x.classList.toggle('on', x === b));
  document.querySelectorAll('.tab-body').forEach((x) => x.classList.toggle('hidden', x.id !== 'tab-' + tab));
  renderTab();
}));

function renderTab() {
  if (tab === 'effects') renderEffects();
  if (tab === 'models') renderModels();
  if (tab === 'compare') renderCompare();
  if (tab === 'runs') renderRuns().catch(() => {});
}

// effects panel holds live inputs: only re-render it when the selection/project changes, not on every patch
let lastEffectsKey = '';
function maybeRenderEffects() {
  const key = JSON.stringify([S.project?.id, S.sel, S.project?.objects.map((o) => o.id), S.project?.background.image]);
  if (key !== lastEffectsKey || !S.project) {
    lastEffectsKey = key;
    if (tab === 'effects') renderEffects();
  }
}

$('btnAddObject').addEventListener('click', addObject);

// open a tab from the URL, e.g. http://127.0.0.1:7860/#tab=models
const hashTab = new URLSearchParams(location.hash.slice(1)).get('tab');
if (hashTab) document.querySelector(`#rightTabs [data-tab="${hashTab}"]`)?.click();

on('models', () => { renderModelSelects(); if (tab === 'models') renderModels(); });
on('project', () => {
  renderObjects();
  renderModelSelects();
  maybeRenderEffects();
  if (tab === 'compare' && !S.compare) renderCompare();
  if (tab === 'runs' && S.project?.id !== lastRunsProject) {
    lastRunsProject = S.project?.id;
    renderRuns().catch(() => {});
  }
});
let lastRunsProject;
on('timeline', renderObjects);
on('frame', renderObjects);
on('runs', () => { if (tab === 'runs') renderRuns(); });
