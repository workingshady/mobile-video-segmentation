// Boot + wiring: top bar, transport, dialogs, keyboard shortcuts, playback, status polling.
import {
  S, on, emit, api, guard, toast, runJob, refreshAll, refreshModels, refreshProject, refreshTimeline,
  setFrame, lastFrame, promptAt, clearPrompt, undo, addObject, loadModel, track, cancelJob, viewModels,
} from './app.js';
import { fit, isReady, preload, requestDraw, setCursor } from './viewer.js';
import { setIn, setOut } from './timeline.js';
import './panels.js';

const $ = (id) => document.getElementById(id);

// ---------------- tools ----------------
function setTool(t) {
  S.tool = t;
  document.querySelectorAll('#toolbar [data-tool]').forEach((b) => b.classList.toggle('on', b.dataset.tool === t));
  $('brushCtl').classList.toggle('hidden', !(t === 'brush' || t === 'erase' || t === 'magic'));
  setCursor();
  requestDraw();
}
document.querySelectorAll('#toolbar [data-tool]').forEach((b) => b.addEventListener('click', () => setTool(b.dataset.tool)));
function setBrush(v) {
  S.brushSize = Math.max(2, Math.min(160, v));
  $('brushSize').value = S.brushSize;
  $('brushSizeLabel').textContent = `${S.brushSize} px`;
  requestDraw();
}
$('brushSize').addEventListener('input', (e) => setBrush(+e.target.value));
const nudgeBrush = (d) => setBrush(S.brushSize + d);
$('btnFit').addEventListener('click', fit);
$('btnUndo').addEventListener('click', undo);
$('btnClearFrame').addEventListener('click', () => S.sel != null && clearPrompt(S.sel, S.frame));

// ---------------- view mode ----------------
function setView(v) {
  S.view = v;
  document.querySelectorAll('#viewMode button').forEach((b) => b.classList.toggle('on', b.dataset.view === v));
  requestDraw();
}
document.querySelectorAll('#viewMode button').forEach((b) => b.addEventListener('click', () => setView(b.dataset.view)));

// ---------------- projects menu ----------------
const menu = $('projectsMenu');
$('btnProjects').addEventListener('click', guard(async (e) => {
  e.stopPropagation();
  if (menu.classList.toggle('open')) {
    const { projects } = await api('/projects');
    menu.innerHTML = projects.length ? '' : '<div class="empty-item">No projects yet</div>';
    for (const p of projects) {
      const it = document.createElement('div');
      it.className = 'item';
      it.innerHTML = `<span>${p.kind === 'image' ? '🖼' : '🎞'}</span><span>${p.name}</span><span class="meta">${p.num_frames} fr</span><button class="del" title="Delete project">✕</button>`;
      it.addEventListener('click', () => { menu.classList.remove('open'); openProject(p.id); });
      it.querySelector('.del').addEventListener('click', guard(async (ev) => {
        ev.stopPropagation();
        if (!confirm(`Delete project "${p.name}" from disk?`)) return;
        await api('/projects/' + p.id, { method: 'DELETE' });
        it.remove();
      }));
      menu.appendChild(it);
    }
  }
}));
document.addEventListener('click', () => menu.classList.remove('open'));

const openProject = guard(async (pid) => {
  const job = await runJob(await api(`/projects/${pid}/open`, { method: 'POST' }));
  if (job.status === 'done') await afterProjectChange();
});

async function afterProjectChange() {
  S.compare = null;
  S.compareObj = null;
  S.frame = 0;
  S.viewModel = null;
  await refreshAll();
  emit('newproject');
  emit('compare');
  preload(0, 20);
}

// ---------------- import ----------------
const importDlg = $('importDialog');
let importFile = null;
function openImport(file) {
  importFile = file || null;
  $('importFileName').textContent = file ? file.name : 'Choose a file…';
  $('importName').value = '';
  importDlg.showModal();
}
$('btnImport').addEventListener('click', () => openImport());
$('btnEmptyImport').addEventListener('click', () => openImport());
$('importFile').addEventListener('change', (e) => {
  importFile = e.target.files[0];
  $('importFileName').textContent = importFile ? importFile.name : 'Choose a file…';
});
$('importGo').addEventListener('click', guard(async (e) => {
  if (!importFile) { e.preventDefault(); return toast('Choose a video or image file'); }
  const fd = new FormData();
  fd.append('file', importFile);
  fd.append('name', $('importName').value.trim());
  fd.append('max_side', $('importMaxSide').value);
  fd.append('start', $('importStart').value || 0);
  fd.append('end', $('importEnd').value || 0);
  fd.append('step', $('importStep').value || 1);
  fd.append('max_frames', $('importMaxFrames').value || 400);
  S.job = { status: 'running', progress: 0, message: 'Uploading…' };
  emit('job');
  const job = await runJob(await api('/projects/import', { method: 'POST', body: fd }));
  if (job.status === 'done') {
    await afterProjectChange();
    toast('Imported. Next: pick a model (top right) → Load → click the object.', 'ok', 6000);
  }
}));

const wrap = $('viewerWrap');
wrap.addEventListener('dragover', (e) => { e.preventDefault(); wrap.classList.add('drag'); });
wrap.addEventListener('dragleave', () => wrap.classList.remove('drag'));
wrap.addEventListener('drop', (e) => {
  e.preventDefault();
  wrap.classList.remove('drag');
  const f = e.dataTransfer.files[0];
  if (f) openImport(f);
});

// ---------------- export ----------------
const exportDlg = $('exportDialog');
$('btnExport').addEventListener('click', () => {
  if (!S.project) return toast('Import a video first');
  const vms = viewModels();
  $('exportModel').innerHTML = vms.length
    ? vms.map((id) => `<option value="${id}">${S.models.find((m) => m.id === id)?.name || id}</option>`).join('')
    : '<option value="">(no masks)</option>';
  if (S.viewModel) $('exportModel').value = S.viewModel;
  exportDlg.showModal();
});
$('exportGo').addEventListener('click', guard(async () => {
  const fmt = document.querySelector('input[name=fmt]:checked').value;
  const body = {
    format: fmt,
    resolution: $('exportRes').value,
    use_range: $('exportRange').checked,
    audio: $('exportAudio').checked,
    model: $('exportModel').value || null,
  };
  const job = await runJob(await api('/export', { json: body }));
  if (job.status === 'done') {
    const r = job.result;
    const el = document.createElement('div');
    el.innerHTML = r.is_file
      ? `Exported <b>${r.name}</b> — <a href="/api/download/${encodeURIComponent(r.name)}" download>download</a><br><span class="muted mono" style="font-size:11px">${r.path}</span>`
      : `Exported frames to<br><span class="mono" style="font-size:11px">${r.path}</span>`;
    toast(el, 'ok', 15000);
  }
}));

// ---------------- models ----------------
$('btnLoadModel').addEventListener('click', () => {
  const id = $('modelSelect').value;
  if (id) loadModel(id);
});

// ---------------- transport ----------------
$('btnStart').addEventListener('click', () => setFrame(0));
$('btnEnd').addEventListener('click', () => setFrame(lastFrame()));
$('btnPrev').addEventListener('click', () => setFrame(S.frame - 1));
$('btnNext').addEventListener('click', () => setFrame(S.frame + 1));
$('btnPlay').addEventListener('click', togglePlay);
$('btnIn').addEventListener('click', setIn);
$('btnOut').addEventListener('click', setOut);
$('frameInput').addEventListener('change', (e) => setFrame(+e.target.value));
$('btnTrackBack').addEventListener('click', () => track('backward'));
$('btnTrackBoth').addEventListener('click', () => track('both'));
$('btnTrackFwd').addEventListener('click', () => track('forward'));
$('btnStop').addEventListener('click', cancelJob);

function fmt(sec) {
  const m = Math.floor(sec / 60);
  return `${String(m).padStart(2, '0')}:${(sec - m * 60).toFixed(2).padStart(5, '0')}`;
}
function updateTransport() {
  const p = S.project;
  $('timecode').textContent = p ? fmt(S.frame / (p.fps || 30)) : '00:00.00';
  $('frameInput').value = S.frame;
  $('frameInput').max = p ? p.num_frames - 1 : 0;
  $('btnPlay').textContent = S.playing ? '❚❚' : '▶';
  $('projectName').textContent = p ? `${p.name} · ${p.width}×${p.height}` : 'No project';
  $('frameTotal').textContent = p ? `/ ${p.num_frames - 1}` : '/ 0';
  $('frameTotal').title = p ? `${p.fps.toFixed(2)} fps` : '';
  $('emptyState').classList.toggle('hidden', !!p);
}

// ---------------- playback ----------------
let lastTick = 0;
function togglePlay() {
  if (!S.project) return;
  S.playing = !S.playing;
  if (S.playing) {
    const end = $('useRange').checked ? S.project.in_out[1] : lastFrame();
    if (S.frame >= end) setFrame($('useRange').checked ? S.project.in_out[0] : 0);
    lastTick = 0;
    requestAnimationFrame(tick);
  }
  emit('play');
}
function tick(ts) {
  if (!S.playing || !S.project) return;
  const dt = 1000 / (S.project.fps || 30);
  if (!lastTick || ts - lastTick >= dt) {
    const end = $('useRange').checked ? S.project.in_out[1] : lastFrame();
    const next = S.frame + 1;
    if (next > end) {
      S.playing = false;
      emit('play');
      return;
    }
    preload(next, 15);
    if (isReady(next)) {   // wait for frame + mask so playback never shows a stale mask
      setFrame(next);
      lastTick = ts;
    }
  }
  requestAnimationFrame(tick);
}

// ---------------- job bar ----------------
function updateJob() {
  const j = S.job;
  const bar = $('jobBar');
  const running = j?.status === 'running';
  $('jobFill').style.width = running ? `${Math.round((j.progress || 0) * 100)}%` : '0';
  let text = 'Ready';
  if (j?.status === 'error') text = 'Error: ' + j.error;
  else if (j?.status === 'cancelled') text = 'Stopped';
  else if (running) text = j.message || 'Working…';
  else if (j?.status === 'done') text = j.message ? `Done · ${j.message}` : 'Done';
  $('jobText').textContent = text;
  bar.title = text;
  bar.classList.toggle('error', j?.status === 'error');
  bar.classList.toggle('active', running);
  $('btnStop').classList.toggle('hidden', !(running && (j.kind === 'propagate' || j.kind === 'export')));
  for (const id of ['btnTrackBack', 'btnTrackBoth', 'btnTrackFwd', 'btnLoadModel', 'btnImport']) $(id).disabled = running;
}

// ---------------- status polling (VRAM chip) ----------------
async function pollStatus() {
  try {
    const s = await api('/status');
    $('vramChip').textContent = s.vram ? `${s.vram.gpu.replace('NVIDIA GeForce ', '')} · ${s.vram.used_mb}/${s.vram.total_mb} MB` : 'CPU only';
    if (s.loaded !== S.loaded) await refreshModels();
  } catch { /* server restarting */ }
  setTimeout(pollStatus, 3000);
}

// ---------------- keyboard ----------------
window.addEventListener('keydown', (e) => {
  const tag = e.target.tagName;
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || document.querySelector('dialog[open]')) return;
  const k = e.key;
  if (e.ctrlKey && k.toLowerCase() === 'z') { e.preventDefault(); undo(); return; }
  if (e.ctrlKey && k.toLowerCase() === 'o') { e.preventDefault(); openImport(); return; }
  if (e.ctrlKey && k.toLowerCase() === 'e') { e.preventDefault(); $('btnExport').click(); return; }
  if (e.ctrlKey) return;
  const step = e.shiftKey ? 10 : 1;
  switch (k) {
    case ' ': e.preventDefault(); togglePlay(); break;
    case 'ArrowLeft': e.preventDefault(); setFrame(S.frame - step); break;
    case 'ArrowRight': e.preventDefault(); setFrame(S.frame + step); break;
    case 'Home': setFrame(0); break;
    case 'End': setFrame(lastFrame()); break;
    case 'v': case 'V': setTool('select'); break;
    case 'a': case 'A': setTool('pos'); break;
    case 's': case 'S': setTool('neg'); break;
    case 'b': case 'B': setTool('box'); break;
    case 'r': case 'R': setTool('brush'); break;
    case 'm': case 'M': setTool('magic'); break;
    case 'e': case 'E': setTool('erase'); break;
    case '[': nudgeBrush(-4); break;
    case ']': nudgeBrush(4); break;
    case 'f': case 'F': fit(); break;
    case 'i': case 'I': setIn(); break;
    case 'o': case 'O': setOut(); break;
    case 'n': case 'N': addObject(); break;
    case 't': case 'T': track('both'); break;
    case 'p': case 'P': setView(S.view === 'edit' ? 'preview' : 'edit'); break;
    case 'Escape': cancelJob(); if (S.playing) togglePlay(); break;
    case 'Delete': case 'Backspace':
      if (S.sel != null && promptAt(S.sel, S.frame)) clearPrompt(S.sel, S.frame);
      break;
    default:
      if (/^[1-9]$/.test(k) && S.project?.objects[+k - 1]) { S.sel = S.project.objects[+k - 1].id; emit('project'); }
  }
});

// ---------------- boot ----------------
on('frame', () => { updateTransport(); preload(S.frame + 1, 6); });
on('project', updateTransport);
on('play', updateTransport);
on('job', updateJob);

(async () => {
  setTool('pos');
  try {
    await refreshAll();
  } catch (e) {
    toast('Server not reachable: ' + e.message, 'error', 10000);
  }
  updateTransport();
  pollStatus();
  if (S.project) preload(0, 20);
})();
