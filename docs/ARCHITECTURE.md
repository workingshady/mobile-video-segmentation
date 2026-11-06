# Track Studio: Architecture (for developers)

```
browser (vanilla JS, no build step)            Python (FastAPI + PyTorch)
┌───────────────────────────────┐   HTTP    ┌──────────────────────────────────────────────┐
│ main.js    wiring, keys, play │ ───────▶  │ server.py   routes + Studio state            │
│ app.js     state, api, actions│           │   ├ jobs.py      background jobs + GPU lock  │
│ viewer.js  canvas + tools     │ ◀───────  │   ├ project.py   files on disk, prompts, masks│
│ timeline.js tracks, scrubbing │  JSON /   │   ├ effects.py   renderer (preview = export) │
│ panels.js  objects + tabs     │  PNG/JPG  │   ├ exporter.py  ffmpeg / PNG output         │
└───────────────────────────────┘           │   └ trackers/   one interface, many models   │
                                            │        ├ hf.py          SAM2.1/EdgeTAM/SAM3  │
                                            │        ├ efficienttam.py EfficientTAM        │
                                            │        ├ onnx_edgetam.py phone ONNX loop     │
                                            │        └ registry.py    catalog + status     │
                                            └──────────────────────────────────────────────┘
```

## Key design decisions

| Decision | Why |
|---|---|
| **Local web app** (FastAPI + browser UI) | The models need Python + CUDA, and a browser gives a real canvas/timeline UI with no build tooling. One `run_editor.bat` |
| **Prompts are the source of truth; the model session is a cache** | Every adapter gets `set_prompts()` (reset + replay all prompts). Deleting a prompt, undo, or adding an object after tracking just replays. No adapter-specific edge cases, and every model sees identical inputs |
| **Fast path for clicks** | `update_prompt()` adds a single (object, frame) prompt in place when the session allows it, so a click costs one decoder pass |
| **Masks stored per model** (`masks/<model_id>/00000.png`) | Tracking with model B never overwrites model A, which makes Compare and A/B viewing possible |
| **Label maps** (uint8 PNG, pixel = object id) | One small file per frame for all objects. Limitation: objects can't overlap (later object wins) |
| **Server-side overlay and preview rendering** | One renderer for preview and export, so what you see is what you get. Images are cached by URL with revision numbers |
| **Per-frame revision counters** | The browser refetches only frames whose mask changed (the URL includes `r=<rev>`), even during live propagation |
| **One GPU lock** | Clicks, tracking, and export never run model code at the same time. A click during tracking gets HTTP 409 "busy" |

## Project on disk

```
projects/<slug>/
  project.json         name, fps, sizes, objects[], prompts{obj:{frame:{points,labels,box,mask}}}, background, settings, in_out
  source.mp4           copy of the original (audio + original-resolution export)
  frames/00000.jpg     working frames (EfficientTAM needs exactly this layout)
  thumbs/              timeline thumbnails (lazy)
  masks/<model>/00000.png   label maps
  prompt_masks/*.png   brush prompts
  runs.json            benchmark log
  exports/             outputs
```

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/models` | Catalog + status (`ready`, `missing`, `gated`, `no-backend`, `downloaded`, `unavailable`) |
| POST | `/api/models/load` `{id}` | Job: load the model (unloads the previous one) and build the video session |
| POST | `/api/models/unload`, `/api/models/refresh` | |
| GET | `/api/projects` · POST `/api/projects/import` (multipart) · POST `/api/projects/{id}/open` · DELETE `/api/projects/{id}` | |
| GET/PATCH | `/api/project` | Metadata. PATCH: `background`, `settings`, `in_out`, `active_model`, `name` |
| POST | `/api/project/background-image` | Upload the background replacement image |
| GET | `/api/frame/{i}.jpg`, `/api/thumb/{i}.jpg` | Frames |
| GET | `/api/overlay/{model}/{i}.png` | Colored RGBA mask overlay (edit view) |
| GET | `/api/objmask/{model}/{i}/{obj}.png` | One object's black/white mask (brush start) |
| GET | `/api/render/{i}.jpg?model=` | Frame with all effects (preview view) |
| GET | `/api/timeline?model=` | Coverage ranges, keyframes, per-frame revisions |
| POST/PATCH/DELETE | `/api/objects[/{id}]` | Objects (PATCH: name, color, visible, foreground, effect{…}) |
| POST | `/api/prompts` `{obj,frame,points,labels,box}` | Set prompt → runs the model on that frame |
| POST | `/api/prompts/mask` `{obj,frame,png}` | Brush mask prompt |
| DELETE | `/api/prompts/{obj}/{frame}` · POST `/api/undo` | |
| POST | `/api/propagate` `{direction, use_range}` | Job: track. The result includes fps / ms per frame / peak VRAM |
| GET | `/api/runs` | Benchmark log |
| POST | `/api/compare` `{a,b}` | Per-object, per-frame IoU between two models' masks |
| POST | `/api/export` `{format,resolution,use_range,audio,model}` · GET `/api/download/{name}` | |
| GET | `/api/jobs/{id}` · POST `/api/jobs/{id}/cancel` · GET `/api/status` | Jobs, VRAM |

## Tracker interface (`editor/trackers/base.py`)

```python
load(device) / unload()
init_video(frames_dir, frames, height, width) / close_video()
set_prompts({obj: {frame: Prompt}}) -> {frame: {obj: mask}}   # reset + replay
update_prompt(obj, frame, Prompt) -> {obj: mask}               # fast path
can_update_in_place(obj) -> bool
propagate(start, max_frames, reverse, should_stop) -> iter[(frame, {obj: mask})]
```

## Phone pipeline

`scripts/export_edgetam_onnx.py` wraps EdgeTAM into 4 ONNX graphs (memory attention is rewritten so the RoPE
layout is an input instead of Python ints, giving one graph for any memory length).
`editor/trackers/onnx_edgetam.py` is the numpy loop that mirrors `EdgeTamVideoModel` step by step.
Its runs report per-stage ms in the Speed tab. See [PHONE_DEPLOYMENT.md](PHONE_DEPLOYMENT.md).

## Tests

```bash
conda activate mvseg
pytest -q            # effects, project storage, prompts/undo, coverage, export (PNG/MP4/masks), catalog
```
GPU and model paths are checked with the end-to-end script `scripts/smoke_test.py` (loads each downloaded model,
clicks, tracks a synthetic video, and prints fps).
