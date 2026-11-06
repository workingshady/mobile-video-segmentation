# Track Studio: User Guide

Track Studio is a local video editor for **testing click-to-select and track segmentation models**
(SAM 2.1, EdgeTAM, EfficientTAM, SAM 3).
You import a video or image, select an object (hand, brush, anything) with clicks, a box, or the brush,
**track it through the whole video**, apply an effect, and export.
Every model uses the same prompts, so you can compare quality and speed side by side.

Start it with `run_editor.bat`. It opens `http://127.0.0.1:7860` in your browser.

---

## 1. Screen layout

```
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ ◆ Track Studio  [Import] [Projects▾]  name    Model [EdgeTAM ▾] [Load] ●chip  [Export]│
├────┬──────────────┬───────────────────────────────────────────┬───────────────────────┤
│ ✋ │ OBJECTS  +New│ [Edit|Preview]  Masks from [model ▾]      │ Effects│Models│Compare│
│ ＋ │ ■ Hand    👁★✕│                                           │ │Benchmarks           │
│ － │ ■ Brush   👁★✕│                                           │                       │
│ ▭  │              │            VIDEO + MASKS + CLICKS         │  per-object effect    │
│ 🖌 │              │                                           │  background mode      │
│ ⌫  │              │                                           │  temporal smoothing   │
│ ▮  │ Clear prompt │                                           │                       │
├────┴──────────────┴───────────────────────────────────────────┴───────────────────────┤
│ ⏮ ◀ ▶ ▶| ⏭  00:03.20  [96] /299 · 30fps │ [In  Out] □in/out │ ◀Track  ◀Track▶  Track▶ │
│ ruler   |0:00.0      |0:01.0      |0:02.0 ...                                         │
│ Video   [thumb][thumb][thumb][thumb][thumb]...                                        │
│ Hand    ◆████████████████████◆█████████                                               │
│ Brush        ◆██████████████                                                          │
│ IoU A/B ▇▇▇▇▇▇▅▅▃▃▇▇▇  (after Compare)                                                │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

| Area | What it does |
|---|---|
| **Top bar** | Import, open/delete projects, choose and **Load** a model, see GPU memory, Export |
| **Toolbar** (left) | The tools for selecting objects (see §3) |
| **Objects** | One row per thing you track. Click to select. 👁 show/hide, ★ foreground (kept when the background is replaced), ✕ delete. Double-click the name to rename, and click the color square to recolor |
| **Viewer** | **Edit** shows masks and your clicks. **Preview** shows the final look with effects. Use the mouse wheel to zoom, and the middle button (or Space+drag, or ✋) to pan |
| **Masks from** | Which model's results are shown. Each model keeps its own masks, so you can flip between them |
| **Timeline** | Ruler, thumbnails, one track per object. Colored bars = frames that have a mask. ◆ = frames where you gave a prompt (click to jump). The dark areas are outside the in/out range |
| **Right tabs** | **Effects** · **Models** (all models + status + Load) · **Compare** (IoU between two models) · **Benchmarks** (fps / ms per frame / VRAM of every run) |

---

## 2. The basic workflow (60 seconds)

1. **Import**: press *Import* or drag a video or image onto the viewer.
   - *Max side*: frames are resized so the longest side is at most this (1280 default).
   - *Every Nth frame* / *Start / End / Max frames*: keep RAM in check. Each frame costs ~6 MB of RAM inside the model.
2. **Load a model**: pick one in the *Model* dropdown and click **Load**. **EdgeTAM** is the phone model, and **SAM 2.1 Large** is the quality reference.
3. **Select the object**: with the **＋** tool, click on your hand or brush. The mask appears right away.
   - Mask too big? Use the **－** tool and click the part that is *not* the object.
   - Thin object (a brush)? Drag a **▭ box** around it, then add ＋ / － clicks.
   - Easiest: the **✨ magic brush** (`M`). Scribble over the object and it finds the whole thing. Scribble over a missed part to add it. Alt-scribble to remove.
   - Want pixel control? Use **🖌 brush** / **⌫ eraser** to paint the mask directly. The painted mask becomes the prompt (no detection).
4. **Track**: press **◀ Track ▶** (or `T`). The mask is propagated through the whole range. You can watch the coverage bar fill in on the timeline.
5. **Fix mistakes**: scrub to the frame where it went wrong, add a ＋/－ click there, and press **Track** again.
   The new click becomes another "memory" for the model and fixes the frames after it.
6. **Effect**: in the *Effects* tab choose e.g. *Highlight*, *Blur*, or *Recolor*, or set *Background → Remove / Blur / Image*.
   Press `P` to see the **Preview**.
7. **Export**: MP4, transparent WebM, PNG sequence with alpha, or black/white mask mattes.

---

## 3. Tools

| Tool | Key | Use |
|---|---|---|
| ✋ Pan | `V` | Drag to move the view (also: middle mouse, or hold Space) |
| ＋ Point | `A` | Click on the object. Click an existing point again to remove it |
| － Point | `S` | Click on what is **not** the object (other hand, background, handle) |
| ▭ Box | `B` | Drag a rectangle around the object. Works together with points |
| ✨ **Magic brush** | `M` | **Paint roughly over part of the object: the model detects the whole object, live while you paint.** Paint over a part it missed to keep filling. **Alt + drag** (or right-drag) removes a wrong area. One stroke = one undo step |
| 🖌 Brush | `R` | Paint the object's mask on this frame (starts from the current mask) |
| ⌫ Eraser | `E` | Erase part of the mask on this frame |
| Brush size | `[` `]` | Brush diameter in video pixels |

**How prompts work:** each object can have prompts on any number of frames (◆ on the timeline).
A frame's prompt is *either* points + box *or* a brush mask:

- Painting on a frame replaces its points.
- Clicking replaces its brush mask.
- **Clear prompt on this frame** (or `Delete`) removes it.
- `Ctrl+Z` undoes the last prompt change.

---

## 4. Tracking (propagation)

| Button | What happens |
|---|---|
| **◀ Track ▶** (`T`) | Forward from the first prompted frame to the end of the range, then backward to the start |
| **Track ▶** | Forward only, from the first prompted frame |
| **◀ Track** | Backward only, from the last prompted frame |
| **■ Stop** (`Esc`) | Stops. Frames done so far are kept |

- **in/out only** checked: only frames between **In** (`I`) and **Out** (`O`) are tracked. Use this to test a hard section quickly.
- All objects are tracked together in one pass.
- The result is stored under the loaded model's name, so tracking with another model never overwrites it.
- Every run is timed and logged in **Benchmarks** (frames, fps, ms/frame, peak VRAM).

---

## 5. Comparing models (the reason this editor exists)

1. Make your prompts once (they are shared by all models).
2. Load **SAM 2.1 Large** → Track. This is your reference.
3. Load **EdgeTAM** → Track. Then **EfficientTAM S** → Track, and so on.
4. **Compare** tab: A = the phone model, B = SAM 2.1 Large → *Compare masks*.
   - **mean / min IoU** per object (1.0 = identical to the reference).
   - A sparkline per object, plus the **10 worst frames** (click to jump there).
   - *show on timeline* adds a colored IoU strip under the tracks: green = agrees, red = disagrees.
5. **Masks from** (above the viewer) switches which model's masks you see, so you can check the bad frames visually.
6. **Benchmarks** tab: the speed of each model on this PC.

> The desktop fps you measure is not phone fps. Use it to rank models against each other.
> For phone numbers, see `REPORT.md` (EdgeTAM ≈ 16 FPS on iPhone 15 Pro Max).

---

## 6. Effects

Per **object** (Effects tab, selected object):

| Effect | Result |
|---|---|
| None | Nothing, but the object still counts as foreground (★) for background effects |
| Highlight | Colored overlay + white outline (opacity, outline px) |
| Outline | Colored outline only |
| Blur | Blurs the object (privacy). *Strength* = blur radius |
| Pixelate | Mosaic. *Strength* = block size |
| Recolor | Changes the hue to the chosen color and keeps the shading (paint the brush red!) |
| Erase | Removes the object and fills the hole from its surroundings (inpainting) |

Mask shaping (per object): **Grow/shrink** (px, negative shrinks) and **Feather** (soft edge, px).

**Background** (everything outside ★ foreground objects): Keep · Remove (transparent) · Solid color (green screen) ·
Blur (portrait mode) · Grayscale (color pop) · Image (replace the background with a picture).

**Temporal smoothing** averages each mask with the previous and next frame. This reduces edge flicker.

The **Preview** view and **Export** use the same renderer, so what you see is what you get.

---

## 7. Export

| Format | Output |
|---|---|
| MP4 | H.264 with all effects + the original audio |
| WebM transparent | VP9 with alpha: a cutout of the foreground objects (for other editors and the web) |
| PNG sequence + alpha | One RGBA PNG per frame |
| PNG sequence | One PNG per frame with effects |
| Mask mattes | One black/white PNG per frame **per object** (for compositing or training data) |

- **Resolution:** *Working* (as edited) or *Original source*. Masks are scaled up, and Feather hides the upscaling.
- **Masks from:** pick which model's masks to render.
- Files go to `projects/<project>/exports/`. The toast at the end has a download link.

---

## 8. Keyboard shortcuts

| Key | Action | Key | Action |
|---|---|---|---|
| `Space` | Play / pause | `T` | Track (both directions) |
| `←` / `→` | Previous / next frame | `Shift+←/→` | 10 frames |
| `Home` / `End` | First / last frame | `I` / `O` | Set in / out |
| `A` `S` `B` `M` `R` `E` `V` | Tools | `[` `]` | Brush size |
| `N` | New object | `1`–`9` | Select object 1–9 |
| `P` | Edit ↔ Preview | `F` | Fit video to screen |
| `Delete` | Clear prompt on this frame | `Ctrl+Z` | Undo prompt |
| `Ctrl+O` | Import | `Ctrl+E` | Export |
| `Esc` | Stop tracking / playback | Wheel | Zoom viewer. `Ctrl+wheel` on the timeline zooms it |

---

## 9. Quality tips (hands & brushes)

- **Hand:** ＋ on the palm and ＋ on one finger. － on the *other* hand, so the mask doesn't jump hands.
- **Brush:** a box around the bristles, plus a － on the handle (or the reverse, if you want the handle).
- **Fast motion / blur frames:** add an extra ＋ on a frame in the middle of the motion.
- **Object leaves and comes back:** add a prompt on the frame where it comes back.
- **Thin edges look jagged:** Feather 2–4 px. **Flicker:** enable temporal smoothing.
- **Fingers get cut off:** Grow +2–4 px.

---

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| Model shows `missing` | Run `python scripts\download_models.py --only <id>` (see the Models tab for the exact id) |
| Model shows `gated` (SAM 3) | Request access at the model's Hugging Face page, wait for approval, then re-run the download script |
| Model shows `no-backend` | The Python package is missing: re-run `setup.bat` |
| "Model is busy" | Tracking or export is running. Wait, or press ■ Stop |
| Out of GPU memory | Use a smaller model, or re-import with a lower *Max side* / fewer frames. Click *Unload model* before loading SAM 3 |
| Out of RAM on import | Fewer frames: *Every Nth frame* = 2, or a shorter Start/End range |
| Export: original resolution fails | The original file moved. It is kept as `projects/<p>/source.*`, so check that it is still there |
