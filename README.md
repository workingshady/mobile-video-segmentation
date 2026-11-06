# Mobile Video Segmentation: click an object, track it through the whole video

Select a **hand, a brush, or anything** with a click, box, or brush stroke → track it across **every frame** →
apply an effect → export. Every phone-class "SAM 2" model is downloaded so you can **test and compare them**
before putting one on a phone.

This repo has four parts:

| Part | What | Where |
|---|---|---|
| 🔬 **Research** | Which model is best for phones, and why (all options, numbers, sources) | [REPORT.md](REPORT.md) |
| 📲 **Phone pipeline** | EdgeTAM exported to ONNX **with tracking memory** (fp32 / fp16 / int8). Same masks as the original (IoU 0.998). Android + iPhone guide | `models/edgetam-onnx-video` · [docs/PHONE_DEPLOYMENT.md](docs/PHONE_DEPLOYMENT.md) |
| 🎬 **Track Studio** | A local video editor: import, timeline, tagged model picker, click/box/brush selection, tracking, effects, model comparison, speed, export | `editor/` · [docs/EDITOR_GUIDE.md](docs/EDITOR_GUIDE.md) |
| 📦 **Model weights** | Phone models first: EdgeTAM, EfficientTAM. Desktop references: SAM 2.1, SAM 3 | `models/` · [docs/MODELS.md](docs/MODELS.md) |

**Recommendation:** **EdgeTAM** for the phone (16 FPS on iPhone 15 Pro Max using the Neural Engine, Apache 2.0).
Ship the **int8** ONNX for CPU or **fp16** for the NPU/GPU. **EfficientTAM Ti/2** is the backup.
Big models (SAM 2.1 Large etc.) are **desktop-only**, used only to measure what the phone model loses.

### Which models are for phones? (tags used everywhere in the editor)

| Tag | Meaning | Models |
|---|---|---|
| 📲 Phone pipeline (ONNX) | The exact phone files + loop, run on CPU like a phone | EdgeTAM · phone ONNX int8 / fp16 / fp32 |
| 📱 Phone · real-time | Live tracking on a phone NPU/GPU (≈10+ FPS) | EdgeTAM, EfficientTAM Ti, Ti/1, Ti/2, Ti-512 |
| 📱 Phone · slow | ≈1–4 FPS on a phone: recorded video only | EfficientTAM S family, SAM 2.1 Tiny |
| 📦 Phone file | Converted files for phone apps | EdgeTAM CoreML Video / ExecuTorch / ONNX |
| 🖥 Desktop only | Too big for phones: quality reference only | SAM 2.1 Small/Base+/Large, SAM 3, SAM 3.1 |

**Measured (this PC):** the phone pipeline gives the same masks as the original (IoU 0.998 in fp32, fp16, and int8).
On CPU alone it takes **≈1.1 s per frame**: the image encoder and memory attention are about 85% of that.
So **live tracking needs the phone's NPU/GPU** (CoreML on iPhone, QNN/NNAPI on Android). CPU-only is fine for recorded video.
Details: [docs/PHONE_DEPLOYMENT.md](docs/PHONE_DEPLOYMENT.md).

---

## Quick start (Windows)

```bat
setup.bat          :: once: conda env "mvseg" + PyTorch CUDA + libraries + EfficientTAM + all weights (~6 GB)
run_editor.bat     :: opens http://127.0.0.1:7860
```

Then in the editor:
1. **Import** a video (or drag it in).
2. **Model ▾ → EdgeTAM → Load**.
3. Click your hand with **＋** (and **－** on things that are not your hand).
4. **◀ Track ▶**.
5. Pick an effect in the right panel, then press **P** to preview.
6. **Export**.

**Light setup (what this PC uses):** reuse an existing PyTorch 2.5 env instead of downloading PyTorch again (~45 MB of downloads):

```bat
C:\Users\worki\anaconda3\envs\torch3070\python.exe -m venv --system-site-packages .venv
.venv\Scripts\python -m pip install "transformers>=4.57,<5" "hydra-core>=1.3.2" imageio-ffmpeg timm onnxruntime onnx
:: link EfficientTAM code into the venv (no install needed)
echo %CD%\third_party\EfficientTAM> .venv\Lib\site-packages\efficienttam_local.pth
run_editor.bat
```
(Transformers 5.x needs PyTorch ≥ 2.6, so 4.57 is pinned for PyTorch 2.5. EdgeTAM also needs `timm`.)

Manual setup (any OS):

```bash
conda create -n mvseg python=3.11 -y && conda activate mvseg
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
git clone --depth 1 https://github.com/yformer/EfficientTAM.git third_party/EfficientTAM
cd third_party/EfficientTAM && Efficient_Track_Anything_BUILD_CUDA=0 pip install --no-build-isolation --no-deps -e . && cd ../..
python scripts/download_models.py
python -m editor
```

---

## Track Studio features

| | |
|---|---|
| **Import** | Video or image. Choose resolution, start/end, every-Nth-frame, and max frames |
| **Timeline** | Ruler with time and frame numbers, thumbnails, one track per object (mask coverage bars + ◆ prompt keyframes), in/out range, scrubbing, zoom, playback |
| **Select** | ＋/－ points, box, brush, eraser, on any frame and for any number of objects. Undo. Prompts can sit on many frames (correction clicks) |
| **Models** | Every model in one dropdown with status (ready / missing / gated). Load one and the same prompts are reused |
| **Track** | Forward, backward, or both; the whole video or the in/out range. Live progress and stop |
| **Effects** | Per object: highlight, outline, blur, pixelate, recolor, erase (inpaint), grow/shrink, feather. Background: remove, color, blur, grayscale, image. Temporal smoothing |
| **Compare** | IoU of model A vs B per object and per frame, worst frames (click to jump), and an IoU strip on the timeline |
| **Benchmarks** | Every run logged: fps, ms/frame, peak VRAM |
| **Export** | MP4 (+audio), transparent WebM, PNG ± alpha, per-object black/white mattes. Working or original resolution |

**✨ Magic brush:** paint over part of an object and the model detects the whole object, live. Paint more to keep filling it, Alt-paint to remove.

Full guide with every button and shortcut: **[docs/EDITOR_GUIDE.md](docs/EDITOR_GUIDE.md)**.
How each model works (masks, tracking across frames, IDs): **[docs/models/](docs/models/README.md)**.

---

## Model status

```bash
python scripts/download_models.py --list     # what is on disk
python scripts/smoke_test.py                 # load each model, click, track a synthetic video, report IoU + fps
```

| Model | In editor | Notes |
|---|---|---|
| EdgeTAM | ✅ | Phone model |
| SAM 2.1 Tiny / Small / Base+ / Large | ✅ | Large = reference |
| EfficientTAM Ti, Ti/1, Ti/2, Ti-512, S, S/1, S/2, S-512 | ✅ | Via `third_party/EfficientTAM` |
| SAM 3 | ✅ after access | **Gated**: request access on Hugging Face first |
| EdgeTAM CoreML Video / ExecuTorch / ONNX, SAM 2.1 `.pt`, SAM 3.1, EfficientSAM3 | 📦 | Downloaded for phone/other tests ([details](docs/MODELS.md)) |
| TinySAM 2 | ❌ | No public weights yet |

---

## Repo layout

```
mobile-video-segmentation/
├── README.md                 ← this file
├── REPORT.md                 ← research: every option, numbers, choices + why, sources
├── setup.bat / run_editor.bat
├── requirements.txt
├── models_catalog.json       ← every model: repo, files, backend, notes (single source of truth)
├── docs/
│   ├── EDITOR_GUIDE.md       ← how to use Track Studio
│   ├── MODELS.md             ← every model + its tag (phone real-time / slow / desktop)
│   ├── PHONE_DEPLOYMENT.md   ← the ONNX phone pipeline: graphs, loop, Android + iPhone
│   ├── models/               ← one README per model: how it masks, tracks, and who gives IDs
│   └── ARCHITECTURE.md       ← code structure, API, design decisions
├── editor/                   ← Track Studio (FastAPI backend + static/ web UI)
│   ├── server.py  project.py  effects.py  exporter.py  jobs.py
│   ├── trackers/  (hf.py = SAM2.1/EdgeTAM/SAM3, efficienttam.py, onnx_edgetam.py = phone loop, registry.py)
│   └── static/    (index.html, css/, js/)
├── scripts/
│   ├── download_models.py    ← download all weights (resumes, retries)
│   ├── smoke_test.py         ← end-to-end test of every model
│   ├── export_edgetam_onnx.py ← export EdgeTAM video pipeline to ONNX (fp32/fp16/int8) + checks
│   ├── validate_onnx.py      ← phone pipeline vs original EdgeTAM: IoU + CPU speed
│   └── track_video.py        ← minimal CLI: click + track with EdgeTAM
├── tests/                    ← pytest (no GPU needed)
├── models/                   ← weights (git-ignored)
├── third_party/EfficientTAM  ← EfficientTAM code (git-ignored)
└── projects/                 ← your editor projects (git-ignored)
```

## Deploying to a phone (summary)
- **iPhone:** `models/edgetam-coreml-video` (4 CoreML packages with video memory), iOS 18+, `.cpuAndGPU`.
- **Android:** `models/edgetam-executorch` (XNNPACK), or Qualcomm AI Hub for the Snapdragon NPU.
- Step-by-step details, the optimization checklist, and the per-frame loop are in [REPORT.md §6](REPORT.md) and [docs/MODELS.md](docs/MODELS.md).

## License notes
Code in this repo: yours. Models keep their own licenses. SAM 2.1, EdgeTAM, and EfficientTAM are Apache 2.0.
SAM 3/3.1 use Meta's SAM license (gated). The CoreML/ONNX/ExecuTorch ports are community conversions.
