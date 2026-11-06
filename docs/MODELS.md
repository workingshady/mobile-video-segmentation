# Models: what is downloaded, where, and how to use each one

Everything is listed in [`models_catalog.json`](../models_catalog.json), the single source of truth.
The downloader, the editor's Models tab, and this page all follow it.

```bash
python scripts/download_models.py --list          # what is on disk
python scripts/download_models.py                 # download everything (default groups, ~6 GB)
python scripts/download_models.py --only sam2.1-large
python scripts/download_models.py --include-huge  # + ~100 GB of EfficientSAM3 image variants
```

The script is safe to re-run. Finished files are skipped, partial downloads resume, and flaky connections are retried.

**How each model works (masks, tracking, IDs):** [docs/models/](models/README.md)

## Tags: which models are for phones?

Every model has a tag. The editor groups the model picker by tag and puts the phone models first.

| Tag | Meaning | Models |
|---|---|---|
| 📲 **Phone pipeline (ONNX)** | The exported phone files + per-frame loop, running on CPU like a phone. **What you would ship.** | EdgeTAM · phone ONNX int8 / fp16 / fp32 |
| 📱 **Phone · real-time** | Small enough to track live on a phone NPU/GPU (≈10+ FPS) | EdgeTAM, EfficientTAM Ti / Ti-1 / Ti-2 / Ti-512 |
| 📱 **Phone · slow** | Runs on a phone at ≈1–4 FPS. Recorded video only | EfficientTAM S family, SAM 2.1 Tiny |
| 📱 **Phone · image only** | Edge-sized but can't track across frames yet | EfficientSAM3 |
| 📦 **Phone file** | Ready-converted files for phone apps | EdgeTAM CoreML Video, ExecuTorch, ONNX (image parts), original `.pt` |
| 🖥 **Desktop only** | Too big for phones. Only used to measure what phone models lose | SAM 2.1 Small / Base+ / Large, SAM 3, SAM 3.1 |

Desktop models are hidden in the editor's Models tab unless you tick **Show desktop models**.

---

## 0. Phone pipeline (made in this repo, not downloaded)

| Id | Files | Size | Same masks as original EdgeTAM | CPU (4 threads) |
|---|---|---|---|---|
| `edgetam-onnx-int8` | `models/edgetam-onnx-video/int8` | 44 MB | IoU 0.998 | ≈1.1 s/frame |
| `edgetam-onnx-fp16` | `models/edgetam-onnx-video/fp16` | 35 MB | IoU 0.998 | for phone GPU/NPU (slow on PC CPU) |
| `edgetam-onnx-fp32` | `models/edgetam-onnx-video/fp32` | 69 MB | IoU 0.998 | ≈1.3 s/frame |

Built by `python scripts/export_edgetam_onnx.py` and checked by `python scripts/validate_onnx.py`.
How to put it on Android/iPhone: **[PHONE_DEPLOYMENT.md](PHONE_DEPLOYMENT.md)**.

---

## 1. Runnable in the editor (click → track)

| Id | Model | Size | Backend | Why it's here |
|---|---|---|---|---|
| `edgetam` | **EdgeTAM** (Meta, CVPR 2025) | 56 MB | Transformers `EdgeTamVideoModel` | **The phone model.** 16 FPS on iPhone 15 Pro Max |
| `sam2.1-tiny` | SAM 2.1 Tiny | 156 MB | Transformers `Sam2VideoModel` | Smallest official SAM 2.1 (~1 FPS on iPhone) |
| `sam2.1-small` | SAM 2.1 Small | 184 MB | Transformers `Sam2VideoModel` | |
| `sam2.1-base-plus` | SAM 2.1 Base+ | 324 MB | Transformers `Sam2VideoModel` | |
| `sam2.1-large` | SAM 2.1 Large | 898 MB | Transformers `Sam2VideoModel` | **Quality reference** for the Compare tab |
| `sam3` | SAM 3 (tracker part) | 3.4 GB | Transformers `Sam3TrackerVideoModel` | Newest Meta model. **Gated** (see below) |
| `efficienttam-ti` | EfficientTAM Ti | 72 MB | `third_party/EfficientTAM` | ~10 FPS reported on iPhone 15 Pro Max |
| `efficienttam-ti-1` / `-ti-2` | EfficientTAM Ti with efficient memory | 72 MB each | same | Faster memory variants |
| `efficienttam-ti-512` | EfficientTAM Ti at 512×512 | 72 MB | same | Much faster, less detail |
| `efficienttam-s` | EfficientTAM S | 136 MB | same | Best phone-class quality (SA-V 74.5 J&F), slower |
| `efficienttam-s-1` / `-s-2` | EfficientTAM S with efficient memory | 136 MB each | same | S/2: ~450 ms/frame on iPhone 15 |
| `efficienttam-s-512` | EfficientTAM S at 512×512 | 136 MB | same | |

All of them get exactly the same prompts (points, box, brush mask) through one interface
(`editor/trackers/base.py`), so the comparison is fair.

### Gated: SAM 3 and SAM 3.1
Meta requires an access request:
1. Open https://huggingface.co/facebook/sam3 (and https://huggingface.co/facebook/sam3.1) while logged in, and accept the terms.
2. Wait for approval (manual, usually hours to days).
3. You are already logged in on this PC (`huggingface-cli whoami`). Re-run `python scripts/download_models.py --only sam3,sam3.1`.
4. Click **↻ Re-scan weights** in the editor's Models tab.

SAM 3 in bf16 needs about 2 GB of VRAM for weights plus the video state. On the 8 GB RTX 3070, unload other models first.

---

## 2. Downloaded for phone / later testing (not runnable in the editor)

| Id | What | Folder | Use it for |
|---|---|---|---|
| `edgetam-coreml-video` | EdgeTAM **full video** CoreML: ImageEncoder, Initializer, MemoryEncoder, Propagator | `models/edgetam-coreml-video` | iPhone app (iOS 18+, `.cpuAndGPU`). Community port, Apache 2.0 |
| `edgetam-executorch` | EdgeTAM encoder + decoder `.pte` (XNNPACK fp32/fp16, CoreML delegate) | `models/edgetam-executorch` | Android (XNNPACK) / iOS via ExecuTorch. Image parts only |
| `edgetam-onnx` | EdgeTAM encoder + decoder ONNX in fp32/fp16/int8/uint8/q4/bnb4 | `models/edgetam-onnx` | ONNX Runtime Mobile / Web, quantization tests. Image parts only |
| `edgetam-original` | Meta's original `edgetam.pt` | `models/edgetam-original` | `facebookresearch/EdgeTAM` repo, `export_to_coreml.py` |
| `sam2.1-originals` | Original SAM 2.1 `.pt` checkpoints (all 4 sizes) | `models/sam2.1-originals` | Meta's `sam2` repo and export tools |
| `sam3.1` | SAM 3.1 multiplex checkpoint (**gated**) | `models/sam3.1` | Meta's `sam3` repo on a big GPU |
| `efficientsam3` | EfficientSAM3 EV-M / RV-M / TV-M | `models/efficientsam3` | Distilled SAM 3 with text prompts. **Upstream only released image models.** The video memory stage is still on their to-do list, so it can't track yet |

"Image parts only" means encoder + decoder: enough to segment one frame from a click,
but the memory modules needed to *track* are not in those files. Only the CoreML Video port has them.

## 3. Not available yet

| Model | Status (2026-10-20) |
|---|---|
| TinySAM 2 (arXiv 2605.18013) | Paper only. No public code or weights |
| Qualcomm EdgeTAM (Snapdragon NPU) | Not a download: compiled per phone on Qualcomm AI Hub. `pip install qai-hub-models` then `python -m qai_hub_models.models.edgetam.export` (free AI Hub account) |

---

## 4. Adding another model

1. Add an entry to `models_catalog.json` (id, name, group `editor`, backend, repo, dir).
2. If its API matches SAM 2 in Transformers, map the class names in `editor/trackers/hf.py` (`CLASSES`).
   Otherwise write a new `Tracker` subclass (see `editor/trackers/efficienttam.py`, about 100 lines) and register it in
   `editor/trackers/registry.py` (`RUNNABLE_BACKENDS` + `create()`).
3. `python scripts/download_models.py --only <id>`, then click ↻ Re-scan weights in the editor.
