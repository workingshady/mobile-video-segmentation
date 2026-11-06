# Research Report — "SAM 2 on a phone": click an object, track it through the whole video

**Date:** 2026-10-19
**Goal:** Tap a hand, a brush, or any object on one frame → get a clean mask on **every** frame
(good quality, no flicker, no losing the object) → run it **on the phone**, offline, optimized.

---

## 1. Short answer

- **Yes, it is possible.** But **not with full SAM 2**. Full SAM 2 (even the smallest one)
  runs at about **1 FPS** on an iPhone 15 Pro Max, so it is too slow.
- **Best pick: EdgeTAM** (Meta, CVPR 2025). It is SAM 2 rebuilt for phones:
  same "click → track with memory" behavior, **16 FPS on iPhone 15 Pro Max**,
  free license (Apache 2.0), works in Hugging Face Transformers, and has ready
  CoreML (iPhone) and Qualcomm (Android) builds.
- **Best backup when quality matters more than speed: EfficientTAM-S**
  (almost SAM 2 accuracy, but around 1–4 FPS on a phone → good for *recorded* video, not live camera).

---

## 2. Every option we looked at

### 2.1 Models that CAN select + propagate (have video memory)

| Model | Year | Phone speed (reported) | Accuracy (J&F, higher = better) | License / code | Verdict |
|---|---|---|---|---|---|
| **EdgeTAM** | CVPR 2025 | **16 FPS** iPhone 15 Pro Max. Encoder **16.8 ms** on Galaxy S24 NPU (Qualcomm AI Hub) | DAVIS17 **87.7**, MOSE 70.0, SA-V val 72.3, SA-V test **71.7** | Apache 2.0, official repo + HF Transformers + CoreML + Qualcomm AI Hub | ✅ **MAIN CHOICE** |
| **EfficientTAM** (Ti / S) | ICCV 2025 | Sources disagree: ~10 FPS (Ti) and ~28 FPS (abstract) on iPhone 15 Pro Max; S is ~1010 ms/frame, S/2 ~450 ms/frame on iPhone 15 | SA-V test **74.5** (S), 74.0 (S/2) — SAM 2 is 74.7 | Code on GitHub (yformer/EfficientTAM) | 🟡 **BACKUP for best quality on recorded video** |
| **TinySAM 2** | May 2026 (arXiv) | No phone FPS published yet | ~90% of SAM 2.1 on DAVIS / SA-V, using only 7% of memory tokens | Paper (Huawei Noah's Ark); code status not confirmed | 👀 Watch — very small, but no phone numbers yet |
| **EfficientSAM3** | Nov 2025 → 2026 updates | Not verified | Not verified (paper says "strong trade-offs") | GitHub + HF weights | 👀 Watch — the only small model that can also take a **text** prompt ("hand"). Its memory module comes from EdgeTAM |
| **SAM 2.1 tiny/small** | 2024 | ~1 FPS on iPhone (CoreML) | The quality reference | Apache 2.0 | ❌ Too slow on phone. Use on PC/server only |
| **SAM 3 / SAM 3.1** | Nov 2025 / Mar 2026 | Server only (benchmarks are on H100 GPUs) | Best accuracy + text prompts | Meta | ❌ Not for phones |

### 2.2 Models that CANNOT propagate (single image only)

| Model | Why rejected |
|---|---|
| EdgeSAM, MobileSAM, EfficientSAM | Very fast (EdgeSAM 30+ FPS on iPhone 14) but **image only**: no memory, so you would have to re-click every frame. |
| MediaPipe Interactive Segmenter (Google) | Click → mask, works on video frames, but **no tracking memory** across frames. |
| Apple "Lift subject" / Vision | Picks the *main* subject automatically, still images. You can't choose "the brush". |
| EdgeDAM (2026) | A detector + tracker (25 FPS on iPhone), not "click anything". |

---

## 3. Why EdgeTAM is the main choice

1. **It does exactly the task.** It keeps SAM 2's design: click/box prompts → mask → a
   memory bank → every next frame is matched against memory. So it follows the object
   through motion, rotation and short occlusions, and you can add **correction clicks** on any frame.
2. **It fixed the real phone bottleneck.** Other "small SAM" models only shrink the image
   encoder. EdgeTAM's authors measured that **memory attention** is also a bottleneck on
   phones and replaced it with a **2D Spatial Perceiver** (fixed learnable queries that
   compress each frame's memory while keeping its spatial layout). Result: 6.4× faster than
   their baseline, and even +0.9 J&F better.
3. **Fastest *verified* phone number** of all the video models (16 FPS, iPhone 15 Pro Max).
4. **Easiest to deploy:**
   - Python: `EdgeTamVideoModel` in Hugging Face Transformers (used in `scripts/track_video.py`).
   - iPhone: official CoreML export + a community **full-video** CoreML port (all 4 parts, including memory).
   - Android: listed on **Qualcomm AI Hub** (Snapdragon NPU).
5. **License is safe for products:** Apache 2.0.
6. **Small:** encoder ~33 MB + decoder ~24 MB (float). FP16 or INT8 makes it smaller.

**Trade-off:** it scores a little lower than full SAM 2 (SA-V test 71.7 vs ~74–75).
For hands and brushes this is usually fine. If it isn't, use the "quality mode" below.

---

## 4. Two modes — pick per feature

| | **Live mode** (camera preview, real time) | **Quality mode** (recorded video, tap "process") |
|---|---|---|
| Model | EdgeTAM | EfficientTAM-S (or EdgeTAM at full 1024 + both directions) |
| Speed target | 10–16 FPS | 1–4 FPS is OK (it runs once, in the background) |
| Direction | Forward only (streaming) | Forward **and** backward from the clicked frame |
| Why | Must keep up with the camera | Only accuracy matters, the user waits a few seconds |

---

## 5. Getting good quality AND consistency (no flicker, no losing the object)

What to do, roughly in order of impact:

1. **Better prompts.** Several positive clicks on the object, plus **negative clicks** on
   nearby things that look similar (the other hand, the brush handle vs bristles, the background).
   A **box** prompt is often better than one point for thin things like a brush.
2. **Correction clicks.** If tracking goes wrong at frame N, add a click on frame N and
   propagate again. SAM 2-style models store it as a new "conditioning" memory, which
   fixes the following frames too.
3. **Watch the object score.** The model returns `object_score_logits` per frame. When it
   drops (the object is hidden or lost), show "tap to re-select" instead of a bad mask.
4. **Cheap post-processing** (already in the script): keep the largest blob plus a
   morphological close to fill holes. Optionally average the mask over 2–3 frames (EMA) for smoother edges.
5. **Memory settings.** EdgeTAM keeps 1 conditioning frame + 6 recent frames + up to 16
   object pointers (`num_maskmem=7`). Keep these on the phone: memory stays fixed, so
   speed stays steady even on long videos.
6. **Known weak cases** (from research): very thin structures (bristles, fingers spread
   apart), small/textureless objects, and lookalike distractors (two hands). Fixes:
   - Crop around the object and run at higher resolution (a published trick for SAM 2 on small objects).
   - **DAM4SAM** (CVPR 2025, training-free): a distractor-aware memory that fixes
     "jumps to the other hand". **SAMURAI**: motion-aware memory for fast motion.
     Both were built for SAM 2. Porting them to EdgeTAM is **not verified**. Treat it as an experiment.

---

## 6. Deployment plan

### iPhone (CoreML)
- **Official** `export_to_coreml.py` (EdgeTAM repo) exports **image encoder, prompt encoder,
  and mask decoder**. That is enough for **single-image** use, but **not** for video memory.
- **Community full-video port** `AhmadYarAI/EdgeTAM-CoreML-Video` (Apache 2.0, not from Meta):
  4 packages — `ImageEncoder`, `Initializer`, `MemoryEncoder`, `Propagator`. iOS 18+.
  Use `.cpuAndGPU` (some ops don't compile for the Neural Engine). Its demo runs at **~10 FPS**.
  Parity with PyTorch: mask IoU 0.93–0.98.
- Swift side you must write: resize frame → 1024², run encoder, run initializer on the first
  frame with the clicks, run memory encoder, keep the memory bank (1 conditioning + 6 recent + 16 pointers),
  run the propagator per new frame, threshold logits at 0, resize to screen.

### Android (Snapdragon)
- **Qualcomm AI Hub → EdgeTAM**: compile and profile for your exact phone. Measured encoder
  **16.8 ms** on Galaxy S24 (Snapdragon 8 Gen 3 NPU), peak memory ≤ 351 MB.
  ⚠️ The model page is inconsistent (it says "not supported on any chipset" and also lists S24),
  so profile on your own device first.
- Run through LiteRT (TFLite) or the QNN runtime on the NPU.
- Non-Snapdragon phones (Exynos, MediaTek, Tensor): no ready build found. You'd export
  yourself (ExecuTorch or LiteRT), which is **unverified** and more work.

### Optimizing
- FP16 first (almost no quality loss). INT8 only after you check the masks yourself.
- Encoder input 1024² is the default. Lower it for speed only if quality holds.
- Run the encoder once per frame. Clicks only re-run the small decoder, so taps feel instant.
- Keep the memory bank fixed-size → steady memory and speed.
- Watch thermals: phones slow down after ~1–2 minutes of full load.

---

## 6b. Measured in this repo (2026-10-20): phone pipeline on CPU vs NPU

EdgeTAM was exported to 4 ONNX graphs **with the video memory** (`scripts/export_edgetam_onnx.py`) and run by a
numpy loop on CPU with 4 threads, the way a phone without an accelerator would run it:

| | Size | IoU vs original EdgeTAM | CPU ms/frame |
|---|---|---|---|
| fp32 | 69 MB | 0.998 | 1262 |
| fp16 | 35 MB | 0.998 | 2092 (no fp16 math on x86) |
| int8 | 44 MB | 0.998 | 1123 |

- Shrinking to int8/fp16 costs **no measurable quality**.
- **CPU-only ≈ 1 FPS.** Image encoder (~470 ms) + memory attention (~500 ms) are about 85% of the time.
  The published **16 FPS** comes from the iPhone **Neural Engine/GPU**, so live tracking needs CoreML (iPhone)
  or QNN/NNAPI/GPU (Android). CPU-only is fine for "process a recorded clip".
- For NPUs, the memory attention runs with fixed shapes (padded + masked memory). Verified identical to the exact version.

## 7. What is NOT verified (be honest)

- No Android FPS for the **full** EdgeTAM video pipeline (only the encoder: 16.8 ms).
- EfficientTAM phone FPS numbers conflict between its own sources (10 vs 28 FPS).
- No phone benchmarks yet for TinySAM 2 or EfficientSAM3.
- No public SAM 3 / 3.1 phone build.
- DAM4SAM / SAMURAI on EdgeTAM: untested.

---

## 8. Sources

- EdgeTAM paper — https://arxiv.org/abs/2501.07256 · CVPR 2025 — https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_EdgeTAM_On-Device_Track_Anything_Model_CVPR_2025_paper.html
- EdgeTAM code — https://github.com/facebookresearch/EdgeTAM
- EdgeTAM in Transformers — https://huggingface.co/docs/transformers/model_doc/edgetam_video
- EdgeTAM CoreML video port — https://huggingface.co/AhmadYarAI/EdgeTAM-CoreML-Video
- Qualcomm AI Hub EdgeTAM — https://aihub.qualcomm.com/models/edgetam · S24 profile job — https://aihub.qualcomm.com/jobs/jgzm72ozp
- EfficientTAM — https://github.com/yformer/EfficientTAM · https://openaccess.thecvf.com/content/ICCV2025/papers/Xiong_Efficient_Track_Anything_ICCV_2025_paper.pdf · https://arxiv.org/pdf/2411.18933
- TinySAM 2 — https://arxiv.org/html/2605.18013
- EfficientSAM3 — https://github.com/SimonZeng7108/efficientsam3 · https://arxiv.org/pdf/2511.15833
- SAM 3.1 — https://ai.meta.com/blog/segment-anything-model-3/ · https://www.labellerr.com/blog/sam-3-vs-sam-3-1-performance-comparison/
- SAM 2 CoreML (image only) — https://github.com/alexhaugland/segment-anything-2-coreml
- EdgeSAM — https://mmlab-ntu.com/project/edgesam
- MediaPipe Interactive Segmenter — https://ai.google.dev/mediapipe/solutions/vision/interactive_segmenter
- Apple subject lifting — https://machinelearning.apple.com/research/salient-object-segmentation
- EdgeDAM — https://arxiv.org/pdf/2603.05463
- DAM4SAM — https://arxiv.org/html/2411.17576v1 · SAM2 trackers benchmark — https://arxiv.org/pdf/2512.09633
- SAM 2 weak on small objects (CTU thesis) — https://dspace.cvut.cz/handle/10467/123552
- sam2-lite — https://huggingface.co/Pablopigue/sam2-lite
