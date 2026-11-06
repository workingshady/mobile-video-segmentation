# Putting EdgeTAM on the phone (Android + iPhone)

This repo contains a **verified phone version of EdgeTAM with full video tracking**:
4 ONNX graphs plus a small per-frame loop, exported from Meta's model and checked against it.

```
models/edgetam-onnx-video/
  fp32/   68.7 MB   reference (exact)
  fp16/   34.8 MB   phone GPU / NPU   (iPhone Neural Engine-style, Android GPU/NNAPI/QNN)
  int8/   44.3 MB   phone CPU          (Android XNNPACK / ONNX Runtime CPU)
  constants.npz     small tensors used by the loop
  meta.json         settings + export check results
```

Re-create it any time (no download needed, uses `models/edgetam-hf`):

```bash
python scripts/export_edgetam_onnx.py     # export + check every graph against PyTorch + make fp16/int8
python scripts/validate_onnx.py           # track a test video: phone pipeline vs original EdgeTAM
```

---

## 1. Measured results (this repo, 2026-10-20)

Test: synthetic 640×360 video, 48 frames, clicks on frames 0 and 24, tracked through the whole video.
CPU = laptop CPU with **4 threads** (phone-like). In the editor, the demo video gave IoU 0.996 against the original.

| Version | Size | Same masks as original EdgeTAM? (IoU) | Accuracy vs truth | CPU ms/frame | Per-stage ms (encoder / decoder / mem-enc / mem-attn) |
|---|---|---|---|---|---|
| Original (PyTorch) | 56 MB | n/a | 0.9925 | n/a | n/a |
| ONNX fp32 | 68.7 MB | **0.998** | 0.992 | 1262 | 481 / 57 / 112 / 598 |
| ONNX fp16 | 34.8 MB | **0.998** | 0.992 | 2092* | 665 / 132 / 496 / 835 |
| ONNX **int8** | 44.3 MB | **0.998** | 0.992 | **1123** | 464 / 44 / 82 / 519 |

\* x86 CPUs have no fp16 math, so fp16 is slower on a PC. On phone GPUs and NPUs, fp16 is the fast format.

**What this means:**
- **Quality:** the phone files produce the same masks as the original model (IoU 0.998), so shrinking costs nothing measurable.
- **Speed: CPU alone is not real-time.** About 1 FPS, with the **image encoder and memory attention taking ~85% of the time**.
  Meta's **16 FPS** on iPhone 15 Pro Max uses the **Neural Engine / GPU**. For live tracking you must run those 2 graphs on the NPU/GPU:
  - **iPhone:** CoreML (Neural Engine/GPU).
  - **Android:** Qualcomm QNN (Snapdragon NPU), NNAPI, or GPU.
  - CPU-only (≈1 FPS) is still fine for "process a recorded video".

---

## 2. The 4 graphs

| Graph | Inputs | Outputs | When |
|---|---|---|---|
| `image_encoder` | `pixel_values` (1,3,1024,1024) float, normalized RGB | `feat_s0` (1,32,256,256), `feat_s1` (1,64,128,128), `vision_feat` (1,256,64,64), `vision_pos` (1,256,64,64) | once per frame |
| `decoder` | `pix_feat` (1,256,64,64), `feat_s0`, `feat_s1`, `point_coords` (1,1,N,2) in 1024-px space, `point_labels` (1,1,N) int64, `mask_input` (1,1,256,256), `has_mask` (1) | `masks` (1,1,4,256,256) logits, `iou` (1,1,4), `pointers` (1,1,4,256), `obj_score` (1,1,1) | per object per frame |
| `memory_encoder` | `vision_feat`, `mask_for_mem` (1,1,1024,1024) | `mem_feat` (1,512,64), `mem_pos` (1,512,64) | per object per frame |
| `memory_attention` | `curr` (1,4096,256), `curr_pos` (1,4096,256), `memory` (1,K,64), `memory_pos` (1,K,64), `k_cos`/`k_sin` (K,256), `k_bias` (1,K) | `conditioned` (1,4096,256) | per object, tracked frames |

Point labels: `1` object, `0` not object, `-1` "no point" (send `[[0,0]]`/`[-1]` when tracking), `2`/`3` box corners (top-left, bottom-right, placed first).

**Fixed shapes for NPUs.** `memory_attention` accepts any memory length K. CoreML and QNN prefer fixed shapes, so always send
**K = 7×512 + 64** (7 memory frames + 16 pointers × 4), fill empty slots with zeros, and set their `k_bias` to `-1e4`.
This is verified identical to the exact version (max difference 0.0).

---

## 3. The per-frame loop (port this)

The reference implementation is `editor/trackers/onnx_edgetam.py` (numpy + OpenCV + ONNX Runtime only, ~300 lines).

```
on user click (object o, frame f):
    F = image_encoder(resize(frame,1024) normalized)                 # cache per frame
    if f was already tracked:  pix = memory_attention(F, bank(o, f))   # correction click
    else:                      pix = F.vision_feat + no_memory_embedding
    masks, iou, ptrs, score = decoder(pix, F, clicks scaled to 1024)
    pick mask: 0–1 click → best IoU of masks[1..3]; 2+ clicks → masks[0] unless unstable (stability < 0.98)
    if score <= 0: mask = -1024 (object absent);  pointer = no_object_pointer
    mem = memory_encoder(F.vision_feat, (upsample(mask,1024) > 0) * 20 - 10)
    bank[o].prompted[f] = {mem, pointer}

for each next frame f (forward or backward):
    F = image_encoder(frame)
    for each object o:
        pix = memory_attention(F, bank(o, f))   # prompted frames + 6 most recent tracked + ≤16 pointers
        masks, iou, ptrs, score = decoder(pix, F, no point)
        mask = best IoU of masks[1..3]; absent if score <= 0
        mem = memory_encoder(F.vision_feat, sigmoid(upsample(mask)) * 20 - 10)
        bank[o].tracked[f] = {mem, pointer};  drop tracked frames older than 16
        show resize(mask, screen) > 0
```

Memory per object stays small and fixed: 7 × 512 × 64 floats ≈ 0.9 MB + pointers.

---

## 4. Android

1. Add **ONNX Runtime Android** (`com.microsoft.onnxruntime:onnxruntime-android`).
2. Ship the **int8** folder for CPU, or **fp16** when using an accelerator.
3. Execution providers:
   - Snapdragon: **QNN EP** (NPU) for `image_encoder` + `memory_attention` with fixed shapes (§2) → real-time.
     Alternative: Qualcomm AI Hub's compiled EdgeTAM (`pip install qai-hub-models`).
   - Others: **NNAPI EP** or GPU. CPU (XNNPACK) works everywhere at ≈1 FPS.
4. Port the loop from `onnx_edgetam.py` to Kotlin. Camera frames → RGB → 1024×1024 → normalize (ImageNet mean/std).

## 5. iPhone

- **Fastest route:** the community CoreML video port (4 CoreML packages with memory, iOS 18+, `.cpuAndGPU`, ~10 FPS demo).
  It is listed as `edgetam-coreml-video` in the catalog and **not downloaded yet**: `python scripts/download_models.py --only edgetam-coreml-video`.
- **Same files as Android:** ONNX Runtime iOS with the **CoreML EP**, using the fp16 graphs and fixed shapes (§2), and the same loop ported to Swift.
- Converting these 4 graphs to native CoreML (`coremltools`, from the PyTorch wrappers in `scripts/export_edgetam_onnx.py`) needs a Mac to test.

## 6. Checklist before shipping
- [ ] `python scripts/validate_onnx.py` passes (IoU vs PyTorch ≥ 0.98).
- [ ] Test your own hand/brush clips in Track Studio with **EdgeTAM · phone ONNX int8**, and Compare against **EdgeTAM** (PyTorch).
- [ ] On the device: time `image_encoder` and `memory_attention` on the NPU/GPU. Both must be well under 30 ms each for ~15 FPS.
- [ ] Watch thermals over 2+ minutes of continuous tracking.
