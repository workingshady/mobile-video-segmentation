"""Parity + speed test: phone ONNX pipeline (fp32 / fp16 / int8) vs the original PyTorch EdgeTAM.

Same video, same clicks, same tracking. Reports per variant:
  - IoU vs PyTorch EdgeTAM on every frame (1.0 = identical masks)
  - IoU vs ground truth (synthetic video)
  - ms per frame on CPU with N threads (a rough phone proxy) and per-stage ms

    python scripts/validate_onnx.py [--threads 4] [--frames 48] [--variants fp32,fp16,int8]
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from editor.trackers import ModelRegistry, Prompt  # noqa: E402
from editor.trackers.onnx_edgetam import OnnxEdgeTAMTracker  # noqa: E402
from scripts.smoke_test import iou, synth_video  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def track(t, prompts, n):
    """Click on frame 0 and on the middle frame, then track forward and backward like the editor does."""
    out = {}
    for f, ms in t.set_prompts(prompts).items():
        out.update({(f, o): m for o, m in ms.items()})
    t0 = time.perf_counter()
    count = 0
    for f, ms in t.propagate(0, n - 1, False, lambda: False):
        out.update({(f, o): m for o, m in ms.items()})
        count += 1
    return out, (time.perf_counter() - t0) * 1000 / max(1, count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--variants", default="fp32,fp16,int8")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="onnxval_"))
    frames, gts, click = synth_video(tmp, args.frames)
    pil = [Image.fromarray(f) for f in frames]
    h, w = frames[0].shape[:2]
    mid = args.frames // 2
    ys, xs = np.nonzero(gts[mid])
    prompts = {1: {0: Prompt(points=[list(c) for c in click], labels=[1, 1]),
                   mid: Prompt(points=[[int(xs.mean()), int(ys.mean())]], labels=[1])}}

    print(f"video {w}x{h}, {args.frames} frames, clicks on frames 0 and {mid}, CPU threads={args.threads}\n")
    ref_t = ModelRegistry().create("edgetam")
    ref_t.load("cuda" if torch.cuda.is_available() else "cpu")
    ref_t.init_video(str(tmp), pil, h, w)
    ref, _ = track(ref_t, prompts, args.frames)
    ref_t.unload()
    ref_gt = np.mean([iou(ref[(f, 1)], gts[f]) for f in range(args.frames)])
    print(f"PyTorch EdgeTAM (reference): IoU vs ground truth {ref_gt:.4f}\n")

    print(f"{'variant':<8} {'size MB':>8} {'IoU vs PyTorch':>15} {'min':>6} {'IoU vs GT':>10} {'ms/frame':>9}   per-stage ms")
    ok = True
    for v in args.variants.split(","):
        size = sum(f.stat().st_size for f in (ROOT / "models/edgetam-onnx-video" / v).glob("*.onnx")) / 1e6
        t = OnnxEdgeTAMTracker({"id": f"onnx-{v}", "name": v, "variant": v, "threads": args.threads},
                               ROOT / "models/edgetam-onnx-video")
        t.load("cpu")
        t.init_video(str(tmp), pil, h, w)
        t.reset_timing()
        got, ms = track(t, prompts, args.frames)
        vs_ref = [iou(got[(f, 1)], ref[(f, 1)]) for f in range(args.frames)]
        vs_gt = [iou(got[(f, 1)], gts[f]) for f in range(args.frames)]
        stages = "  ".join(f"{k.replace('_', ' ')} {v:.0f}" for k, v in t.timing.items())
        print(f"{v:<8} {size:8.1f} {np.mean(vs_ref):15.4f} {np.min(vs_ref):6.3f} {np.mean(vs_gt):10.4f} {ms:9.0f}   {stages}")
        ok &= np.mean(vs_ref) >= (0.98 if v == "fp32" else 0.95)
    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
