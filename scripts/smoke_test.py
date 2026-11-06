"""End-to-end check of every runnable model: load -> click -> track -> IoU vs ground truth -> fps.

Builds a synthetic video (a textured disc moving over a noisy background) so the true mask is known.

    python scripts/smoke_test.py                    # all models whose weights are on disk
    python scripts/smoke_test.py --models edgetam,efficienttam-ti --frames 60
"""

import argparse
import os
import sys

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from editor.trackers import ModelRegistry, Prompt  # noqa: E402


def synth_video(folder, n, w=640, h=360):
    rng = np.random.default_rng(0)
    bg = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), (0, 0), 3)
    frames, gts = [], []
    for i in range(n):
        t = i / max(1, n - 1)
        cx, cy = int(120 + 400 * t), int(180 + 80 * np.sin(t * 6.28))
        img = bg.copy()
        # a single-part object (no nested highlight: one click on a sub-part is ambiguous for SAM by design)
        cv2.circle(img, (cx, cy), 55, (40, 90, 230), -1)
        gt = np.zeros((h, w), np.uint8)
        cv2.circle(gt, (cx, cy), 55, 1, -1)
        cv2.imwrite(str(folder / f"{i:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        gts.append(gt.astype(bool))
    return frames, gts, [(120, 180), (140, 200)]


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="comma-separated ids (default: all ready)")
    ap.add_argument("--frames", type=int, default=48)
    args = ap.parse_args()

    from PIL import Image

    reg = ModelRegistry()
    ids = args.models.split(",") if args.models else [m["id"] for m in reg.listing() if m["runnable"]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tmp = Path(tempfile.mkdtemp(prefix="smoke_"))
    frames, gts, click = synth_video(tmp, args.frames)
    pil = [Image.fromarray(f) for f in frames]
    h, w = frames[0].shape[:2]
    print(f"device={device}  frames={args.frames}  size={w}x{h}  models={ids}\n")

    rows = []
    for mid in ids:
        try:
            t = reg.create(mid)
            t0 = time.time()
            t.load(device)
            load_s = time.time() - t0
            t.init_video(str(tmp), pil, h, w)
            res = t.set_prompts({1: {0: Prompt(points=[list(c) for c in click], labels=[1] * len(click))}})
            click_iou = iou(res[0][1], gts[0])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            ious = [iou(m[1], gts[f]) for f, m in t.propagate(0, None, False, lambda: False)]
            el = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0
            rows.append((mid, "OK", load_s, click_iou, float(np.mean(ious)), float(np.min(ious)), len(ious) / el, peak))
            t.unload()
        except Exception as e:  # report and continue with the next model
            rows.append((mid, f"FAIL {type(e).__name__}: {e}"[:110], 0, 0, 0, 0, 0, 0))
        r = rows[-1]
        print(f"{r[0]:<20} {r[1]:<6} load {r[2]:5.1f}s  click IoU {r[3]:.3f}  track mIoU {r[4]:.3f} (min {r[5]:.3f})  {r[6]:6.1f} fps  {r[7]:6.0f} MB"
              if r[1] == "OK" else f"{r[0]:<20} {r[1]}", flush=True)
    bad = [r for r in rows if r[1] != "OK" or r[4] < 0.8]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} models passed (OK and mean IoU >= 0.8)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
