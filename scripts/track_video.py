"""Click an object (hand, brush, anything) on one frame and track it through the whole video.

Uses EdgeTAM (the phone-sized SAM 2) through Hugging Face Transformers.

Examples:
    # Interactive: a window opens on the first frame.
    #   Left click  = "this is the object"   (positive point)
    #   Right click = "this is NOT the object" (negative point)
    #   ENTER = start tracking, R = reset clicks, ESC = quit
    python scripts/track_video.py --video my_hand.mp4

    # Headless: give the points yourself (x,y in original video pixels).
    python scripts/track_video.py --video my_hand.mp4 --points 420,310 --neg-points 600,200
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import EdgeTamVideoModel, Sam2VideoProcessor

MODEL_ID = "yonigozlan/edgetam-video-1"
OVERLAY_COLOR = np.array([255, 64, 160], dtype=np.uint8)  # pink, RGB


def parse_points(text):
    if not text:
        return []
    return [[float(v) for v in p.split(",")] for p in text.split(";")]


def pick_device():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def load_frames(path, max_frames, resize_long):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while len(frames) < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        if resize_long:
            h, w = bgr.shape[:2]
            scale = resize_long / max(h, w)
            if scale < 1:
                bgr = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise SystemExit("Video has no readable frames.")
    return frames, fps


def click_points(frame):
    """Let the user click positive/negative points on a frame. Returns (pos, neg)."""
    pos, neg = [], []
    base = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    win = "Left=object  Right=not object  ENTER=track  R=reset  ESC=quit"

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            pos.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN:
            neg.append([x, y])

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        view = base.copy()
        for x, y in pos:
            cv2.circle(view, (x, y), 7, (0, 255, 0), -1)
        for x, y in neg:
            cv2.circle(view, (x, y), 7, (0, 0, 255), -1)
        cv2.imshow(win, view)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and pos:
            break
        if key in (ord("r"), ord("R")):
            pos.clear()
            neg.clear()
        if key == 27:
            raise SystemExit("Cancelled.")
    cv2.destroyAllWindows()
    return pos, neg


def clean_mask(mask):
    """Keep only the largest blob and close small holes - cheap consistency boost."""
    mask = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 2:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel).astype(bool)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=0, help="frame index to click on")
    ap.add_argument("--points", help='positive points "x,y;x,y" (skips the click window)')
    ap.add_argument("--neg-points", help='negative points "x,y;x,y"')
    ap.add_argument("--max-frames", type=int, default=600, help="all frames are held in memory")
    ap.add_argument("--resize-long", type=int, default=1280, help="downscale long side (0 = keep)")
    ap.add_argument("--no-clean", action="store_true", help="disable largest-blob/hole-fill cleanup")
    ap.add_argument("--save-masks", action="store_true", help="also write one PNG mask per frame")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    frames, fps = load_frames(args.video, args.max_frames, args.resize_long)
    if not 0 <= args.frame < len(frames):
        raise SystemExit(f"--frame must be between 0 and {len(frames) - 1}")
    print(f"Loaded {len(frames)} frames ({frames[0].width}x{frames[0].height}) at {fps:.1f} fps")

    if args.points:
        pos, neg = parse_points(args.points), parse_points(args.neg_points)
    else:
        pos, neg = click_points(frames[args.frame])

    device, dtype = pick_device()
    print(f"Loading {MODEL_ID} on {device} ({dtype})")
    model = EdgeTamVideoModel.from_pretrained(MODEL_ID).to(device, dtype=dtype)
    processor = Sam2VideoProcessor.from_pretrained(MODEL_ID)

    session = processor.init_video_session(video=frames, inference_device=device, dtype=dtype)
    processor.add_inputs_to_inference_session(
        inference_session=session,
        frame_idx=args.frame,
        obj_ids=1,
        input_points=[[pos + neg]],
        input_labels=[[[1] * len(pos) + [0] * len(neg)]],
    )
    model(inference_session=session, frame_idx=args.frame)

    size = [[session.video_height, session.video_width]]
    masks = {}
    t0 = time.perf_counter()
    with torch.inference_mode():
        # Forward from the clicked frame, then backward so frames before it are covered too.
        for reverse in (False, True):
            if reverse and args.frame == 0:
                continue
            for out in model.propagate_in_video_iterator(session, start_frame_idx=args.frame, reverse=reverse):
                logits = processor.post_process_masks([out.pred_masks], original_sizes=size, binarize=False)[0]
                masks[out.frame_idx] = (logits[0, 0] > 0).float().cpu().numpy().astype(bool)
    dt = time.perf_counter() - t0
    print(f"Tracked {len(masks)} frames in {dt:.1f}s ({len(masks) / dt:.1f} fps on {device})")

    os.makedirs(args.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.video))[0]
    out_path = os.path.join(args.out, f"{name}_tracked.mp4")
    h, w = session.video_height, session.video_width
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if args.save_masks:
        os.makedirs(os.path.join(args.out, "masks"), exist_ok=True)

    for i, frame in enumerate(frames):
        rgb = np.array(frame)
        mask = masks.get(i)
        if mask is not None:
            if not args.no_clean:
                mask = clean_mask(mask)
            rgb[mask] = (0.45 * rgb[mask] + 0.55 * OVERLAY_COLOR).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, contours, -1, (255, 255, 255), 2)
            if args.save_masks:
                cv2.imwrite(os.path.join(args.out, "masks", f"{i:05d}.png"), mask.astype(np.uint8) * 255)
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
