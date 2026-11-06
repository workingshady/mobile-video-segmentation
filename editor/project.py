"""A project = one imported video (or image) on disk.

projects/<slug>/
  project.json            metadata, objects, prompts, effects, settings
  frames/00000.jpg        working-resolution frames (what models see; EfficientTAM needs this layout)
  thumbs/00000.jpg        small frames for the timeline (made on demand)
  masks/<model_id>/00000.png   uint8 label map per frame: pixel value = object id (0 = nothing)
  prompt_masks/*.png      brush masks used as prompts
  runs.json               benchmark log of every propagation
  exports/                rendered outputs
"""

import copy
import json
import re
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .paths import PROJECTS
from .trackers.base import Prompt

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PALETTE = ["#ff4fa0", "#3fb8ff", "#ffd23f", "#4ade80", "#c084fc", "#fb923c", "#22d3ee", "#f87171"]
DEFAULT_EFFECT = {"type": "highlight", "color": None, "opacity": 0.5, "strength": 15, "grow": 0, "feather": 2}
DEFAULT_BACKGROUND = {"mode": "keep", "color": "#00b140", "strength": 25, "image": None}
UNDO_LIMIT = 60


def slugify(name):
    s = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower()
    return s or "project"


def list_projects():
    PROJECTS.mkdir(parents=True, exist_ok=True)
    out = []
    for d in sorted(PROJECTS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        meta = d / "project.json"
        if meta.exists():
            m = json.loads(meta.read_text(encoding="utf-8"))
            out.append({"id": d.name, "name": m["name"], "num_frames": m["num_frames"], "kind": m["kind"]})
    return out


def import_media(src, name, max_side=1280, start=0.0, end=None, step=1, max_frames=400, progress=None):
    """Extract working frames from a video or image into a new project. Returns the project id."""
    src = Path(src)
    PROJECTS.mkdir(parents=True, exist_ok=True)
    pid = slugify(name)
    base = pid
    n = 2
    while (PROJECTS / pid).exists():
        pid = f"{base}-{n}"
        n += 1
    root = PROJECTS / pid
    (root / "frames").mkdir(parents=True)

    def fit(img):
        h, w = img.shape[:2]
        s = min(1.0, max_side / max(h, w)) if max_side else 1.0
        if s < 1:
            img = cv2.resize(img, (round(w * s / 2) * 2, round(h * s / 2) * 2), interpolation=cv2.INTER_AREA)
        return img

    if src.suffix.lower() in IMAGE_EXT:
        img = cv2.imdecode(np.fromfile(str(src), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            shutil.rmtree(root)
            raise ValueError("Could not read image")
        oh, ow = img.shape[:2]
        img = fit(img)
        cv2.imwrite(str(root / "frames" / "00000.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        kind, fps, count, start_frame = "image", 1.0, 1, 0
    else:
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            shutil.rmtree(root)
            raise ValueError("Could not open video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ow, oh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        start_frame = int(round(start * fps))
        end_frame = int(round(end * fps)) if end else (total or 10**9)
        if start_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        expected = max(1, min(max_frames, (end_frame - start_frame + step - 1) // step))
        count, idx = 0, start_frame
        while idx < end_frame and count < max_frames:
            ok, img = cap.read()
            if not ok:
                break
            if (idx - start_frame) % step == 0:
                cv2.imwrite(str(root / "frames" / f"{count:05d}.jpg"), fit(img), [cv2.IMWRITE_JPEG_QUALITY, 95])
                count += 1
                if progress:
                    progress(count / expected, f"Extracting frame {count}/{expected}")
            idx += 1
        cap.release()
        if count == 0:
            shutil.rmtree(root)
            raise ValueError("Video has no readable frames")
        kind = "video"

    first = cv2.imread(str(root / "frames" / "00000.jpg"))
    meta = {
        "name": name,
        "kind": kind,
        "source": str(src.resolve()),
        "created": time.time(),
        "fps": fps / step,
        "source_fps": fps,
        "start_frame": start_frame,
        "frame_step": step,
        "num_frames": count,
        "width": first.shape[1],
        "height": first.shape[0],
        "orig_width": ow,
        "orig_height": oh,
        "objects": [],
        "prompts": {},
        "background": dict(DEFAULT_BACKGROUND),
        "settings": {"temporal_smooth": False},
        "in_out": [0, count - 1],
        "active_model": None,
    }
    (root / "project.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return pid


class Project:
    def __init__(self, pid):
        self.id = pid
        self.root = PROJECTS / pid
        self.meta = json.loads((self.root / "project.json").read_text(encoding="utf-8"))
        self.lock = threading.RLock()
        self.undo_stack = []
        self.revs = {}  # model_id -> np.int32[num_frames], bumped whenever a frame's mask changes
        self._presence = {}  # model_id -> {obj_id: bool[num_frames]}, kept in sync by write_masks

    # ---------- basics ----------
    @property
    def n(self):
        return self.meta["num_frames"]

    def save(self):
        with self.lock:
            tmp = self.root / "project.json.tmp"
            tmp.write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
            tmp.replace(self.root / "project.json")

    def frame_path(self, i):
        return self.root / "frames" / f"{i:05d}.jpg"

    def frames_dir(self):
        return str(self.root / "frames")

    def read_frame(self, i):
        return cv2.cvtColor(cv2.imread(str(self.frame_path(i))), cv2.COLOR_BGR2RGB)

    def load_frames_pil(self):
        from PIL import Image

        return [Image.open(self.frame_path(i)).convert("RGB") for i in range(self.n)]

    def thumb_path(self, i, height=72):
        p = self.root / "thumbs" / f"{i:05d}.jpg"
        if not p.exists():
            p.parent.mkdir(exist_ok=True)
            img = cv2.imread(str(self.frame_path(i)))
            h, w = img.shape[:2]
            img = cv2.resize(img, (max(1, round(w * height / h)), height), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return p

    # ---------- objects ----------
    def obj(self, oid):
        for o in self.meta["objects"]:
            if o["id"] == oid:
                return o
        raise KeyError(oid)

    def add_object(self, name=None):
        with self.lock:
            used = {o["id"] for o in self.meta["objects"]}
            oid = next(i for i in range(1, 256) if i not in used)
            color = PALETTE[(oid - 1) % len(PALETTE)]
            o = {
                "id": oid,
                "name": name or f"Object {oid}",
                "color": color,
                "visible": True,
                "foreground": True,
                "effect": {**DEFAULT_EFFECT, "color": color},
            }
            self.meta["objects"].append(o)
            self.save()
            return o

    def update_object(self, oid, patch):
        with self.lock:
            o = self.obj(oid)
            if "effect" in patch:
                o["effect"].update(patch.pop("effect"))
            o.update({k: v for k, v in patch.items() if k in ("name", "color", "visible", "foreground")})
            self.save()
            return o

    def delete_object(self, oid):
        with self.lock:
            self._push_undo()
            self.meta["objects"] = [o for o in self.meta["objects"] if o["id"] != oid]
            self.meta["prompts"].pop(str(oid), None)
            for model_dir in (self.root / "masks").glob("*"):
                for f in model_dir.glob("*.png"):
                    lab = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                    if (lab == oid).any():
                        lab[lab == oid] = 0
                        cv2.imwrite(str(f), lab)
                        self.bump(model_dir.name, int(f.stem))
            for pres in self._presence.values():
                pres.pop(oid, None)
            self.save()

    # ---------- prompts ----------
    def _push_undo(self):
        self.undo_stack.append(copy.deepcopy(self.meta["prompts"]))
        del self.undo_stack[:-UNDO_LIMIT]

    def undo(self):
        with self.lock:
            if not self.undo_stack:
                return False
            self.meta["prompts"] = self.undo_stack.pop()
            self.save()
            return True

    def get_prompt_raw(self, oid, frame):
        return self.meta["prompts"].get(str(oid), {}).get(str(frame))

    def set_prompt(self, oid, frame, points=None, labels=None, box=None, mask=None, push_undo=True):
        """Store a prompt. `mask` (bool array) switches the frame to a brush-mask prompt.

        push_undo=False lets the magic brush send live updates during one stroke while undo restores
        the state from before the whole stroke."""
        with self.lock:
            if push_undo:
                self._push_undo()
            rec = {"points": points or [], "labels": labels or [], "box": box, "mask": None}
            if mask is not None:
                d = self.root / "prompt_masks"
                d.mkdir(exist_ok=True)
                fname = f"o{oid}_f{frame:05d}_{int(time.time() * 1000)}.png"
                cv2.imwrite(str(d / fname), mask.astype(np.uint8) * 255)
                rec = {"points": [], "labels": [], "box": None, "mask": fname}
            if not rec["points"] and rec["box"] is None and rec["mask"] is None:
                self.meta["prompts"].get(str(oid), {}).pop(str(frame), None)
            else:
                self.meta["prompts"].setdefault(str(oid), {})[str(frame)] = rec
            self.save()

    def clear_prompt(self, oid, frame):
        with self.lock:
            self._push_undo()
            self.meta["prompts"].get(str(oid), {}).pop(str(frame), None)
            self.save()

    def to_prompt(self, rec):
        mask = None
        if rec.get("mask"):
            m = cv2.imread(str(self.root / "prompt_masks" / rec["mask"]), cv2.IMREAD_GRAYSCALE)
            mask = m > 127
        return Prompt(points=rec["points"], labels=rec["labels"], box=rec["box"], mask=mask)

    def all_prompts(self):
        """{obj_id: {frame: Prompt}} for existing objects."""
        valid = {o["id"] for o in self.meta["objects"]}
        out = {}
        for oid, frames in self.meta["prompts"].items():
            if int(oid) in valid and frames:
                out[int(oid)] = {int(f): self.to_prompt(r) for f, r in frames.items()}
        return out

    def keyframes(self):
        return {oid: sorted(int(f) for f in frames) for oid, frames in self.meta["prompts"].items() if frames}

    # ---------- masks ----------
    def mask_path(self, model_id, i):
        return self.root / "masks" / model_id / f"{i:05d}.png"

    def read_labels(self, model_id, i):
        p = self.mask_path(model_id, i)
        if not p.exists():
            return None
        return cv2.imread(str(p), cv2.IMREAD_UNCHANGED)

    def rev_array(self, model_id):
        if model_id not in self.revs:
            arr = np.zeros(self.n, np.int32)
            d = self.root / "masks" / model_id
            if d.exists():
                for f in d.glob("*.png"):
                    i = int(f.stem)
                    if i < self.n:
                        arr[i] = 1
            self.revs[model_id] = arr
        return self.revs[model_id]

    def bump(self, model_id, i):
        self.rev_array(model_id)[i] += 1

    def write_masks(self, model_id, i, masks, replace_all=False):
        """Merge {obj_id: bool mask} into frame i's label map."""
        with self.lock:
            lab = None if replace_all else self.read_labels(model_id, i)
            if lab is None:
                lab = np.zeros((self.meta["height"], self.meta["width"]), np.uint8)
            for oid, m in masks.items():
                if m is None:
                    continue
                if m.shape != lab.shape:
                    m = cv2.resize(m.astype(np.uint8), (lab.shape[1], lab.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                lab[lab == oid] = 0
                lab[m] = oid
            p = self.mask_path(model_id, i)
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), lab)
            self.bump(model_id, i)
            pres = self.presence(model_id)
            ids = set(np.unique(lab).tolist())
            for oid, arr in pres.items():
                arr[i] = oid in ids
            for oid in ids - set(pres) - {0}:
                pres[oid] = np.zeros(self.n, bool)
                pres[oid][i] = True

    def presence(self, model_id):
        """{obj_id: bool[num_frames]} - which frames have a mask. Scans the PNGs once, then stays in memory."""
        if model_id not in self._presence:
            pres = {}
            d = self.root / "masks" / model_id
            if d.exists():
                for f in d.glob("*.png"):
                    i = int(f.stem)
                    if i >= self.n:
                        continue
                    for oid in np.unique(cv2.imread(str(f), cv2.IMREAD_UNCHANGED)).tolist():
                        if oid:
                            pres.setdefault(oid, np.zeros(self.n, bool))[i] = True
            self._presence[model_id] = pres
        return self._presence[model_id]

    def coverage(self, model_id):
        """Per object: list of [start, end] frame ranges that have a non-empty mask."""
        out = {o["id"]: [] for o in self.meta["objects"]}
        if not model_id:
            return out
        pres = self.presence(model_id)
        present = {oid: pres.get(oid, np.zeros(self.n, bool)) for oid in out}
        for oid, arr in present.items():
            ranges, s = [], None
            for i, v in enumerate(arr):
                if v and s is None:
                    s = i
                if not v and s is not None:
                    ranges.append([s, i - 1])
                    s = None
            if s is not None:
                ranges.append([s, self.n - 1])
            out[oid] = ranges
        return out

    # ---------- runs ----------
    def add_run(self, run):
        p = self.root / "runs.json"
        runs = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
        runs.append(run)
        p.write_text(json.dumps(runs, indent=2), encoding="utf-8")

    def runs(self):
        p = self.root / "runs.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []

    def models_with_masks(self):
        d = self.root / "masks"
        return sorted(p.name for p in d.iterdir() if p.is_dir() and any(p.glob("*.png"))) if d.exists() else []
