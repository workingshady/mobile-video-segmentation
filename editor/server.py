"""Track Studio backend: FastAPI app serving the editor UI and the model/edit API."""

import base64
import hashlib
import shutil
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import exporter
from .effects import hex_to_rgb, render
from .jobs import GPU_LOCK, JobManager
from .paths import STATIC
from .project import Project, import_media, list_projects
from .trackers import ModelRegistry, TrackerUnavailable


def pick_device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class Studio:
    """All server state. The editor is single-user, so one open project and one loaded model."""

    def __init__(self):
        self.registry = ModelRegistry()
        self.jobs = JobManager()
        self.project: Project | None = None
        self.tracker = None
        self.session_ready = False
        self.synced = False  # tracker session holds exactly the project's prompts
        self.device = pick_device()
        self.overlay_cache = OrderedDict()
        self.bg_image = None

    # ----- helpers -----
    def need_project(self):
        if not self.project:
            raise HTTPException(400, "Open or import a project first")
        return self.project

    def model_id(self, override=None):
        if override:
            return override
        if self.tracker:
            return self.tracker.id
        return self.project.meta.get("active_model") if self.project else None

    def load_bg_image(self):
        self.bg_image = None
        if self.project and self.project.meta["background"].get("image"):
            p = self.project.root / self.project.meta["background"]["image"]
            if p.exists():
                self.bg_image = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)

    def build_session(self, job=None):
        """Create the tracker's video session for the open project and replay prompts (GPU lock held)."""
        p, t = self.project, self.tracker
        if not (p and t):
            return
        if job:
            job.update(0.1, f"Preparing {p.n} frames for {t.name}...")
        t.init_video(p.frames_dir(), p.load_frames_pil(), p.meta["height"], p.meta["width"])
        self.session_ready = True
        self.replay(job)

    def replay(self, job=None):
        if job:
            job.update(0.8, "Applying prompts...")
        result = self.tracker.set_prompts(self.project.all_prompts())
        for frame, masks in result.items():
            self.project.write_masks(self.tracker.id, frame, masks)
        self.synced = True


S = Studio()
app = FastAPI(title="Track Studio")


# ======================= models =======================
@app.get("/api/models")
def models():
    return {
        "models": S.registry.listing(),
        "tiers": S.registry.tiers,
        "tier_order": S.registry.tier_order,
        "loaded": S.tracker.id if S.tracker else None,
        "device": S.device,
    }


@app.post("/api/models/refresh")
def models_refresh():
    S.registry = ModelRegistry()
    return models()


class LoadReq(BaseModel):
    id: str


@app.post("/api/models/load")
def load_model(req: LoadReq):
    try:
        tracker = S.registry.create(req.id)
    except (TrackerUnavailable, KeyError) as e:
        raise HTTPException(400, str(e))

    def run(job):
        import torch

        if S.tracker:
            job.update(0.02, f"Unloading {S.tracker.name}...")
            S.tracker.unload()
            S.tracker = None
            S.session_ready = S.synced = False
        job.update(0.05, f"Loading {tracker.name} on {S.device}...")
        t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        tracker.load(S.device)
        S.tracker = tracker
        job.update(0.5, f"Loaded in {time.time() - t0:.1f}s", load_seconds=round(time.time() - t0, 2))
        if S.project:
            S.project.meta["active_model"] = tracker.id
            S.project.save()
            S.build_session(job)
        return {"loaded": tracker.id}

    return S.jobs.start("load", run).to_dict()


@app.post("/api/models/unload")
def unload_model():
    with GPU_LOCK:
        if S.tracker:
            S.tracker.unload()
        S.tracker = None
        S.session_ready = S.synced = False
    return {"ok": True}


# ======================= projects =======================
@app.get("/api/projects")
def projects():
    return {"projects": list_projects(), "open": S.project.id if S.project else None}


@app.post("/api/projects/import")
async def import_project(
    file: UploadFile = File(...),
    name: str = Form(""),
    max_side: int = Form(1280),
    start: float = Form(0.0),
    end: float = Form(0.0),
    step: int = Form(1),
    max_frames: int = Form(400),
):
    uploads = Path(tempfile.gettempdir()) / "track_studio_uploads"
    uploads.mkdir(exist_ok=True)
    src = uploads / f"{int(time.time())}_{Path(file.filename).name}"
    with src.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    pname = name or Path(file.filename).stem

    def run(job):
        pid = import_media(src, pname, max_side, start, end or None, max(1, step), max_frames,
                           progress=lambda p, m: job.update(p * 0.7, m))
        # keep a copy of the original next to the project so audio/original-res export always works
        p = Project(pid)
        kept = p.root / f"source{src.suffix.lower()}"
        shutil.move(str(src), kept)
        p.meta["source"] = str(kept)
        p.save()
        open_project_impl(pid, job)
        return {"project": pid}

    return S.jobs.start("import", run).to_dict()


def open_project_impl(pid, job=None):
    if S.tracker and S.tracker.id:
        S.tracker.close_video()
    S.project = Project(pid)
    S.session_ready = S.synced = False
    S.overlay_cache.clear()
    S.load_bg_image()
    if S.tracker:
        S.project.meta["active_model"] = S.tracker.id
        S.project.save()
        S.build_session(job)


@app.post("/api/projects/{pid}/open")
def open_project(pid: str):
    def run(job):
        open_project_impl(pid, job)
        return {"project": pid}

    return S.jobs.start("open", run).to_dict()


@app.delete("/api/projects/{pid}")
def delete_project(pid: str):
    if S.project and S.project.id == pid:
        raise HTTPException(400, "Close the project first (open another one)")
    p = Project(pid)
    shutil.rmtree(p.root)
    return {"ok": True}


@app.get("/api/project")
def project_info():
    p = S.need_project()
    return {**p.meta, "id": p.id, "models_with_masks": p.models_with_masks(), "can_undo": bool(p.undo_stack)}


class ProjectPatch(BaseModel):
    background: dict | None = None
    settings: dict | None = None
    in_out: list | None = None
    active_model: str | None = None
    name: str | None = None


@app.patch("/api/project")
def patch_project(req: ProjectPatch):
    p = S.need_project()
    with p.lock:
        if req.background is not None:
            p.meta["background"].update(req.background)
        if req.settings is not None:
            p.meta["settings"].update(req.settings)
        if req.in_out is not None:
            a, b = sorted(int(x) for x in req.in_out)
            p.meta["in_out"] = [max(0, a), min(p.n - 1, b)]
        if req.active_model is not None:
            p.meta["active_model"] = req.active_model
        if req.name:
            p.meta["name"] = req.name
        p.save()
    S.overlay_cache.clear()
    return project_info()


@app.post("/api/project/background-image")
async def background_image(file: UploadFile = File(...)):
    p = S.need_project()
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Not an image")
    cv2.imwrite(str(p.root / "background.jpg"), img)
    p.meta["background"].update({"image": "background.jpg", "mode": "image"})
    p.save()
    S.load_bg_image()
    return project_info()


# ======================= frames / masks =======================
@app.get("/api/frame/{i}.jpg")
def frame(i: int):
    p = S.need_project()
    return FileResponse(p.frame_path(i), headers={"Cache-Control": "max-age=31536000"})


@app.get("/api/thumb/{i}.jpg")
def thumb(i: int):
    p = S.need_project()
    return FileResponse(p.thumb_path(min(max(i, 0), p.n - 1)), headers={"Cache-Control": "max-age=31536000"})


def _png(img):
    ok, buf = cv2.imencode(".png", img)
    return Response(buf.tobytes(), media_type="image/png", headers={"Cache-Control": "max-age=31536000"})


@app.get("/api/overlay/{model_id}/{i}.png")
def overlay(model_id: str, i: int, sel: int = 0):
    """Colored RGBA overlay of all visible objects (edit view)."""
    p = S.need_project()
    style = "|".join(f"{o['id']}{o['color']}{o['visible']}" for o in p.meta["objects"])
    key = (p.id, model_id, i, int(p.rev_array(model_id)[i]), style, sel)
    if key in S.overlay_cache:
        S.overlay_cache.move_to_end(key)
        return Response(S.overlay_cache[key], media_type="image/png")
    lab = p.read_labels(model_id, i)
    h, w = p.meta["height"], p.meta["width"]
    out = np.zeros((h, w, 4), np.uint8)
    if lab is not None:
        kernel = np.ones((3, 3), np.uint8)
        for o in p.meta["objects"]:
            if not o["visible"]:
                continue
            m = lab == o["id"]
            if not m.any():
                continue
            r, g, b = hex_to_rgb(o["color"])
            out[m] = (b, g, r, 110 if o["id"] != sel else 140)
            edge = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
            out[edge] = (b, g, r, 255)
    ok, buf = cv2.imencode(".png", out)
    data = buf.tobytes()
    S.overlay_cache[key] = data
    while len(S.overlay_cache) > 400:
        S.overlay_cache.popitem(last=False)
    return Response(data, media_type="image/png")


@app.get("/api/objmask/{model_id}/{i}/{oid}.png")
def objmask(model_id: str, i: int, oid: int):
    """Black/white mask of one object (the brush tool starts from it)."""
    p = S.need_project()
    lab = p.read_labels(model_id, i)
    m = np.zeros((p.meta["height"], p.meta["width"]), np.uint8) if lab is None else (lab == oid).astype(np.uint8) * 255
    return _png(m)


@app.get("/api/render/{i}.jpg")
def render_frame(i: int, model: str = ""):
    """Frame with all effects applied (preview view) - same code path as export."""
    p = S.need_project()
    mid = S.model_id(model)
    frame = p.read_frame(i)
    lab = p.read_labels(mid, i) if mid else None
    neighbors = None
    if lab is not None and p.meta["settings"].get("temporal_smooth"):
        neighbors = [nb for nb in (p.read_labels(mid, j) for j in (i - 1, i + 1) if 0 <= j < p.n) if nb is not None]
    out = render(frame, lab, p.meta["objects"], p.meta["background"], S.bg_image, neighbors, preview=True)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(buf.tobytes(), media_type="image/jpeg")


@app.get("/api/timeline")
def timeline(model: str = ""):
    p = S.need_project()
    mid = S.model_id(model)
    return {
        "model": mid,
        "n": p.n,
        "fps": p.meta["fps"],
        "coverage": p.coverage(mid) if mid else {},
        "keyframes": p.keyframes(),
        "revs": p.rev_array(mid).tolist() if mid else [0] * p.n,
    }


# ======================= objects =======================
class ObjReq(BaseModel):
    name: str | None = None


@app.post("/api/objects")
def add_object(req: ObjReq):
    p = S.need_project()
    return p.add_object(req.name)


@app.patch("/api/objects/{oid}")
def patch_object(oid: int, patch: dict):
    p = S.need_project()
    o = p.update_object(oid, dict(patch))
    S.overlay_cache.clear()
    return o


@app.delete("/api/objects/{oid}")
def delete_object(oid: int):
    p = S.need_project()
    _gpu_or_busy()
    try:
        p.delete_object(oid)
        S.synced = False
        if S.tracker and S.session_ready:
            S.replay()
    finally:
        GPU_LOCK.release()
    return project_info()


# ======================= prompts =======================
class PromptReq(BaseModel):
    obj: int
    frame: int
    points: list = []
    labels: list = []
    box: list | None = None
    record_undo: bool = True  # magic brush: only the first update of a stroke records undo


class MaskPromptReq(BaseModel):
    obj: int
    frame: int
    png: str  # data URL or base64 of a black/white PNG


def _gpu_or_busy():
    if not GPU_LOCK.acquire(timeout=2):
        raise HTTPException(409, "Model is busy (tracking/export running). Stop it or wait.")


def _apply_prompt_change(p, oid, frame, removed=False):
    """Run the model on the changed frame (or replay everything) and store the resulting masks."""
    t = S.tracker
    if not (t and S.session_ready):
        return {"ran": False, "reason": "No model loaded - prompt saved; load a model to see the mask."}
    t0 = time.time()
    rec = p.get_prompt_raw(oid, frame)
    if removed or rec is None or not S.synced or not t.can_update_in_place(oid):
        S.replay()
        if rec is None:  # prompt removed: clear this object's mask on this frame
            p.write_masks(t.id, frame, {oid: np.zeros((p.meta["height"], p.meta["width"]), bool)})
    else:
        masks = t.update_prompt(oid, frame, p.to_prompt(rec))
        p.write_masks(t.id, frame, masks)
    return {"ran": True, "ms": round((time.time() - t0) * 1000), "model": t.id}


@app.post("/api/prompts")
def set_prompt(req: PromptReq):
    p = S.need_project()
    p.obj(req.obj)
    _gpu_or_busy()
    try:
        p.set_prompt(req.obj, req.frame, req.points, req.labels, req.box, push_undo=req.record_undo)
        return _apply_prompt_change(p, req.obj, req.frame)
    finally:
        GPU_LOCK.release()


@app.post("/api/prompts/mask")
def set_mask_prompt(req: MaskPromptReq):
    p = S.need_project()
    p.obj(req.obj)
    data = req.png.split(",", 1)[-1]
    img = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(400, "Bad PNG")
    if img.ndim == 3:  # browser canvases export opaque RGBA: white = object, black = not
        img = img[..., :3].max(axis=2)
    mask = cv2.resize(img, (p.meta["width"], p.meta["height"]), interpolation=cv2.INTER_NEAREST) > 127
    _gpu_or_busy()
    try:
        p.set_prompt(req.obj, req.frame, mask=mask)
        if not (S.tracker and S.session_ready):
            # no model: still show the painted mask, stored under a "manual" model id
            p.write_masks(S.model_id() or "manual", req.frame, {req.obj: mask})
            return {"ran": False, "reason": "No model loaded - brush mask saved."}
        return _apply_prompt_change(p, req.obj, req.frame)
    finally:
        GPU_LOCK.release()


@app.delete("/api/prompts/{oid}/{frame}")
def clear_prompt(oid: int, frame: int):
    p = S.need_project()
    _gpu_or_busy()
    try:
        p.clear_prompt(oid, frame)
        S.synced = False
        return _apply_prompt_change(p, oid, frame, removed=True)
    finally:
        GPU_LOCK.release()


@app.post("/api/undo")
def undo():
    p = S.need_project()
    _gpu_or_busy()
    try:
        if not p.undo():
            return {"ok": False}
        S.synced = False
        if S.tracker and S.session_ready:
            S.replay()
        return {"ok": True}
    finally:
        GPU_LOCK.release()


# ======================= propagation =======================
class PropReq(BaseModel):
    direction: str = "both"  # forward | backward | both
    use_range: bool = True


@app.post("/api/propagate")
def propagate(req: PropReq):
    p = S.need_project()
    if not (S.tracker and S.session_ready):
        raise HTTPException(400, "Load a model first")
    prompts = p.all_prompts()
    if not prompts:
        raise HTTPException(400, "Add at least one click/box/brush on an object first")
    a, b = p.meta["in_out"] if req.use_range else [0, p.n - 1]
    pframes = sorted({f for frames in prompts.values() for f in frames})
    inside = [f for f in pframes if a <= f <= b]
    if not inside:
        raise HTTPException(400, f"No prompt inside the in/out range ({a}-{b}). Add a click there or untick 'in/out only'.")
    t = S.tracker

    def run(job):
        import torch

        if not S.synced:
            job.update(0.0, "Applying prompts...")
            S.replay()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if hasattr(t, "reset_timing"):  # phone ONNX pipeline reports per-stage ms
            t.reset_timing()
        passes = []
        if req.direction in ("forward", "both"):
            s = inside[0]
            passes.append((s, b - s, False))
        if req.direction == "backward":
            s = inside[-1]
            passes.append((s, s - a, True))
        if req.direction == "both" and inside[0] > a:
            passes.append((inside[0], inside[0] - a, True))
        total = sum(n + 1 for _, n, _ in passes) or 1
        done = 0
        t0 = time.time()
        for start, n, reverse in passes:
            for fidx, masks in t.propagate(start, n, reverse, job.should_stop):
                p.write_masks(t.id, fidx, masks, replace_all=True)
                done += 1
                el = time.time() - t0
                job.update(done / total, f"{'◀' if reverse else '▶'} frame {fidx}  ({done / el:.1f} fps)",
                           frame=fidx, fps=round(done / el, 2))
            if job.should_stop():
                break
        el = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0
        run_info = {
            "model": t.id,
            "model_name": t.name,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "direction": req.direction,
            "frames": done,
            "seconds": round(el, 2),
            "fps": round(done / el, 2) if el else 0,
            "ms_per_frame": round(1000 * el / done, 1) if done else 0,
            "peak_vram_mb": round(peak),
            "objects": len(prompts),
            "resolution": f"{p.meta['width']}x{p.meta['height']}",
            "device": f"cpu · {t.threads} threads" if hasattr(t, "threads") else S.device,
            "stage_ms": getattr(t, "timing", None),
            "cancelled": job.should_stop(),
        }
        if hasattr(t, "threads"):
            run_info["peak_vram_mb"] = 0  # runs on CPU like a phone
        p.add_run(run_info)
        return run_info

    return S.jobs.start("propagate", run).to_dict()


@app.get("/api/runs")
def runs():
    return {"runs": S.need_project().runs()}


# ======================= compare =======================
class CompareReq(BaseModel):
    a: str
    b: str


@app.post("/api/compare")
def compare(req: CompareReq):
    """IoU between two models' masks, per object and per frame (b is usually the 'reference', e.g. SAM 2.1 Large)."""
    p = S.need_project()
    objs = [o["id"] for o in p.meta["objects"]]
    per_frame = {o: [None] * p.n for o in objs}
    for i in range(p.n):
        la, lb = p.read_labels(req.a, i), p.read_labels(req.b, i)
        if la is None or lb is None:
            continue
        for o in objs:
            ma, mb = la == o, lb == o
            union = np.logical_or(ma, mb).sum()
            if union:
                per_frame[o][i] = round(float(np.logical_and(ma, mb).sum() / union), 4)
    summary = {}
    for o, vals in per_frame.items():
        v = [x for x in vals if x is not None]
        worst = sorted((x, i) for i, x in enumerate(vals) if x is not None)[:10]
        summary[o] = {
            "mean_iou": round(float(np.mean(v)), 4) if v else None,
            "min_iou": round(float(np.min(v)), 4) if v else None,
            "frames": len(v),
            "worst": [{"frame": i, "iou": x} for x, i in worst],
        }
    return {"a": req.a, "b": req.b, "summary": summary, "per_frame": per_frame}


# ======================= export =======================
class ExportReq(BaseModel):
    format: str = "mp4"
    resolution: str = "working"  # working | original
    use_range: bool = True
    audio: bool = True
    model: str | None = None


@app.post("/api/export")
def export(req: ExportReq):
    p = S.need_project()
    mid = S.model_id(req.model)
    opts = {"format": req.format, "resolution": req.resolution, "audio": req.audio,
            "range": p.meta["in_out"] if req.use_range else None}

    def run(job):
        out = exporter.export(p, mid, opts, S.bg_image, lambda f, m: job.update(f, m), job.should_stop)
        return {"path": str(out), "name": out.name, "is_file": out.is_file()}

    return S.jobs.start("export", run).to_dict()


@app.get("/api/download/{name}")
def download(name: str):
    p = S.need_project()
    f = (p.root / "exports" / name).resolve()
    if f.parent != (p.root / "exports").resolve() or not f.is_file():
        raise HTTPException(404)
    return FileResponse(f, filename=name)


# ======================= jobs / status =======================
@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    try:
        return S.jobs.get(job_id).to_dict()
    except KeyError:
        raise HTTPException(404)


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str):
    S.jobs.get(job_id).cancelled = True
    return {"ok": True}


@app.get("/api/status")
def status():
    import torch

    vram = None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        vram = {"used_mb": round((total - free) / 2**20), "total_mb": round(total / 2**20),
                "gpu": torch.cuda.get_device_name(0)}
    return {
        "device": S.device,
        "vram": vram,
        "loaded": S.tracker.id if S.tracker else None,
        "loaded_name": S.tracker.name if S.tracker else None,
        "session_ready": S.session_ready,
        "project": S.project.id if S.project else None,
        "jobs": S.jobs.active(),
    }


@app.exception_handler(TrackerUnavailable)
def unavailable(_, exc):
    return JSONResponse({"detail": str(exc)}, status_code=400)


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
