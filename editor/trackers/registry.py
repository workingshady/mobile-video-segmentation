"""Reads models_catalog.json and reports, for every model, whether it can run here and why not."""

import importlib.util
import json
import threading

from ..paths import CATALOG, ROOT
from .base import TrackerUnavailable

RUNNABLE_BACKENDS = {"hf-sam2", "hf-edgetam", "hf-sam3", "efficienttam", "onnx-edgetam"}
_IMPORT_LOCK = threading.Lock()


def _weights_present(entry):
    if not entry.get("dir"):
        return False
    d = ROOT / entry["dir"]
    if entry.get("checkpoint"):
        return (d / entry["checkpoint"]).exists()
    if entry["backend"] == "onnx-edgetam":
        return (d / "meta.json").exists() and (d / entry["variant"] / "memory_attention.onnx").exists()
    if entry["backend"].startswith("hf-"):
        return (d / "config.json").exists() and any(d.glob("*.safetensors"))
    return d.exists() and any(
        p.is_file() and p.stat().st_size > 1_000_000 and ".cache" not in p.relative_to(d).parts for p in d.rglob("*")
    )


def _backend_problem(backend):
    if backend.startswith("hf-"):
        from .hf import resolve_classes

        try:
            resolve_classes(backend)
        except (TrackerUnavailable, ImportError) as e:
            return str(e)
        return None
    if backend == "efficienttam":
        if importlib.util.find_spec("efficient_track_anything") is None:
            return "EfficientTAM package not installed (run setup.bat)"
        return None
    if backend == "onnx-edgetam":
        if importlib.util.find_spec("onnxruntime") is None:
            return "onnxruntime not installed: pip install onnxruntime"
        return None
    return "not runnable in the editor"


class ModelRegistry:
    def __init__(self):
        data = json.loads(CATALOG.read_text(encoding="utf-8"))
        self.entries = data["models"]
        self.tiers = data.get("tiers", {})
        self.tier_order = data.get("tier_order", [])
        self._backend_cache = {}

    def get(self, model_id):
        for e in self.entries:
            if e["id"] == model_id:
                return e
        raise KeyError(model_id)

    def status(self, entry):
        """Return (runnable: bool, status: str, reason: str)."""
        on_disk = _weights_present(entry)
        if entry["group"] == "unavailable":
            return False, "unavailable", entry["notes"]
        if entry["backend"] not in RUNNABLE_BACKENDS:
            return False, "downloaded" if on_disk else "missing", "Weights only, for phone/other testing"
        if not on_disk:
            if entry["backend"] == "onnx-edgetam":
                return False, "missing", "Run: python scripts/export_edgetam_onnx.py"
            if entry.get("gated"):
                return False, "gated", f"Request access at https://huggingface.co/{entry['repo']}, then run download_models.py"
            return False, "missing", "Run: python scripts/download_models.py --only " + entry["id"]
        with _IMPORT_LOCK:  # lazy transformers imports are not thread-safe; requests run in a threadpool
            if entry["backend"] not in self._backend_cache:
                problem = _backend_problem(entry["backend"])
                if problem:
                    return False, "no-backend", problem  # not cached: retried on the next listing
                self._backend_cache[entry["backend"]] = None
        problem = self._backend_cache[entry["backend"]]
        if problem:
            return False, "no-backend", problem
        return True, "ready", ""

    def listing(self):
        out = []
        for e in self.entries:
            runnable, status, reason = self.status(e)
            out.append({**e, "runnable": runnable, "status": status, "reason": reason})
        return out

    def create(self, model_id):
        entry = self.get(model_id)
        runnable, status, reason = self.status(entry)
        if not runnable:
            raise TrackerUnavailable(f"{entry['name']}: {reason}")
        weights = ROOT / entry["dir"]
        if entry["backend"] == "efficienttam":
            from .efficienttam import EfficientTAMTracker

            return EfficientTAMTracker(entry, weights)
        if entry["backend"] == "onnx-edgetam":
            from .onnx_edgetam import OnnxEdgeTAMTracker

            return OnnxEdgeTAMTracker(entry, weights)
        from .hf import HFTracker

        return HFTracker(entry, weights)
