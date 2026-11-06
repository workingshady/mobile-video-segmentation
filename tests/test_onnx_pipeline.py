"""Phone ONNX pipeline: fixed-shape padding is exact, and the loop tracks a moving object (CPU, no GPU needed)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ONNX_DIR = ROOT / "models" / "edgetam-onnx-video"

ort = pytest.importorskip("onnxruntime")
pytestmark = pytest.mark.skipif(not (ONNX_DIR / "int8" / "memory_attention.onnx").exists(),
                                reason="run scripts/export_edgetam_onnx.py first")


def test_padded_memory_attention_equals_exact():
    from editor.trackers.onnx_edgetam import key_rope

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    s = ort.InferenceSession(str(ONNX_DIR / "fp32" / "memory_attention.onnx"), opts, providers=["CPUExecutionProvider"])
    c = np.load(ONNX_DIR / "constants.npz")
    rng = np.random.default_rng(0)
    curr, cpos = rng.standard_normal((2, 1, 4096, 256), dtype=np.float32)
    mem, mpos = rng.standard_normal((2, 1, 2 * 512 + 8, 64), dtype=np.float32)

    kc, ks = key_rope(c["rope_k_cos"], c["rope_k_sin"], 2, 8, 512)
    exact = s.run(None, dict(curr=curr, curr_pos=cpos, memory=mem, memory_pos=mpos, k_cos=kc, k_sin=ks,
                             k_bias=np.zeros((1, mem.shape[1]), np.float32)))[0]

    pad_f, pad_p = 5 * 512, 56  # fixed NPU shape: 7 frames + 64 pointer tokens
    z = lambda n: np.zeros((1, n, 64), np.float32)  # noqa: E731
    pm = np.concatenate([mem[:, :1024], z(pad_f), mem[:, 1024:], z(pad_p)], 1)
    pp = np.concatenate([mpos[:, :1024], z(pad_f), mpos[:, 1024:], z(pad_p)], 1)
    kc, ks = key_rope(c["rope_k_cos"], c["rope_k_sin"], 7, 64, 512)
    bias = np.zeros((1, pm.shape[1]), np.float32)
    bias[0, 1024:1024 + pad_f] = -1e4
    bias[0, -pad_p:] = -1e4
    padded = s.run(None, dict(curr=curr, curr_pos=cpos, memory=pm, memory_pos=pp, k_cos=kc, k_sin=ks, k_bias=bias))[0]
    assert np.abs(exact - padded).max() < 1e-4


def test_int8_pipeline_tracks_moving_object():
    from PIL import Image

    from editor.trackers.base import Prompt
    from editor.trackers.onnx_edgetam import OnnxEdgeTAMTracker
    from scripts.smoke_test import iou, synth_video

    tmp = Path(tempfile.mkdtemp())
    frames, gts, clicks = synth_video(tmp, 8)
    h, w = frames[0].shape[:2]
    t = OnnxEdgeTAMTracker({"id": "t", "name": "t", "variant": "int8", "threads": 4}, ONNX_DIR)
    t.load("cpu")
    t.init_video(str(tmp), [Image.fromarray(f) for f in frames], h, w)
    first = t.set_prompts({1: {0: Prompt(points=[list(p) for p in clicks], labels=[1, 1])}})
    assert iou(first[0][1], gts[0]) > 0.9
    ious = [iou(m[1], gts[f]) for f, m in t.propagate(0, None, False, lambda: False)]
    assert len(ious) == 8 and min(ious) > 0.9
    assert set(t.timing) == {"image_encoder", "decoder", "memory_encoder", "memory_attention"}
