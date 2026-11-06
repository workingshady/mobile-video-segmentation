"""EdgeTAM video tracking with the exported ONNX graphs - the exact loop a phone app runs.

No PyTorch here: only numpy, OpenCV and ONNX Runtime (CPU, limited threads like a phone).
Port this file 1:1 to Kotlin (ONNX Runtime Android) or Swift (CoreML / ONNX Runtime iOS).

Per frame, per object:
  1. image_encoder      once per frame (shared by all objects)
  2. memory_attention   condition the frame on this object's memory bank (skipped on the first prompted frame)
  3. decoder            clicks / memory -> 4 candidate masks; pick one (multimask + stability rules)
  4. memory_encoder     mask + frame -> 512 memory tokens, stored for later frames
Memory bank = every prompted frame + the 6 most recent tracked frames + up to 16 object pointers.
Mirrors transformers' EdgeTamVideoModel step by step (see scripts/validate_onnx.py for the parity test).
"""

import json
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from .base import Prompt, Tracker, TrackerUnavailable

NO_OBJ_SCORE = -1024.0


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def key_rope(rope_k_cos, rope_k_sin, n_frames, n_ptr_tokens, tokens_per_frame):
    d = rope_k_cos.shape[-1]
    temporal = tokens_per_frame - rope_k_cos.shape[0]
    one_c = np.concatenate([np.ones((temporal, d), np.float32), rope_k_cos], 0)
    one_s = np.concatenate([np.zeros((temporal, d), np.float32), rope_k_sin], 0)
    cos = np.concatenate([np.tile(one_c, (n_frames, 1)), np.ones((n_ptr_tokens, d), np.float32)], 0)
    sin = np.concatenate([np.tile(one_s, (n_frames, 1)), np.zeros((n_ptr_tokens, d), np.float32)], 0)
    return cos, sin


def sine_pe_1d(pos, dim, temperature=10000.0):
    half = dim // 2
    dim_t = temperature ** (2 * (np.arange(half, dtype=np.float32) // 2) / half)
    p = pos[:, None] / dim_t
    return np.concatenate([np.sin(p), np.cos(p)], -1).astype(np.float32)


class EdgeTamOnnx:
    """Pure pipeline, independent of the editor."""

    GRAPHS = ("image_encoder", "decoder", "memory_encoder", "memory_attention")

    def __init__(self, model_dir, variant="fp32", threads=4):
        import onnxruntime as ort

        self.dir = Path(model_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        self.c = dict(np.load(self.dir / "constants.npz"))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3  # errors only (constant-folding notices are harmless)
        self.sess = {}
        for g in self.GRAPHS:
            s = ort.InferenceSession(str(self.dir / variant / f"{g}.onnx"), opts, providers=["CPUExecutionProvider"])
            self.sess[g] = (s, {i.name for i in s.get_inputs()})
        self.size = self.meta["image_size"]
        self.mean = np.array(self.meta["image_mean"], np.float32) * 255
        self.std = np.array(self.meta["image_std"], np.float32) * 255
        self.timing = {g: 0.0 for g in self.GRAPHS}
        self.calls = {g: 0 for g in self.GRAPHS}

    def run(self, graph, **feeds):
        s, names = self.sess[graph]
        t0 = time.perf_counter()
        out = s.run(None, {k: v for k, v in feeds.items() if k in names})
        self.timing[graph] += (time.perf_counter() - t0) * 1000
        self.calls[graph] += 1
        return out

    def encode(self, rgb):
        x = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        x = ((x - self.mean) / self.std).transpose(2, 0, 1)[None]
        s0, s1, vf, vp = self.run("image_encoder", pixel_values=np.ascontiguousarray(x))
        return {"feat_s0": s0, "feat_s1": s1, "vision_feat": vf, "vision_pos": vp}


class OnnxEdgeTAMTracker(Tracker):
    """Editor adapter: same prompts/propagation semantics as the PyTorch trackers."""

    def __init__(self, entry, weights_dir):
        self.id = entry["id"]
        self.name = entry["name"]
        self.variant = entry.get("variant", "fp32")
        self.threads = int(entry.get("threads", 4))
        self.weights_dir = Path(weights_dir)
        self.p = None

    # ---------- lifecycle ----------
    def load(self, device):
        try:
            import onnxruntime  # noqa: F401
        except ImportError as e:
            raise TrackerUnavailable("onnxruntime not installed: pip install onnxruntime") from e
        self.p = EdgeTamOnnx(self.weights_dir, self.variant, self.threads)
        m = self.p.meta
        self.n_maskmem = m["num_maskmem"]
        self.tokens_per_frame = m["tokens_per_memory_frame"]

    def unload(self):
        self.close_video()
        self.p = None

    def init_video(self, frames_dir, frames, height, width):
        self.frames = [np.asarray(f) for f in frames]
        self.h, self.w = height, width
        self.num_frames = len(self.frames)
        self.feat_cache = OrderedDict()
        self._reset()

    def close_video(self):
        self.frames = []
        self.feat_cache = OrderedDict()
        self.objs = {}

    def _reset(self):
        self.objs = {}  # obj_id -> {"cond": {f: rec}, "noncond": {f: rec}, "tracked": set()}

    @property
    def timing(self):
        if not self.p:
            return None
        return {g: round(self.p.timing[g] / max(1, self.p.calls[g]), 1) for g in self.p.GRAPHS}

    def reset_timing(self):
        if self.p:
            self.p.timing = {g: 0.0 for g in self.p.GRAPHS}
            self.p.calls = {g: 0 for g in self.p.GRAPHS}

    # ---------- building blocks ----------
    def features(self, frame):
        if frame in self.feat_cache:
            self.feat_cache.move_to_end(frame)
            return self.feat_cache[frame]
        f = self.p.encode(self.frames[frame])
        self.feat_cache[frame] = f
        while len(self.feat_cache) > 3:
            self.feat_cache.popitem(last=False)
        return f

    def _to_1024(self, pts):
        return [[x * self.p.size / self.w, y * self.p.size / self.h] for x, y in pts]

    def _decode(self, pix, f, coords, labels, mask_input=None):
        mask_in = mask_input if mask_input is not None else np.zeros((1, 1, 256, 256), np.float32)
        masks, iou, ptrs, score = self.p.run(
            "decoder",
            pix_feat=pix.astype(np.float32),
            feat_s0=f["feat_s0"],
            feat_s1=f["feat_s1"],
            point_coords=np.array([[coords]], np.float32),
            point_labels=np.array([[labels]], np.int64),
            mask_input=mask_in,
            has_mask=np.array([1.0 if mask_input is not None else 0.0], np.float32),
        )
        return masks[0, 0], iou[0, 0], ptrs[0, 0], float(score.reshape(-1)[0])

    def _select(self, masks, iou, ptrs, multimask):
        m = self.p.meta
        if multimask:
            i = 1 + int(np.argmax(iou[1:]))
            return masks[i], ptrs[i]
        m0 = masks[0]
        d = m["stability_delta"]
        area_u = (m0 > -d).sum()
        stability = (m0 > d).sum() / area_u if area_u > 0 else 1.0
        if stability >= m["stability_thresh"]:
            return m0, ptrs[0]
        return masks[1 + int(np.argmax(iou[1:]))], ptrs[0]

    def _mix_ptr(self, ptr, is_obj):
        return is_obj * ptr + (1 - is_obj) * self.p.c["no_object_pointer"]

    def _conditioned(self, st, frame, f, reverse):
        """Memory attention over: all prompted frames + 6 most recent tracked frames + object pointers."""
        c, meta = self.p.c, self.p.meta
        tpe = c["memory_temporal_pe"]
        mems, poss = [], []
        for out in st["cond"].values():
            mems.append(out["mem_feat"])
            poss.append(out["mem_pos"] + tpe[-1])  # temporal offset 0 -> index -1 (as in the original)
        for off in range(self.n_maskmem - 1, 0, -1):
            out = st["noncond"].get(frame + off if reverse else frame - off)
            if out is not None:
                mems.append(out["mem_feat"])
                poss.append(out["mem_pos"] + tpe[off - 1])
        n_frames = len(mems)

        max_ptr = min(self.num_frames, meta["max_object_pointers"])
        sign = -1 if reverse else 1
        offs, ptrs = [], []
        for t, out in st["cond"].items():
            if (t >= frame) if reverse else (t <= frame):
                offs.append((frame - t) * sign)
                ptrs.append(out["ptr"])
        for d in range(1, max_ptr):
            ref = frame + d if reverse else frame - d
            if ref < 0 or ref >= self.num_frames:
                break
            out = st["noncond"].get(ref)
            if out is not None:
                offs.append(d)
                ptrs.append(out["ptr"])
        mem_dim = meta["mem_dim"]
        splits = meta["hidden_dim"] // mem_dim
        n_ptr_tokens = len(ptrs) * splits
        if ptrs:
            ptr_tok = np.stack(ptrs).reshape(len(ptrs), splits, mem_dim).reshape(-1, mem_dim)
            if meta["pointer_temporal_pe"]:
                pe = sine_pe_1d(np.array(offs, np.float32) / max(1.0, max_ptr - 1), meta["hidden_dim"])
                pe = pe @ c["ptr_tpos_w"].T + c["ptr_tpos_b"]
                ptr_pos = np.repeat(pe, splits, axis=0)
            else:
                ptr_pos = np.zeros((n_ptr_tokens, mem_dim), np.float32)
            mems.append(ptr_tok)
            poss.append(ptr_pos)

        memory = np.concatenate(mems, 0)[None].astype(np.float32)
        memory_pos = np.concatenate(poss, 0)[None].astype(np.float32)
        kc, ks = key_rope(c["rope_k_cos"], c["rope_k_sin"], n_frames, n_ptr_tokens, self.tokens_per_frame)
        vf, vp = f["vision_feat"], f["vision_pos"]
        (cond,) = self.p.run(
            "memory_attention",
            curr=np.ascontiguousarray(vf.reshape(1, vf.shape[1], -1).transpose(0, 2, 1)),
            curr_pos=np.ascontiguousarray(vp.reshape(1, vp.shape[1], -1).transpose(0, 2, 1)),
            memory=memory, memory_pos=memory_pos, k_cos=kc, k_sin=ks,
            k_bias=np.zeros((1, memory.shape[1]), np.float32),
        )
        return np.ascontiguousarray(cond.transpose(0, 2, 1).reshape(vf.shape))

    def _run_frame(self, st, frame, prompt: Prompt | None, is_init_cond, reverse):
        """One tracking step (mirrors EdgeTamVideoModel._run_single_frame_inference)."""
        p, meta, size = self.p, self.p.meta, self.p.size
        f = self.features(frame)
        no_pts = ([[0.0, 0.0]], [-1])

        if prompt is not None and prompt.mask is not None:
            # mask prompt: the mask itself is the output; the decoder only provides the object pointer
            m1024 = cv2.resize(prompt.mask.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR) > 0.5
            hr = m1024.astype(np.float32) * 20.0 - 10.0
            low = cv2.resize(hr, (size // 4, size // 4), interpolation=cv2.INTER_AREA)
            w = p.c["mask_downsample_w"]
            down = np.einsum("iajb,ab->ij", m1024.astype(np.float32).reshape(size // 4, 4, size // 4, 4), w)
            down = (down + p.c["mask_downsample_b"][0])[None, None].astype(np.float32)
            masks, iou, ptrs, score = self._decode(f["vision_feat"], f, *no_pts, mask_input=down)
            _, ptr = self._select(masks, iou, ptrs, multimask=True)
            ptr = self._mix_ptr(ptr, float(score > 0))
            is_obj = float(m1024.any())
            ptr = self._mix_ptr(ptr, is_obj)
            score = 20.0 * is_obj - 10.0
            mask_for_mem = (hr > 0).astype(np.float32)
        else:
            if is_init_cond:
                pix = f["vision_feat"] + p.c["no_memory_embedding"].reshape(1, -1, 1, 1)
            else:
                pix = self._conditioned(st, frame, f, reverse)
            coords, labels = no_pts
            num_pts = 0
            if prompt is not None and (prompt.points or prompt.box is not None):
                coords, labels = [], []
                if prompt.box is not None:
                    x1, y1, x2, y2 = prompt.box
                    coords += [[x1, y1], [x2, y2]]
                    labels += [2, 3]
                coords += [list(q) for q in prompt.points]
                labels += [int(v) for v in prompt.labels]
                coords = self._to_1024(coords)
                num_pts = len(labels)
            multimask = (meta["multimask_output_in_sam"]
                         and (is_init_cond or meta["multimask_output_for_tracking"])
                         and meta["multimask_min_pt_num"] <= num_pts <= meta["multimask_max_pt_num"])
            masks, iou, ptrs, score = self._decode(pix, f, coords, labels)
            low, ptr = self._select(masks, iou, ptrs, multimask)
            is_obj = float(score > 0)
            if not is_obj:
                low = np.full_like(low, NO_OBJ_SCORE)
            ptr = self._mix_ptr(ptr, is_obj)
            hr = cv2.resize(low, (size, size), interpolation=cv2.INTER_LINEAR)
            from_prompt = prompt is not None
            mask_for_mem = (hr > 0).astype(np.float32) if from_prompt else _sigmoid(hr)

        mask_for_mem = mask_for_mem * meta["sigmoid_scale_for_mem_enc"] + meta["sigmoid_bias_for_mem_enc"]
        mem_feat, mem_pos = p.run(
            "memory_encoder", vision_feat=f["vision_feat"], mask_for_mem=mask_for_mem[None, None].astype(np.float32),
            is_obj=np.array([is_obj], np.float32),
        )
        return {"mem_feat": mem_feat[0], "mem_pos": mem_pos[0], "ptr": ptr.astype(np.float32),
                "score": score, "hr": hr}

    def _video_mask(self, rec):
        return cv2.resize(rec["hr"], (self.w, self.h), interpolation=cv2.INTER_LINEAR) > 0

    def _add_prompt(self, obj_id, frame, prompt):
        st = self.objs.setdefault(obj_id, {"cond": {}, "noncond": {}, "tracked": set()})
        is_init = frame not in st["tracked"]
        rec = self._run_frame(st, frame, prompt, is_init_cond=is_init, reverse=False)
        st["cond"][frame] = rec
        return self._video_mask(rec)

    # ---------- Tracker interface ----------
    def set_prompts(self, prompts):
        self._reset()
        result = {}
        for obj_id in sorted(prompts):
            for frame in sorted(prompts[obj_id]):
                pr = prompts[obj_id][frame]
                if pr.is_empty():
                    continue
                result.setdefault(frame, {})[obj_id] = self._add_prompt(obj_id, frame, pr)
        return result

    def can_update_in_place(self, obj_id):
        return True

    def update_prompt(self, obj_id, frame, prompt):
        return {obj_id: self._add_prompt(obj_id, frame, prompt)}

    def propagate(self, start, max_frames, reverse, should_stop):
        n = self.num_frames
        max_frames = n if max_frames is None else max_frames
        if reverse:
            end = max(start - max_frames, 0)
            order = range(start, end - 1, -1) if start > 0 else []
        else:
            end = min(start + max_frames, n - 1)
            order = range(start, end + 1)
        for frame in order:
            out = {}
            for obj_id in sorted(self.objs):
                st = self.objs[obj_id]
                if frame in st["cond"]:
                    rec = st["cond"][frame]
                else:
                    rec = self._run_frame(st, frame, None, is_init_cond=False, reverse=reverse)
                    st["noncond"][frame] = rec
                    st["tracked"].add(frame)
                out[obj_id] = self._video_mask(rec)
            yield frame, out
            if should_stop():
                break
