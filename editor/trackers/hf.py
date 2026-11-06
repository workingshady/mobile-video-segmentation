"""SAM 2.1, SAM 3 (tracker) and EdgeTAM through Hugging Face Transformers.

All three share the same video API (init_video_session / add_inputs_to_inference_session /
propagate_in_video_iterator), so one adapter covers them.
"""

import gc

import numpy as np
import torch

from .base import Prompt, Tracker, TrackerUnavailable

CLASSES = {
    "hf-sam2": ("Sam2VideoModel", "Sam2VideoProcessor"),
    "hf-edgetam": ("EdgeTamVideoModel", "Sam2VideoProcessor"),
    "hf-sam3": ("Sam3TrackerVideoModel", "Sam3TrackerVideoProcessor"),
}


def resolve_classes(backend):
    import transformers

    model_name, proc_name = CLASSES[backend]
    try:
        return getattr(transformers, model_name), getattr(transformers, proc_name)
    except AttributeError as e:
        raise TrackerUnavailable(
            f"transformers {transformers.__version__} has no {model_name}; needs transformers >= 4.57"
        ) from e
    except Exception as e:  # lazy import failed: report the real cause (e.g. a missing package like timm)
        raise TrackerUnavailable(f"{model_name} could not be imported: {type(e).__name__}: {e}"[:300]) from e


class HFTracker(Tracker):
    def __init__(self, entry, weights_dir):
        self.id = entry["id"]
        self.name = entry["name"]
        self.backend = entry["backend"]
        self.weights_dir = str(weights_dir)
        self.model = None
        self.processor = None
        self.session = None
        self.size = None
        self.tracked = False  # True once propagation ran since the last replay

    def load(self, device):
        model_cls, proc_cls = resolve_classes(self.backend)
        self.device = device
        self.dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = model_cls.from_pretrained(self.weights_dir, dtype=self.dtype).to(device).eval()
        self.processor = proc_cls.from_pretrained(self.weights_dir)

    def unload(self):
        self.close_video()
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def init_video(self, frames_dir, frames, height, width):
        self.close_video()
        self.size = [[height, width]]
        self.session = self.processor.init_video_session(
            video=frames,
            inference_device=self.device,
            video_storage_device="cpu",
            processing_device="cpu",
            dtype=self.dtype,
        )

    def close_video(self):
        self.session = None
        self.tracked = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _to_masks(self, pred_masks, obj_ids):
        logits = self.processor.post_process_masks([pred_masks], original_sizes=self.size, binarize=False)[0]
        logits = logits.float().cpu().numpy()
        return {int(o): logits[i, 0] > 0 for i, o in enumerate(obj_ids)}

    def _add(self, obj_id, frame, prompt: Prompt):
        kw = dict(inference_session=self.session, frame_idx=frame, obj_ids=obj_id)
        if prompt.mask is not None:
            kw["input_masks"] = prompt.mask.astype(np.float32)
        else:
            if prompt.points:
                kw["input_points"] = [[[list(map(float, p)) for p in prompt.points]]]
                kw["input_labels"] = [[[int(l) for l in prompt.labels]]]
            if prompt.box is not None:
                kw["input_boxes"] = [[list(map(float, prompt.box))]]
        self.processor.add_inputs_to_inference_session(**kw)
        out = self.model(inference_session=self.session, frame_idx=frame)
        ids = getattr(out, "object_ids", None) or [obj_id]
        return self._to_masks(out.pred_masks, ids)

    @torch.inference_mode()
    def set_prompts(self, prompts):
        self.session.reset_tracking_data()
        self.tracked = False
        result = {}
        for obj_id in sorted(prompts):
            for frame in sorted(prompts[obj_id]):
                p = prompts[obj_id][frame]
                if p.is_empty():
                    continue
                masks = self._add(obj_id, frame, p)
                result.setdefault(frame, {}).update({obj_id: masks.get(obj_id)})
        return result

    def can_update_in_place(self, obj_id):
        known = obj_id in list(self.session.obj_ids)
        return known or not self.tracked

    @torch.inference_mode()
    def update_prompt(self, obj_id, frame, prompt):
        return self._add(obj_id, frame, prompt)

    def propagate(self, start, max_frames, reverse, should_stop):
        self.tracked = True
        with torch.inference_mode():
            it = self.model.propagate_in_video_iterator(
                self.session, start_frame_idx=start, max_frame_num_to_track=max_frames, reverse=reverse
            )
            for out in it:
                ids = getattr(out, "object_ids", None) or list(self.session.obj_ids)
                yield out.frame_idx, self._to_masks(out.pred_masks, ids)
                if should_stop():
                    break
