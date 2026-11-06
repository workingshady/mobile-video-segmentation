"""EfficientTAM through its official package (third_party/EfficientTAM, SAM 2-style predictor API)."""

import contextlib
import gc

import numpy as np
import torch

from .base import Prompt, Tracker, TrackerUnavailable


def import_builder():
    try:
        from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor
    except ImportError as e:
        raise TrackerUnavailable(
            "EfficientTAM package not installed. Run setup.bat (it installs third_party/EfficientTAM)."
        ) from e
    return build_efficienttam_video_predictor


class EfficientTAMTracker(Tracker):
    def __init__(self, entry, weights_dir):
        self.id = entry["id"]
        self.name = entry["name"]
        self.checkpoint = str(weights_dir / entry["checkpoint"])
        self.config = entry["config"]
        self.predictor = None
        self.state = None
        self.tracked = False

    def _autocast(self):
        if self.device == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def load(self, device):
        build = import_builder()
        self.device = device
        # torch.compile needs Triton, which doesn't exist on Windows; eager mode works everywhere
        self.predictor = build(
            self.config, self.checkpoint, device=device, hydra_overrides_extra=["++model.compile_image_encoder=False"]
        )

    def unload(self):
        self.close_video()
        self.predictor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def init_video(self, frames_dir, frames, height, width):
        self.close_video()
        with torch.inference_mode(), self._autocast():
            self.state = self.predictor.init_state(
                video_path=frames_dir, offload_video_to_cpu=True, offload_state_to_cpu=False
            )

    def close_video(self):
        self.state = None
        self.tracked = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _to_masks(obj_ids, logits):
        logits = logits.float().cpu().numpy()
        return {int(o): logits[i, 0] > 0 for i, o in enumerate(obj_ids)}

    def _add(self, obj_id, frame, prompt: Prompt):
        if prompt.mask is not None:
            _, ids, logits = self.predictor.add_new_mask(self.state, frame, obj_id, prompt.mask.astype(bool))
        else:
            points = np.array(prompt.points, dtype=np.float32) if prompt.points else None
            labels = np.array(prompt.labels, dtype=np.int32) if prompt.points else None
            box = np.array(prompt.box, dtype=np.float32) if prompt.box is not None else None
            _, ids, logits = self.predictor.add_new_points_or_box(
                self.state, frame, obj_id, points=points, labels=labels, box=box, clear_old_points=True
            )
        return self._to_masks(ids, logits)

    def set_prompts(self, prompts):
        with torch.inference_mode(), self._autocast():
            self.predictor.reset_state(self.state)
            self.tracked = False
            result = {}
            for obj_id in sorted(prompts):
                for frame in sorted(prompts[obj_id]):
                    p = prompts[obj_id][frame]
                    if p.is_empty():
                        continue
                    masks = self._add(obj_id, frame, p)
                    result.setdefault(frame, {})[obj_id] = masks.get(obj_id)
            return result

    def can_update_in_place(self, obj_id):
        return obj_id in self.state["obj_id_to_idx"] or not self.tracked

    def update_prompt(self, obj_id, frame, prompt):
        with torch.inference_mode(), self._autocast():
            return self._add(obj_id, frame, prompt)

    def propagate(self, start, max_frames, reverse, should_stop):
        self.tracked = True
        with torch.inference_mode(), self._autocast():
            it = self.predictor.propagate_in_video(
                self.state, start_frame_idx=start, max_frame_num_to_track=max_frames, reverse=reverse
            )
            for frame_idx, ids, logits in it:
                yield frame_idx, self._to_masks(ids, logits)
                if should_stop():
                    break
