"""Common interface every model adapter implements.

The editor treats the project's prompts as the source of truth and a tracker
session as a cache: whenever prompts change in a way the session can't absorb
(deleting a prompt, adding a new object after tracking), the session is reset
and every prompt is replayed. That keeps all adapters simple and identical in
behavior, which matters because the whole point is comparing models fairly.
"""

from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np


class TrackerUnavailable(RuntimeError):
    """Raised when a model can't run here (missing weights, package, or access)."""


@dataclass
class Prompt:
    """Prompts for one object on one frame. Either points/box, or a brush mask."""

    points: list = field(default_factory=list)  # [[x, y], ...] in working-frame pixels
    labels: list = field(default_factory=list)  # 1 = object, 0 = not object
    box: list | None = None  # [x1, y1, x2, y2]
    mask: np.ndarray | None = None  # bool HxW, replaces points/box when set

    def is_empty(self):
        return not self.points and self.box is None and self.mask is None


# frame_idx -> {obj_id: bool mask HxW}
FrameMasks = dict[int, np.ndarray]


class Tracker:
    id: str
    name: str

    def load(self, device: str) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        raise NotImplementedError

    def init_video(self, frames_dir: str, frames: list, height: int, width: int) -> None:
        """Prepare a session for the project's working frames."""
        raise NotImplementedError

    def close_video(self) -> None:
        raise NotImplementedError

    def set_prompts(self, prompts: dict[int, dict[int, Prompt]]) -> dict[int, dict[int, np.ndarray]]:
        """Reset the session and replay all prompts {obj_id: {frame: Prompt}}.

        Returns {frame: {obj_id: mask}} for every prompted frame.
        """
        raise NotImplementedError

    def update_prompt(self, obj_id: int, frame: int, prompt: Prompt) -> dict[int, np.ndarray]:
        """Fast path: add/replace one object's prompt on one frame. Returns {obj_id: mask} for that frame."""
        raise NotImplementedError

    def can_update_in_place(self, obj_id: int) -> bool:
        return False

    def propagate(
        self, start: int, max_frames: int | None, reverse: bool, should_stop: Callable[[], bool]
    ) -> Iterator[tuple[int, dict[int, np.ndarray]]]:
        """Yield (frame_idx, {obj_id: mask}) for each tracked frame."""
        raise NotImplementedError
