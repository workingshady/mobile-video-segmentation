"""Render the project with effects and write it out.

Formats:
  mp4         H.264 video with effects (+ original audio)
  webm_alpha  VP9 video with transparency (background mode 'remove')
  png_alpha   PNG sequence with transparency
  png         PNG sequence (no alpha)
  masks       black/white matte PNG per frame for every object (for other editors / training data)
"""

import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from .effects import render


def ffmpeg_exe():
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


class SourceReader:
    """Yields frames at original resolution matching the project's working frames."""

    def __init__(self, project, first):
        m = project.meta
        self.project = project
        self.kind = m["kind"]
        self.step = m["frame_step"]
        if self.kind == "image":
            img = cv2.imdecode(np.fromfile(m["source"], np.uint8), cv2.IMREAD_COLOR)
            self.image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            self.cap = cv2.VideoCapture(m["source"])
            if not self.cap.isOpened():
                raise FileNotFoundError(f"Original video not found: {m['source']}")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, m["start_frame"] + first * self.step)

    def next(self):
        if self.kind == "image":
            return self.image
        frame = None
        for k in range(self.step):
            ok, img = self.cap.read()
            if not ok:
                break
            if k == 0:
                frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return frame

    def close(self):
        if self.kind != "image":
            self.cap.release()


def export(project, model_id, opts, bg_image, progress, should_stop):
    """Returns the output path (file or folder)."""
    m = project.meta
    fmt = opts.get("format", "mp4")
    a, b = opts.get("range") or [0, m["num_frames"] - 1]
    a, b = max(0, int(a)), min(m["num_frames"] - 1, int(b))
    original = opts.get("resolution") == "original"
    smooth = m["settings"].get("temporal_smooth", False)
    objects = m["objects"]
    background = m["background"]
    out_dir = project.root / "exports"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = f"{project.id}_{model_id or 'nomodel'}_{stamp}"

    reader = SourceReader(project, a) if original else None
    proc = None
    folder = None
    if fmt in ("mp4", "webm_alpha"):
        first = reader.next() if reader else project.read_frame(a)
        h, w = first.shape[:2]
        w2, h2 = w - w % 2, h - h % 2
        alpha = fmt == "webm_alpha"
        out_path = out_dir / f"{base}.{'webm' if alpha else 'mp4'}"
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgba" if alpha else "rgb24",
               "-s", f"{w2}x{h2}", "-r", f"{m['fps']:.6f}", "-i", "-"]
        use_audio = opts.get("audio", True) and m["kind"] == "video"
        if use_audio:
            offset = (m["start_frame"] + a * m["frame_step"]) / m["source_fps"]
            cmd += ["-ss", f"{offset:.3f}", "-i", m["source"], "-map", "0:v", "-map", "1:a?", "-shortest"]
        if alpha:
            cmd += ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "30"]
            if use_audio:
                cmd += ["-c:a", "libopus"]
        else:
            cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(opts.get("crf", 18)), "-preset", "medium"]
            if use_audio:
                cmd += ["-c:a", "aac", "-b:a", "192k"]
        cmd.append(str(out_path))
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        pending_first = first
    else:
        folder = out_dir / base
        folder.mkdir()
        out_path = folder
        pending_first = None

    try:
        for i in range(a, b + 1):
            if should_stop():
                break
            frame = pending_first if pending_first is not None else (reader.next() if reader else project.read_frame(i))
            pending_first = None
            if frame is None:
                break
            labels = project.read_labels(model_id, i) if model_id else None
            neighbors = None
            if smooth and labels is not None:
                neighbors = [nb for nb in (project.read_labels(model_id, j) for j in (i - 1, i + 1)) if nb is not None]

            if fmt == "masks":
                h, w = frame.shape[:2]
                if labels is not None and labels.shape != (h, w):
                    labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
                for o in objects:
                    m_o = (labels == o["id"]) if labels is not None else np.zeros((h, w), bool)
                    d = folder / f"obj{o['id']}_{o['name'].replace(' ', '_')}"
                    d.mkdir(exist_ok=True)
                    cv2.imwrite(str(d / f"{i:05d}.png"), m_o.astype(np.uint8) * 255)
            else:
                want_alpha = fmt in ("webm_alpha", "png_alpha")
                bg = dict(background)
                if want_alpha:
                    bg["mode"] = "remove"
                out = render(frame, labels, objects, bg, bg_image, neighbors, preview=False)
                if want_alpha and out.shape[2] == 3:  # no foreground objects -> fully opaque
                    out = np.dstack([out, np.full(out.shape[:2], 255, np.uint8)])
                if proc:
                    out = out[: out.shape[0] - out.shape[0] % 2, : out.shape[1] - out.shape[1] % 2]
                    if fmt == "mp4" and out.shape[2] == 4:
                        out = out[..., :3]
                    proc.stdin.write(np.ascontiguousarray(out).tobytes())
                else:
                    if fmt == "png" and out.shape[2] == 4:
                        out = out[..., :3]
                    code = cv2.COLOR_RGBA2BGRA if out.shape[2] == 4 else cv2.COLOR_RGB2BGR
                    cv2.imwrite(str(folder / f"{i:05d}.png"), cv2.cvtColor(out, code))
            progress((i - a + 1) / (b - a + 1), f"Rendering frame {i - a + 1}/{b - a + 1}")
    finally:
        if reader:
            reader.close()
        if proc:
            proc.stdin.close()
            err = proc.stderr.read().decode(errors="replace")
            if proc.wait() != 0:
                raise RuntimeError(f"ffmpeg failed: {err[-800:]}")
    return out_path
