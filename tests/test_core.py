"""Tests for the parts of Track Studio that don't need a GPU or model weights."""

import json

import cv2
import numpy as np
import pytest

from editor import effects, exporter, project as projmod
from editor.trackers.registry import ModelRegistry


@pytest.fixture
def projects_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(projmod, "PROJECTS", tmp_path / "projects")
    return tmp_path


def make_video(path, n=12, w=64, h=48):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (w, h))
    for i in range(n):
        img = np.full((h, w, 3), 40, np.uint8)
        cv2.rectangle(img, (5 + i, 10), (25 + i, 30), (0, 0, 255), -1)
        vw.write(img)
    vw.release()
    return path


# ---------------- effects ----------------
def test_highlight_changes_only_object_pixels():
    frame = np.full((40, 40, 3), 100, np.uint8)
    labels = np.zeros((40, 40), np.uint8)
    labels[10:20, 10:20] = 1
    obj = {"id": 1, "color": "#ff0000", "visible": True, "foreground": True,
           "effect": {"type": "highlight", "opacity": 0.5, "feather": 0, "grow": 0, "outline": 1}}
    out = effects.render(frame, labels, [obj], {"mode": "keep"})
    assert out.shape == frame.shape
    assert (out[0, 0] == 100).all(), "background must be untouched"
    assert out[15, 15, 0] > out[15, 15, 1], "inside should be tinted red"


def test_remove_background_exports_alpha_matching_mask():
    frame = np.full((30, 30, 3), 80, np.uint8)
    labels = np.zeros((30, 30), np.uint8)
    labels[5:15, 5:15] = 2
    obj = {"id": 2, "color": "#00ff00", "visible": True, "foreground": True,
           "effect": {"type": "none", "feather": 0, "grow": 0}}
    out = effects.render(frame, labels, [obj], {"mode": "remove"}, preview=False)
    assert out.shape == (30, 30, 4)
    assert out[10, 10, 3] == 255 and out[25, 25, 3] == 0


@pytest.mark.parametrize("kind", ["blur", "pixelate", "recolor", "erase", "outline", "none"])
@pytest.mark.parametrize("bg", ["keep", "color", "blur", "grayscale", "remove"])
def test_every_effect_and_background_runs(kind, bg):
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (50, 70, 3), dtype=np.uint8)
    labels = np.zeros((50, 70), np.uint8)
    labels[10:30, 20:40] = 1
    obj = {"id": 1, "color": "#3fb8ff", "visible": True, "foreground": True,
           "effect": {"type": kind, "opacity": 0.6, "strength": 6, "feather": 1.5, "grow": 1}}
    out = effects.render(frame, labels, [obj], {"mode": bg, "color": "#000000", "strength": 5})
    assert out.shape == (50, 70, 3) and out.dtype == np.uint8


def test_grow_and_temporal_smoothing():
    m = np.zeros((20, 20), bool)
    m[8:12, 8:12] = True
    assert effects.soft_mask(m, grow=2).sum() > effects.soft_mask(m).sum()
    labels = m.astype(np.uint8)
    obj = {"id": 1, "visible": True, "effect": {"feather": 0, "grow": 0}}
    a = effects.object_alphas(labels, [obj], neighbors=[np.zeros_like(labels)])[1]
    assert a.max() == pytest.approx(0.5)


def test_label_map_resized_to_frame():
    frame = np.zeros((40, 60, 3), np.uint8)
    labels = np.ones((20, 30), np.uint8)
    obj = {"id": 1, "color": "#ffffff", "visible": True, "effect": {"type": "highlight", "feather": 0}}
    assert effects.render(frame, labels, [obj], {"mode": "keep"}).shape == (40, 60, 3)


# ---------------- project ----------------
def test_import_prompts_masks_coverage_undo(projects_dir, tmp_path):
    video = make_video(tmp_path / "clip.avi")
    pid = projmod.import_media(video, "My Clip", max_side=0)
    p = projmod.Project(pid)
    assert pid == "my-clip"
    assert p.n == 12 and p.meta["width"] == 64 and p.meta["height"] == 48

    o = p.add_object()
    p.set_prompt(o["id"], 3, points=[[10, 20]], labels=[1])
    prompts = p.all_prompts()
    assert prompts[o["id"]][3].points == [[10, 20]]
    assert p.keyframes() == {str(o["id"]): [3]}

    mask = np.zeros((48, 64), bool)
    mask[10:30, 10:30] = True
    for f in (2, 3, 4, 8):
        p.write_masks("m1", f, {o["id"]: mask})
    assert p.coverage("m1")[o["id"]] == [[2, 4], [8, 8]]
    assert p.rev_array("m1")[3] > 0 and p.rev_array("m1")[0] == 0

    p.set_prompt(o["id"], 5, mask=mask)
    assert p.all_prompts()[o["id"]][5].mask.sum() == mask.sum()
    assert p.undo()
    assert 5 not in p.all_prompts()[o["id"]]

    p.delete_object(o["id"])
    assert p.coverage("m1") == {}
    assert (p.read_labels("m1", 3) == 0).all()


def test_masks_of_two_objects_coexist(projects_dir, tmp_path):
    p = projmod.Project(projmod.import_media(make_video(tmp_path / "c.avi"), "c", max_side=0))
    a, b = p.add_object(), p.add_object()
    ma = np.zeros((48, 64), bool)
    ma[:, :20] = True
    mb = np.zeros((48, 64), bool)
    mb[:, 40:] = True
    p.write_masks("m", 0, {a["id"]: ma})
    p.write_masks("m", 0, {b["id"]: mb})
    lab = p.read_labels("m", 0)
    assert (lab[:, :20] == a["id"]).all() and (lab[:, 40:] == b["id"]).all()


def test_import_options_step_and_max_frames(projects_dir, tmp_path):
    p = projmod.Project(projmod.import_media(make_video(tmp_path / "c.avi", n=12), "c", max_side=32, step=2, max_frames=4))
    assert p.n == 4
    assert max(p.meta["width"], p.meta["height"]) <= 32
    assert p.meta["fps"] == pytest.approx(5.0)


def test_import_image(projects_dir, tmp_path):
    img = tmp_path / "pic.png"
    cv2.imwrite(str(img), np.full((30, 40, 3), 200, np.uint8))
    p = projmod.Project(projmod.import_media(img, "pic"))
    assert p.meta["kind"] == "image" and p.n == 1


# ---------------- export ----------------
def _project_with_mask(tmp_path):
    p = projmod.Project(projmod.import_media(make_video(tmp_path / "c.avi", n=5), "e", max_side=0))
    o = p.add_object()
    m = np.zeros((48, 64), bool)
    m[10:30, 10:30] = True
    for f in range(5):
        p.write_masks("m", f, {o["id"]: m})
    return p


@pytest.mark.parametrize("fmt", ["png", "png_alpha", "masks"])
def test_export_folders(projects_dir, tmp_path, fmt):
    p = _project_with_mask(tmp_path)
    out = exporter.export(p, "m", {"format": fmt}, None, lambda *_: None, lambda: False)
    files = list(out.rglob("*.png"))
    assert len(files) == 5
    if fmt == "png_alpha":
        assert cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED).shape[2] == 4


def test_export_mp4(projects_dir, tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    p = _project_with_mask(tmp_path)
    out = exporter.export(p, "m", {"format": "mp4", "audio": True, "resolution": "original"}, None,
                          lambda *_: None, lambda: False)
    assert out.exists() and out.stat().st_size > 0
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 5


# ---------------- registry ----------------
def test_registry_reports_every_model():
    reg = ModelRegistry()
    listing = reg.listing()
    ids = {m["id"] for m in listing}
    assert {"edgetam", "sam2.1-tiny", "efficienttam-s", "tinysam2"} <= ids
    by_id = {m["id"]: m for m in listing}
    assert by_id["tinysam2"]["status"] == "unavailable" and not by_id["tinysam2"]["runnable"]
    assert not by_id["edgetam-onnx"]["runnable"]


def test_catalog_is_consistent():
    from editor.paths import CATALOG

    models = json.loads(CATALOG.read_text(encoding="utf-8"))["models"]
    assert len({m["id"] for m in models}) == len(models)
    for m in models:
        if m["backend"] == "efficienttam":
            assert m["checkpoint"] and m["config"].endswith(".yaml")
