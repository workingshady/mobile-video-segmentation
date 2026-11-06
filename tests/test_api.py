"""HTTP API flow without any model loaded (no GPU needed): import → objects → prompts → brush → render → export."""

import base64
import time

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from editor import project as projmod  # noqa: E402
from editor import server  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(projmod, "PROJECTS", tmp_path / "projects")
    monkeypatch.setattr(server, "list_projects", projmod.list_projects)
    server.S.project = None
    server.S.tracker = None
    return TestClient(server.app)


def wait(client, job):
    for _ in range(200):
        j = client.get(f"/api/jobs/{job['id']}").json()
        if j["status"] != "running":
            return j
        time.sleep(0.05)
    raise TimeoutError


def video_bytes(tmp_path, n=6):
    path = tmp_path / "v.avi"
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for i in range(n):
        vw.write(np.full((48, 64, 3), 30 * i, np.uint8))
    vw.release()
    return path.read_bytes()


def test_full_flow_without_model(client, tmp_path):
    r = client.post("/api/projects/import", files={"file": ("v.avi", video_bytes(tmp_path))},
                    data={"name": "flow", "max_side": "0"})
    job = wait(client, r.json())
    assert job["status"] == "done", job
    p = client.get("/api/project").json()
    assert p["num_frames"] == 6

    o = client.post("/api/objects", json={}).json()
    r = client.post("/api/prompts", json={"obj": o["id"], "frame": 2, "points": [[10, 10]], "labels": [1]}).json()
    assert r["ran"] is False  # no model: prompt stored, reason returned
    assert client.get("/api/project").json()["prompts"][str(o["id"])]["2"]["points"] == [[10, 10]]

    mask = np.zeros((48, 64), np.uint8)
    mask[5:20, 5:20] = 255
    png = base64.b64encode(cv2.imencode(".png", cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA))[1]).decode()
    client.post("/api/prompts/mask", json={"obj": o["id"], "frame": 3, "png": "data:image/png;base64," + png})
    tl = client.get("/api/timeline?model=manual").json()
    assert tl["coverage"][str(o["id"])] == [[3, 3]]
    assert tl["keyframes"][str(o["id"])] == [2, 3]

    assert client.get("/api/overlay/manual/3.png").status_code == 200
    client.patch(f"/api/objects/{o['id']}", json={"effect": {"type": "blur", "strength": 4}})
    client.patch("/api/project", json={"background": {"mode": "grayscale"}, "in_out": [1, 4]})
    assert client.get("/api/render/3.jpg?model=manual").status_code == 200

    assert client.post("/api/undo").json()["ok"]
    assert "3" not in client.get("/api/project").json()["prompts"][str(o["id"])]

    job = wait(client, client.post("/api/export", json={"format": "png", "model": "manual"}).json())
    assert job["status"] == "done", job
    assert job["result"]["name"].startswith("flow")

    assert client.post("/api/propagate", json={}).status_code == 400  # no model loaded
    assert client.get("/api/models").json()["models"]


def test_magic_brush_stroke_is_one_undo_step(client, tmp_path):
    job = wait(client, client.post("/api/projects/import", files={"file": ("v.avi", video_bytes(tmp_path))},
                                   data={"name": "magic", "max_side": "0"}).json())
    assert job["status"] == "done"
    o = client.post("/api/objects", json={}).json()["id"]
    client.post("/api/prompts", json={"obj": o, "frame": 0, "points": [[5, 5]], "labels": [1]})
    # one stroke = first live update records undo, the rest don't
    stroke = [[10, 10], [14, 12], [18, 14], [22, 16]]
    for k in range(1, len(stroke) + 1):
        client.post("/api/prompts", json={"obj": o, "frame": 0, "points": [[5, 5]] + stroke[:k],
                                          "labels": [1] * (k + 1), "record_undo": k == 1})
    assert len(client.get("/api/project").json()["prompts"][str(o)]["0"]["points"]) == 5
    assert client.post("/api/undo").json()["ok"]
    assert client.get("/api/project").json()["prompts"][str(o)]["0"]["points"] == [[5, 5]]
