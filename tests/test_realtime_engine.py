"""End-to-end engine + HTTP API test without model weights.

Synthetic camera videos carry the frame and camera index in a pixel block; a
fake detector/pose model returns the true projected joints (+ noise) for that
frame. Everything else – decoding threads, tracking, async detection,
triangulation, Kalman/RTS, streaming, replay and export – is the real code.
"""

import json
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.realtime import engine as engine_module  # noqa: E402
from backend.realtime.synthetic import ring_calibration, running_motion  # noqa: E402
from backend.realtime.tracker import keypoint_box  # noqa: E402
from backend.realtime.triangulation import CameraRig  # noqa: E402

SIZE = (640, 360)
FPS = 30.0


def _write_tag(img, idx, cam):
    """Binary black/white blocks survive lossy video compression."""
    for bit in range(11):
        value = (idx >> bit) & 1 if bit < 8 else (cam >> (bit - 8)) & 1
        img[4:20, 4 + 20 * bit : 20 + 20 * bit] = 255 * value


def _read_tag(image):
    bits = [int(image[6:18, 6 + 20 * b : 18 + 20 * b].mean() > 127) for b in range(11)]
    idx = sum(v << b for b, v in enumerate(bits[:8]))
    cam = sum(v << b for b, v in enumerate(bits[8:]))
    return idx, cam


@pytest.fixture(scope="module")
def capture(tmp_path_factory):
    out = tmp_path_factory.mktemp("capture")
    cal = ring_calibration(4, size=SIZE, focal=420.0)
    rig = CameraRig.from_calibration(cal)
    times, gt = running_motion(2.0, FPS)
    uv = np.stack([rig.project(gt[t])[0] for t in range(len(gt))])
    for c, name in enumerate(rig.names):
        writer = cv2.VideoWriter(
            str(out / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE
        )
        assert writer.isOpened()
        for t in range(len(gt)):
            img = np.full((SIZE[1], SIZE[0], 3), 90, np.uint8)
            _write_tag(img, t, c)
            writer.write(img)
        writer.release()
    (out / "calibration.json").write_text(json.dumps(cal))
    return out, cal, rig, gt, uv


class FakeModel:
    provider = "fake"
    latency_ms = 0.0


class FakeDetector:
    def __init__(self, model, input_size):
        self.uv = FakeDetector.uv

    def __call__(self, image):
        idx, cam = _read_tag(image)
        box = keypoint_box(self.uv[min(idx, len(self.uv) - 1), cam])
        return np.array([[*box, 0.95]]) if box is not None else np.zeros((0, 5))


class FakePose:
    def __init__(self, model, skeleton, input_size, flip_test):
        self.skeleton = skeleton
        self.input_size = input_size
        self.rng = np.random.default_rng(0)

    def __call__(self, jobs):
        kps, scores = [], []
        for image, _box in jobs:
            idx, cam = _read_tag(image)
            kp = FakePose.uv[min(idx, len(FakePose.uv) - 1), cam] + self.rng.normal(
                0, 1.5, (17, 2)
            )
            kps.append(kp)
            scores.append(np.full(17, 0.9))
        return np.array(kps).reshape(-1, 17, 2), np.array(scores).reshape(-1, 17)


@pytest.fixture
def fake_models(monkeypatch, capture):
    _, _, _, _, uv = capture
    FakeDetector.uv = uv
    FakePose.uv = uv
    from backend.realtime.models import MODELS

    monkeypatch.setattr(
        engine_module,
        "load_models",
        lambda store, preset, provider, batch, log: (
            MODELS["yolox-m"],
            FakeModel(),
            MODELS["rtmpose-m"],
            FakeModel(),
        ),
    )
    monkeypatch.setattr(engine_module, "PersonDetector", FakeDetector)
    monkeypatch.setattr(engine_module, "PoseEstimator", FakePose)


def parse(stream_bytes):
    packets, buf = [], stream_bytes
    while len(buf) >= 4:
        total = struct.unpack(">I", buf[:4])[0]
        body = buf[4 : 4 + total]
        head_len = struct.unpack(">I", body[:4])[0]
        packets.append((json.loads(body[4 : 4 + head_len]), body[4 + head_len :]))
        buf = buf[4 + total :]
    return packets


def wait(client, rid, timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        s = client.get(f"/api/runs/{rid}").json()
        if s["status"] in ("finished", "stopped", "error"):
            return s
        time.sleep(0.2)
    raise AssertionError("run did not finish")


def test_engine_stream_replay_and_save(capture, fake_models, tmp_path):
    from starlette.testclient import TestClient

    from backend.realtime.models import ModelStore
    from backend.realtime.server import create_app

    folder, cal, rig, gt, _ = capture
    client = TestClient(create_app(ModelStore(tmp_path / "models")))
    assert client.get("/api/system").status_code == 200
    assert (
        client.get("/api/system", headers={"host": "evil.example"}).status_code == 403
    )

    body = {
        "calibration_path": str(folder / "calibration.json"),
        "videos": {n: str(folder / f"{n}.mp4") for n in rig.names},
        "options": {"name": "fake run", "det_interval": 10},
    }
    r = client.post("/api/runs", json=body)
    assert r.status_code == 200, r.text
    rid = r.json()["id"]
    packets = parse(client.get(f"/api/runs/{rid}/stream").content)
    status = wait(client, rid)
    assert status["status"] == "finished", status
    assert len(packets) == len(gt)
    header, jpegs = packets[-1]
    assert len(header["cams"]) == 4 and sum(header["jpeg_sizes"]) == len(jpegs)
    assert jpegs[:2] == b"\xff\xd8"

    run = client.app.state.live.runs[rid]
    n = len(gt)
    err_live = np.linalg.norm(run.rec["live"][:n] - gt, axis=-1)
    err_final = np.linalg.norm(run.final["smoothed"] - gt, axis=-1)
    assert np.isfinite(err_final).mean() > 0.97
    assert np.nanmean(err_final) * 1000 < 12  # mm
    assert np.nanmean(err_final) < np.nanmean(err_live)
    assert status["quality"]["median_reprojection_px_smoothed"] < 3

    replay = parse(client.get(f"/api/runs/{rid}/replay?variant=smoothed").content)
    assert len(replay) == n and replay[0][0]["variant"] == "smoothed"

    r = client.post(
        f"/api/runs/{rid}/save",
        json={
            "folder": str(tmp_path / "exports"),
            "options": {"csv": True, "trc": True, "videos": True},
        },
    )
    assert r.status_code == 200, r.text
    for _ in range(200):
        state = client.get(f"/api/runs/{rid}/save").json()
        if state["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert state["status"] == "done", state
    out = Path(state["folder"])
    names = {p.name for p in out.iterdir()}
    assert {
        "pose3d.json",
        "pose3d.csv",
        "pose3d.trc",
        "report.json",
        "pose2d_cam1.csv",
        "cam1_overlay.mp4",
    } <= names
    data = json.loads((out / "pose3d.json").read_text())
    assert len(data["frames"]) == n and data["units"] == "meters"
    assert client.post("/api/open", json={"path": "/etc"}).status_code == 403


def test_bad_inputs_are_explained(capture, fake_models, tmp_path):
    from starlette.testclient import TestClient

    from backend.realtime.models import ModelStore
    from backend.realtime.server import create_app

    folder, cal, rig, _, _ = capture
    client = TestClient(create_app(ModelStore(tmp_path / "models")))
    r = client.post(
        "/api/runs",
        json={
            "calibration_path": str(folder / "calibration.json"),
            "videos": {"cam1": str(folder / "cam1.mp4")},
        },
    )
    assert r.status_code == 200  # run starts, then reports the problem
    s = wait(client, r.json()["id"])
    assert s["status"] == "error" and "at least two" in s["error"]
    r = client.post(
        "/api/runs",
        json={"calibration_path": str(folder / "missing.json"), "videos": {}},
    )
    assert r.status_code == 400
