"""Optional real-model integration; supply a local public/person test image.

Identical images in two virtual cameras deliberately do not constitute a real
stereo capture. This test validates inference/selection/storage, not 3D accuracy.
"""

import json
import os
from pathlib import Path
import time

import cv2
import pytest

from backend.detector import Detector


@pytest.mark.skipif(
    not os.getenv("MOCAP_TEST_IMAGE"),
    reason="Set MOCAP_TEST_IMAGE for the real ONNX integration test",
)
def test_real_model_import_job_actor_and_backup(workspace, calibration, monkeypatch):
    client, _, _, module = workspace
    image_path = Path(os.environ["MOCAP_TEST_IMAGE"])
    image = cv2.imread(str(image_path))
    assert image is not None
    height, width = image.shape[:2]
    for camera in calibration["cameras"].values():
        camera["image_size"] = [width, height]
        camera["K"] = [[700.0, 0, width / 2], [0, 700.0, height / 2], [0, 0, 1]]
        camera["dist"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    calibration["cameras"].pop("cam3")
    detector = Detector(module.MODEL_DIR)
    assert detector.ready(), "Install the published model weights first"
    monkeypatch.setattr(module, "detector", detector)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    response = client.post(
        "/api/sessions",
        data={"name": "Inference integration fixture"},
        files=[
            ("calibration", ("rig.json", json.dumps(calibration))),
            ("cam1", ("frame000.jpg", encoded.tobytes())),
            ("cam2", ("frame000.jpg", encoded.tobytes())),
        ],
    )
    assert response.status_code == 200, response.text
    s = response.json()
    response = client.post(
        f"/api/sessions/{s['id']}/detect", params={"revision": s["revision"]}
    )
    assert response.status_code == 200, response.text
    job = response.json()
    deadline = time.monotonic() + 30
    while job["status"] in ["queued", "running"] and time.monotonic() < deadline:
        time.sleep(0.05)
        job = client.get(f"/api/jobs/{job['id']}").json()
    assert job["status"] == "complete", job
    assert job["completed_frames"] == 1
    s = client.get(f"/api/sessions/{s['id']}").json()
    assert len(s["frames"][0]["views"]["cam1"]["candidates"]) >= 1
    for name in ["cam1", "cam2"]:
        response = client.post(
            f"/api/sessions/{s['id']}/actor",
            json={"frame_id": 0, "camera": name, "index": 0, "revision": s["revision"]},
        )
        assert response.status_code == 200, response.text
        s = response.json()
        raw = s["frames"][0]["views"][name]["raw"]
        assert len(raw) == 17
        assert any(j["confidence"] > 0.35 for j in raw)
    # The same image in both cameras has no useful stereo parallax.
    assert all(j["point"] is None for j in s["frames"][0]["pose"])
    archive = client.get(f"/api/sessions/{s['id']}/project").content
    response = client.post("/api/restore", files={"project": ("project.zip", archive)})
    assert response.status_code == 200, response.text
    restored = response.json()
    for name in ["cam1", "cam2"]:
        assert (
            restored["frames"][0]["views"][name]["candidates"]
            == s["frames"][0]["views"][name]["candidates"]
        )
        assert restored["frames"][0]["views"][name]["actor_index"] == 0
