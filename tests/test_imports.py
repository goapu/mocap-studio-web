"""Exercise media import using real image/video codecs and malformed calibration."""

import io
import json

import cv2
import numpy as np
import pytest

from backend.calibration import read_calibration


def capture_files(calibration, image_shape=(480, 640, 3)):
    ok, image = cv2.imencode(".jpg", np.zeros(image_shape, np.uint8))
    assert ok
    return [
        ("calibration", ("rig.json", json.dumps(calibration), "application/json"))
    ] + [
        (camera, (f"frame{fid}.jpg", image.tobytes(), "image/jpeg"))
        for camera in ["cam1", "cam2"]
        for fid in [100, 104]
    ]


def test_image_import_preserves_source_gaps(workspace, calibration):
    client, _, _, _ = workspace
    response = client.post(
        "/api/sessions",
        data={"fps": "25", "name": "Image test"},
        files=capture_files(calibration),
    )
    assert response.status_code == 200, response.text
    s = response.json()
    assert [f["id"] for f in s["frames"]] == [100, 104]
    assert [f["timestamp"] for f in s["frames"]] == [0, 0.16]
    assert s["capture"]["source_frame_offset"] == 100
    assert all(j["point"] is None for f in s["frames"] for j in f["pose"])


def test_wrong_resolution_rejects_and_removes_partial_import(workspace, calibration):
    client, store, _, _ = workspace
    before = set(store.root.iterdir())
    response = client.post(
        "/api/sessions", files=capture_files(calibration, (240, 320, 3))
    )
    assert response.status_code == 400
    assert "matching resolution" in response.json()["detail"]
    assert set(store.root.iterdir()) == before


def make_video(tmp_path, fps):
    path = tmp_path / f"video{fps}.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (640, 480)
    )
    assert writer.isOpened()
    for i in range(7):
        writer.write(np.full((480, 640, 3), i * 20, np.uint8))
    writer.release()
    return path.read_bytes()


def test_video_stride_keeps_original_timestamps(workspace, calibration, tmp_path):
    client, _, _, _ = workspace
    video = make_video(tmp_path, 25)
    files = [("calibration", ("rig.json", json.dumps(calibration)))] + [
        (cam, ("capture.avi", video, "video/x-msvideo")) for cam in ["cam1", "cam2"]
    ]
    response = client.post("/api/sessions", data={"stride": "3"}, files=files)
    assert response.status_code == 200, response.text
    s = response.json()
    assert s["fps"] == 25
    assert [f["id"] for f in s["frames"]] == [0, 3, 6]
    assert [f["timestamp"] for f in s["frames"]] == [0, 0.12, 0.24]


def test_mismatched_video_rates_rejected(workspace, calibration, tmp_path):
    client, _, _, _ = workspace
    files = [("calibration", ("rig.json", json.dumps(calibration)))] + [
        (cam, ("capture.avi", make_video(tmp_path, fps)))
        for cam, fps in [("cam1", 25), ("cam2", 30)]
    ]
    response = client.post("/api/sessions", files=files)
    assert response.status_code == 400
    assert "matching constant FPS" in response.json()["detail"]


def test_incomplete_npz_reports_actionable_error():
    stream = io.BytesIO()
    np.savez(stream, camera_names=np.array(["cam1", "cam2"]))
    with pytest.raises(ValueError, match="Incomplete NPZ"):
        read_calibration(stream.getvalue(), "rig.npz")
