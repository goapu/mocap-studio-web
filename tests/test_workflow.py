import io
import json
import zipfile
import numpy as np
import pytest
from backend.calibration import read_calibration, calibrate
from backend.app import run_detection


def test_edit_save_reload_undo_keeps_raw_predictions(workspace):
    client, store, s, _ = workspace
    fid = s["frames"][0]["id"]
    view = s["frames"][0]["views"]["cam2"]
    original = view["raw"][10]["xy"][:]
    changed = [original[0] + 2, original[1] + 1]
    response = client.patch(
        f"/api/sessions/{s['id']}/joint",
        json={
            "frame_id": fid,
            "camera": "cam2",
            "joint": 10,
            "value": changed,
            "revision": s["revision"],
        },
    )
    assert response.status_code == 200, response.text
    saved = client.get(f"/api/sessions/{s['id']}").json()
    v = saved["frames"][0]["views"]["cam2"]
    assert v["raw"][10]["xy"] == original
    assert v["edits"]["10"] == changed
    assert saved["frames"][0]["pose"][10]["status"] == "corrected"
    undone = client.post(f"/api/sessions/{s['id']}/undo").json()
    assert "10" not in undone["frames"][0]["views"]["cam2"]["edits"]
    assert undone["frames"][0]["views"]["cam2"]["effective"][10]["xy"] == original


def test_stale_revision_rejected(workspace):
    client, _, s, _ = workspace
    r = client.patch(
        f"/api/sessions/{s['id']}/joint",
        json={
            "frame_id": 100,
            "camera": "cam1",
            "joint": 10,
            "value": [10, 20],
            "revision": s["revision"] - 1,
        },
    )
    assert r.status_code == 409


def test_missing_labels_preserve_nulls_and_gapped_timeline(workspace):
    client, store, s, _ = workspace
    for frame in s["frames"]:
        for v in frame["views"].values():
            v["edits"]["10"] = None
    store.reconstruct(s)
    export = client.get(f"/api/sessions/{s['id']}/export").json()
    assert [f["frame"] for f in export["frames"]] == [100, 101, 104]
    assert [f["timestamp_seconds"] for f in export["frames"]] == [0, 1 / 30, 4 / 30]
    assert all(f["joints"]["right_wrist"] is None for f in export["frames"])
    assert "NaN" not in json.dumps(export)


def test_archive_roundtrip_preserves_images_labels_and_timestamps(workspace):
    client, store, s, _ = workspace
    s["frames"][0]["views"]["cam2"]["edits"]["10"] = s["frames"][0]["views"]["cam2"][
        "raw"
    ][10]["xy"][:]
    store.reconstruct(s)
    archive = client.get(f"/api/sessions/{s['id']}/project").content
    response = client.post(
        "/api/restore", files={"project": ("project.zip", archive, "application/zip")}
    )
    assert response.status_code == 200, response.text
    restored = response.json()
    assert restored["id"] != s["id"]
    assert (
        restored["frames"][0]["views"]["cam2"]["edits"]
        == s["frames"][0]["views"]["cam2"]["edits"]
    )
    assert [f["timestamp"] for f in restored["frames"]] == [0, 1 / 30, 4 / 30]
    image = restored["frames"][0]["views"]["cam1"]["image"]
    assert (
        client.get(f"/api/sessions/{restored['id']}/media/{image}").status_code == 200
    )


def test_archive_path_traversal_rejected(workspace):
    client, _, _, _ = workspace
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        z.writestr("../escaped.txt", "bad")
    assert (
        client.post(
            "/api/restore", files={"project": ("bad.zip", stream.getvalue())}
        ).status_code
        == 400
    )


def test_detection_rerun_preserves_labels(workspace, monkeypatch):
    _, store, s, module = workspace
    label = s["frames"][0]["views"]["cam1"]["raw"][10]["xy"][:]
    s["frames"][0]["views"]["cam1"]["edits"]["10"] = label
    store.save(s)
    candidate = s["frames"][0]["views"]["cam1"]["raw"]
    monkeypatch.setattr(module.detector, "detect", lambda image: [candidate])
    module.jobs["test"] = {"id": "test", "session_id": s["id"]}
    run_detection(s["id"], "test", 100)
    assert module.jobs["test"]["status"] == "complete"
    assert store.load(s["id"])["frames"][0]["views"]["cam1"]["edits"]["10"] == label


def test_ambiguous_people_require_selection(workspace, monkeypatch):
    _, store, s, module = workspace
    candidate = s["frames"][0]["views"]["cam1"]["raw"]
    monkeypatch.setattr(module.detector, "detect", lambda image: [candidate, candidate])
    module.jobs["test"] = {"id": "test", "session_id": s["id"]}
    run_detection(s["id"], "test", 100)
    view = store.load(s["id"])["frames"][0]["views"]["cam1"]
    assert view["actor_index"] is None
    assert all(o["xy"] is None for o in view["raw"])


def test_failed_calibration_not_promoted(monkeypatch):
    from backend import calibration as module

    images = {
        f"cam{i}": {f: np.zeros((480, 640, 3), np.uint8) for f in range(12)}
        for i in [1, 2]
    }
    K = np.array([[700.0, 0, 320], [0, 700, 240], [0, 0, 1]])
    monkeypatch.setattr(
        module.cv2,
        "findChessboardCornersSB",
        lambda *a, **k: (True, np.zeros((54, 1, 2), np.float32)),
    )
    monkeypatch.setattr(
        module.cv2, "calibrateCamera", lambda *a, **k: (0.1, K, np.zeros(5), [], [])
    )
    monkeypatch.setattr(
        module.cv2,
        "stereoCalibrate",
        lambda *a, **k: (
            9.0,
            K,
            np.zeros(5),
            K,
            np.zeros(5),
            np.eye(3),
            np.array([[-0.7], [0], [0]]),
            None,
            None,
        ),
    )
    with pytest.raises(ValueError, match="quality gate"):
        calibrate(images)


def test_conflicting_npz_formats_rejected(calibration):
    payload = {
        "camera_names": np.array(["cam1", "cam2"]),
        "R": np.eye(3),
        "T": np.array([[-0.7], [0], [0]]),
    }
    for name in ["cam1", "cam2"]:
        for k, v in calibration["cameras"][name].items():
            payload[f"{name}_{k}"] = np.array(v)
    payload["cam2_T"] = np.array([-1.5, 0, 0])
    stream = io.BytesIO()
    np.savez(stream, **payload)
    with pytest.raises(ValueError, match="conflicting"):
        read_calibration(stream.getvalue(), "calibration.npz")


def test_non_local_origin_blocked(workspace):
    client, _, _, _ = workspace
    assert (
        client.post(
            "/api/demo", headers={"Origin": "https://unrelated.example"}
        ).status_code
        == 403
    )
