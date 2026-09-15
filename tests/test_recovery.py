from copy import deepcopy
from uuid import uuid4

import pytest

from backend.store import Store, atomic_json


def queued_job(module, store, session):
    job = {
        "id": uuid4().hex,
        "session_id": session["id"],
        "status": "queued",
        "frame_ids": [f["id"] for f in session["frames"]],
        "completed_frame_ids": [],
        "cancel_requested": False,
        "progress": 0,
        "message": "Queued",
        "created_at": "2026-09-15T12:00:00+00:00",
    }
    module.jobs[job["id"]] = job
    store.save_job(job)
    return job


class DeferredPool:
    def __init__(self):
        self.calls = []

    def submit(self, callback, *args):
        self.calls.append((callback, args))

    def run(self):
        callback, args = self.calls.pop(0)
        callback(*args)


def test_failed_frame_is_atomic_and_resume_preserves_labels(workspace, monkeypatch):
    client, store, session, module = workspace
    label = session["frames"][1]["views"]["cam1"]["raw"][10]["xy"][:]
    session["frames"][1]["views"]["cam1"]["edits"]["10"] = label
    store.save(session)
    before = deepcopy(session["frames"][1])
    candidate = deepcopy(session["frames"][0]["views"]["cam1"]["raw"])
    calls = []

    def detect(image):
        calls.append(1)
        if len(calls) == 5:
            raise RuntimeError("Model call failed")
        return [candidate]

    monkeypatch.setattr(module.detector, "detect", detect)
    monkeypatch.setattr(module.detector, "ready", lambda: True)
    job = queued_job(module, store, session)
    module.run_detection(session["id"], job["id"])
    saved = store.load(session["id"])
    assert saved["revision"] == session["revision"] + 1
    assert saved["frames"][1] == before
    assert job["status"] == "failed"
    assert job["completed_frame_ids"] == [100]
    discovered = client.get(f"/api/sessions/{session['id']}/job").json()
    assert discovered["can_resume"] is True
    assert discovered["remaining_frames"] == 2
    assert Store(store.root).recover_jobs()[job["id"]]["status"] == "failed"

    pool = DeferredPool()
    monkeypatch.setattr(module, "pool", pool)
    response = client.post(f"/api/jobs/{job['id']}/resume?revision={saved['revision']}")
    assert response.status_code == 200, response.text
    resumed = response.json()
    assert resumed["completed_frames"] == 1
    calls.clear()
    monkeypatch.setattr(
        module.detector, "detect", lambda image: calls.append(1) or [candidate]
    )
    pool.run()
    assert len(calls) == 6  # two unfinished frames, three cameras each
    assert client.get(f"/api/jobs/{resumed['id']}").json()["status"] == "complete"
    assert (
        store.load(session["id"])["frames"][1]["views"]["cam1"]["edits"]["10"] == label
    )
    assert client.post(f"/api/jobs/{job['id']}/resume").status_code == 409


def test_restart_reconciles_frame_saved_before_job_checkpoint(workspace, monkeypatch):
    client, store, session, module = workspace
    job = queued_job(module, store, session)
    job.update(status="running", message="Working")
    store.save_job(job)
    session["frames"][0]["inference"] = {
        "job_id": job["id"],
        "completed_at": "2026-09-15T12:00:01+00:00",
    }
    session["revision"] += 1
    store.save(session)
    fresh_store = Store(store.root)
    recovered = fresh_store.recover_jobs()
    assert recovered[job["id"]]["status"] == "interrupted"
    assert recovered[job["id"]]["completed_frame_ids"] == [100]
    monkeypatch.setattr(module, "store", fresh_store)
    monkeypatch.setattr(module, "jobs", recovered)
    job_response = client.get(f"/api/sessions/{session['id']}/jobs").json()[0]
    assert job_response["completed_frames"] == 1
    assert job_response["can_resume"] is True
    # Interrupted sessions are editable; they do not stay permanently locked.
    assert (
        client.patch(
            f"/api/sessions/{session['id']}",
            json={"name": "Recovered", "revision": session["revision"]},
        ).status_code
        == 200
    )


def test_restart_after_last_frame_marks_complete(workspace):
    _, store, session, module = workspace
    job = queued_job(module, store, session)
    job["status"] = "running"
    store.save_job(job)
    for frame in session["frames"]:
        frame["inference"] = {"job_id": job["id"]}
    store.save(session)
    recovered = Store(store.root).recover_jobs()[job["id"]]
    assert recovered["status"] == "complete"
    assert recovered["progress"] == 100


def test_cancel_during_last_camera_discards_entire_frame(workspace, monkeypatch):
    client, store, session, module = workspace
    original = store.load(session["id"])
    candidate = session["frames"][0]["views"]["cam1"]["raw"]
    job = queued_job(module, store, session)
    calls = []

    def detect(image):
        calls.append(1)
        if len(calls) == 3:
            response = module.cancel(job["id"])
            assert response["cancel_requested"] is True
        return [candidate]

    monkeypatch.setattr(module.detector, "detect", detect)
    module.run_detection(session["id"], job["id"])
    assert len(calls) == 3
    assert store.load(session["id"]) == original
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "cancelled"
    assert Store(store.root).recover_jobs()[job["id"]]["status"] == "cancelled"


def test_cancel_queued_job_never_runs_model(workspace, monkeypatch):
    _, store, session, module = workspace
    original = store.load(session["id"])
    job = queued_job(module, store, session)
    monkeypatch.setattr(
        module.detector,
        "detect",
        lambda image: pytest.fail("Cancelled job ran a model"),
    )
    assert module.cancel(job["id"])["status"] == "cancelled"
    module.run_detection(session["id"], job["id"])
    assert store.load(session["id"]) == original


def test_unexpected_revision_change_never_overwritten(workspace, monkeypatch):
    _, store, session, module = workspace
    job = queued_job(module, store, session)
    candidate = session["frames"][0]["views"]["cam1"]["raw"]
    changed = store.load(session["id"])
    changed["name"] = "External writer"
    changed["revision"] += 1

    def detect(image):
        store.save(changed)
        return [candidate]

    monkeypatch.setattr(module.detector, "detect", detect)
    module.run_detection(session["id"], job["id"])
    assert store.load(session["id"]) == changed
    assert job["status"] == "failed"
    assert job["completed_frame_ids"] == []
    assert "Session changed" in job["message"]


def test_archive_rename_restore_and_stale_mutations(workspace):
    client, store, session, module = workspace
    sid = session["id"]
    revision = session["revision"]
    assert (
        client.patch(
            f"/api/sessions/{sid}", json={"name": "  ", "revision": revision}
        ).status_code
        == 400
    )
    renamed = client.patch(
        f"/api/sessions/{sid}", json={"name": "  First take  ", "revision": revision}
    ).json()
    assert renamed["name"] == "First take"
    assert (
        client.post(f"/api/sessions/{sid}/archive?revision={revision}").status_code
        == 409
    )
    for route in ["undo", "detect"]:
        assert (
            client.post(f"/api/sessions/{sid}/{route}?revision={revision}").status_code
            == 409
        )
    assert (
        client.patch(
            f"/api/sessions/{sid}/settings",
            json=dict(session["settings"], revision=revision),
        ).status_code
        == 409
    )
    assert (
        client.post(
            f"/api/sessions/{sid}/actor",
            json={"frame_id": 100, "camera": "cam1", "index": 0, "revision": revision},
        ).status_code
        == 409
    )
    archived = client.post(
        f"/api/sessions/{sid}/archive?revision={renamed['revision']}"
    ).json()
    assert archived["archived"] is True
    assert client.get("/api/sessions").json() == []
    assert client.get("/api/sessions?archived=true").json()[0]["id"] == sid
    assert (store.path(sid) / session["frames"][0]["views"]["cam1"]["image"]).exists()
    restored = client.post(
        f"/api/sessions/{sid}/unarchive?revision={archived['revision']}"
    ).json()
    assert restored["archived"] is False
    job = queued_job(module, store, restored)
    assert client.post(f"/api/sessions/{sid}/archive").status_code == 409
    assert client.post(f"/api/sessions/{sid}/detect").status_code == 409
    assert job["status"] == "queued"


def test_atomic_json_keeps_previous_document_on_replace_failure(tmp_path, monkeypatch):
    import backend.store as module

    target = tmp_path / "session.json"
    atomic_json(target, {"revision": 1})

    def fail(*args):
        raise OSError("Disk fault")

    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError, match="Disk fault"):
        atomic_json(target, {"revision": 2})
    assert '"revision": 1' in target.read_text()
    assert list(tmp_path.iterdir()) == [target]


def project_bytes(session, store):
    import io
    import json
    import zipfile

    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("session.json", json.dumps(session))
        for image in {
            v["image"] for f in session["frames"] for v in f["views"].values()
        }:
            archive.write(store.path(session["id"]) / image, image)
    return data.getvalue()


def test_restore_preserves_ambiguous_person_candidates(workspace):
    client, store, session, _ = workspace
    view = session["frames"][0]["views"]["cam1"]
    candidates = [deepcopy(view["raw"]), deepcopy(view["raw"])]
    candidates[1][10]["xy"][0] += 5
    view["candidates"] = candidates
    view["actor_index"] = None
    view["raw"] = [{"xy": None, "confidence": 0.0} for _ in range(17)]
    store.reconstruct(session)
    content = client.get(f"/api/sessions/{session['id']}/project").content
    response = client.post("/api/restore", files={"project": ("project.zip", content)})
    assert response.status_code == 200, response.text
    restored = response.json()
    restored_view = restored["frames"][0]["views"]["cam1"]
    assert restored_view["candidates"] == candidates
    assert restored_view["actor_index"] is None
    selected = client.post(
        f"/api/sessions/{restored['id']}/actor",
        json={
            "frame_id": 100,
            "camera": "cam1",
            "index": 1,
            "revision": restored["revision"],
        },
    )
    assert selected.status_code == 200
    assert selected.json()["frames"][0]["views"]["cam1"]["raw"] == candidates[1]


@pytest.mark.parametrize("invalid", ["confidence", "coordinates", "actor_index"])
def test_restore_rejects_invalid_candidates(workspace, invalid):
    client, store, session, _ = workspace
    view = session["frames"][0]["views"]["cam1"]
    view["candidates"] = [deepcopy(view["raw"])]
    if invalid == "confidence":
        view["candidates"][0][0]["confidence"] = 1.2
    elif invalid == "coordinates":
        view["candidates"][0][0]["xy"] = [-1, 10]
    else:
        view["actor_index"] = 2
    response = client.post(
        "/api/restore",
        files={"project": ("project.zip", project_bytes(session, store))},
    )
    assert response.status_code == 400


@pytest.mark.parametrize(
    "metadata",
    [
        [],
        {
            "format": "mocap_web_session_v1",
            "joint_names": [
                "nose",
                "left_eye",
                "right_eye",
                "left_ear",
                "right_ear",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
            ],
        },
    ],
)
def test_restore_malformed_metadata_returns_user_error(workspace, metadata):
    import io
    import json
    import zipfile

    client, _, _, _ = workspace
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("session.json", json.dumps(metadata))
    assert (
        client.post(
            "/api/restore", files={"project": ("project.zip", data.getvalue())}
        ).status_code
        == 400
    )


def test_import_storage_limit_cleans_partial_session(workspace, monkeypatch):
    import json

    _, store, session, module = workspace
    image = (
        store.path(session["id"]) / session["frames"][0]["views"]["cam1"]["image"]
    ).read_bytes()
    files = {
        "calibration": [
            ("calibration.json", json.dumps(session["calibration"]).encode())
        ],
        "cam1": [("frame000.jpg", image)],
        "cam2": [("frame000.jpg", image)],
    }
    monkeypatch.setattr(module, "MAX_STORED_MEDIA_BYTES", 1)
    with pytest.raises(ValueError, match="450 MB"):
        module.import_capture({"name": "Too large"}, files)
    assert [p.name for p in store.root.iterdir() if p.is_dir()] == [session["id"]]


def test_graceful_shutdown_waits_for_worker_without_saving_partial_frame(
    workspace, monkeypatch
):
    from threading import Event
    from fastapi.testclient import TestClient

    _, store, session, module = workspace
    original = store.load(session["id"])
    entered, stopping = Event(), Event()
    save_job = store.save_job

    def observe_shutdown(job):
        save_job(job)
        if job.get("cancel_requested"):
            stopping.set()

    def detect(image):
        entered.set()
        assert stopping.wait(3), "Shutdown did not request cancellation"
        return [session["frames"][0]["views"]["cam1"]["raw"]]

    monkeypatch.setattr(store, "save_job", observe_shutdown)
    monkeypatch.setattr(module.detector, "ready", lambda: True)
    monkeypatch.setattr(module.detector, "detect", detect)
    with TestClient(module.app) as client:
        response = client.post(f"/api/sessions/{session['id']}/detect")
        assert response.status_code == 200
        job_id = response.json()["id"]
        assert entered.wait(3), "Inference worker did not start"
    assert module.jobs[job_id]["status"] == "cancelled"
    assert store.load(session["id"]) == original
