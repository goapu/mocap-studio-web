from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from uuid import uuid4
import zipfile

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .calibration import calibrate, read_calibration, validate
from .demo import create_demo
from .detector import Detector
from .geometry import JOINTS, effective_observations, reconstruct_frame
from .store import Store

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.getenv("MOCAP_DATA_DIR", ROOT / ".local" / "sessions"))
MODEL_DIR = Path(os.getenv("MOCAP_MODEL_DIR", ROOT / ".local" / "models"))
MAX_STORED_MEDIA_BYTES = 450 * 1024 * 1024
store = Store(DATA)
detector = Detector(MODEL_DIR)
pool = ThreadPoolExecutor(max_workers=1)
jobs = store.recover_jobs()


@asynccontextmanager
async def lifespan(app):
    global pool, jobs
    # One application process owns the inference queue and its session directory.
    pool = ThreadPoolExecutor(max_workers=1)
    jobs = store.recover_jobs()
    try:
        yield
    finally:
        with store.lock:
            for job in jobs.values():
                if job["status"] == "queued":
                    job.update(
                        status="cancelled",
                        cancel_requested=False,
                        message="Server stopped before inference started. Resume to continue.",
                    )
                    store.save_job(job)
                elif job["status"] == "running":
                    job.update(
                        cancel_requested=True,
                        message="Server stopping. Finishing the current model call without saving the unfinished frame.",
                    )
                    store.save_job(job)
        # Join the current model call so a worker cannot keep writing after exit.
        await run_in_threadpool(pool.shutdown, wait=True, cancel_futures=True)


app = FastAPI(title="Mocap Studio", version="0.2.0", lifespan=lifespan)


@app.exception_handler(ValueError)
async def value_error(request, exc):
    return JSONResponse({"detail": str(exc)}, status_code=400)


@app.middleware("http")
async def local_origin(request, call_next):
    from urllib.parse import urlparse

    origin = request.headers.get("origin")
    # Validate Host as well as Origin: blocks DNS-rebinding pages that make
    # same-origin requests (which carry no Origin header) to this server.
    host = urlparse(f"//{request.headers.get('host', '')}").hostname
    if host not in ["localhost", "127.0.0.1", "::1", "testserver"]:
        return JSONResponse({"detail": "Local access only."}, status_code=403)
    if origin and urlparse(origin).hostname not in ["localhost", "127.0.0.1", "::1"]:
        return JSONResponse(
            {"detail": "This workstation accepts local browser requests only."},
            status_code=403,
        )
    return await call_next(request)


def busy(sid):
    return any(
        j["session_id"] == sid and j["status"] in ["queued", "running"]
        for j in jobs.values()
    )


def job_response(job):
    response = deepcopy(job)
    response["total_frames"] = len(job.get("frame_ids", []))
    response["completed_frames"] = len(job.get("completed_frame_ids", []))
    response["remaining_frames"] = (
        response["total_frames"] - response["completed_frames"]
    )
    response["can_resume"] = (
        job["status"] in ["failed", "cancelled", "interrupted"]
        and response["remaining_frames"] > 0
        and not job.get("resumed_by")
    )
    return response


def session_jobs(sid):
    store.load(sid)
    return sorted(
        (j for j in jobs.values() if j["session_id"] == sid),
        key=lambda j: j.get("created_at", ""),
        reverse=True,
    )


@app.get("/api/sessions/{sid}/jobs")
def get_session_jobs(sid: str):
    with store.lock:
        return [job_response(j) for j in session_jobs(sid)]


@app.get("/api/sessions/{sid}/job")
def get_session_job(sid: str):
    with store.lock:
        matches = session_jobs(sid)
        return job_response(matches[0]) if matches else None


def editable(sid, revision=None):
    if busy(sid):
        raise HTTPException(
            409,
            "Detection is running. Wait for it to finish or cancel it before editing.",
        )
    session = store.load(sid)
    if revision is not None and session["revision"] != revision:
        raise HTTPException(
            409, "This session changed in another tab. Reload before editing."
        )
    return session


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "model_ready": detector.ready(),
        "model": "RTMPose-M + YOLOX-M",
        "device": "CPU",
        "version": "0.2.0",
    }


@app.get("/api/sessions")
def sessions(archived: bool = False):
    with store.lock:
        records = [
            json.loads(p.read_text())
            for p in sorted(
                DATA.glob("*/session.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        ]
        return [
            store.summarize(s)
            for s in records
            if bool(s.get("archived", False)) == archived
        ]


class SessionName(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    revision: int


@app.patch("/api/sessions/{sid}")
def rename_session(sid: str, update: SessionName):
    with store.lock:
        record = editable(sid, update.revision)
        name = update.name.strip()
        if not name:
            raise ValueError("Session name cannot be blank.")
        record["name"] = name
        record["revision"] += 1
        store.save(record)
        return record


def set_archived(sid, archived, revision):
    with store.lock:
        record = editable(sid, revision)
        record["archived"] = archived
        record["revision"] += 1
        store.save(record)
        return record


@app.post("/api/sessions/{sid}/archive")
def archive_session(sid: str, revision: int | None = None):
    return set_archived(sid, True, revision)


@app.post("/api/sessions/{sid}/unarchive")
def unarchive_session(sid: str, revision: int | None = None):
    return set_archived(sid, False, revision)


@app.post("/api/demo")
def demo():
    with store.lock:
        return create_demo(store)


@app.get("/api/sessions/{sid}")
def session(sid: str):
    with store.lock:
        return store.load(sid)


@app.get("/api/sessions/{sid}/media/{filename}")
def media(sid: str, filename: str):
    if not re.fullmatch(r"cam[1-6]_\d+\.jpg", filename):
        raise HTTPException(404)
    path = store.path(sid) / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


async def uploads(request):
    form = await request.form(
        max_files=2000, max_fields=30, max_part_size=10 * 1024 * 1024
    )
    files, total = {}, 0
    for key, value in form.multi_items():
        if hasattr(value, "filename"):
            content = await value.read(500 * 1024 * 1024 + 1)
            total += len(content)
            if total > 500 * 1024 * 1024:
                raise ValueError("Upload limit is 500 MB per session.")
            files.setdefault(key, []).append((Path(value.filename).name, content))
    return form, files


def frame_number(filename):
    stem = Path(filename).stem
    match = re.search(r"(?:frame|img|image)[_-]?(\d+)", stem, re.I) or re.search(
        r"(\d+)", re.sub(r"cam[1-6]", "", stem, flags=re.I)
    )
    if not match:
        raise ValueError(
            f"{filename}: image filenames need a shared frame number, e.g. frame001.jpg."
        )
    return int(match.group(1))


def decode_images(files):
    result = {}
    for filename, content in files:
        fid = frame_number(filename)
        if fid in result:
            raise ValueError(f"Duplicate frame number {fid}.")
        image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode {filename}.")
        result[fid] = image
    return result


@app.post("/api/calibrate")
async def build_calibration(request: Request):
    form, files = await uploads(request)
    cameras = {
        k: decode_images(v) for k, v in files.items() if re.fullmatch(r"cam[1-6]", k)
    }
    if not 2 <= len(cameras) <= 6:
        raise ValueError("Upload paired board images for at least two cameras.")
    return await run_in_threadpool(
        calibrate,
        cameras,
        int(form.get("cols", 9)),
        int(form.get("rows", 6)),
        float(form.get("square", 0.025)),
    )


def import_capture(form_values, files):
    calibration_files = files.get("calibration", [])
    if len(calibration_files) != 1:
        raise ValueError(
            "A calibration JSON or NPZ is required for metric 3D reconstruction."
        )
    filename, content = calibration_files[0]
    calibration = read_calibration(content, filename)
    fps = float(form_values.get("fps", 30))
    stride = int(form_values.get("stride", 1))
    if not 1 <= fps <= 240 or not 1 <= stride <= 60:
        raise ValueError("FPS must be 1–240; frame stride must be 1–60.")
    cameras = {k: v for k, v in files.items() if re.fullmatch(r"cam[1-6]", k) and v}
    if not 2 <= len(cameras) <= 6 or any(
        k not in calibration["cameras"] for k in cameras
    ):
        raise ValueError("Provide at least two cameras that exist in the calibration.")
    kinds = {
        name: len(items) == 1
        and Path(items[0][0]).suffix.lower() in [".mp4", ".mov", ".avi", ".mkv", ".m4v"]
        for name, items in cameras.items()
    }
    if len(set(kinds.values())) != 1:
        raise ValueError("Use all videos or all image sequences in one session.")
    session = store.create(
        str(form_values.get("name") or "Untitled capture"), calibration, fps
    )
    folder = store.path(session["id"])
    groups, actual_fps = {}, []
    stored_media_bytes = 0
    try:
        for name, items in cameras.items():
            size = calibration["cameras"][name]["image_size"]

            def save_frame(fid, image):
                nonlocal stored_media_bytes
                if image.shape[1::-1] != tuple(size):
                    raise ValueError(
                        f"{name}: capture is {image.shape[1]}×{image.shape[0]}, calibration is {size[0]}×{size[1]}. Use matching resolution."
                    )
                if len(groups) >= 300 and fid not in groups:
                    raise ValueError(
                        "Session limit: 300 sampled frames. Use a shorter clip or larger frame stride."
                    )
                target = f"{name}_{fid:06d}.jpg"
                if not cv2.imwrite(
                    str(folder / target), image, [cv2.IMWRITE_JPEG_QUALITY, 93]
                ):
                    raise ValueError(
                        f"{name}: unable to save frame {fid}. Check available disk space."
                    )
                stored_media_bytes += (folder / target).stat().st_size
                if stored_media_bytes > MAX_STORED_MEDIA_BYTES:
                    raise ValueError(
                        "Decoded camera images exceed the 450 MB session storage limit. "
                        "Use a shorter clip or larger video frame stride so the project remains portable."
                    )
                groups.setdefault(fid, {})[name] = {
                    "image": target,
                    "width": size[0],
                    "height": size[1],
                    "raw": [{"xy": None, "confidence": 0.0} for _ in JOINTS],
                    "edits": {},
                    "candidates": [],
                    "actor_index": None,
                }

            if kinds[name]:
                with tempfile.NamedTemporaryFile(
                    suffix=Path(items[0][0]).suffix, dir=folder
                ) as tmp:
                    tmp.write(items[0][1])
                    tmp.flush()
                    cap = cv2.VideoCapture(tmp.name)
                    try:
                        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
                        if (
                            not cap.isOpened()
                            or not np.isfinite(source_fps)
                            or source_fps <= 0
                        ):
                            raise ValueError(
                                f"{name}: video cannot be read or has invalid FPS."
                            )
                        actual_fps.append(source_fps)
                        fid = 0
                        while True:
                            ok, image = cap.read()
                            if not ok:
                                break
                            if fid % stride == 0:
                                save_frame(fid, image)
                            fid += 1
                    finally:
                        cap.release()
            else:
                seen = set()
                for item in items:
                    # Decode one image at a time, retaining only compressed uploads.
                    for fid, image in decode_images([item]).items():
                        if fid in seen:
                            raise ValueError(f"Duplicate frame number {fid}.")
                        seen.add(fid)
                        # Image IDs are timestamps on the source FPS timeline.
                        save_frame(fid, image)
        if actual_fps:
            fps = actual_fps[0]
            if any(abs(fps - f) > 0.01 for f in actual_fps):
                raise ValueError(
                    "Camera videos must have matching constant FPS and aligned start times."
                )
            session["fps"] = fps
        if not groups or not any(len(v) >= 2 for v in groups.values()):
            raise ValueError("No shared frame numbers between cameras.")
        first = min(groups)
        session["frames"] = [
            {"id": fid, "timestamp": (fid - first) / fps, "views": views}
            for fid, views in sorted(groups.items())
        ]
        session["capture"] = {
            "sync_assumption": "User supplies synchronized constant-FPS cameras with aligned frame zero.",
            "stride": stride if actual_fps else 1,
            "source_frame_offset": first,
        }
        return store.reconstruct(session)
    except Exception:
        shutil.rmtree(folder)
        raise


@app.post("/api/sessions")
async def import_session(request: Request):
    form, files = await uploads(request)
    values = {k: v for k, v in form.items() if not hasattr(v, "filename")}
    return await run_in_threadpool(import_capture, values, files)


class Edit(BaseModel):
    frame_id: int
    camera: str
    joint: int = Field(ge=0, le=16)
    value: tuple[float, float] | None = None
    reset: bool = False
    revision: int


@app.patch("/api/sessions/{sid}/joint")
def edit_joint(sid: str, edit: Edit):
    with store.lock:
        session = editable(sid, edit.revision)
        frame = next((f for f in session["frames"] if f["id"] == edit.frame_id), None)
        if frame is None or edit.camera not in frame["views"]:
            raise ValueError("Frame/camera not found.")
        view = frame["views"][edit.camera]
        if edit.value is not None and (
            not np.isfinite(edit.value).all()
            or not 0 <= edit.value[0] < view["width"]
            or not 0 <= edit.value[1] < view["height"]
        ):
            raise ValueError("Joint must be inside the camera image.")
        key = str(edit.joint)
        session["history"].append(
            {
                "frame_id": edit.frame_id,
                "camera": edit.camera,
                "joint": key,
                "previous_present": key in view["edits"],
                "previous": view["edits"].get(key),
                "value": edit.value,
                "reset": edit.reset,
                "at": datetime.now(timezone.utc).isoformat(),
            }
        )
        if edit.reset:
            view["edits"].pop(key, None)
        else:
            view["edits"][key] = edit.value
        reconstruct_frame(frame, session["calibration"], session["settings"])
        session["revision"] += 1
        store.save(session)
        return session


@app.post("/api/sessions/{sid}/undo")
def undo(sid: str, revision: int | None = None):
    with store.lock:
        session = editable(sid, revision)
        if session["history"]:
            h = session["history"].pop()
            frame = next(f for f in session["frames"] if f["id"] == h["frame_id"])
            edits = frame["views"][h["camera"]]["edits"]
            if h["previous_present"]:
                edits[h["joint"]] = h["previous"]
            else:
                edits.pop(h["joint"], None)
            reconstruct_frame(frame, session["calibration"], session["settings"])
            session["revision"] += 1
            store.save(session)
        return session


class Actor(BaseModel):
    frame_id: int
    camera: str
    index: int = Field(ge=0)
    revision: int | None = None


@app.post("/api/sessions/{sid}/actor")
def choose_actor(sid: str, choice: Actor):
    with store.lock:
        session = editable(sid, choice.revision)
        frame = next((f for f in session["frames"] if f["id"] == choice.frame_id), None)
        if not frame or choice.camera not in frame["views"]:
            raise ValueError("Camera/frame not found.")
        view = frame["views"][choice.camera]
        if choice.index >= len(view["candidates"]):
            raise ValueError("Person index is invalid.")
        view.update(
            actor_index=choice.index, raw=view["candidates"][choice.index], edits={}
        )
        session["history"] = [
            h
            for h in session["history"]
            if not (h["frame_id"] == choice.frame_id and h["camera"] == choice.camera)
        ]
        reconstruct_frame(frame, session["calibration"], session["settings"])
        session["revision"] += 1
        store.save(session)
        return session


class Settings(BaseModel):
    min_confidence: float = Field(ge=0, le=1)
    max_error_px: float = Field(ge=0.1, le=30)
    min_angle_deg: float = Field(ge=0.1, le=30)
    revision: int | None = None


@app.patch("/api/sessions/{sid}/settings")
def settings(sid: str, settings: Settings):
    with store.lock:
        session = editable(sid, settings.revision)
        session["settings"] = settings.model_dump(exclude={"revision"})
        return store.reconstruct(session)


def finish_job(job, status, message):
    job.update(status=status, message=message, cancel_requested=False)
    store.save_job(job)


def cancel_requested(job):
    if job.get("cancel_requested"):
        finish_job(
            job,
            "cancelled",
            "Cancelled. Completed frames were saved; the unfinished frame was discarded.",
        )
        return True
    return False


def run_detection(sid, job_id, frame_id=None):
    """Checkpoint complete frames, never partly detected views or stale sessions."""
    with store.lock:
        job = jobs[job_id]
        if job.get("status") in ["cancelled", "complete", "failed", "interrupted"]:
            return
        session = store.load(sid)
        # Defaults also permit running this worker directly in offline tooling.
        job.setdefault(
            "frame_ids",
            [
                f["id"]
                for f in session["frames"]
                if frame_id is None or f["id"] == frame_id
            ],
        )
        job.setdefault("completed_frame_ids", [])
        job.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        job.setdefault("progress", 0)
        job.setdefault("cancel_requested", False)
        if cancel_requested(job):
            return
        job.update(status="running", message="Loading pose models…")
        expected_revision = session["revision"]
        store.save_job(job)
    try:
        for fid in job["frame_ids"]:
            if fid in job["completed_frame_ids"]:
                continue
            with store.lock:
                if cancel_requested(job):
                    return
                session = store.load(sid)
                if session["revision"] != expected_revision:
                    raise ValueError(
                        "Session changed during inference. Unfinished results were discarded; resume after reviewing the session."
                    )
                frame = deepcopy(next(f for f in session["frames"] if f["id"] == fid))
            for name, view in frame["views"].items():
                with store.lock:
                    if cancel_requested(job):
                        return
                    job["message"] = (
                        f"Frame {fid} · {name} · {len(job['completed_frame_ids'])}/{len(job['frame_ids'])} frames saved"
                    )
                    store.save_job(job)
                image = cv2.imread(str(store.path(sid) / view["image"]))
                if image is None:
                    raise ValueError(f"Unable to read {name}, frame {fid}.")
                candidates = detector.detect(image)
                view["candidates"] = candidates
                view["actor_index"] = 0 if len(candidates) == 1 else None
                view["raw"] = (
                    candidates[0]
                    if len(candidates) == 1
                    else [{"xy": None, "confidence": 0.0} for _ in JOINTS]
                )
            reconstruct_frame(frame, session["calibration"], session["settings"])
            with store.lock:
                # A cancellation during the last camera is still a partial frame.
                if cancel_requested(job):
                    return
                current = store.load(sid)
                if current["revision"] != expected_revision:
                    raise ValueError(
                        "Session changed during inference. Unfinished results were discarded; resume after reviewing the session."
                    )
                frame["inference"] = {
                    "job_id": job_id,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                index = next(
                    i for i, item in enumerate(current["frames"]) if item["id"] == fid
                )
                current["frames"][index] = frame
                current["revision"] += 1
                store.save(current)
                expected_revision = current["revision"]
                job["completed_frame_ids"].append(fid)
                job["progress"] = round(
                    len(job["completed_frame_ids"])
                    / max(len(job["frame_ids"]), 1)
                    * 100
                )
                store.save_job(job)
        with store.lock:
            job["progress"] = 100
            finish_job(job, "complete", "Detection complete. Manual labels preserved.")
    except Exception as exc:
        with store.lock:
            finish_job(
                job,
                "failed",
                f"{exc} Completed frames were saved; resume retries unfinished frames.",
            )


def queue_detection(session, frame_ids, previous=None):
    job_id = uuid4().hex
    job = {
        "id": job_id,
        "session_id": session["id"],
        "status": "queued",
        "progress": 0,
        "message": "Queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frame_ids": frame_ids,
        "completed_frame_ids": [],
        "cancel_requested": False,
    }
    if previous:
        job["resumed_from"] = previous["id"]
        job["completed_frame_ids"] = previous["completed_frame_ids"][:]
        job["progress"] = round(
            len(job["completed_frame_ids"]) / max(len(frame_ids), 1) * 100
        )
    jobs[job_id] = job
    store.save_job(job)
    if previous:
        previous["resumed_by"] = job_id
        store.save_job(previous)
    try:
        pool.submit(run_detection, session["id"], job_id)
    except RuntimeError as exc:
        finish_job(job, "failed", f"Unable to start inference: {exc}. Resume to retry.")
    return job_response(job)


@app.post("/api/sessions/{sid}/detect")
def detect(sid: str, frame_id: int | None = None, revision: int | None = None):
    with store.lock:
        session = editable(sid, revision)
        if session["source"] == "synthetic":
            raise ValueError(
                "The synthetic demo already contains simulated detections. Import camera footage to run RTMPose."
            )
        if frame_id is not None and not any(
            f["id"] == frame_id for f in session["frames"]
        ):
            raise ValueError("Frame not found.")
        if not detector.ready():
            raise ValueError(
                "Models are missing. Run python scripts/download_models.py first."
            )
        return queue_detection(
            session,
            [
                f["id"]
                for f in session["frames"]
                if frame_id is None or f["id"] == frame_id
            ],
        )


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with store.lock:
        if job_id not in jobs:
            raise HTTPException(404, "Inference job not found.")
        return job_response(jobs[job_id])


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str):
    with store.lock:
        get_job(job_id)
        job = jobs[job_id]
        if job["status"] == "queued":
            finish_job(
                job,
                "cancelled",
                "Cancelled before inference started. Resume to process unfinished frames.",
            )
        elif job["status"] == "running":
            job["cancel_requested"] = True
            job["message"] = (
                "Cancelling after the current model call. The unfinished frame will be discarded."
            )
        store.save_job(job)
        return job_response(job)


@app.post("/api/jobs/{job_id}/resume")
def resume_detection(job_id: str, revision: int | None = None):
    with store.lock:
        get_job(job_id)
        previous = jobs[job_id]
        session = editable(previous["session_id"], revision)
        matches = session_jobs(session["id"])
        if not job_response(previous)["can_resume"] or matches[0]["id"] != job_id:
            raise HTTPException(
                409, "Only the latest unfinished inference job can be resumed."
            )
        if not detector.ready():
            raise ValueError(
                "Models are missing. Run python scripts/download_models.py first."
            )
        return queue_detection(session, previous["frame_ids"][:], previous)


@app.get("/api/sessions/{sid}/export")
def export(sid: str):
    session = store.load(sid)
    frames = []
    for frame in session["frames"]:
        joints = {name: frame["pose"][i]["point"] for i, name in enumerate(JOINTS)}
        for name, a, b in [
            ("pelvis", "left_hip", "right_hip"),
            ("neck", "left_shoulder", "right_shoulder"),
        ]:
            joints[name] = (
                ((np.array(joints[a]) + joints[b]) / 2).tolist()
                if joints[a] is not None and joints[b] is not None
                else None
            )
        joints["spine"] = (
            (
                np.array(joints["pelvis"]) * 0.45 + np.array(joints["neck"]) * 0.55
            ).tolist()
            if joints["pelvis"] is not None and joints["neck"] is not None
            else None
        )
        joints["head"] = joints["nose"]
        frames.append(
            {
                "frame": frame["id"],
                "timestamp_seconds": frame["timestamp"],
                "joints": joints,
                "quality": frame["quality"],
                "joint_quality": frame["pose"],
            }
        )
    payload = {
        "format": "mocap_web_animation_v1",
        "units": "meters",
        "world_frame": session["calibration"]["world_frame"],
        "fps": session["fps"],
        "session_id": sid,
        "revision": session["revision"],
        "source": session["source"],
        "model": session["model"],
        "calibration": session["calibration"],
        "settings": session["settings"],
        "missing_data_policy": "null; no temporal interpolation",
        "frames": frames,
    }
    return Response(
        json.dumps(payload, indent=2, allow_nan=False),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="pose_animation.json"'},
    )


@app.get("/api/sessions/{sid}/project")
def export_project(sid: str):
    with store.lock:
        session = store.load(sid)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("session.json", json.dumps(session, allow_nan=False))
            for image in sorted(
                {v["image"] for f in session["frames"] for v in f["views"].values()}
            ):
                archive.write(store.path(sid) / image, image)
    return Response(
        buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="mocap_project.zip"'},
    )


@app.post("/api/restore")
async def restore_project(request: Request):
    try:
        return await restore_project_data(request)
    except (
        KeyError,
        TypeError,
        AttributeError,
        IndexError,
        OverflowError,
        cv2.error,
    ) as exc:
        raise ValueError(
            "Project metadata or media is malformed. Choose an unmodified Mocap Studio project ZIP."
        ) from exc


async def restore_project_data(request: Request):
    _, files = await uploads(request)
    if len(files.get("project", [])) != 1:
        raise ValueError("Choose one exported project ZIP.")
    _, content = files["project"][0]
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise ValueError("Invalid project ZIP.") from exc
    with archive:
        entries = archive.infolist()
        if len(entries) > 1801 or sum(e.file_size for e in entries) > 600 * 1024 * 1024:
            raise ValueError("Project archive exceeds the session limits.")
        if len({e.filename for e in entries}) != len(entries):
            raise ValueError("Duplicate archive paths are not supported.")
        if any(
            e.filename != "session.json"
            and not re.fullmatch(r"cam[1-6]_\d+\.jpg", e.filename)
            for e in entries
        ):
            raise ValueError("Project contains unexpected paths.")
        if "session.json" not in archive.namelist():
            raise ValueError("Project has no session.json.")
        s = json.loads(archive.read("session.json"))
        if s.get("format") != "mocap_web_session_v1" or s.get("joint_names") != JOINTS:
            raise ValueError("Unsupported project version or joint schema.")
        validate(s["calibration"])
        Settings(**s["settings"])
        if not 1 <= float(s["fps"]) <= 240 or not 1 <= len(s["frames"]) <= 300:
            raise ValueError("Invalid frame rate or frame count.")
        previous_time = -1.0
        ids = set()
        media_bytes = {}
        for frame in s["frames"]:
            if (
                not isinstance(frame["id"], int)
                or frame["id"] < 0
                or frame["id"] in ids
            ):
                raise ValueError("Frame IDs must be unique nonnegative integers.")
            ids.add(frame["id"])
            if (
                not np.isfinite(frame["timestamp"])
                or frame["timestamp"] < 0
                or frame["timestamp"] <= previous_time
            ):
                raise ValueError(
                    "Frame timestamps must be finite, nonnegative, and increasing."
                )
            previous_time = frame["timestamp"]
            for name, view in frame["views"].items():
                if name not in s["calibration"]["cameras"] or not re.fullmatch(
                    r"cam[1-6]_\d+\.jpg", view["image"]
                ):
                    raise ValueError("Invalid project camera/image.")
                candidates = view.setdefault("candidates", [])
                if not isinstance(candidates, list):
                    raise ValueError("Invalid person candidates.")
                for observations in [view["raw"], *candidates]:
                    if not isinstance(observations, list) or len(observations) != 17:
                        raise ValueError("Invalid observation count.")
                    for obs in observations:
                        if (
                            not np.isfinite(obs["confidence"])
                            or not 0 <= obs["confidence"] <= 1
                        ):
                            raise ValueError("Invalid detector confidence.")
                        xy = obs.get("xy")
                        if xy is not None and (
                            len(xy) != 2
                            or not np.isfinite(xy).all()
                            or not 0 <= xy[0] < view["width"]
                            or not 0 <= xy[1] < view["height"]
                        ):
                            raise ValueError(
                                "Invalid candidate or raw joint coordinates."
                            )
                actor_index = view.get("actor_index")
                if actor_index is not None:
                    if (
                        type(actor_index) is not int
                        or actor_index < 0
                        or actor_index >= max(len(candidates), 1)
                    ):
                        raise ValueError("Invalid selected person index.")
                    # Early projects used index zero even without candidate lists.
                    if not candidates:
                        view["actor_index"] = None
                for key in view["edits"]:
                    if key not in [str(i) for i in range(17)]:
                        raise ValueError("Invalid manual joint index.")
                for obs in effective_observations(view) + view["raw"]:
                    xy = obs.get("xy")
                    if xy is not None and (
                        len(xy) != 2
                        or not np.isfinite(xy).all()
                        or not 0 <= xy[0] < view["width"]
                        or not 0 <= xy[1] < view["height"]
                    ):
                        raise ValueError("Invalid joint coordinates.")
                if view["image"] not in archive.namelist():
                    raise ValueError("Project image is missing.")
                image_data = archive.read(view["image"])
                image = cv2.imdecode(
                    np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR
                )
                if (
                    image is None
                    or list(image.shape[1::-1])
                    != s["calibration"]["cameras"][name]["image_size"]
                    or list(image.shape[1::-1]) != [view["width"], view["height"]]
                ):
                    raise ValueError("Project image size does not match calibration.")
                media_bytes[view["image"]] = image_data
        s["id"] = uuid4().hex
        s["archived"] = False
        s["name"] = str(s.get("name", "Restored project"))[:100]
        s["revision"] = 0
        s["history"] = []  # labels restored; prior undo history is intentionally reset
        folder = store.path(s["id"])
        folder.mkdir()
        try:
            for filename, image_data in media_bytes.items():
                (folder / filename).write_bytes(image_data)
            return store.reconstruct(s)
        except Exception:
            shutil.rmtree(folder)
            raise


# Real-time multi-camera 2D/3D pipeline (desktop app UI at /live/).
from fastapi.responses import RedirectResponse  # noqa: E402

from .realtime.server import create_app as create_live_app  # noqa: E402


@app.get("/live", include_in_schema=False)
def live_redirect():
    return RedirectResponse("/live/")


app.mount("/live", create_live_app(desktop=os.getenv("MOCAP_DESKTOP") == "1"))

# The production build is served by this same local backend; no desktop framework.
DIST = ROOT / "frontend" / "dist"
if DIST.is_dir():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
