import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

from .geometry import JOINTS, BONES, reconstruct_frame


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, allow_nan=False)
    # Flush the complete document before replacing the previous checkpoint.
    # Unique temporary names also avoid collisions between independent writes.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()

    def path(self, sid):
        if not re.fullmatch(r"[a-f0-9]{32}", sid):
            raise ValueError("Invalid session ID.")
        return self.root / sid

    def load(self, sid):
        path = self.path(sid) / "session.json"
        if not path.exists():
            raise ValueError("Session not found.")
        return json.loads(path.read_text())

    def save(self, session):
        atomic_json(self.path(session["id"]) / "session.json", session)

    def save_job(self, job):
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", job["id"]):
            raise ValueError("Invalid job ID.")
        job["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(self.root / ".jobs" / f"{job['id']}.json", job)

    def recover_jobs(self):
        """Reload jobs, reconciling the session checkpoint before offering resume.

        The frame and its inference marker share one atomic session write. This
        also covers a process exit between saving the frame and saving its job.
        Startup never silently restarts expensive inference.
        """
        jobs = {}
        for path in (self.root / ".jobs").glob("*.json"):
            job = json.loads(path.read_text())
            if job.get("status") in ["queued", "running"]:
                session = self.load(job["session_id"])
                completed = set(job.get("completed_frame_ids", []))
                completed.update(
                    frame["id"]
                    for frame in session["frames"]
                    if frame.get("inference", {}).get("job_id") == job["id"]
                )
                targets = job.get("frame_ids", [])
                job["completed_frame_ids"] = [
                    fid for fid in targets if fid in completed
                ]
                finished = len(completed.intersection(targets)) == len(targets)
                job["status"] = "complete" if finished else "interrupted"
                job["progress"] = round(
                    len(job["completed_frame_ids"]) / max(len(targets), 1) * 100
                )
                job["message"] = (
                    "Detection complete. Saved frames recovered after restart."
                    if finished
                    else "Server restarted. Completed frames are saved; resume to process unfinished frames."
                )
                job["cancel_requested"] = False
                self.save_job(job)
            jobs[job["id"]] = job
        return jobs

    def create(self, name, calibration, fps, source="capture"):
        sid = uuid4().hex
        self.path(sid).mkdir()
        return {
            "id": sid,
            "name": name.strip()[:120] or "Untitled capture",
            "format": "mocap_web_session_v1",
            "revision": 0,
            "archived": False,
            "source": source,
            "fps": fps,
            "calibration": calibration,
            "joint_names": JOINTS,
            "bones": BONES,
            "frames": [],
            "history": [],
            "settings": {
                "min_confidence": 0.35,
                "max_error_px": 6.0,
                "min_angle_deg": 1.0,
            },
            "model": {"name": "RTMPose-M / YOLOX-M", "backend": "ONNX Runtime CPU"},
        }

    def summarize(self, session):
        return {k: session[k] for k in ["id", "name", "source", "fps", "revision"]} | {
            "frame_count": len(session["frames"]),
            "archived": bool(session.get("archived", False)),
        }

    def reconstruct(self, session):
        for frame in session["frames"]:
            reconstruct_frame(frame, session["calibration"], session["settings"])
        session["revision"] += 1
        self.save(session)
        return session
