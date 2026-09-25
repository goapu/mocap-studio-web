"""Save a processed run to a folder on this computer."""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .engine import VERSION, Run, _clean, _max_axis
from .sources import MultiVideo

DEFAULT_EXPORT_DIR = Path(
    os.getenv("MOCAP_EXPORT_DIR", Path.home() / "Documents" / "MocapStudio" / "Exports")
)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")[:60] or "capture"


def _to_y_up(X: np.ndarray, world_frame: str) -> np.ndarray:
    """World coordinates → OpenSim/Blender-style Y-up (right-handed)."""
    if world_frame == "z_up":
        return np.stack([X[..., 0], X[..., 2], -X[..., 1]], -1)
    return np.stack([X[..., 0], -X[..., 1], -X[..., 2]], -1)  # camera: y down, z fwd


def _num(v, digits=5):
    return "" if not np.isfinite(v) else f"{v:.{digits}f}"


class Exporter:
    def __init__(self, run: Run, folder: str | Path | None, options: dict):
        if run.final is None:
            raise ValueError("Processing is not finished yet.")
        base = Path(folder).expanduser() if folder else DEFAULT_EXPORT_DIR
        if not base.is_absolute():
            raise ValueError("Choose an absolute folder path.")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = base / f"{_slug(run.config.name)}_{stamp}"
        self.run = run
        self.options = {
            "csv": True,
            "trc": True,
            "videos": False,
            "bone_constraint": False,
            **(options or {}),
        }
        self.state = {
            "status": "queued",
            "progress": 0,
            "folder": str(self.dir),
            "files": [],
        }
        self.thread = threading.Thread(target=self._save, daemon=True, name="export")

    def start(self):
        self.thread.start()
        return self

    def _add(self, path: Path):
        self.state["files"].append(path.name)

    def _save(self):
        try:
            self.state["status"] = "saving"
            self.dir.mkdir(parents=True, exist_ok=False)
            run, f = self.run, self.run.final
            X = f["constrained"] if self.options["bone_constraint"] else f["smoothed"]
            names = list(run.skeleton.joints)
            n = len(f["times"])
            rec = {k: v[:n] for k, v in run.rec.items()}
            wf = run.config.calibration.get("world_frame", "camera")
            self._json(X, rec, names, wf)
            self.state["progress"] = 20
            if self.options["csv"]:
                self._csv(X, rec, names)
            self.state["progress"] = 35
            if self.options["trc"]:
                self._trc(X, names, wf)
            self.state["progress"] = 45
            if self.options["videos"]:
                self._videos(rec)
            report = {
                "app": "Mocap Studio real-time",
                "version": VERSION,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "config": run.config_json(),
                "quality": f["quality"],
                "coordinate_note": (
                    f"pose3d.json/csv use the calibration world frame '{wf}' in meters. "
                    "pose3d.trc is converted to Y-up millimeters for OpenSim/Blender."
                ),
                "platform": sys.platform,
            }
            (self.dir / "report.json").write_text(
                json.dumps(report, indent=2, allow_nan=False)
            )
            self._add(self.dir / "report.json")
            self.state.update(status="done", progress=100)
        except Exception as exc:
            self.state.update(status="error", message=str(exc))

    def _json(self, X, rec, names, wf):
        run, f = self.run, self.run.final
        frames = []
        for i in range(len(f["times"])):
            frames.append(
                {
                    "frame": i,
                    "time_s": round(float(f["times"][i]), 6),
                    "joints3d": _clean(X[i], 5),
                    "sigma3d_mm": _clean(_max_axis(f["sigma"][i]) * 1000, 2),
                    "joints3d_raw": _clean(rec["X"][i], 5),
                    "joints3d_live": _clean(rec["live"][i], 5),
                    "views": {
                        cam: {
                            "keypoints2d": _clean(
                                np.concatenate(
                                    [rec["kp"][i, c], rec["score"][i, c][:, None]], -1
                                ),
                                2,
                            ),
                            "reprojected2d": _clean(f["reproj"][i, c], 2),
                            "used_for_3d": rec["used"][i, c].tolist(),
                        }
                        for c, cam in enumerate(run.names)
                    },
                }
            )
        payload = {
            "format": "mocap_realtime_v1",
            "units": "meters",
            "world_frame": wf,
            "fps": run.fps,
            "skeleton": run.skeleton.to_json(),
            "joints3d_variant": "rts_smoothed_bone_constrained"
            if self.options["bone_constraint"]
            else "rts_smoothed",
            "missing_data_policy": "null when no joint could be estimated; Kalman gaps up to 0.3 s are bridged",
            "cameras": run.names,
            "calibration": run.config.calibration,
            "models": run.config_json()["preset"],
            "quality": f["quality"],
            "frames": frames,
        }
        path = self.dir / "pose3d.json"
        path.write_text(json.dumps(payload, allow_nan=False))
        self._add(path)

    def _csv(self, X, rec, names):
        run, f = self.run, self.run.final
        path = self.dir / "pose3d.csv"
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                ["frame", "time_s"]
                + [f"{j}_{a}" for j in names for a in "xyz"]
                + [f"{j}_sigma_mm" for j in names]
            )
            for i in range(len(f["times"])):
                sig = _max_axis(f["sigma"][i]) * 1000
                w.writerow(
                    [i, f"{f['times'][i]:.6f}"]
                    + [_num(v) for v in X[i].ravel()]
                    + [_num(v, 2) for v in sig]
                )
        self._add(path)
        for c, cam in enumerate(run.names):
            path = self.dir / f"pose2d_{cam}.csv"
            with path.open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(
                    ["frame", "time_s"]
                    + [f"{j}_{a}" for j in names for a in ("x", "y", "score")]
                    + [f"{j}_reproj_{a}" for j in names for a in "xy"]
                )
                for i in range(len(f["times"])):
                    kp, sc, rp = rec["kp"][i, c], rec["score"][i, c], f["reproj"][i, c]
                    row = [i, f"{f['times'][i]:.6f}"]
                    for j in range(len(names)):
                        row += [_num(kp[j, 0], 2), _num(kp[j, 1], 2), _num(sc[j], 3)]
                    row += [_num(v, 2) for v in rp.ravel()]
                    w.writerow(row)
            self._add(path)

    def _trc(self, X, names, wf):
        f = self.run.final
        Y = _to_y_up(X, wf) * 1000
        n, fps = len(f["times"]), self.run.fps
        path = self.dir / "pose3d.trc"
        with path.open("w", newline="") as fh:
            fh.write(f"PathFileType\t4\t(X/Y/Z)\t{path.name}\n")
            fh.write(
                "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\t"
                "OrigDataStartFrame\tOrigNumFrames\n"
            )
            fh.write(
                f"{fps:.3f}\t{fps:.3f}\t{n}\t{len(names)}\tmm\t{fps:.3f}\t1\t{n}\n"
            )
            fh.write("Frame#\tTime\t" + "\t\t\t".join(names) + "\t\t\n")
            fh.write(
                "\t\t"
                + "\t".join(f"X{k}\tY{k}\tZ{k}" for k in range(1, len(names) + 1))
                + "\n\n"
            )
            for i in range(n):
                vals = "\t".join(_num(v, 3) for v in Y[i].ravel())
                fh.write(f"{i + 1}\t{f['times'][i]:.5f}\t{vals}\n")
        self._add(path)

    def _videos(self, rec):
        run, f = self.run, self.run.final
        n = len(f["times"])
        bones = run.skeleton.bones
        video = MultiVideo(run.config.videos, run.config.offsets).start()
        writers = {}
        try:
            for c, cam in enumerate(run.names):
                w, h = (int(v) for v in run.rig.image_size[c])
                path = self.dir / f"{cam}_overlay.mp4"
                writer = None
                for code in ("avc1", "mp4v"):
                    writer = cv2.VideoWriter(
                        str(path), cv2.VideoWriter_fourcc(*code), run.fps, (w, h)
                    )
                    if writer.isOpened():
                        break
                if writer is None or not writer.isOpened():
                    raise RuntimeError(
                        "OpenCV cannot write MP4 video on this computer."
                    )
                writers[cam] = (writer, path)
            for i in range(n):
                item = video.read()
                if item is None:
                    break
                _, frames = item
                for c, cam in enumerate(run.names):
                    img = frames[cam]
                    rp = f["reproj"][i, c]
                    thick = max(2, int(img.shape[0] / 360))
                    for a, b in bones:
                        if np.isfinite(rp[[a, b]]).all():
                            color = (
                                (255, 160, 60)
                                if "left" in run.skeleton.joints[b]
                                else (60, 160, 255)
                                if "right" in run.skeleton.joints[b]
                                else (230, 230, 230)
                            )
                            cv2.line(
                                img,
                                tuple(np.int32(rp[a])),
                                tuple(np.int32(rp[b])),
                                color,
                                thick,
                                cv2.LINE_AA,
                            )
                    for j in range(len(rp)):
                        if np.isfinite(rp[j]).all():
                            cv2.circle(
                                img,
                                tuple(np.int32(rp[j])),
                                thick + 1,
                                (255, 255, 255),
                                -1,
                                cv2.LINE_AA,
                            )
                    cv2.putText(
                        img,
                        f"{cam}  t={f['times'][i]:.3f}s",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    writers[cam][0].write(img)
                self.state["progress"] = 45 + int(50 * (i + 1) / n)
        finally:
            video.stop()
            for writer, path in writers.values():
                writer.release()
                self._add(path)
