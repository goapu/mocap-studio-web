import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.geometry import JOINTS, project
from backend.store import Store


@pytest.fixture
def calibration():
    return {
        "format": "mocap_calibration_v1",
        "units": "meters",
        "world_frame": "camera",
        "quality": {"source": "test", "accepted": True, "stereo_rms_px": 0.1},
        "cameras": {
            f"cam{i}": {
                "K": [[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]],
                "dist": [0.02, -0.005, 0.001, 0.002, 0.0],
                "R": np.eye(3).tolist(),
                "T": [-(i - 1) * 0.7, 0.0, 0.0],
                "image_size": [640, 480],
            }
            for i in range(1, 4)
        },
    }


@pytest.fixture
def workspace(tmp_path, monkeypatch, calibration):
    from backend import app as module

    store = Store(tmp_path / "sessions")
    monkeypatch.setattr(module, "store", store)
    monkeypatch.setattr(module, "DATA", store.root)
    monkeypatch.setattr(module, "jobs", {})
    session = store.create("Reference", calibration, 30.0)
    for fid in [100, 101, 104]:
        point = np.array([0.15, 0.05, 4.0])
        views = {}
        for name, camera in calibration["cameras"].items():
            xy = project(point, camera)[0].tolist()
            filename = f"{name}_{fid:06d}.jpg"
            cv2.imwrite(
                str(store.path(session["id"]) / filename),
                np.zeros((480, 640, 3), np.uint8),
            )
            views[name] = {
                "image": filename,
                "width": 640,
                "height": 480,
                "raw": [{"xy": xy[:], "confidence": 0.9} for _ in JOINTS],
                "edits": {},
                "candidates": [],
                "actor_index": 0,
            }
        session["frames"].append(
            {"id": fid, "timestamp": (fid - 100) / 30, "views": views}
        )
    store.reconstruct(session)
    return TestClient(module.app), store, session, module
