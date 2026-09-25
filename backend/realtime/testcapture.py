"""Generate a synthetic multi-camera capture for end-to-end testing.

A photograph of a person is mapped onto a vertical plane that moves quickly
through the capture volume and is filmed by calibrated virtual cameras. Because
the "person" lives on a known 3D plane, the true 3D position of every keypoint
the network finds on the flat photograph is known, which lets the complete
pipeline (detector, pose network, tracking, triangulation, filters) be scored
in millimetres. It is a pipeline test, not a substitute for real footage.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .synthetic import look_at


def arc_calibration(n=4, size=(1280, 720), focal=1100.0, radius=4.5, span_deg=100):
    cams = {}
    for i in range(n):
        a = np.radians(-span_deg / 2 + span_deg * i / max(n - 1, 1)) - np.pi / 2
        c = np.array([radius * np.cos(a), radius * np.sin(a), 1.3 + 0.15 * (i % 2)])
        R, T = look_at(c, np.array([0, 0, 1.0]))
        cams[f"cam{i + 1}"] = {
            "K": [[focal, 0, size[0] / 2], [0, focal, size[1] / 2], [0, 0, 1]],
            "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
            "R": R.tolist(),
            "T": T.tolist(),
            "image_size": list(size),
        }
    return {
        "format": "mocap_calibration_v1",
        "units": "meters",
        "world_frame": "z_up",
        "quality": {
            "source": "synthetic test rig",
            "accepted": True,
            "stereo_rms_px": 0.0,
        },
        "cameras": cams,
    }


def plane_pose(t, speed=2.5, extent=1.2):
    """Plane origin (top-left corner of the photo) moving side to side fast."""
    x = extent * np.sin(speed / extent * t)
    return np.array([x - 0.9, 0.2 * np.sin(3.1 * t), 1.9 + 0.05 * np.sin(9 * t)])


def make_capture(
    out_dir,
    texture_bgr,
    seconds=3.0,
    fps=30.0,
    cams=4,
    size=(1280, 720),
    plane_width=1.8,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cal = arc_calibration(cams, size)
    (out / "calibration.json").write_text(json.dumps(cal, indent=2))
    th, tw = texture_bgr.shape[:2]
    m_per_px = plane_width / tw
    rng = np.random.default_rng(3)
    background = rng.integers(60, 120, (size[1] // 8, size[0] // 8, 3), dtype=np.uint8)
    background = cv2.resize(background, size, interpolation=cv2.INTER_CUBIC)
    writers = {}
    for name in cal["cameras"]:
        writers[name] = cv2.VideoWriter(
            str(out / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
        )
    frames = int(seconds * fps)
    origins = []
    for i in range(frames):
        o = plane_pose(i / fps)
        origins.append(o.tolist())
        for name, cam in cal["cameras"].items():
            K, R, T = np.array(cam["K"]), np.array(cam["R"]), np.array(cam["T"])
            # texture pixel (u,v) → world: o + u*m*x̂ − v*m*ẑ (plane faces −y)
            A = np.column_stack([R[:, 0] * m_per_px, -R[:, 2] * m_per_px, R @ o + T])
            Hm = K @ A
            img = background.copy()
            warped = cv2.warpPerspective(texture_bgr, Hm, size, flags=cv2.INTER_LINEAR)
            mask = cv2.warpPerspective(np.full((th, tw), 255, np.uint8), Hm, size)
            img[mask > 0] = warped[mask > 0]
            writers[name].write(img)
    for w in writers.values():
        w.release()
    meta = {"fps": fps, "frames": frames, "m_per_px": m_per_px, "origins": origins}
    (out / "truth.json").write_text(json.dumps(meta))
    return cal, meta


def truth_points(texture_kp, meta, index):
    """World coordinates of texture keypoints (K,2 pixels) at frame ``index``."""
    o = np.array(meta["origins"][index])
    m = meta["m_per_px"]
    u, v = texture_kp[:, 0], texture_kp[:, 1]
    return np.stack([o[0] + u * m, np.full_like(u, o[1]), o[2] - v * m], -1)
