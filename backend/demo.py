"""Deterministic synthetic calibration fixture; clearly labelled in the UI."""

import cv2
import numpy as np

from .geometry import BONES, project


def create_demo(store):
    K = [[780.0, 0, 480.0], [0, 780.0, 360.0], [0, 0, 1.0]]
    cameras = {}
    for i, center in enumerate(
        [[-2.7, -4.2, 2.1], [3.3, -3.0, 2.2], [2.8, 3.6, 2.3]], 1
    ):
        center = np.array(center)
        forward = np.array([0.0, 0.0, 1.0]) - center
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        R = np.vstack([right, down, forward])
        cameras[f"cam{i}"] = {
            "K": K,
            "dist": [0.0] * 5,
            "R": R.tolist(),
            "T": (-R @ center).tolist(),
            "image_size": [960, 720],
        }
    calibration = {
        "format": "mocap_calibration_v1",
        "units": "meters",
        "world_frame": "z_up",
        "cameras": cameras,
        "quality": {"source": "synthetic", "accepted": True, "stereo_rms_px": 0.0},
    }
    session = store.create(
        "Reach study · synthetic demo", calibration, 24.0, source="synthetic"
    )
    rng = np.random.default_rng(7)
    for fid in range(48):
        a = np.sin(fid / 47 * np.pi)
        points = np.array(
            [
                [0, -0.08, 1.77],
                [-0.035, -0.09, 1.81],
                [0.035, -0.09, 1.81],
                [-0.08, 0, 1.78],
                [0.08, 0, 1.78],
                [-0.23, 0, 1.5],
                [0.23, 0, 1.5],
                [-0.42, -0.03, 1.2 + 0.3 * a],
                [0.42 + 0.1 * a, 0, 1.24 + 0.5 * a],
                [-0.46, -0.1, 0.97 + 0.6 * a],
                [0.51 + 0.14 * a, -0.05, 1.0 + 0.95 * a],
                [-0.15, 0, 0.95],
                [0.15, 0, 0.95],
                [-0.17, -0.03, 0.53],
                [0.17, 0.06, 0.53],
                [-0.18, -0.02, 0.07],
                [0.19, 0.09, 0.07],
            ]
        )
        views = {}
        for name, camera in cameras.items():
            xy = project(points, camera)
            image = np.full((720, 960, 3), [35, 42, 44], dtype=np.uint8)
            for g in np.arange(-2, 2.1, 0.5):
                for line in [
                    np.array([[g, -2, 0], [g, 2, 0]]),
                    np.array([[-2, g, 0], [2, g, 0]]),
                ]:
                    projected = project(line, camera).astype(int)
                    cv2.line(
                        image,
                        tuple(projected[0]),
                        tuple(projected[1]),
                        (53, 62, 64),
                        1,
                        cv2.LINE_AA,
                    )
            for start, end in BONES:
                cv2.line(
                    image,
                    tuple(xy[start].astype(int)),
                    tuple(xy[end].astype(int)),
                    (145, 155, 154),
                    13,
                    cv2.LINE_AA,
                )
            for point in xy:
                cv2.circle(
                    image, tuple(point.astype(int)), 8, (183, 192, 187), -1, cv2.LINE_AA
                )
            cv2.putText(
                image,
                "SYNTHETIC REFERENCE / " + name.upper(),
                (28, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (180, 190, 185),
                1,
                cv2.LINE_AA,
            )
            predicted = xy + rng.normal(0, 0.65, xy.shape)
            if name == "cam2" and 10 <= fid <= 31:
                predicted[10] += [
                    42.0,
                    -26.0,
                ]  # deliberate misalignment for manual repair
            observations = [
                {"xy": point.tolist(), "confidence": 0.95} for point in predicted
            ]
            filename = f"{name}_{fid:06d}.jpg"
            cv2.imwrite(
                str(store.path(session["id"]) / filename),
                image,
                [cv2.IMWRITE_JPEG_QUALITY, 85],
            )
            views[name] = {
                "image": filename,
                "width": 960,
                "height": 720,
                "raw": observations,
                "edits": {},
                "candidates": [],
                "actor_index": 0,
            }
        session["frames"].append({"id": fid, "timestamp": fid / 24.0, "views": views})
    return store.reconstruct(session)
