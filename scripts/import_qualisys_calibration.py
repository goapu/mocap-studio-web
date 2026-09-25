"""Convert a Qualisys QTM calibration (.qca / .qca.txt) to mocap_calibration_v1.

Qualisys stores each camera's position (mm) and camera-to-world rotation, with
intrinsics in 1/64 sub-pixel units. This converts them to the OpenCV
world-to-camera convention used by Mocap Studio (``X_cam = R @ X_world + T``,
meters, z-up world) and validates the result.

    python scripts/import_qualisys_calibration.py Calib.qca.txt calibration.json \
        --video cam1=videos/cam01.mp4 --video cam2=videos/cam02.mp4 ...

``--video`` sets each camera's image size from its footage (recommended when
videos were exported with a slightly different width than the sensor window).
Cameras are named cam1…camN in file order.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.calibration import validate  # noqa: E402

# Qualisys camera axes (x right, y up, z backwards) → OpenCV (x right, y down, z forward).
QUALISYS_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


def convert(path: Path, video_sizes: dict[str, tuple[int, int]] | None = None) -> dict:
    root = ET.parse(path).getroot()
    cameras = {}
    for i, cam in enumerate(root.findall("cameras/camera")):
        name = f"cam{i + 1}"
        t = cam.find("transform").attrib
        it = cam.find("intrinsic").attrib
        fov = cam.find("fov_video").attrib
        rot = np.array([[float(t[f"r{a}{b}"]) for b in "123"] for a in "123"])
        position = np.array([float(t["x"]), float(t["y"]), float(t["z"])]) / 1000.0
        R = QUALISYS_TO_OPENCV @ rot
        T = -R @ position
        left, top = float(fov["left"]), float(fov["top"])
        K = [
            [
                float(it["focalLengthU"]) / 64,
                0.0,
                float(it["centerPointU"]) / 64 - left,
            ],
            [0.0, float(it["focalLengthV"]) / 64, float(it["centerPointV"]) / 64 - top],
            [0.0, 0.0, 1.0],
        ]
        dist = [
            float(it["radialDistortion1"]) / 64,
            float(it["radialDistortion2"]) / 64,
            float(it["tangentalDistortion1"]) / 64,
            float(it["tangentalDistortion2"]) / 64,
        ]
        size = [
            int(float(fov["right"]) - left + 1),
            int(float(fov["bottom"]) - top + 1),
        ]
        if video_sizes and name in video_sizes:
            size = list(video_sizes[name])
        cameras[name] = {
            "K": K,
            "dist": dist,
            "R": R.tolist(),
            "T": T.tolist(),
            "image_size": size,
            "source_serial": cam.attrib.get("serial"),
            "avg_residual_px": float(cam.attrib.get("avg-residual", "nan")),
        }
    return validate(
        {
            "format": "mocap_calibration_v1",
            "units": "meters",
            "world_frame": "z_up",
            "quality": {"source": f"Qualisys import: {path.name}", "accepted": None},
            "cameras": cameras,
        }
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("qca", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--video", action="append", default=[], help="camN=path/to/video")
    a = p.parse_args()
    sizes = {}
    if a.video:
        import cv2

        for item in a.video:
            name, path = item.split("=", 1)
            cap = cv2.VideoCapture(path)
            sizes[name] = (int(cap.get(3)), int(cap.get(4)))
            cap.release()
    cal = convert(a.qca, sizes)
    a.output.write_text(json.dumps(cal, indent=2))
    print(
        f"Wrote {a.output} with {len(cal['cameras'])} cameras (world frame z_up, meters)."
    )


if __name__ == "__main__":
    main()
