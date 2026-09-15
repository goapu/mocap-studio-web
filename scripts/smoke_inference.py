"""Run actual CPU inference on one local image; report shape and timing."""

import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
from backend.detector import Detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.getenv("MOCAP_MODEL_DIR", ROOT / ".local/models")),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error("Image cannot be decoded.")
    detector = Detector(args.model_dir)
    start = perf_counter()
    people = detector.detect(image)
    cold = perf_counter() - start
    start = perf_counter()
    detector.detect(image)
    warm = perf_counter() - start
    result = {
        "model": "RTMPose-M + YOLOX-M",
        "device": "CPU",
        "image_size": list(image.shape[1::-1]),
        "people": len(people),
        "joints_per_person": [len(p) for p in people],
        "confident_joints": [sum(j["confidence"] >= 0.35 for j in p) for p in people],
        "cold_seconds": cold,
        "warm_seconds": warm,
        "scope": "2D inference smoke test; not a 3D accuracy benchmark",
    }
    report = json.dumps(result, indent=2)
    print(report)
    if args.output:
        args.output.write_text(report + "\n")


if __name__ == "__main__":
    main()
