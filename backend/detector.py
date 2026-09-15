"""RTMPose-M + YOLOX-M, ONNX Runtime CPU. Model files are explicit assets."""

from pathlib import Path
from threading import Lock

import numpy as np

MODELS = {
    "detector": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
    "pose": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
}


class Detector:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.body = None
        self.lock = Lock()

    def paths(self):
        return {
            key: self.model_dir / (Path(url).stem + ".onnx")
            for key, url in MODELS.items()
        }

    def ready(self):
        return all(p.is_file() for p in self.paths().values())

    def detect(self, image):
        with self.lock:
            if not self.ready():
                raise ValueError(
                    "Model files are missing. Run python scripts/download_models.py, then retry detection."
                )
            if self.body is None:
                from rtmlib import Body

                paths = self.paths()
                self.body = Body(
                    det=str(paths["detector"]),
                    det_input_size=(640, 640),
                    pose=str(paths["pose"]),
                    pose_input_size=(192, 256),
                    backend="onnxruntime",
                    device="cpu",
                    to_openpose=False,
                )
            keypoints, scores = self.body(image)
        candidates = []
        for points, confidence in zip(keypoints, scores):
            if np.asarray(points).shape != (17, 2):
                raise ValueError("Detector returned an unexpected skeleton schema.")
            observations = []
            for point, score in zip(points, confidence):
                valid = (
                    np.isfinite(point).all()
                    and np.isfinite(score)
                    and 0 <= point[0] < image.shape[1]
                    and 0 <= point[1] < image.shape[0]
                )
                observations.append(
                    {
                        "xy": [float(v) for v in point] if valid else None,
                        "confidence": float(np.clip(score, 0, 1)) if valid else 0.0,
                    }
                )
            candidates.append(observations)
        return sorted(
            candidates, key=lambda c: sum(o["confidence"] for o in c), reverse=True
        )
