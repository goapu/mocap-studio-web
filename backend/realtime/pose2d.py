"""Person detection and batched top-down 2D keypoint estimation (ONNX).

Accuracy details beyond the reference RTMLib adapter:
* crops are converted BGR→RGB, matching the RTMPose training pipeline;
* SimCC peaks are refined to sub-bin precision with a parabola fit;
* optional flip test-time augmentation averages the original and mirrored crop;
* all cameras (and their flipped crops) run as one batch, which keeps the
  GPU/Neural Engine busy and amortises per-call overhead.
"""

from __future__ import annotations

import cv2
import numpy as np

from .models import LoadedModel
from .skeletons import Skeleton

MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)


class PersonDetector:
    """YOLOX (OpenMMLab SDK export with embedded NMS)."""

    def __init__(self, model: LoadedModel, input_size=(640, 640), score_thr=0.4):
        self.model = model
        self.w, self.h = input_size
        self.score_thr = score_thr

    @staticmethod
    def sample(input_size=(640, 640)) -> np.ndarray:
        return np.zeros((1, 3, input_size[1], input_size[0]), np.float32)

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        """Return person boxes as (N, 5) [x1, y1, x2, y2, score] in image pixels."""
        ih, iw = bgr.shape[:2]
        ratio = min(self.h / ih, self.w / iw)
        canvas = np.full((self.h, self.w, 3), 114, np.uint8)
        resized = cv2.resize(
            bgr, (int(iw * ratio), int(ih * ratio)), interpolation=cv2.INTER_LINEAR
        )
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        x = canvas.transpose(2, 0, 1)[None].astype(np.float32)
        outputs = self.model.run(x)
        dets = outputs[0][0]
        labels = outputs[1][0] if len(outputs) > 1 else np.zeros(len(dets))
        keep = (dets[:, 4] >= self.score_thr) & (labels == 0)
        boxes = dets[keep].astype(np.float64)
        boxes[:, :4] /= ratio
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, iw - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, ih - 1)
        ok = (boxes[:, 2] - boxes[:, 0] > 8) & (boxes[:, 3] - boxes[:, 1] > 16)
        return boxes[ok]


def _crop_transform(box, input_size, padding=1.25):
    """Center/scale of a padded, aspect-corrected crop (mmpose convention)."""
    w, h = input_size
    x1, y1, x2, y2 = box[:4]
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float64)
    bw, bh = (x2 - x1) * padding, (y2 - y1) * padding
    if bw > bh * w / h:
        bh = bw * h / w
    else:
        bw = bh * w / h
    return center, np.array([bw, bh], np.float64)


def _affine(center, scale, input_size):
    w, h = input_size
    src = np.float32(
        [
            center - scale / 2,
            [center[0] + scale[0] / 2, center[1] - scale[1] / 2],
            [center[0] - scale[0] / 2, center[1] + scale[1] / 2],
        ]
    )
    dst = np.float32([[0, 0], [w, 0], [0, h]])
    return cv2.getAffineTransform(src, dst)


def simcc_decode(simcc: np.ndarray, split_ratio: float = 2.0):
    """Arg-max with parabolic sub-bin refinement. simcc: (N, K, bins)."""
    idx = simcc.argmax(-1)
    peak = np.take_along_axis(simcc, idx[..., None], -1)[..., 0]
    bins = simcc.shape[-1]
    left = np.take_along_axis(simcc, np.clip(idx - 1, 0, bins - 1)[..., None], -1)[
        ..., 0
    ]
    right = np.take_along_axis(simcc, np.clip(idx + 1, 0, bins - 1)[..., None], -1)[
        ..., 0
    ]
    denom = left - 2 * peak + right
    with np.errstate(divide="ignore", invalid="ignore"):
        offset = np.where(denom < -1e-6, 0.5 * (left - right) / denom, 0.0)
    offset = np.clip(offset, -0.5, 0.5)
    interior = (idx > 0) & (idx < bins - 1)
    loc = idx + np.where(interior, offset, 0.0)
    return loc / split_ratio, peak


class PoseEstimator:
    """Batched RTMPose (SimCC) inference on per-camera person boxes."""

    def __init__(
        self,
        model: LoadedModel,
        skeleton: Skeleton,
        input_size=(192, 256),
        flip_test=False,
    ):
        self.model = model
        self.skeleton = skeleton
        self.input_size = tuple(input_size)
        self.flip_test = flip_test
        self.flip_index = np.array(skeleton.flip_index())

    @staticmethod
    def sample(input_size, batch) -> np.ndarray:
        w, h = input_size
        return np.zeros((batch, 3, h, w), np.float32)

    def crop(self, bgr: np.ndarray, box) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        center, scale = _crop_transform(np.asarray(box, float), self.input_size)
        matrix = _affine(center, scale, self.input_size)
        patch = cv2.warpAffine(bgr, matrix, self.input_size, flags=cv2.INTER_LINEAR)
        return patch, center, scale

    def __call__(self, jobs):
        """jobs: list of (bgr_image, box). Returns (N,K,2) keypoints, (N,K) scores."""
        if not jobs:
            k = self.skeleton.size
            return np.zeros((0, k, 2)), np.zeros((0, k))
        w, h = self.input_size
        patches, centers, scales = [], [], []
        for image, box in jobs:
            patch, center, scale = self.crop(image, box)
            patches.append(patch)
            centers.append(center)
            scales.append(scale)
        batch = np.stack(patches)[..., ::-1]  # BGR -> RGB
        if self.flip_test:
            batch = np.concatenate([batch, batch[:, :, ::-1]])
        x = ((batch.astype(np.float32) - MEAN) / STD).transpose(0, 3, 1, 2)
        simcc_x, simcc_y = self.model.run(np.ascontiguousarray(x))[:2]
        xs, sx = simcc_decode(simcc_x)
        ys, sy = simcc_decode(simcc_y)
        scores = 0.5 * (sx + sy)
        n = len(jobs)
        if self.flip_test:
            fx = (w - 1) - xs[n:][:, self.flip_index]
            fy = ys[n:][:, self.flip_index]
            fs = scores[n:][:, self.flip_index]
            xs = 0.5 * (xs[:n] + fx)
            ys = 0.5 * (ys[:n] + fy)
            scores = 0.5 * (scores[:n] + fs)
        centers = np.array(centers)[:, None, :]
        scales = np.array(scales)[:, None, :]
        local = np.stack([xs[:n], ys[:n]], -1) / np.array([w, h])
        keypoints = local * scales + centers - scales / 2
        return keypoints, np.clip(scores[:n], 0.0, 1.0)
