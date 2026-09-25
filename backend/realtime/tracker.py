"""Per-camera single-actor tracking for top-down pose estimation.

Running the person detector on every frame is the main cost of a top-down
pipeline. Instead, each camera follows its actor with the previous frame's
keypoints, extrapolated by their velocity and padded for fast motion, so the
crop keeps up with sprinting limbs. The detector runs asynchronously – on a
staggered schedule to catch drift and immediately when a track is lost – and
never blocks the frame loop.
"""

from __future__ import annotations

import numpy as np


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def keypoint_box(kpts, scores=None, min_score=0.2, margin=0.12):
    ok = np.isfinite(kpts).all(-1)
    if scores is not None:
        ok &= scores >= min_score
    if ok.sum() < 3:
        return None
    pts = kpts[ok]
    x1, y1 = pts.min(0)
    x2, y2 = pts.max(0)
    w, h = x2 - x1, y2 - y1
    # Keypoints sit inside the body outline (head top / feet extend past them).
    x1, x2 = x1 - margin * w, x2 + margin * w
    y1, y2 = y1 - (margin + 0.08) * h, y2 + margin * h
    return np.array([x1, y1, x2, y2], float)


class CameraTracker:
    def __init__(
        self,
        name,
        image_size,
        joints,
        det_interval=30,
        phase=0,
        min_score=0.3,
        min_joints=None,
    ):
        self.name = name
        self.w, self.h = image_size
        self.J = joints
        self.det_interval = max(1, int(det_interval))
        self.phase = phase
        self.min_score = min_score
        self.min_joints = min_joints or max(4, joints // 3)
        self.kpts = None
        self.scores = None
        self.vel = np.zeros((joints, 2))
        self.seed_box = None  # detector box used when there are no keypoints
        self.seed_age = 0
        self.last_box = None
        self.history: dict[int, np.ndarray] = {}
        self.pending_detection = False
        self.lost_frames = 0

    @property
    def tracking(self) -> bool:
        return self.kpts is not None

    def wants_detection(self, frame: int) -> bool:
        if self.pending_detection:
            return False
        if not self.tracking and self.seed_box is None:
            return True
        return (frame + self.phase) % self.det_interval == 0

    def crop_box(self):
        """Box for this frame's pose crop, or None if the actor is unknown."""
        if self.tracking:
            pred = self.kpts + self.vel
            box = keypoint_box(pred, self.scores, self.min_score * 0.7)
            if box is None:
                return None
            cur = keypoint_box(self.kpts, self.scores, self.min_score * 0.7)
            if cur is not None:  # cover both current and predicted positions
                box = np.array(
                    [
                        min(box[0], cur[0]),
                        min(box[1], cur[1]),
                        max(box[2], cur[2]),
                        max(box[3], cur[3]),
                    ]
                )
            speed = np.nanmax(np.abs(self.vel)) if np.isfinite(self.vel).any() else 0
            box += np.array([-1, -1, 1, 1]) * min(speed * 0.5, 0.25 * (box[3] - box[1]))
        elif self.seed_box is not None:
            box = self.seed_box.copy()
            grow = 1.0 + min(0.08 * self.seed_age, 0.8)
            c = (box[:2] + box[2:]) / 2
            half = (box[2:] - box[:2]) / 2 * grow
            box = np.concatenate([c - half, c + half])
            self.seed_age += 1
        else:
            return None
        box[[0, 2]] = box[[0, 2]].clip(-0.1 * self.w, 1.1 * self.w)
        box[[1, 3]] = box[[1, 3]].clip(-0.1 * self.h, 1.1 * self.h)
        if box[2] - box[0] < 8 or box[3] - box[1] < 16:
            return None
        return box

    def seed(self, box, frame_lag=0):
        self.seed_box = np.asarray(box[:4], float)
        self.seed_age = frame_lag

    def update(self, frame: int, box, kpts, scores) -> bool:
        """Accept the pose for this frame. Returns False if the track is lost."""
        good = np.isfinite(kpts).all(-1) & (scores >= self.min_score)
        in_image = (
            (kpts[:, 0] > -0.05 * self.w)
            & (kpts[:, 0] < 1.05 * self.w)
            & (kpts[:, 1] > -0.05 * self.h)
            & (kpts[:, 1] < 1.05 * self.h)
        )
        good &= in_image
        if good.sum() < self.min_joints:
            self.lose()
            return False
        if self.tracking:
            both = good & (self.scores >= self.min_score)
            step = kpts - self.kpts
            self.vel = np.where(
                both[:, None], 0.6 * step + 0.4 * self.vel, self.vel * 0.5
            )
        else:
            self.vel = np.zeros((self.J, 2))
        prev = self.kpts if self.tracking else np.full_like(kpts, np.nan)
        self.kpts = np.where(good[:, None], kpts, prev)
        self.vel = np.nan_to_num(self.vel)
        self.scores = np.where(good, scores, 0.0)
        self.last_box = box
        self.history[frame] = keypoint_box(kpts, scores, self.min_score)
        for old in [f for f in self.history if f < frame - 240]:
            del self.history[old]
        self.seed_box = None
        self.lost_frames = 0
        return True

    def lose(self):
        self.kpts = None
        self.scores = None
        self.vel = np.zeros((self.J, 2))
        self.lost_frames += 1

    def check_drift(self, frame: int, boxes: np.ndarray, min_iou=0.3):
        """Detector result for ``frame`` arrived: is our track still on a person?"""
        ref = self.history.get(frame)
        if ref is None or len(boxes) == 0:
            return True
        return max(iou(ref, b) for b in boxes) >= min_iou


def choose_box(boxes: np.ndarray, reference=None):
    """Pick the actor among detections: best overlap with a reference box
    (e.g. the projected 3D pose), otherwise the largest confident person."""
    if len(boxes) == 0:
        return None
    if reference is not None:
        scores = [iou(reference, b) for b in boxes]
        best = int(np.argmax(scores))
        if scores[best] > 0.1:
            return boxes[best]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return boxes[int(np.argmax(area * boxes[:, 4]))]
