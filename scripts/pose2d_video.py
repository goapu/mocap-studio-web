"""Single-camera 2D pose overlay using the real-time 2D stack.

Runs the same detector, pose network, keypoint tracker and One-Euro temporal
filter as the desktop app on one ordinary video and writes an MP4 with the
skeleton drawn on every frame. The on-screen numbers are measured during the
run on this computer. One camera gives 2D pose only; metric 3D needs two or
more calibrated cameras (use the desktop app for that).

    python scripts/pose2d_video.py input.mp4 output.mp4
    python scripts/pose2d_video.py input.mp4 output.mp4 --preset accurate --flip
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.realtime.engine import load_models  # noqa: E402
from backend.realtime.models import MODELS, PRESETS, ModelStore  # noqa: E402
from backend.realtime.pose2d import PersonDetector, PoseEstimator  # noqa: E402
from backend.realtime.skeletons import SKELETONS  # noqa: E402
from backend.realtime.temporal import OneEuro  # noqa: E402
from backend.realtime.tracker import CameraTracker, choose_box, iou  # noqa: E402

LEFT, RIGHT, CENTER = (255, 181, 63), (67, 159, 255), (235, 235, 235)


class Writer:
    """H.264 through ffmpeg when available, otherwise OpenCV's MPEG-4."""

    def __init__(self, path, size, fps):
        self.proc = None
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            w, h = size
            self.proc = subprocess.Popen(
                [
                    ffmpeg,
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "bgr24",
                    "-s",
                    f"{w}x{h}",
                    "-r",
                    str(fps),
                    "-i",
                    "-",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "23",
                    "-preset",
                    "medium",
                    "-movflags",
                    "+faststart",
                    str(path),
                ],
                stdin=subprocess.PIPE,
            )
        else:
            self.cv = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
            )

    def write(self, frame):
        if self.proc:
            self.proc.stdin.write(frame.tobytes())
        else:
            self.cv.write(frame)

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.cv.release()


def draw(frame, kp, score, skeleton, min_score, scale):
    thick = max(2, int(round(3 * scale)))
    for a, b in skeleton.bones:
        if (
            score[a] >= min_score
            and score[b] >= min_score
            and np.isfinite(kp[[a, b]]).all()
        ):
            name = skeleton.joints[b]
            color = (
                LEFT
                if name.startswith("left")
                else RIGHT
                if name.startswith("right")
                else CENTER
            )
            cv2.line(
                frame,
                tuple(np.int32(kp[a])),
                tuple(np.int32(kp[b])),
                color,
                thick,
                cv2.LINE_AA,
            )
    for j in range(len(kp)):
        if score[j] >= min_score and np.isfinite(kp[j]).all():
            cv2.circle(
                frame,
                tuple(np.int32(kp[j])),
                thick + 2,
                (255, 255, 255),
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                frame, tuple(np.int32(kp[j])), thick + 2, (20, 20, 20), 1, cv2.LINE_AA
            )


def hud(frame, lines, scale):
    h = int(34 * scale * len(lines) + 16 * scale)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], h), (20, 16, 13), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    for i, (text, color) in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (int(14 * scale), int((30 + 34 * i) * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72 * scale,
            color,
            max(1, int(2 * scale)),
            cv2.LINE_AA,
        )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--preset", default="balanced", choices=list(PRESETS))
    p.add_argument("--flip", action="store_true", help="flip test-time augmentation")
    p.add_argument("--provider", default="auto", choices=["auto", "gpu", "cpu"])
    p.add_argument("--width", type=int, default=720, help="output width (0 = source)")
    p.add_argument("--det-interval", type=int, default=15)
    p.add_argument("--min-score", type=float, default=0.3)
    p.add_argument("--no-hud", action="store_true")
    a = p.parse_args()

    cap = cv2.VideoCapture(a.input)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {a.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pose_key = PRESETS[a.preset]["pose"]
    skeleton = SKELETONS[MODELS[pose_key].skeleton]
    print(f"Loading {PRESETS[a.preset]['label']} …", flush=True)
    det_spec, det, pose_spec, pose = load_models(
        ModelStore(), a.preset, a.provider, 2 if a.flip else 1, print
    )
    detector = PersonDetector(det, det_spec.input_size)
    estimator = PoseEstimator(pose, skeleton, pose_spec.input_size, a.flip)
    tracker = CameraTracker(
        "video", (W, H), skeleton.size, a.det_interval, min_score=a.min_score
    )
    euro = OneEuro(min_cutoff=1.5, beta=0.03)

    out_w = a.width or W
    s = out_w / W
    out_h = int(round(H * s / 2)) * 2
    writer = Writer(a.output, (out_w, out_h), fps)
    times, detections, idx = [], 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / fps
            start = time.perf_counter()
            if idx == 0 or tracker.wants_detection(idx):
                boxes = detector(frame)
                detections += 1
                if len(boxes):
                    if not tracker.tracking:
                        tracker.seed(choose_box(boxes, tracker.last_box))
                    elif max(iou(tracker.last_box, b) for b in boxes) < 0.2:
                        tracker.lose()
                        tracker.seed(choose_box(boxes))
            box = tracker.crop_box()
            kp = np.full((skeleton.size, 2), np.nan)
            score = np.zeros(skeleton.size)
            if box is not None:
                k, sc = estimator([(frame, box)])
                if tracker.update(idx, box, k[0], sc[0]):
                    kp, score = k[0], sc[0]
            smooth = euro(np.where((score >= a.min_score)[:, None], kp, np.nan), t)
            times.append(time.perf_counter() - start)

            view = (
                cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
                if s != 1
                else frame
            )
            draw(view, smooth * s, score, skeleton, a.min_score, max(s * W / 1080, 0.6))
            if not a.no_hud:
                recent = times[-30:]
                live_fps = len(recent) / sum(recent) if sum(recent) else 0
                hud(
                    view,
                    [
                        (
                            "Mocap Studio - single camera, 2D pose + temporal filter",
                            (241, 235, 230),
                        ),
                        (
                            f"{MODELS[pose_key].key} | {pose.provider} | {live_fps:4.1f} fps processing "
                            f"| source {fps:.0f} fps",
                            (255, 181, 63),
                        ),
                        (f"frame {idx + 1}/{total}  t={t:5.2f}s", (166, 151, 139)),
                    ],
                    max(out_w / 720, 0.6),
                )
            writer.write(view)
            idx += 1
            if idx % 25 == 0:
                print(
                    f"  {idx}/{total} frames, {len(times) / sum(times):.1f} fps",
                    flush=True,
                )
    finally:
        cap.release()
        writer.close()
    mean_fps = len(times) / sum(times) if times else 0
    print(
        f"Done: {idx} frames · {mean_fps:.1f} fps processing (detector + pose + tracking + filter) "
        f"· {detections} detector runs · pose on {pose.provider} · wrote {a.output}"
    )


if __name__ == "__main__":
    main()
