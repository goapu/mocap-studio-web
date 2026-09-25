"""Real-time multi-camera 2D + 3D pose pipeline.

Threads (all bounded queues, so memory stays flat for any video length):

  decoders (1/camera) ─► inference ─► fusion ─► packet ring buffer ─► UI stream
                             ▲   │
                 detector ◄──┘   └ tracker crops (keypoint-driven)

* inference: tracker crops for every camera, one batched RTMPose call/frame;
* detector: YOLOX on a staggered schedule and on track loss (asynchronous);
* fusion: sigma model → left/right repair → robust triangulation → Kalman
  filter → bone lengths → reprojection → preview JPEGs → stream packet.

Every source frame is processed (no frame skipping). The UI never stalls the
pipeline: if a viewer falls behind, it skips ahead in the ring buffer while
the full-resolution result keeps being recorded for saving.
"""

from __future__ import annotations

import itertools
import json
import queue
import struct
import threading
import time
import traceback
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .models import MODELS, PRESETS, ModelStore, load_session
from .pose2d import PersonDetector, PoseEstimator, _crop_transform
from .skeletons import SKELETONS, Skeleton
from .sources import MultiVideo
from .temporal import (
    BoneLengths,
    Kalman3D,
    OneEuro,
    default_jerk,
    smooth_sequence,
)
from .tracker import CameraTracker, choose_box, keypoint_box
from .triangulation import CameraRig, repair_left_right, triangulate

VERSION = "1.0.0"


@dataclass
class RunConfig:
    videos: dict
    calibration: dict
    name: str = "capture"
    preset: str = "balanced"
    flip_test: bool = False
    det_interval: int = 30
    offsets: dict = field(default_factory=dict)
    preview_width: int = 0  # 0 = automatic
    jpeg_quality: int = 78
    realtime_pace: bool = False
    provider: str = "auto"
    min_score: float = 0.3
    smoothing: float = 1.0
    live_bone_constraint: bool = True
    sigma_input_px: float = 2.0

    def validate(self):
        if self.preset not in PRESETS:
            raise ValueError(f"Unknown model preset {self.preset!r}.")
        if not 0.05 <= self.min_score <= 0.95:
            raise ValueError(
                "Minimum keypoint confidence must be between 0.05 and 0.95."
            )
        if not 0.1 <= self.smoothing <= 10:
            raise ValueError("Smoothing must be between 0.1 and 10.")
        if not 1 <= self.det_interval <= 600:
            raise ValueError("Detector interval must be 1–600 frames.")
        missing = [n for n in self.videos if n not in self.calibration["cameras"]]
        if missing:
            raise ValueError(f"No calibration for camera(s): {', '.join(missing)}.")


# ---------------------------------------------------------------------------
# Model cache – CoreML compilation is slow, so sessions are reused across runs.

_MODEL_CACHE: dict = {}
_MODEL_LOCK = threading.Lock()


def load_models(store: ModelStore, preset: str, provider: str, batch: int, log):
    spec = PRESETS[preset]
    det_spec, pose_spec = MODELS[spec["detector"]], MODELS[spec["pose"]]
    for key in (det_spec.key, pose_spec.key):
        if not store.installed(key):
            raise ValueError(
                f"Model {key} is not installed. Download it from the Models panel."
            )
    with _MODEL_LOCK:
        key = (det_spec.key, provider)
        if key not in _MODEL_CACHE:
            store.verify(det_spec.key)
            _MODEL_CACHE[key] = load_session(
                store.path(det_spec.key),
                PersonDetector.sample(det_spec.input_size),
                provider,
                log=log,
            )
        det = _MODEL_CACHE[key]
        key = (pose_spec.key, provider, batch)
        if key not in _MODEL_CACHE:
            store.verify(pose_spec.key)
            _MODEL_CACHE[key] = load_session(
                store.path(pose_spec.key),
                PoseEstimator.sample(pose_spec.input_size, batch),
                provider,
                log=log,
            )
        pose = _MODEL_CACHE[key]
    return det_spec, det, pose_spec, pose


# ---------------------------------------------------------------------------
# Packet ring buffer for streaming to the UI


def encode_packet(header: dict, images: list[bytes]) -> bytes:
    header = dict(header, jpeg_sizes=[len(b) for b in images])
    head = json.dumps(header, separators=(",", ":"), allow_nan=False).encode()
    body = struct.pack(">I", len(head)) + head + b"".join(images)
    return struct.pack(">I", len(body)) + body


class PacketBuffer:
    def __init__(self, capacity: int):
        self.items: deque = deque(maxlen=capacity)
        self.cond = threading.Condition()
        self.next_seq = 0
        self.closed = False

    def push(self, data: bytes):
        with self.cond:
            self.items.append((self.next_seq, data))
            self.next_seq += 1
            self.cond.notify_all()

    def close(self):
        with self.cond:
            self.closed = True
            self.cond.notify_all()

    def iterate(self, start: int = 0, stop_event: threading.Event | None = None):
        cursor = start
        while True:
            with self.cond:
                while (
                    not (self.items and self.items[-1][0] >= cursor) and not self.closed
                ):
                    self.cond.wait(timeout=0.5)
                    if stop_event is not None and stop_event.is_set():
                        return
                available = [d for s, d in self.items if s >= cursor]
                if self.items:
                    oldest = self.items[0][0]
                    cursor = max(cursor, oldest)
                if not available and self.closed:
                    return
                if available:
                    cursor = self.items[-1][0] + 1
            for data in available:
                yield data


def _max_axis(a):
    a = np.asarray(a, float)
    ok = np.isfinite(a).any(-1)
    return np.where(ok, np.max(np.where(np.isfinite(a), a, -np.inf), -1), np.nan)


def _clean(value, digits=4):
    """JSON-safe nested lists with NaN → None."""
    arr = np.asarray(value, float)
    out = np.round(arr, digits).astype(object)
    out[~np.isfinite(arr)] = None
    return out.tolist()


# ---------------------------------------------------------------------------


class Run:
    def __init__(self, config: RunConfig, store: ModelStore):
        config.validate()
        self.id = uuid.uuid4().hex[:12]
        self.config = config
        self.store = store
        self.created = time.time()
        self.status = "starting"
        self.message = "Loading…"
        self.error = None
        self.log_lines: list[str] = []
        self._stop = threading.Event()
        self.buffer = PacketBuffer(capacity=600)
        self.final = None
        self.stats = {
            "frames_done": 0,
            "total_frames": 0,
            "processing_fps": 0.0,
            "source_fps": 0.0,
            "realtime_factor": 0.0,
            "latency_ms": 0.0,
            "pose_ms": 0.0,
            "fusion_ms": 0.0,
            "decode_buffer": 0,
            "detector_runs": 0,
            "provider": None,
            "coverage_3d": 0.0,
            "mean_reproj_px": None,
            "left_right_repairs": 0,
        }
        self.thread = threading.Thread(
            target=self._main, daemon=True, name=f"run-{self.id}"
        )

    # -- public -----------------------------------------------------------
    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self._stop.set()

    def log(self, text):
        self.log_lines.append(f"{time.strftime('%H:%M:%S')} {text}")
        del self.log_lines[:-200]

    def summary(self) -> dict:
        return {
            "id": self.id,
            "name": self.config.name,
            "status": self.status,
            "message": self.message,
            "error": self.error,
            "stats": self.stats,
            "cameras": getattr(self, "names", list(self.config.videos)),
            "skeleton": self.skeleton.to_json() if hasattr(self, "skeleton") else None,
            "world_frame": self.config.calibration.get("world_frame", "camera"),
            "preset": self.config.preset,
            "flip_test": self.config.flip_test,
            "has_final": self.final is not None,
            "quality": self.final["quality"] if self.final is not None else None,
            "rig": [
                {
                    "name": n,
                    "center": _clean(c, 4),
                    "forward": _clean(r[2], 4),
                    "up": _clean(-r[1], 4),
                }
                for n, c, r in zip(self.rig.names, self.rig.centers(), self.rig.R)
            ]
            if hasattr(self, "rig")
            else None,
            "fps": getattr(self, "fps", None),
            "log": self.log_lines[-12:],
        }

    # -- setup ------------------------------------------------------------
    def _setup(self):
        cfg = self.config
        self.video = MultiVideo(cfg.videos, cfg.offsets)
        self.names = self.video.names
        self.fps = self.video.fps
        self.rig = CameraRig.from_calibration(
            cfg.calibration, self.names, self.video.sizes()
        )
        C = len(self.names)
        pose_key = PRESETS[cfg.preset]["pose"]
        self.skeleton: Skeleton = SKELETONS[MODELS[pose_key].skeleton]
        J = self.skeleton.size
        batch = C * (2 if cfg.flip_test else 1)
        self.message = (
            "Loading models (first use on this Mac compiles them for the GPU)…"
        )
        det_spec, det, pose_spec, pose = load_models(
            self.store, cfg.preset, cfg.provider, batch, self.log
        )
        self.detector = PersonDetector(det, det_spec.input_size)
        self.pose = PoseEstimator(
            pose, self.skeleton, pose_spec.input_size, cfg.flip_test
        )
        self.stats["provider"] = f"pose: {pose.provider} · detector: {det.provider}"
        self.stats["source_fps"] = round(self.fps, 3)
        self.stats["total_frames"] = self.video.frames
        self.trackers = [
            CameraTracker(
                n,
                self.video.sizes()[n],
                J,
                cfg.det_interval,
                phase=(i * cfg.det_interval) // max(C, 1),
                min_score=cfg.min_score,
            )
            for i, n in enumerate(self.names)
        ]
        self.kalman = Kalman3D(J, default_jerk(self.skeleton) * cfg.smoothing)
        self.bones = BoneLengths(self.skeleton)
        self.euro = [OneEuro() for _ in self.names]
        self.latest3d = None
        self.lock = threading.Lock()
        widths = [self.video.sizes()[n][0] for n in self.names]
        target = cfg.preview_width or (960 if C <= 2 else 720 if C <= 4 else 560)
        self.preview_scale = [min(1.0, target / w) for w in widths]
        cap = max(self.video.frames, 1)
        self.rec = {
            "t": np.full(cap, np.nan),
            "kp": np.full((cap, C, J, 2), np.nan, np.float32),
            "score": np.zeros((cap, C, J), np.float32),
            "box": np.full((cap, C, 4), np.nan, np.float32),
            "X": np.full((cap, J, 3), np.nan),
            "cov": np.tile(np.eye(3), (cap, J, 1, 1)),
            "valid": np.zeros((cap, J), bool),
            "status": np.zeros((cap, J), np.int8),
            "used": np.zeros((cap, C, J), bool),
            "err": np.full((cap, C, J), np.nan, np.float32),
            "live": np.full((cap, J, 3), np.nan),
            "live_sigma": np.full((cap, J, 3), np.nan),
        }
        self.encoder = ThreadPoolExecutor(
            max_workers=min(C, 4), thread_name_prefix="jpeg"
        )

    def _grow(self, idx):
        cap = len(self.rec["t"])
        if idx < cap:
            return
        extra = max(cap, 256)
        for k, v in self.rec.items():
            pad = np.full(
                (extra,) + v.shape[1:], np.nan if v.dtype.kind == "f" else 0, v.dtype
            )
            if k == "cov":
                pad[:] = np.eye(3)
            self.rec[k] = np.concatenate([v, pad])

    # -- threads ----------------------------------------------------------
    def _main(self):
        try:
            self._setup()
            self.status = "running"
            self.message = "Processing"
            self.log(
                f"{len(self.names)} cameras · {self.fps:.2f} fps · {self.stats['provider']}"
            )
            self.fusion_q: queue.Queue = queue.Queue(maxsize=6)
            self.detect_q: queue.Queue = queue.Queue(maxsize=len(self.names))
            self.detect_results: queue.Queue = queue.Queue()
            fusion = threading.Thread(
                target=self._fusion_loop, daemon=True, name="fusion"
            )
            detector = threading.Thread(
                target=self._detect_loop, daemon=True, name="detector"
            )
            fusion.start()
            detector.start()
            self.video.start()
            try:
                self._inference_loop()
            finally:
                self.video.stop()
                self.fusion_q.put(None)
                fusion.join()
                self.detect_q.put(None)
                detector.join(timeout=5)
                self.encoder.shutdown(wait=True)
            if self.error:
                raise RuntimeError(self.error)
            self.buffer.close()
            n = self.stats["frames_done"]
            if n == 0:
                raise RuntimeError("No frames were processed.")
            self.status = "finalizing"
            self.message = "Smoothing the full recording (uses past and future frames)…"
            self._finalize(n)
            self.status = "stopped" if self._stop.is_set() else "finished"
            self.message = (
                f"Stopped after {n} frames. Results up to that point are ready to save."
                if self._stop.is_set()
                else f"Processed all {n} frames."
            )
        except Exception as exc:
            self.error = self.error or str(exc)
            self.status = "error"
            self.message = self.error
            self.log(traceback.format_exc(limit=3))
            self.buffer.close()

    def _detect_loop(self):
        while True:
            job = self.detect_q.get()
            if job is None:
                return
            c, idx, frame = job
            try:
                boxes = self.detector(frame)
                self.stats["detector_runs"] += 1
            except Exception as exc:
                self.log(f"detector error: {exc}")
                boxes = np.zeros((0, 5))
            self.detect_results.put((c, idx, boxes))

    def _reference_box(self, c):
        with self.lock:
            latest = self.latest3d
        if latest is None:
            return None
        uv, depth = self.rig.project(latest)
        pts = np.where((depth[c] > 0)[:, None], uv[c], np.nan)
        return keypoint_box(pts)

    def _apply_detections(self, idx):
        while True:
            try:
                c, fidx, boxes = self.detect_results.get_nowait()
            except queue.Empty:
                return
            tr = self.trackers[c]
            tr.pending_detection = False
            if len(boxes) == 0:
                continue
            if tr.tracking:
                if not tr.check_drift(fidx, boxes):
                    self.log(f"{tr.name}: track drifted off the actor; re-acquiring")
                    tr.lose()
                else:
                    continue
            box = choose_box(boxes, self._reference_box(c))
            if box is not None:
                tr.seed(box, frame_lag=idx - fidx)

    def _associate(self, idx, frames):
        """First frame: detect everywhere and pick the same person in all views."""
        C = len(self.names)
        cands = []
        for name in self.names:
            boxes = self.detector(frames[name])
            self.stats["detector_runs"] += 1
            if len(boxes):
                order = np.argsort(
                    -(boxes[:, 2] - boxes[:, 0])
                    * (boxes[:, 3] - boxes[:, 1])
                    * boxes[:, 4]
                )
                boxes = boxes[order[: (3 if C <= 4 else 2)]]
            cands.append(boxes)
        if all(len(b) <= 1 for b in cands):
            for c, b in enumerate(cands):
                if len(b):
                    self.trackers[c].seed(b[0])
            return
        jobs, owner = [], []
        for c, boxes in enumerate(cands):
            for k, b in enumerate(boxes):
                jobs.append((frames[self.names[c]], b))
                owner.append((c, k))
        kp, sc = self.pose(jobs)
        lookup = {o: i for i, o in enumerate(owner)}
        torso = [5, 6, 11, 12]
        options = [range(len(b)) if len(b) else [None] for b in cands]
        best, best_cost = None, np.inf
        for combo in itertools.product(*options):
            uv = np.full((C, len(torso), 2), np.nan)
            sig = np.full((C, len(torso)), np.nan)
            for c, k in enumerate(combo):
                if k is None:
                    continue
                i = lookup[(c, k)]
                ok = sc[i, torso] >= self.config.min_score
                uv[c, ok] = kp[i, torso][ok]
                sig[c, ok] = 3.0
            res = triangulate(self.rig, uv, sig, gate_px=1e9)
            err = res["err_px"][res["used"]]
            views = sum(k is not None for k in combo)
            cost = (np.median(err) if err.size else 1e6) + 50 * (C - views)
            if cost < best_cost:
                best, best_cost = combo, cost
        for c, k in enumerate(best or []):
            if k is not None:
                self.trackers[c].seed(cands[c][k])
        self.log(
            f"Actor association across views (torso reprojection {best_cost:.1f} px)"
        )

    def _inference_loop(self):
        cfg = self.config
        C = len(self.names)
        J = self.skeleton.size
        wall0 = time.perf_counter()
        in_h = self.pose.input_size[1]
        while not self._stop.is_set():
            item = self.video.read()
            if item is None:
                break
            idx, frames = item
            t = idx / self.fps
            t_read = time.perf_counter()
            if cfg.realtime_pace:
                delay = wall0 + t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            if idx == 0:
                self._associate(idx, frames)
            self._apply_detections(idx)
            jobs, which, crop_h, boxes = (
                [],
                [],
                np.full(C, np.nan),
                np.full((C, 4), np.nan),
            )
            for c, name in enumerate(self.names):
                tr = self.trackers[c]
                box = tr.crop_box()
                if box is not None:
                    jobs.append((frames[name], box))
                    which.append(c)
                    boxes[c] = box
                    _, scale = _crop_transform(box, self.pose.input_size)
                    crop_h[c] = scale[1]
                if tr.wants_detection(idx):
                    try:
                        self.detect_q.put_nowait((c, idx, frames[name]))
                        tr.pending_detection = True
                    except queue.Full:
                        pass
            start = time.perf_counter()
            kp, sc = self.pose(jobs)
            pose_ms = (time.perf_counter() - start) * 1000
            K2 = np.full((C, J, 2), np.nan)
            S2 = np.zeros((C, J))
            for i, c in enumerate(which):
                if self.trackers[c].update(idx, boxes[c], kp[i], sc[i]):
                    K2[c], S2[c] = kp[i], sc[i]
                else:
                    boxes[c] = np.nan
            sigma = np.where(
                S2 >= cfg.min_score,
                cfg.sigma_input_px * (crop_h / in_h)[:, None] / np.clip(S2, 0.05, 1.0),
                np.nan,
            )
            self.stats["pose_ms"] = round(
                0.9 * self.stats["pose_ms"] + 0.1 * pose_ms, 2
            )
            self.stats["decode_buffer"] = self.video.buffered()
            while not self._stop.is_set():
                try:
                    self.fusion_q.put(
                        (idx, t, frames, K2, S2, sigma, boxes, t_read), timeout=0.2
                    )
                    break
                except queue.Full:
                    continue

    def _fusion_loop(self):
        cfg = self.config
        C = len(self.names)
        rate_window: deque = deque(maxlen=30)
        gate = max(6.0, 0.004 * float(np.hypot(*self.rig.image_size.mean(0))))
        repairs = 0
        while True:
            item = self.fusion_q.get()
            if item is None:
                return
            if self.error:
                continue
            try:
                idx, t, frames, K2, S2, sigma, boxes, t_read = item
                start = time.perf_counter()
                self._grow(idx)
                uv = np.where(np.isfinite(sigma)[..., None], K2, np.nan)
                self.kalman.predict(t)
                pred, _ = self.kalman.state()
                if np.isfinite(pred).any():
                    pred_uv, _ = self.rig.project(pred)
                    uv, sigma, swapped = repair_left_right(
                        uv, sigma, pred_uv, self.skeleton.swap_groups
                    )
                    repairs += int(swapped.sum())
                res = triangulate(self.rig, uv, sigma, gate_px=gate)
                ok = np.isfinite(res["X"]).all(-1)
                cov = np.where(ok[:, None, None], res["cov"], np.eye(3))
                self.kalman.update(np.nan_to_num(res["X"]), cov, ok)
                Xf, sf = self.kalman.state()
                self.bones.observe(Xf, sf)
                X_live = self.bones.apply(Xf) if cfg.live_bone_constraint else Xf
                with self.lock:
                    self.latest3d = Xf
                reproj, depth = self.rig.project(X_live)
                reproj = np.where((depth > 0)[..., None], reproj, np.nan)
                disp2d = np.stack([self.euro[c](K2[c], t) for c in range(C)])
                r = self.rec
                r["t"][idx] = t
                r["kp"][idx], r["score"][idx], r["box"][idx] = K2, S2, boxes
                r["X"][idx], r["cov"][idx], r["valid"][idx] = res["X"], cov, ok
                r["status"][idx], r["used"][idx], r["err"][idx] = (
                    res["status"],
                    res["used"],
                    res["err_px"],
                )
                r["live"][idx], r["live_sigma"][idx] = X_live, sf
                fusion_ms = (time.perf_counter() - start) * 1000
                images = list(
                    self.encoder.map(
                        self._encode, range(C), [frames[n] for n in self.names]
                    )
                )
                done = idx + 1
                now = time.perf_counter()
                rate_window.append(now)
                s = self.stats
                s["frames_done"] = done
                if len(rate_window) > 1:
                    s["processing_fps"] = round(
                        (len(rate_window) - 1) / (rate_window[-1] - rate_window[0]), 2
                    )
                s["realtime_factor"] = (
                    round(s["processing_fps"] / self.fps, 3) if self.fps else 0
                )
                s["latency_ms"] = round((now - t_read) * 1000, 1)
                s["fusion_ms"] = round(0.9 * s["fusion_ms"] + 0.1 * fusion_ms, 2)
                s["left_right_repairs"] = repairs
                valid_hist = r["valid"][max(0, done - 300) : done]
                s["coverage_3d"] = round(float(valid_hist.mean()) * 100, 1)
                errs = r["err"][max(0, done - 300) : done][
                    r["used"][max(0, done - 300) : done]
                ]
                s["mean_reproj_px"] = (
                    round(float(errs.mean()), 2) if errs.size else None
                )
                header = self._header(
                    idx, t, disp2d, S2, boxes, reproj, X_live, sf, res
                )
                self.buffer.push(encode_packet(header, images))
            except Exception as exc:
                self.error = (
                    f"Processing failed at frame {item[0] if item else '?'}: {exc}"
                )
                self.log(traceback.format_exc(limit=4))
                self._stop.set()

    def _encode(self, c, frame):
        s = self.preview_scale[c]
        if s < 1:
            frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
        )
        return buf.tobytes() if ok else b""

    def _header(
        self, idx, t, kp2d, scores, boxes, reproj, X, sigma, res, variant="live"
    ):
        cams = []
        for c, name in enumerate(self.names):
            w, h = self.rig.image_size[c]
            kp = (
                np.concatenate([kp2d[c], scores[c][:, None]], -1)
                if kp2d is not None
                else None
            )
            cams.append(
                {
                    "name": name,
                    "w": int(w),
                    "h": int(h),
                    "kp": _clean(kp, 1) if kp is not None else None,
                    "reproj": _clean(reproj[c], 1),
                    "box": _clean(boxes[c], 1) if boxes is not None else None,
                    "used": res["used"][c].tolist() if res is not None else None,
                }
            )
        return {
            "type": "frame",
            "variant": variant,
            "frame": int(idx),
            "t": round(float(t), 5),
            "fps": self.fps,
            "total": self.stats["total_frames"],
            "cams": cams,
            "X": _clean(X, 4),
            "sigma_mm": _clean(_max_axis(sigma) * 1000, 1)
            if sigma is not None
            else None,
            "stats": self.stats if idx % 5 == 0 else None,
        }

    # -- finalisation -------------------------------------------------------
    def _finalize(self, n):
        r = {k: v[:n] for k, v in self.rec.items()}
        times = np.where(np.isfinite(r["t"]), r["t"], np.arange(n) / self.fps)
        jerk = default_jerk(self.skeleton) * self.config.smoothing
        _, smoothed, sig = smooth_sequence(
            times, np.nan_to_num(r["X"]), r["cov"], r["valid"], jerk
        )
        bones = BoneLengths(self.skeleton, history=100000)
        lengths = bones.fit_sequence(smoothed, sig)
        constrained = bones.apply(smoothed, lengths)
        C = len(self.names)
        J = self.skeleton.size

        def reproject(X):
            flat = X.reshape(-1, 3)
            uv, depth = self.rig.project(flat)
            uv = np.where((depth > 0)[..., None], uv, np.nan)
            return uv.reshape(C, n, J, 2).transpose(1, 0, 2, 3)

        reproj = reproject(smoothed)
        obs = r["kp"].astype(float)
        good = r["score"] >= self.config.min_score
        d = np.linalg.norm(reproj - obs, axis=-1)
        d = d[good & np.isfinite(d)]
        edges = self.skeleton.limb_tree

        def bone_cv(X):
            out = []
            for a, b in edges:
                L = np.linalg.norm(X[:, b] - X[:, a], axis=-1)
                L = L[np.isfinite(L)]
                out.append(float(np.std(L) / np.mean(L)) if len(L) > 5 else None)
            return out

        self.final = {
            "times": times,
            "smoothed": smoothed,
            "sigma": sig,
            "constrained": constrained,
            "reproj": reproj,
            "bone_lengths": lengths,
            "quality": {
                "frames": int(n),
                "coverage_raw_pct": round(float(r["valid"].mean()) * 100, 2),
                "coverage_smoothed_pct": round(
                    float(np.isfinite(smoothed).all(-1).mean()) * 100, 2
                ),
                "per_joint_coverage_pct": _clean(r["valid"].mean(0) * 100, 1),
                "median_reprojection_px_smoothed": round(float(np.median(d)), 2)
                if d.size
                else None,
                "bone_length_cv_raw": bone_cv(r["X"]),
                "bone_length_cv_smoothed": bone_cv(smoothed),
                "bone_lengths_m": _clean(lengths, 4),
                "left_right_repairs": self.stats["left_right_repairs"],
                "processing_fps": self.stats["processing_fps"],
                "source_fps": self.fps,
                "provider": self.stats["provider"],
            },
        }
        self.log(
            f"Final: {self.final['quality']['coverage_smoothed_pct']}% joints, "
            f"median reprojection {self.final['quality']['median_reprojection_px_smoothed']} px"
        )

    # -- replay -------------------------------------------------------------
    def replay(self, variant="smoothed", stop_event=None):
        """Yield packets of the saved result over freshly decoded frames."""
        if self.final is None:
            raise ValueError("The result is not ready yet.")
        f = self.final
        X_all = {
            "smoothed": f["smoothed"],
            "constrained": f["constrained"],
            "live": self.rec["live"][: len(f["times"])],
        }.get(variant, f["smoothed"])
        n = len(f["times"])
        video = MultiVideo(self.config.videos, self.config.offsets).start()
        try:
            for i in range(n):
                if stop_event is not None and stop_event.is_set():
                    return
                item = video.read()
                if item is None:
                    return
                _, frames = item
                X = X_all[i]
                reproj, depth = self.rig.project(X)
                reproj = np.where((depth > 0)[..., None], reproj, np.nan)
                images = [
                    self._encode(c, frames[nm]) for c, nm in enumerate(self.names)
                ]
                kp = self.rec["kp"][i].astype(float)
                header = self._header(
                    i,
                    f["times"][i],
                    kp,
                    self.rec["score"][i],
                    None,
                    reproj,
                    X,
                    f["sigma"][i],
                    None,
                    variant=variant,
                )
                header["stats"] = None
                yield encode_packet(header, images)
        finally:
            video.stop()

    def config_json(self):
        cfg = asdict(self.config)
        cfg.pop("calibration")
        return cfg
