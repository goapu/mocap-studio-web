"""Prefetching, synchronised multi-camera video decoding.

Each camera decodes on its own thread into a bounded queue, so decoding runs
ahead of inference and never makes the pipeline wait on disk or codec latency.
Frames are aligned by index (constant-frame-rate, frame-zero synchronised
footage) with optional per-camera frame offsets for manual sync correction.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2

_END = object()


@dataclass
class VideoInfo:
    name: str
    path: str
    fps: float
    frames: int
    width: int
    height: int

    def to_json(self):
        return self.__dict__.copy()


def probe(name: str, path: str) -> VideoInfo:
    if not Path(path).is_file():
        raise ValueError(f"{name}: video file not found: {path}")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"{name}: cannot open video {Path(path).name}.")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"{name}: video has no readable frames.")
            height, width = frame.shape[:2]
    finally:
        cap.release()
    if not (1 <= fps <= 1000):
        raise ValueError(f"{name}: video reports an invalid frame rate ({fps}).")
    return VideoInfo(name, str(path), fps, frames, width, height)


class _Reader(threading.Thread):
    def __init__(
        self, info: VideoInfo, offset: int, prefetch: int, stop: threading.Event
    ):
        super().__init__(daemon=True, name=f"decode-{info.name}")
        self.info = info
        self.offset = offset
        self.queue: queue.Queue = queue.Queue(maxsize=prefetch)
        self.stop = stop
        self.error: Exception | None = None

    def _put(self, item):
        while not self.stop.is_set():
            try:
                self.queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def run(self):
        cap = cv2.VideoCapture(self.info.path)
        try:
            for _ in range(self.offset):
                if not cap.grab():
                    break
            while not self.stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                if not self._put(frame):
                    break
        except Exception as exc:  # surfaced to the engine
            self.error = exc
        finally:
            cap.release()
            self._put(_END)


class MultiVideo:
    """Synchronised frame groups from several camera videos."""

    def __init__(
        self,
        videos: dict[str, str],
        offsets: dict[str, int] | None = None,
        prefetch: int = 24,
        fps_tolerance: float = 0.01,
    ):
        if len(videos) < 2:
            raise ValueError(
                "Provide at least two camera videos for 3D reconstruction."
            )
        self.offsets = {k: int((offsets or {}).get(k, 0)) for k in videos}
        if any(v < 0 for v in self.offsets.values()):
            raise ValueError("Sync offsets must be zero or positive frame counts.")
        self.infos = {name: probe(name, path) for name, path in videos.items()}
        fps = [i.fps for i in self.infos.values()]
        if max(fps) - min(fps) > fps_tolerance * max(fps):
            detail = ", ".join(f"{i.name} {i.fps:.3f}" for i in self.infos.values())
            raise ValueError(
                f"Camera frame rates differ ({detail}). Record all cameras at the same "
                "constant frame rate."
            )
        self.fps = sum(fps) / len(fps)
        counts = [
            i.frames - self.offsets[n] for n, i in self.infos.items() if i.frames > 0
        ]
        self.frames = max(0, min(counts)) if counts else 0
        self.prefetch = prefetch
        self._stop = threading.Event()
        self.readers: dict[str, _Reader] = {}
        self.index = 0

    @property
    def names(self):
        return list(self.infos)

    def sizes(self):
        return {n: (i.width, i.height) for n, i in self.infos.items()}

    def start(self):
        self.readers = {
            n: _Reader(i, self.offsets[n], self.prefetch, self._stop)
            for n, i in self.infos.items()
        }
        for r in self.readers.values():
            r.start()
        return self

    def buffered(self) -> int:
        return min((r.queue.qsize() for r in self.readers.values()), default=0)

    def read(self, timeout: float = 30.0):
        """Next (index, {camera: frame}) or None at the end of any stream."""
        group = {}
        for name, reader in self.readers.items():
            try:
                item = reader.queue.get(timeout=timeout)
            except queue.Empty:
                raise TimeoutError(f"{name}: video decoding stalled.") from None
            if item is _END:
                if reader.error:
                    raise RuntimeError(f"{name}: decoding failed: {reader.error}")
                return None
            group[name] = item
        idx = self.index
        self.index += 1
        return idx, group

    def stop(self):
        self._stop.set()
        for r in self.readers.values():
            # Drain so a blocked producer can exit promptly.
            try:
                while True:
                    r.queue.get_nowait()
            except queue.Empty:
                pass
        for r in self.readers.values():
            r.join(timeout=2)
