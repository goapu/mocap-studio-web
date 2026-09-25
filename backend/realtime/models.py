"""Model registry, verified downloads and accelerated ONNX Runtime sessions.

On Apple Silicon the CoreML execution provider runs the networks on the GPU /
Neural Engine. CUDA and DirectML are used when present. Every accelerated
session is benchmarked against the CPU at load time and the faster one wins,
so an unsupported operator partition can never make the app slower.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.getenv("MOCAP_MODEL_DIR", ROOT / ".local" / "models"))
_BASE = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    kind: str  # detector | pose
    url: str
    input_size: tuple[int, int]  # (width, height)
    skeleton: str | None = None
    sha256: str | None = None  # pinned hash; None = recorded on first download
    approx_mb: int = 0

    @property
    def file(self) -> str:
        return Path(self.url).stem + ".onnx"


MODELS = {
    m.key: m
    for m in [
        ModelSpec(
            "yolox-m",
            "detector",
            _BASE + "yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
            (640, 640),
            sha256="3dea6513388889f0fff4b77bf7a26013600321b9eb9ceb0e9a400a82572f5f23",
            approx_mb=97,
        ),
        ModelSpec(
            "rtmpose-m",
            "pose",
            _BASE + "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
            (192, 256),
            skeleton="coco17",
            sha256="5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c",
            approx_mb=52,
        ),
        ModelSpec(
            "rtmpose-x",
            "pose",
            _BASE + "rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip",
            (288, 384),
            skeleton="coco17",
            approx_mb=190,
        ),
        ModelSpec(
            "rtmpose-m-feet",
            "pose",
            _BASE
            + "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",
            (192, 256),
            skeleton="halpe26",
            approx_mb=55,
        ),
        ModelSpec(
            "rtmpose-x-feet",
            "pose",
            _BASE
            + "rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip",
            (288, 384),
            skeleton="halpe26",
            approx_mb=195,
        ),
    ]
}

PRESETS = {
    "balanced": {
        "label": "Balanced — RTMPose-M 256×192 (17 joints)",
        "detector": "yolox-m",
        "pose": "rtmpose-m",
    },
    "balanced-feet": {
        "label": "Balanced + feet — RTMPose-M 256×192 (26 joints incl. heels/toes)",
        "detector": "yolox-m",
        "pose": "rtmpose-m-feet",
    },
    "accurate": {
        "label": "High accuracy — RTMPose-X 384×288 (17 joints)",
        "detector": "yolox-m",
        "pose": "rtmpose-x",
    },
    "accurate-feet": {
        "label": "Highest accuracy — RTMPose-X 384×288 (26 joints incl. heels/toes)",
        "detector": "yolox-m",
        "pose": "rtmpose-x-feet",
    },
}


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelStore:
    """Local model folder with a manifest of verified hashes."""

    def __init__(self, directory: Path = MODEL_DIR):
        self.dir = Path(directory)
        self.lock = threading.Lock()
        self.progress: dict[str, dict] = {}

    @property
    def manifest_path(self) -> Path:
        return self.dir / "realtime_manifest.json"

    def _manifest(self) -> dict:
        try:
            return json.loads(self.manifest_path.read_text())
        except (OSError, ValueError):
            return {}

    def path(self, key: str) -> Path:
        return self.dir / MODELS[key].file

    def expected_hash(self, key: str) -> str | None:
        return MODELS[key].sha256 or self._manifest().get(key, {}).get("sha256")

    def installed(self, key: str) -> bool:
        return self.path(key).is_file()

    def status(self) -> dict:
        result = {}
        for key, spec in MODELS.items():
            result[key] = {
                "installed": self.installed(key),
                "kind": spec.kind,
                "skeleton": spec.skeleton,
                "input_size": list(spec.input_size),
                "approx_mb": spec.approx_mb,
                "hash_pinned": spec.sha256 is not None,
                "download": self.progress.get(key),
            }
        return result

    def verify(self, key: str) -> None:
        expected = self.expected_hash(key)
        if expected and _digest(self.path(key)) != expected:
            raise ValueError(
                f"{MODELS[key].file}: checksum mismatch. Delete the file and download it again."
            )

    def download(self, key: str) -> Path:
        spec = MODELS[key]
        target = self.path(key)
        if target.is_file():
            self.verify(key)
            return target
        self.dir.mkdir(parents=True, exist_ok=True)
        self.progress[key] = {"state": "downloading", "bytes": 0}
        try:
            with tempfile.TemporaryDirectory(dir=self.dir) as tmp:
                archive = Path(tmp) / "model.zip"
                with (
                    urlopen(spec.url, timeout=120) as source,
                    archive.open("wb") as out,
                ):
                    total = int(source.headers.get("Content-Length") or 0)
                    self.progress[key]["total"] = total
                    while chunk := source.read(1 << 20):
                        out.write(chunk)
                        self.progress[key]["bytes"] += len(chunk)
                with zipfile.ZipFile(archive) as bundle:
                    members = [
                        m for m in bundle.infolist() if m.filename.endswith(".onnx")
                    ]
                    if len(members) != 1:
                        raise ValueError(f"{key}: archive must contain one ONNX file.")
                    extracted = Path(tmp) / spec.file
                    with bundle.open(members[0]) as src, extracted.open("wb") as out:
                        shutil.copyfileobj(src, out)
                digest = _digest(extracted)
                if spec.sha256 and digest != spec.sha256:
                    raise ValueError(f"{key}: checksum mismatch after download.")
                extracted.replace(target)
            with self.lock:
                manifest = self._manifest()
                manifest[key] = {
                    "file": spec.file,
                    "source": spec.url,
                    "sha256": digest,
                    "pinned": spec.sha256 is not None,
                    "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                self.manifest_path.write_text(json.dumps(manifest, indent=2))
            self.progress[key] = {"state": "done"}
            return target
        except Exception as exc:
            self.progress[key] = {"state": "error", "message": str(exc)}
            raise


# ---------------------------------------------------------------------------
# Execution providers


def available_providers() -> list[str]:
    import onnxruntime as ort

    return ort.get_available_providers()


def _provider_candidates(prefer: str) -> list[tuple[str, list]]:
    """Ordered provider configurations: (label, providers argument)."""
    available = available_providers()
    cache = str(MODEL_DIR / "coreml_cache")
    out: list[tuple[str, list]] = []
    if prefer in ("auto", "gpu"):
        if "CUDAExecutionProvider" in available:
            out.append(("CUDA", ["CUDAExecutionProvider", "CPUExecutionProvider"]))
        if "CoreMLExecutionProvider" in available:
            out.append(
                (
                    "CoreML (GPU/Neural Engine)",
                    [
                        (
                            "CoreMLExecutionProvider",
                            {
                                "ModelFormat": "MLProgram",
                                "MLComputeUnits": "ALL",
                                "ModelCacheDirectory": cache,
                            },
                        ),
                        "CPUExecutionProvider",
                    ],
                )
            )
            # Older runtimes reject the option dictionary; keep a plain entry.
            out.append(
                (
                    "CoreML (GPU/Neural Engine)",
                    ["CoreMLExecutionProvider", "CPUExecutionProvider"],
                )
            )
        if "DmlExecutionProvider" in available:
            out.append(("DirectML", ["DmlExecutionProvider", "CPUExecutionProvider"]))
    out.append(("CPU", ["CPUExecutionProvider"]))
    return out


@dataclass
class LoadedModel:
    session: object
    provider: str
    latency_ms: float
    input_name: str
    output_names: list[str]

    def run(self, x: np.ndarray) -> list[np.ndarray]:
        return self.session.run(self.output_names, {self.input_name: x})


def load_session(
    path: Path,
    sample: np.ndarray,
    prefer: str = "auto",
    threads: int = 0,
    benchmark_runs: int = 3,
    log=print,
) -> LoadedModel:
    """Create the fastest working session for ``sample``-shaped inputs."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        options.intra_op_num_threads = threads
    best: LoadedModel | None = None
    tried = set()
    for label, providers in _provider_candidates(prefer):
        if label in tried:
            continue
        try:
            session = ort.InferenceSession(
                str(path), sess_options=options, providers=providers
            )
            name = session.get_inputs()[0].name
            outputs = [o.name for o in session.get_outputs()]
            session.run(outputs, {name: sample})  # compile + warm up
            start = time.perf_counter()
            for _ in range(benchmark_runs):
                session.run(outputs, {name: sample})
            latency = (time.perf_counter() - start) / benchmark_runs * 1000
        except Exception as exc:  # provider or option not supported here
            log(f"{Path(path).name}: {label} unavailable ({exc.__class__.__name__}).")
            continue
        tried.add(label)
        log(f"{Path(path).name}: {label} {latency:.1f} ms/run")
        if best is None or latency < best.latency_ms * 0.9:
            best = LoadedModel(session, label, latency, name, outputs)
    if best is None:
        raise RuntimeError(f"Unable to load {path}.")
    return best
