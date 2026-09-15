"""Download published RTMPose/YOLOX ONNX checkpoints to the local model folder."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backend.detector import MODELS


def verify(file, expected):
    with file.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(
            f"Checksum mismatch for {file.name}. Remove this file and retry the download."
        )


def download(model_dir, name, spec):
    target = model_dir / spec["file"]
    if target.is_file():
        verify(target, spec["sha256"])
        return
    print(f"Downloading {name}…", flush=True)
    # Extract only the named ONNX member; never trust archive paths.
    with tempfile.TemporaryDirectory(dir=model_dir) as tmp:
        archive = Path(tmp) / "checkpoint.zip"
        with urlopen(spec["source"], timeout=120) as source, archive.open("wb") as out:
            shutil.copyfileobj(source, out)
        with zipfile.ZipFile(archive) as bundle:
            matches = [
                m for m in bundle.infolist() if Path(m.filename).name == spec["file"]
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"{name}: checkpoint archive does not contain the expected ONNX file."
                )
            extracted = Path(tmp) / spec["file"]
            with bundle.open(matches[0]) as source, extracted.open("wb") as out:
                shutil.copyfileobj(source, out)
        verify(extracted, spec["sha256"])
        extracted.replace(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.getenv("MOCAP_MODEL_DIR", ROOT / ".local" / "models")),
    )
    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "models.lock.json").read_text())
    for name, url in MODELS.items():
        spec = manifest[name]
        if spec["source"] != url or spec["file"] != Path(url).stem + ".onnx":
            raise ValueError(f"{name}: model adapter and lock file disagree.")
        download(args.model_dir, name, spec)
    (args.model_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Models ready:", args.model_dir)


if __name__ == "__main__":
    main()
