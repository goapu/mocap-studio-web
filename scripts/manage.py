"""Portable setup, diagnostics, and launch commands for Mocap Studio."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
PYTHON = Path(
    os.getenv(
        "MOCAP_PYTHON",
        str(VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")),
    )
)


def run(args, **kwargs):
    return subprocess.run([str(a) for a in args], cwd=ROOT, check=True, **kwargs)


def require_python():
    if not PYTHON.is_file():
        raise RuntimeError(
            "Python environment is missing. Run: python3 scripts/manage.py setup"
        )


def setup(skip_models=False):
    if sys.version_info[:2] not in [(3, 12), (3, 13)]:
        raise RuntimeError("Use Python 3.12 or 3.13 for the locked dependency set.")
    npm = shutil.which("npm")
    node = shutil.which("node")
    if not npm or not node:
        raise RuntimeError(
            "Install Node.js 22 or newer (including npm), then run setup again."
        )
    version = subprocess.check_output([node, "--version"], text=True).strip()
    if int(version.lstrip("v").split(".")[0]) < 22:
        raise RuntimeError(f"Node.js 22+ is required; found {version}.")
    if not PYTHON.is_file():
        run([sys.executable, "-m", "venv", VENV])
    run([PYTHON, "-m", "pip", "install", "-r", "requirements.lock.txt"])
    run([npm, "--prefix", "frontend", "ci"])
    run([npm, "--prefix", "frontend", "run", "build"])
    if not skip_models:
        run([PYTHON, "scripts/download_models.py"])
    print("\nSetup complete. Start with: python3 scripts/manage.py start")


def doctor():
    checks = []

    def check(label, good, detail):
        checks.append(bool(good))
        print(f"{'OK' if good else 'FAIL'}  {label}: {detail}")

    check("Python environment", PYTHON.is_file(), str(PYTHON))
    if PYTHON.is_file():
        result = subprocess.run(
            [
                str(PYTHON),
                "-c",
                'import cv2, numpy, scipy, fastapi, uvicorn, rtmlib, onnxruntime; print("all runtime imports succeeded")',
            ],
            capture_output=True,
            text=True,
        )
        check(
            "Runtime dependencies",
            result.returncode == 0,
            result.stdout.strip() or result.stderr.strip(),
        )
        result = subprocess.run(
            [str(PYTHON), "-m", "pip", "check"], capture_output=True, text=True
        )
        check(
            "Dependency compatibility",
            result.returncode == 0,
            result.stdout.strip() or result.stderr.strip(),
        )
    check(
        "Browser build",
        (ROOT / "frontend/dist/index.html").is_file(),
        "frontend/dist/index.html",
    )
    model_dir = Path(os.getenv("MOCAP_MODEL_DIR", str(ROOT / ".local/models")))
    for name, spec in json.loads((ROOT / "models.lock.json").read_text()).items():
        target = model_dir / spec["file"]
        digest = None
        if target.is_file():
            with target.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
        check(
            name + " weights",
            digest == spec["sha256"],
            str(target)
            if target.is_file()
            else "missing; run scripts/download_models.py",
        )
    data_dir = Path(os.getenv("MOCAP_DATA_DIR", str(ROOT / ".local/sessions")))
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        import tempfile

        with tempfile.TemporaryFile(dir=data_dir) as stream:
            stream.write(b"storage-check")
        check("Session storage", True, str(data_dir))
    except OSError as exc:
        check("Session storage", False, str(exc))
    return 0 if all(checks) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    install = commands.add_parser(
        "setup",
        help="Install locked dependencies, build UI, and download verified models",
    )
    install.add_argument(
        "--skip-models",
        action="store_true",
        help="Install only the editor and synthetic demo",
    )
    serve = commands.add_parser("start", help="Run the local browser workstation")
    serve.add_argument("--port", type=int, default=8765)
    commands.add_parser(
        "doctor", help="Check dependencies, build, model hashes, and writable storage"
    )
    commands.add_parser(
        "test", help="Run backend tests and frontend format/build checks"
    )
    args = parser.parse_args()
    try:
        if args.command == "setup":
            setup(args.skip_models)
        elif args.command == "doctor":
            return doctor()
        elif args.command == "start":
            require_python()
            if not 1024 <= args.port <= 65535:
                raise RuntimeError("Choose a port between 1024 and 65535.")
            if not (ROOT / "frontend/dist/index.html").is_file():
                raise RuntimeError(
                    "Browser build is missing. Run: python3 scripts/manage.py setup"
                )
            print(
                f"Open http://127.0.0.1:{args.port} — press Ctrl+C to stop.", flush=True
            )
            run(
                [
                    PYTHON,
                    "-m",
                    "uvicorn",
                    "backend.app:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.port),
                ]
            )
        elif args.command == "test":
            require_python()
            npm = shutil.which("npm")
            if not npm:
                raise RuntimeError("npm is required for frontend checks.")
            run([PYTHON, "-m", "pytest"])
            run([npm, "--prefix", "frontend", "run", "format:check"])
            run([npm, "--prefix", "frontend", "test"])
            run([npm, "--prefix", "frontend", "run", "build"])
    except KeyboardInterrupt:
        return 0
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
