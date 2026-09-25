"""Local HTTP API and UI for the real-time pipeline (Starlette).

Mounted by ``backend.app`` at ``/live`` and opened in a native desktop window
by ``desktop/launch.py``. Binds to loopback only; requests with a foreign Host
or Origin header are refused (protects against DNS rebinding and CSRF).
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .engine import VERSION, Run, RunConfig
from .export import DEFAULT_EXPORT_DIR, Exporter
from .models import MODELS, PRESETS, ModelStore, available_providers

UI_DIR = Path(__file__).resolve().parent / "ui"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "testserver"}


class LocalOnly:
    """Pure ASGI middleware (keeps streaming responses unbuffered)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = {
                k.decode().lower(): v.decode() for k, v in scope.get("headers", [])
            }
            host = headers.get("host", "")
            hostname = urlparse(f"//{host}").hostname or ""
            origin = headers.get("origin")
            bad_origin = (
                origin
                and origin != "null"
                and urlparse(origin).hostname not in LOCAL_HOSTS
            )
            if hostname not in LOCAL_HOSTS or bad_origin:
                response = JSONResponse(
                    {"detail": "Local access only."}, status_code=403
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def _error(message, status=400):
    return JSONResponse({"detail": message}, status_code=status)


def _read_calibration_file(path: str) -> dict:
    from backend.calibration import read_calibration

    p = Path(path).expanduser()
    if not p.is_file():
        raise ValueError(f"Calibration file not found: {path}")
    return read_calibration(p.read_bytes(), p.name)


class State:
    def __init__(self, store: ModelStore, desktop: bool):
        self.store = store
        self.desktop = desktop
        self.runs: dict[str, Run] = {}
        self.exports: dict[str, Exporter] = {}
        self.upload_dir = Path(tempfile.mkdtemp(prefix="mocap-live-uploads-"))
        self.lock = threading.Lock()

    def run(self, rid) -> Run:
        run = self.runs.get(rid)
        if run is None:
            raise KeyError(rid)
        return run

    def add(self, run: Run):
        with self.lock:
            for old in list(self.runs.values()):
                if old.status in ("running", "starting", "finalizing"):
                    old.stop()
            # Keep a few finished runs so results can still be saved.
            finished = [
                r for r in self.runs.values() if r.status not in ("running", "starting")
            ]
            for old in sorted(finished, key=lambda r: r.created)[:-3]:
                self.runs.pop(old.id, None)
            self.runs[run.id] = run


def create_app(store: ModelStore | None = None, desktop: bool = False) -> Starlette:
    state = State(store or ModelStore(), desktop)

    async def index(request):
        return FileResponse(
            UI_DIR / "index.html", headers={"Cache-Control": "no-store"}
        )

    async def system(request):
        try:
            providers = available_providers()
        except Exception as exc:  # onnxruntime missing
            providers = [f"unavailable: {exc}"]
        return JSONResponse(
            {
                "version": VERSION,
                "platform": f"{platform.system()} {platform.machine()}",
                "python": sys.version.split()[0],
                "providers": providers,
                "gpu": any(
                    p in providers
                    for p in (
                        "CoreMLExecutionProvider",
                        "CUDAExecutionProvider",
                        "DmlExecutionProvider",
                    )
                ),
                "desktop": state.desktop,
                "models": state.store.status(),
                "presets": PRESETS,
                "export_dir": str(DEFAULT_EXPORT_DIR),
                "runs": [r.summary() | {"log": None} for r in state.runs.values()],
            }
        )

    async def download_model(request):
        key = request.path_params["key"]
        if key not in MODELS:
            return _error("Unknown model.", 404)
        if (state.store.progress.get(key) or {}).get("state") == "downloading":
            return JSONResponse({"started": False})
        threading.Thread(
            target=lambda: _safe(state.store.download, key), daemon=True
        ).start()
        return JSONResponse({"started": True})

    async def upload(request: Request):
        name = request.headers.get("x-filename", "upload.bin")
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name)[-120:] or "upload.bin"
        target = state.upload_dir / f"{int(time.time() * 1000)}_{safe}"
        size = 0
        with target.open("wb") as out:
            async for chunk in request.stream():
                size += len(chunk)
                if size > 20 * 1024**3:
                    out.close()
                    target.unlink(missing_ok=True)
                    return _error("File is larger than 20 GB.", 413)
                out.write(chunk)
        return JSONResponse({"path": str(target), "bytes": size})

    async def calibration(request: Request):
        body = await request.json()
        try:
            cal = _read_calibration_file(body.get("path", ""))
        except ValueError as exc:
            return _error(str(exc))
        return JSONResponse(
            {
                "cameras": {
                    n: {"image_size": c["image_size"]}
                    for n, c in cal["cameras"].items()
                },
                "world_frame": cal.get("world_frame"),
                "quality": cal.get("quality"),
            }
        )

    async def probe(request: Request):
        from .sources import probe as probe_video

        body = await request.json()
        out = {}
        for name, path in (body.get("videos") or {}).items():
            try:
                out[name] = probe_video(name, path).to_json()
            except ValueError as exc:
                out[name] = {"error": str(exc)}
        return JSONResponse(out)

    async def start_run(request: Request):
        body = await request.json()
        try:
            cal = body.get("calibration") or _read_calibration_file(
                body.get("calibration_path", "")
            )
            if not isinstance(cal, dict):
                raise ValueError("Calibration is required.")
            if body.get("calibration"):
                from backend.calibration import validate

                cal = validate(cal)
            videos = {
                k: str(Path(v).expanduser())
                for k, v in (body.get("videos") or {}).items()
                if v
            }
            options = body.get("options") or {}
            allowed = RunConfig.__dataclass_fields__.keys() - {"videos", "calibration"}
            cfg = RunConfig(
                videos=videos,
                calibration=cal,
                **{k: v for k, v in options.items() if k in allowed},
            )
            run = Run(cfg, state.store)
        except (ValueError, TypeError, KeyError) as exc:
            return _error(str(exc))
        state.add(run)
        run.start()
        return JSONResponse(run.summary())

    def _get_run(request):
        try:
            return state.run(request.path_params["rid"])
        except KeyError:
            return None

    async def run_status(request):
        run = _get_run(request)
        return JSONResponse(run.summary()) if run else _error("Run not found.", 404)

    async def stop_run(request):
        run = _get_run(request)
        if not run:
            return _error("Run not found.", 404)
        run.stop()
        return JSONResponse(run.summary())

    async def stream(request):
        run = _get_run(request)
        if not run:
            return _error("Run not found.", 404)
        start = int(request.query_params.get("from", 0))
        return StreamingResponse(
            run.buffer.iterate(start),
            media_type="application/octet-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    async def replay(request):
        run = _get_run(request)
        if not run:
            return _error("Run not found.", 404)
        if run.final is None:
            return _error("The result is not ready yet.", 409)
        variant = request.query_params.get("variant", "smoothed")
        return StreamingResponse(
            run.replay(variant),
            media_type="application/octet-stream",
            headers={"Cache-Control": "no-store"},
        )

    async def save(request):
        run = _get_run(request)
        if not run:
            return _error("Run not found.", 404)
        if request.method == "GET":
            exp = state.exports.get(run.id)
            return JSONResponse(exp.state if exp else {"status": "none"})
        body = await request.json()
        try:
            exp = Exporter(
                run, body.get("folder") or None, body.get("options") or {}
            ).start()
        except ValueError as exc:
            return _error(str(exc))
        state.exports[run.id] = exp
        return JSONResponse(exp.state)

    async def open_path(request):
        body = await request.json()
        target = Path(body.get("path", "")).expanduser().resolve()
        allowed = {Path(e.state["folder"]).resolve() for e in state.exports.values()}
        if target not in allowed or not target.exists():
            return _error("Only folders saved by this app can be opened.", 403)
        opener = {"darwin": ["open"], "win32": ["explorer"]}.get(
            sys.platform, ["xdg-open"]
        )
        subprocess.Popen(opener + [str(target)])
        return JSONResponse({"opened": True})

    routes = [
        Route("/", index),
        Route("/api/system", system),
        Route("/api/models/{key}/download", download_model, methods=["POST"]),
        Route("/api/upload", upload, methods=["POST"]),
        Route("/api/calibration", calibration, methods=["POST"]),
        Route("/api/probe", probe, methods=["POST"]),
        Route("/api/runs", start_run, methods=["POST"]),
        Route("/api/runs/{rid}", run_status),
        Route("/api/runs/{rid}/stop", stop_run, methods=["POST"]),
        Route("/api/runs/{rid}/stream", stream),
        Route("/api/runs/{rid}/replay", replay),
        Route("/api/runs/{rid}/save", save, methods=["GET", "POST"]),
        Route("/api/open", open_path, methods=["POST"]),
        Mount("/static", StaticFiles(directory=UI_DIR), name="static"),
    ]
    app = Starlette(routes=routes, middleware=[Middleware(LocalOnly)])
    app.state.live = state
    return app


def _safe(fn, *args):
    try:
        fn(*args)
    except Exception as exc:  # progress dict carries the message to the UI
        print(f"background task failed: {exc}", file=sys.stderr)


def main():
    """Standalone server (without the offline review tool)."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("MOCAP_PORT", 8766)))
    args = parser.parse_args()
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
