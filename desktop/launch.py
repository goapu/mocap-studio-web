"""Mocap Studio desktop app.

Starts the local engine on a free loopback port and opens the UI in a native
window (pywebview → WKWebView on macOS). Without pywebview it falls back to a
Chrome/Edge app window or the default browser. Closing the window stops the
engine.

    python -m desktop.launch            # native window
    python -m desktop.launch --browser  # open in the default browser
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if sys.platform == "darwin":
    LOG_FILE = Path.home() / "Library" / "Logs" / "MocapStudio.log"
else:
    LOG_FILE = ROOT / ".local" / "mocap-studio.log"

VIDEO_TYPES = "Video files (*.mp4;*.mov;*.m4v;*.avi;*.mkv)"
CAL_TYPES = "Calibration (*.json;*.npz)"


def free_port(preferred: int) -> int:
    for port in (preferred, 0):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return sock.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("No free local port.")


def start_server(port: int):
    import uvicorn

    os.environ["MOCAP_DESKTOP"] = "1"
    config = uvicorn.Config(
        "backend.app:app",
        host="127.0.0.1",
        port=port,
        log_level="warning",
        timeout_graceful_shutdown=3,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="engine")
    thread.start()
    url = f"http://127.0.0.1:{port}/live/"
    for _ in range(300):
        try:
            with urllib.request.urlopen(url + "api/system", timeout=1):
                return server, thread, url
        except OSError:
            if not thread.is_alive():
                raise RuntimeError(f"Engine failed to start; see {LOG_FILE}") from None
            time.sleep(0.1)
    raise RuntimeError("Engine did not start within 30 s.")


class DesktopApi:
    """Methods callable from JavaScript as window.pywebview.api.*"""

    def __init__(self):
        self._window = None

    def _dialog(self, kind, **kwargs):
        import webview

        dialogs = getattr(webview, "FileDialog", None)
        mode = {
            "open": dialogs.OPEN if dialogs else webview.OPEN_DIALOG,
            "folder": dialogs.FOLDER if dialogs else webview.FOLDER_DIALOG,
        }[kind]
        result = self._window.create_file_dialog(mode, **kwargs)
        if not result:
            return None
        return result[0] if isinstance(result, (list, tuple)) else result

    def pick_file(self, kind="video"):
        types = (CAL_TYPES,) if kind == "calibration" else (VIDEO_TYPES,)
        return self._dialog(
            "open", allow_multiple=False, file_types=types + ("All files (*.*)",)
        )

    def pick_folder(self):
        return self._dialog("folder")


def open_browser(url: str):
    if sys.platform == "darwin":
        for app in ("Google Chrome", "Microsoft Edge", "Brave Browser"):
            if Path(f"/Applications/{app}.app").exists():
                subprocess.Popen(["open", "-na", app, "--args", f"--app={url}"])
                return
        subprocess.Popen(["open", url])
        return
    for exe in ("google-chrome", "chromium", "msedge", "chrome"):
        path = shutil.which(exe)
        if path:
            subprocess.Popen([path, f"--app={url}"])
            return
    webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser(description="Mocap Studio desktop app")
    parser.add_argument("--port", type=int, default=int(os.getenv("MOCAP_PORT", 8765)))
    parser.add_argument("--browser", action="store_true", help="Use a browser window")
    args = parser.parse_args()

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.info("Starting Mocap Studio desktop from %s", ROOT)
    os.chdir(ROOT)
    server, thread, url = start_server(free_port(args.port))
    logging.info("Engine listening at %s", url)

    webview = None
    if not args.browser:
        try:
            import webview  # type: ignore
        except ImportError:
            logging.warning("pywebview not installed; using a browser window")
    try:
        if webview is not None:
            api = DesktopApi()
            window = webview.create_window(
                "Mocap Studio",
                url,
                js_api=api,
                width=1600,
                height=1000,
                min_size=(1100, 720),
                background_color="#0d1014",
                text_select=False,
            )
            api._window = window
            webview.start(private_mode=False)
        else:
            open_browser(url)
            print(
                f"Mocap Studio is running at {url} — press Ctrl+C to quit.", flush=True
            )
            while thread.is_alive():
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        logging.info("Stopped")


if __name__ == "__main__":
    main()
