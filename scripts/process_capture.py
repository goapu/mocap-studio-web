"""Process a multi-camera capture without the UI and save the results.

Runs exactly the same engine as the desktop app (tracking, 2D pose, robust
triangulation, Kalman filter, full-take RTS smoothing) and writes the same
files as the app's Save button.

    python scripts/process_capture.py calibration.json out_dir \
        --video cam1=cam01.mp4 --video cam2=cam02.mp4 --video cam3=cam03.mp4 --flip
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.calibration import read_calibration  # noqa: E402
from backend.realtime.engine import Run, RunConfig  # noqa: E402
from backend.realtime.export import Exporter  # noqa: E402
from backend.realtime.models import PRESETS, ModelStore  # noqa: E402


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("calibration", type=Path)
    p.add_argument("output", type=Path, help="folder for the results")
    p.add_argument("--video", action="append", required=True, help="camN=path")
    p.add_argument("--name", default="capture")
    p.add_argument("--preset", default="balanced", choices=list(PRESETS))
    p.add_argument("--flip", action="store_true", help="flip test-time augmentation")
    p.add_argument("--provider", default="auto", choices=["auto", "gpu", "cpu"])
    p.add_argument("--overlay-videos", action="store_true")
    a = p.parse_args()

    calibration = read_calibration(a.calibration.read_bytes(), a.calibration.name)
    videos = dict(item.split("=", 1) for item in a.video)
    run = Run(
        RunConfig(
            videos=videos,
            calibration=calibration,
            name=a.name,
            preset=a.preset,
            flip_test=a.flip,
            provider=a.provider,
        ),
        ModelStore(),
    ).start()
    last = -1
    while run.status in ("starting", "running", "finalizing"):
        time.sleep(0.5)
        s = run.stats
        if s["frames_done"] != last and run.status == "running":
            last = s["frames_done"]
            print(
                f"\r{last}/{s['total_frames']} frames · {s['processing_fps']:.1f} fps",
                end="",
                flush=True,
            )
    print()
    if run.status == "error":
        raise SystemExit(f"Failed: {run.error}")
    q = run.final["quality"]
    print(
        f"{q['frames']} frames · {q['coverage_smoothed_pct']}% joints · median "
        f"reprojection {q['median_reprojection_px_smoothed']} px · "
        f"{run.stats['processing_fps']:.1f} fps on {run.stats['provider']}"
    )
    exporter = Exporter(
        run, a.output.resolve(), {"videos": a.overlay_videos, "csv": True, "trc": True}
    ).start()
    exporter.thread.join()
    if exporter.state["status"] != "done":
        raise SystemExit(f"Save failed: {exporter.state.get('message')}")
    print(f"Saved to {exporter.state['folder']}")
    print(json.dumps({"folder": exporter.state["folder"]}))


if __name__ == "__main__":
    main()
