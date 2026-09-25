# Real-time multi-camera desktop app

Mocap Studio Live processes synchronised videos from 2–6 calibrated cameras. It
shows the 2D skeleton in every view and the metric 3D skeleton while it works.
When processing finishes, it smooths the whole recording using both earlier and
later frames. You can replay that result and save it to your computer.

![Real 4-camera 3D demo](media/realtime-3d-demo.gif)

This demo ([MP4](media/realtime-3d-demo.mp4)) is the public Pose2Sim
single-person capture: 4 synchronized cameras, 1080×1920 at 60 fps, Qualisys
calibration. It was processed by this engine and rendered from the saved
`pose3d.json`. All 100 frames have every joint reconstructed, the median
reprojection error is 12.7 px, and limb lengths vary by 1.5–5 % across the
take. See the README for the commands that reproduce it.

Headless tools that use the same engine as the app:

| Script | Purpose |
|---|---|
| `scripts/process_capture.py` | Process calibration + videos without the UI and save the results |
| `scripts/render_3d_demo.py` | Render camera views + 3D skeleton from a saved result |
| `scripts/import_qualisys_calibration.py` | Convert a Qualisys `.qca` calibration to `mocap_calibration_v1` |
| `scripts/pose2d_video.py` | Single-camera 2D pose overlay video (no calibration needed) |

## Start

```bash
python3 scripts/manage.py setup        # once: dependencies, models, desktop window, .app
python3 scripts/manage.py desktop      # or double-click dist/Mocap Studio.app
```

`setup` installs `pywebview`, which gives the app a native macOS window, and
builds `dist/Mocap Studio.app`. Run `build-app --install` to copy the app to
`~/Applications`. If pywebview is not installed, the app opens in a Chrome or
Edge app window, or in your default browser. The engine only accepts
connections from this computer (`127.0.0.1`). The old offline review tool is
still at `manage.py start`.

## Workflow

1. **Calibration.** Choose a `mocap_calibration_v1` JSON or an NPZ calibration,
   for example one made by the chessboard builder in the offline tool. A video
   may have a lower resolution than its calibration if it is a uniformly scaled
   copy with the same aspect ratio. In that case the camera parameters are
   rescaled automatically.
2. **Videos.** Choose one video per calibrated camera. All videos must have the
   same constant frame rate. If a camera started recording early, use its
   `sync +N` field to skip its first N frames.
3. **Accuracy.**
   - Pick a model preset. Presets that are not installed show a Download
     button.
   - **Flip test-time augmentation**: each crop is also run mirrored and the two
     results are averaged. This is more accurate and costs about twice the
     compute.
   - **Motion responsiveness**: sets how quickly the temporal model follows
     sudden motion. Lower values give a smoother result; higher values react
     faster.
4. **Start.** Every source frame is processed. None are skipped.
5. **Replay smoothed** shows the final result. **Save** writes it to a folder
   (default `~/Documents/MocapStudio/Exports`).

## Pipeline

```text
decoders (1 thread per camera, prefetching)
        │ synchronised frame groups
inference ── per-camera tracker crops ── batched RTMPose (CoreML GPU/ANE or CPU)
   │  └─ async YOLOX detector (staggered checks, immediate on track loss)
fusion ── confidence → 2D σ ── left/right repair ── robust triangulation
       ── 3D Kalman filter ── limb lengths ── reprojection ── preview JPEGs
        │ ring buffer
UI stream (adaptive jitter buffer) ── camera overlays + three.js 3D view
finish ── RTS smoother over the whole take ── replay / save
```

| Stage | What makes it accurate or fast |
|---|---|
| Execution provider | Each model is benchmarked on CoreML (Apple GPU/Neural Engine), CUDA or DirectML against the CPU when it loads. The faster one is used. Compiled CoreML models are cached. |
| Tracking | Each camera's crop is predicted from the previous keypoints plus their velocity, with extra margin for fast motion such as sprinting. The detector runs asynchronously and never blocks the frame loop. |
| Actor selection | On the first frame, the actor is chosen by cross-view triangulation consistency. When a camera loses the actor, it re-acquires the person whose box matches the projected 3D pose. |
| 2D decoding | Crops are converted BGR→RGB to match training. SimCC peaks are refined to sub-bin precision. Flip TTA is optional. |
| Triangulation | All joints are solved at once with a weighted DLT followed by a Gauss–Newton pixel-error refinement. If one camera disagrees, that view is dropped and the joint re-solved; the whole joint is not rejected. Each joint gets a 3×3 covariance, so the filter trusts narrow-angle depth less. |
| Left/right repair | A camera's leg or arm labels are swapped when that clearly matches the predicted 3D pose better. This fixes the usual leg confusion while running. |
| Temporal model | A constant-acceleration Kalman filter per joint uses measurement covariance and chi-square outlier gating. Gaps up to 0.3 s are bridged. |
| Final smoothing | A Rauch–Tung–Striebel smoother uses past and future frames. Higher capture frame rates improve it further. |
| Playback | The UI keeps a jitter buffer. If processing is slower than real time, playback slows down smoothly instead of freezing. Choose "Follow live edge" to always see the newest frame. |

## Accuracy evidence

Run `python3 scripts/manage.py benchmark --cams 4 --fps 60` to reproduce these
numbers. The synthetic runner is ground truth. The 2D error model is
pessimistic: σ = 2.5 px Gaussian noise, 3 % outliers of 20–90 px, 3 % dropouts,
and bursts of left/right leg swaps.

| 4 cameras, 60 fps | 3D coverage | Mean error | p95 | Jitter |
|---|---|---|---|---|
| MVP per-frame solver (`backend.geometry`) | 91.8 % | 8.8 mm | 16.8 mm | 26.4 mm |
| Real-time triangulation | 100 % | 9.2 mm | 16.3 mm | 37.4 mm |
| + left/right repair | 100 % | 8.9 mm | 16.0 mm | 34.7 mm |
| + live Kalman filter (what you see live) | 100 % | 7.0 mm | 13.0 mm | 13.1 mm |
| + RTS smoothing (what is saved) | 100 % | **3.7 mm** | **7.1 mm** | **2.4 mm** |

The MVP solver took about 650 ms per frame and the real-time triangulation took
about 3 ms per frame, both on a 2-core cloud CPU. With 6 cameras at 120 fps the
saved result reaches 2.2 mm mean error.

**Limits of this evidence.** The benchmark tests the geometry and temporal
model with a known error model. It does not measure the 2D network on real
people. Real footage adds model bias, synchronisation error and calibration
error. Before quoting accuracy for your rig, validate against a reference such
as a marker system or known segment lengths.

The end-to-end test (`tests/test_realtime_engine.py`) runs the real decoding,
tracking, fusion, streaming, replay and export code on synthetic multi-camera
videos, using stand-in models so it runs without weights.

## Speed

Speed depends on the machine, the camera count and the preset. The app reports
the processing frame rate and the real-time factor live.

- Measured on a 2-core cloud CPU with no GPU, Balanced preset, 4 cameras at
  720p: about 4 fps.
- On Apple Silicon the CoreML path runs the pose network on the GPU/Neural
  Engine. Treat that speed as unverified until you run the app on your Mac.
- If throughput is below the source frame rate, try these in order: the
  Balanced preset instead of RTMPose-X; flip TTA off; a smaller preview width;
  fewer cameras.

## Outputs

| File | Content |
|---|---|
| `pose3d.json` | Per frame: smoothed 3D joints (m), 3D σ (mm), raw and live 3D, 2D keypoints and scores per camera, reprojections, views used. Also includes calibration, skeleton and quality report. |
| `pose3d.csv` | Smoothed 3D joints plus σ per joint (calibration world frame, meters). |
| `pose2d_camN.csv` | Detected 2D keypoints and scores, plus the reprojected smoothed 3D pose. |
| `pose3d.trc` | Marker trajectories for OpenSim or Blender, converted to Y-up millimetres. |
| `camN_overlay.mp4` | Optional: each video with the final skeleton drawn on it. |
| `report.json` | Settings, providers, coverage, reprojection error, bone-length stability. |

## Known limits

- Tracks one actor. Other people are ignored, apart from the initial selection
  and re-acquisition described above.
- The videos must already be synchronised with a constant frame rate. The app
  does not estimate sub-frame offsets.
- Only uploaded video files are supported. Live camera capture (USB, GigE, NDI)
  is not yet supported; the pipeline is designed so a live source can replace
  the file decoders.
- The output is joint positions. Joint rotations and BVH/FBX retargeting are not
  included yet.
- `pywebview` is not in the locked requirement set and is installed at setup
  time. Checkpoints that are not pinned in the code have their hash recorded at
  first download.
