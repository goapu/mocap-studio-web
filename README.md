# Mocap Studio Web · local MVP

A complete local workflow for **multi-camera pose estimation, calibrated 3D skeleton reconstruction, and human joint correction**.

The frontend is React/TypeScript with Three.js. A Python/FastAPI backend runs RTMPose-M and YOLOX-M through ONNX Runtime. Images, model inference, annotations, and saved sessions remain on the workstation. No desktop GUI framework or cloud inference service is required.

## What works

- Import 2–6 synchronized camera videos or numbered image sequences.
- Build chessboard calibration, or import consistent JSON/NPZ calibration from the original MocapStudio project.
- Run real RTMPose inference and inspect 17 COCO body landmarks per person.
- Select matching people explicitly when multiple people appear in a camera frame.
- Triangulate using distortion-aware geometry, robust inlier selection, positive-depth checks, and a minimum triangulation angle.
- Drag joints, zoom camera views, place missing landmarks, exclude observations, restore predictions, and undo corrections.
- Update the orbitable 3D skeleton after each edit. All usable cameras participate, including cameras not currently displayed.
- Save annotations automatically; rerun detection without erasing manual labels.
- Recover interrupted inference jobs and resume unfinished frames after restarting.
- Rename, archive, and recover saved sessions; restore the last workspace and reviewed frame after a reload.
- Review flagged frames and play the original timestamped sequence.
- Export 3D motion JSON; save/restore project ZIPs containing images, calibration, predictions, and labels.
- Explore a deterministic synthetic demo with an intentionally misaligned wrist.

## Quick start

From the repository root:

Use **Python 3.12 or 3.13**, Node.js 22+ and npm. The validated development machine used Python 3.13.5 and Node.js 24.14.1 on macOS arm64.

```bash
python3 scripts/manage.py setup
python3 scripts/manage.py doctor
python3 scripts/manage.py start
```

Open **http://127.0.0.1:8765**. On Windows, use `python` if that is your Python command. Setup installs the locked Python environment, builds the browser UI and downloads verified weights. `start` binds only to the local workstation. Use `start --port 8766` if another program already uses port 8765.

For an editor-only installation with the synthetic demo, run `setup --skip-models`. Install the models later with `.venv/bin/python scripts/download_models.py` (Windows: `.venv\Scripts\python.exe`). Internet access is needed during setup; operation is local afterward.

<details>
<summary>Equivalent manual setup</summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python scripts/download_models.py
npm --prefix frontend ci
npm --prefix frontend run build
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

On Windows, activate with `.venv\Scripts\Activate.ps1` instead.

</details>

Model downloads are needed once and cached under `.local/models/` by default. The downloader verifies the ONNX weights against the recorded SHA-256 hashes. The combined downloads are approximately 140 MB compressed. You can run the synthetic demo without downloading models.

The UI and the API share the same local server. Keep the terminal running while using the application. There is no automatic startup service.

### Development

Use two terminals with the virtual environment active:

```bash
# Terminal 1, repository root
python -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8765

# Terminal 2, repository root
npm --prefix frontend run dev
```

Open Vite's displayed local URL. The dev server proxies `/api` to port 8765. React fast refresh updates the UI while editing.

```bash
python3 scripts/manage.py test

# Optional contributor lint/format checks
python -m pip install -r requirements-dev.txt
ruff check backend scripts tests
ruff format --check backend scripts tests
```

## First session

1. **Try the demo.** Open Saved sessions → New synthetic demo, then Inspect frame 20. In cam2, drag the right wrist onto its reference wrist or orange reprojection cross. Confirm that the inspector includes cam2 and the joint becomes Corrected. Zoom in for precise placement.
2. **Calibrate your actual rig.** Use Camera calibration with at least 12 synchronized chessboard images per camera. Enter the number of inner corners and the measured square size in meters. Intrinsic/stereo RMS must be at most 1.5 px; per-view fit and board-pose diversity are also checked. An accepted calibration is downloaded and attached to the next import.
3. **Import your capture.** Choose one synchronized video or an image sequence for each camera, plus its calibration. Do not use the demo's camera geometry for real footage.
4. **Run pose estimation.** CPU inference runs in a background job with progress and cancellation. If a frame contains multiple people, use the camera-panel selector to choose the same actor in each view. Automatic cross-camera identity tracking is not implemented.
5. **Review and correct.** Select a joint, drag its handle, use the precise pixel editor, or choose Place missing joint and click the correct image location. Orange crosses show projections of the current 3D solution. Conflicting manual labels stay saved and cause the 3D joint to be marked missing for review; they are never silently excluded from the solve.
6. **Save/export.** Edits save immediately. Save project ZIP makes a portable backup; Export 3D motion downloads the timestamped skeleton. Saved sessions reopen locally. Rename sessions to identify captures; Archive hides a session reversibly and Archived sessions can be restored. Restore project ZIP preserves labels and resets the undo history.
7. **Resume interrupted work.** Reopening a session restores its active job. After a failed job or server restart, Resume processes unfinished frames. A cancellation keeps completed frames and discards work in the current partial frame.

### Capture requirements

- Videos must be synchronized, constant-FPS, and aligned at frame zero. This release does not measure or repair synchronization. Variable-frame-rate videos should be converted to constant FPS first.
- All video cameras in a session must report matching FPS. Camera image resolution must match that camera's calibration exactly.
- For images, provide the true source FPS and shared numbered filenames such as `frame001.jpg`, `frame004.jpg`. Gaps remain gaps in time. Files can be selected separately for each camera, so camera labels do not have to appear in filenames.
- MVP limits: 300 sampled frames, 500 MB per upload, and 450 MB of extracted camera images per session. Video stride reduces sampling density while preserving original frame timestamps.
- The chessboard builder currently requires equal image dimensions across cameras. Imported calibrated cameras may have different resolutions.
- Keep one intended actor in the capture volume when possible. Multiple-person selections are per frame and require operator review.

## Data and accuracy boundaries

This MVP delivers the complete local operator workflow. It is not yet a metrologically validated capture system. Real RTMPose 2D inference was verified on a public image; real synchronized multi-camera human capture accuracy has not yet been benchmarked.

3D positions use calibrated multi-view geometry, not an inferred single-camera body shape. A joint needs at least two usable views. Missing joints stay `null`; there is no automatic gap filling, motion smoothing, bone-length enforcement, or foot locking. This preserves evidence and avoids introducing the timing/contact defects found in the original scripts.

Reprojection error measures consistency with image observations. It does not certify depth accuracy, actor identity, synchronization, anatomical correctness, or calibration coverage. Inspect accepted/excluded views as well as the rendered skeleton.

The output is COCO-17 body pose plus derived pelvis, neck, spine, and head positions. Detailed fingers, heels/toes, and production-character retargeting are outside this release. A head-to-neck segment is displayed using the shoulder midpoint; it is a visual derived segment, not a separately detected landmark.

The 3D grid is a reference aid. With `world_frame: camera`, its height follows the observed skeleton and it is **not a measured floor**. A `z_up` calibration uses the supplied world Z=0 plane.

## Repository layout

```text
backend/                  FastAPI routes, calibration, geometry, model adapter, storage
frontend/src/             Typed React editor, camera interaction, Three.js viewer
tests/                    Geometry and workflow regression tests
scripts/                  Setup/start/doctor, model downloader, inference smoke check
docs/                     Architecture, calibration format, validation evidence
examples/                 Calibration schema example (not a real rig calibration)
.github/workflows/ci.yml   Backend/frontend tests, lint, formatting and build
requirements.txt          Direct dependency versions
requirements.lock.txt     Tested complete Python dependency set
requirements-dev.txt      Contributor lint dependency
frontend/package-lock.json
.local/                   Ignored recordings, saved sessions, model weights
```

The original desktop source is preserved outside this new repository. This implementation uses one reconstruction backend for import, inference, edits, quality reporting, and export.

The repository has an initial local MVP commit on `main`, with no remote configured. Recordings, model binaries, environments, caches, and generated builds are ignored. Review the source and choose your project's license before publishing or connecting a remote. No repository was published.

## Release evidence and operation

- [MVP scope and acceptance criteria](docs/MVP_ACCEPTANCE.md)
- [Validation results](docs/VALIDATION.md)
- [Architecture and data invariants](docs/ARCHITECTURE.md)
- [Calibration schema and capture guidance](docs/CALIBRATION.md)
- [Operator troubleshooting](docs/OPERATIONS.md)

## Configuration

`MOCAP_DATA_DIR` and `MOCAP_MODEL_DIR` optionally relocate session and model storage. `MOCAP_PYTHON` optionally selects an existing environment for the launcher. Set overrides as environment variables; `.env.example` documents them but is not loaded automatically.

The server is intended for a **single local workstation** and binds to loopback. It rejects browser requests with nonlocal origins. It has no authentication, multi-user isolation, remote deployment configuration, or public hosting setup.

## Model choice and provenance

RTMPose-M is a practical CPU baseline with an explicit ONNX path and established body-pose implementation. It is not claimed to be universally the most accurate open model. YOLOX-M proposes person boxes; RTMPose-M estimates the joints. The detector is isolated behind an adapter for future benchmarked replacements.

See [RTMPose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose), [RTMLib](https://github.com/Tau-J/rtmlib), and [third-party notices](THIRD_PARTY_NOTICES.md). Code and checkpoint provenance are distinct; retain upstream notices and review checkpoint terms for your intended distribution.
